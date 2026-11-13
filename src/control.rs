//! Pause protocol between the controller (the REPL) and a running script.
//!
//! The script is a separate process, so `/pause` cannot call a Python function
//! directly. Instead the two sides talk through two tiny files in a *control
//! directory* that is private to one run:
//!
//! ```text
//! controller (main.rs)                         script (python.rs -> api.rs -> here)
//! --------------------                         -------------------------------------
//! TC_CONTROL_DIR=<dir>  ── environment ──────► tc.register_pause(pause)
//!                                              starts a listener thread (Rust, no GIL needed)
//! /pause: create <dir>/pause.request ────────► listener sees it, deletes it,
//!                                              writes <dir>/pause.ack, then calls pause()
//! wait for <dir>/pause.ack  ◄────────────────  (the script's own pause method runs)
//! ```
//!
//! * The listener polls every [`POLL_EVERY`]. It runs on a plain Rust thread
//!   and takes the Python GIL only for the moment the pause method runs, so a
//!   busy training loop does not delay the acknowledgement.
//! * The acknowledgement only says "the request arrived"; the pause method may
//!   take longer (saving a checkpoint). What it prints shows up in the `pause`
//!   panel (lines starting with `[pause]`).
//! * Nothing here stops the script. Stopping is the job of the script's pause
//!   method (set a flag the training loop checks, save a checkpoint, exit).
//!   When the script then exits with code 0 the controller records the run as
//!   *paused*, and `/resume` starts it again with `--resume`.
//!
//! This module uses no PyO3 types, so it is tested with plain `cargo test`.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, Once};
use std::thread;
use std::time::{Duration, Instant};

/// Environment variable naming the control directory. `/run_script` and
/// `/resume` set it; a script started by hand does not have it.
pub const DIR_ENV: &str = "TC_CONTROL_DIR";

const REQUEST_FILE: &str = "pause.request";
const ACK_FILE: &str = "pause.ack";

/// How often the script's listener looks for a pause request.
pub const POLL_EVERY: Duration = Duration::from_millis(200);

/// How long `/pause` waits for the script to acknowledge the request.
pub const ACK_TIMEOUT: Duration = Duration::from_secs(3);

const ACK_POLL: Duration = Duration::from_millis(50);

// ---------------------------------------------------------------------------
// Controller side
// ---------------------------------------------------------------------------

/// Creates the (empty) control directory of one run.
pub fn prepare(dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|error| format!("Failed to create {}: {error}", dir.display()))
}

/// Removes the control directory of a run that has ended. Never fails.
pub fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

/// Asks the script to pause: forgets an old acknowledgement, then creates the
/// request file.
pub fn request_pause(dir: &Path) -> Result<(), String> {
    let _ = fs::remove_file(dir.join(ACK_FILE));
    File::create(dir.join(REQUEST_FILE))
        .map(|_| ())
        .map_err(|error| format!("Failed to write the pause request in {}: {error}", dir.display()))
}

/// Waits until the script acknowledged the request. `true` = it did.
pub fn wait_for_ack(dir: &Path, timeout: Duration) -> bool {
    let ack = dir.join(ACK_FILE);
    let started = Instant::now();
    loop {
        if ack.is_file() {
            return true;
        }
        if started.elapsed() >= timeout {
            return false;
        }
        thread::sleep(ACK_POLL);
    }
}

/// Withdraws a request nobody picked up. Returns `true` if the request was
/// still there (so the script will never see it), `false` if the script took
/// it in the meantime.
pub fn cancel_request(dir: &Path) -> bool {
    fs::remove_file(dir.join(REQUEST_FILE)).is_ok()
}

// ---------------------------------------------------------------------------
// Script side
// ---------------------------------------------------------------------------

/// If a pause request is waiting, consumes it, writes the acknowledgement and
/// returns `true`.
pub fn take_pause_request(dir: &Path) -> bool {
    if fs::remove_file(dir.join(REQUEST_FILE)).is_err() {
        return false; // no request (or it could not be taken yet: try again next poll)
    }
    let _ = fs::write(dir.join(ACK_FILE), b"received");
    true
}

type Handler = Arc<dyn Fn() + Send + Sync>;

static HANDLER: Mutex<Option<Handler>> = Mutex::new(None);
static LISTENER: Once = Once::new();

fn handler_slot() -> MutexGuard<'static, Option<Handler>> {
    // A panic elsewhere must not turn every later call into an error.
    HANDLER.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The control directory the controller gave this process, if any.
pub fn dir_from_env() -> Option<PathBuf> {
    std::env::var_os(DIR_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Runs the registered handler once. Panics in the handler are contained so
/// the listener keeps working.
fn run_handler() {
    let handler = handler_slot().clone();
    match handler {
        Some(handler) => {
            println!("[pause] pause requested: calling the script's pause method");
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler()));
            match outcome {
                Ok(()) => println!("[pause] the pause method returned"),
                Err(_) => eprintln!("[error] the pause method panicked"),
            }
        }
        None => println!("[pause] pause requested, but no pause method is registered"),
    }
}

fn listen(dir: PathBuf, every: Duration) {
    loop {
        if take_pause_request(&dir) {
            run_handler();
        }
        thread::sleep(every);
    }
}

/// Registers the function `/pause` runs and makes sure the listener thread is
/// running. Registering again replaces the function (there is still only one
/// listener).
///
/// Returns `Ok(false)` when this process was not started by the controller
/// (there is no control directory): the function is stored but `/pause` cannot
/// reach it.
pub fn register_pause_handler(handler: Handler) -> Result<bool, String> {
    register_in(dir_from_env(), handler, POLL_EVERY)
}

fn register_in(dir: Option<PathBuf>, handler: Handler, every: Duration) -> Result<bool, String> {
    *handler_slot() = Some(handler);
    let Some(dir) = dir else {
        return Ok(false);
    };

    let mut spawn_error: Option<String> = None;
    LISTENER.call_once(|| {
        let spawned = thread::Builder::new()
            .name("tc-pause-listener".into())
            .spawn(move || listen(dir, every));
        if let Err(error) = spawned {
            spawn_error = Some(format!("Failed to start the pause listener: {error}"));
        }
    });
    match spawn_error {
        Some(error) => Err(error),
        None => Ok(true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "training_controller-control-{label}-{}-{}",
            std::process::id(),
            crate::api::now_ms()
        ));
        prepare(&dir).unwrap();
        dir
    }

    #[test]
    fn a_request_is_taken_once_and_acknowledged() {
        let dir = temp_dir("take");
        assert!(!take_pause_request(&dir), "nothing was requested yet");
        assert!(!wait_for_ack(&dir, Duration::from_millis(60)));

        request_pause(&dir).unwrap();
        assert!(take_pause_request(&dir));
        assert!(wait_for_ack(&dir, Duration::from_millis(500)));
        // The request is consumed: a second poll finds nothing.
        assert!(!take_pause_request(&dir));
        cleanup(&dir);
        assert!(!dir.exists());
    }

    #[test]
    fn a_new_request_forgets_the_old_acknowledgement() {
        let dir = temp_dir("stale-ack");
        request_pause(&dir).unwrap();
        assert!(take_pause_request(&dir));
        assert!(wait_for_ack(&dir, Duration::from_millis(100)));

        request_pause(&dir).unwrap();
        assert!(
            !wait_for_ack(&dir, Duration::from_millis(60)),
            "the old acknowledgement must not count for the new request"
        );
        cleanup(&dir);
    }

    #[test]
    fn an_unanswered_request_can_be_cancelled() {
        let dir = temp_dir("cancel");
        request_pause(&dir).unwrap();
        assert!(cancel_request(&dir));
        assert!(!take_pause_request(&dir));
        // Cancelling when the script already took it reports that.
        request_pause(&dir).unwrap();
        assert!(take_pause_request(&dir));
        assert!(!cancel_request(&dir));
        cleanup(&dir);
    }

    #[test]
    fn requesting_in_a_missing_directory_is_an_error() {
        let dir = std::env::temp_dir().join("training_controller-control-missing-dir/none");
        assert!(request_pause(&dir).is_err());
    }

    #[test]
    fn the_listener_runs_the_handler_for_each_request() {
        // The only test that touches the process-wide handler and listener.
        // Without a control directory the handler is only stored.
        let handler: Handler = Arc::new(|| {});
        assert_eq!(register_in(None, handler, Duration::from_millis(10)), Ok(false));

        let dir = temp_dir("listener");
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let handler: Handler = Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(
            register_in(Some(dir.clone()), handler, Duration::from_millis(10)),
            Ok(true)
        );

        request_pause(&dir).unwrap();
        assert!(wait_for_ack(&dir, Duration::from_secs(5)));
        let started = Instant::now();
        while calls.load(Ordering::SeqCst) < 1 && started.elapsed() < Duration::from_secs(5) {
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Registering again keeps the single listener and swaps the function.
        let replaced = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&replaced);
        let handler: Handler = Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(
            register_in(Some(dir.clone()), handler, Duration::from_millis(10)),
            Ok(true)
        );
        request_pause(&dir).unwrap();
        let started = Instant::now();
        while replaced.load(Ordering::SeqCst) < 1 && started.elapsed() < Duration::from_secs(5) {
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(replaced.load(Ordering::SeqCst), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        cleanup(&dir);
    }
}
