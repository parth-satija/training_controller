//! Persistent shell sessions for `/run_command` and `/run_command_wsl`.
//!
//! Each target (native, WSL2) has **one shell that stays open** while the
//! controller runs, so state carries from one command to the next: `cd`,
//! `set` / `export`, an activated virtualenv, and so on. The shell is started
//! with the same environment a script gets (`PYTHONPATH`, the checkpoint
//! folder, the log database, ...) in the active script's folder, and its
//! output is routed into the output panels exactly like a script's output.
//!
//! ```text
//! /run_command ...  ──►  stdin of the shell  ──►  command runs
//!                        stdout / stderr  ◄──  panels (output, error, ...)
//! ```
//!
//! # How a command's end is found
//!
//! A shell does not say when a command is done, so after every command the
//! controller sends a marker line (`__TC_DONE_<run>__<exit code>`). The reader
//! thread removes that line from the output and ends the run with the exit
//! code. Commands run one after another: a command sent while another is still
//! running waits for it (use `&` in a POSIX shell to run something in the
//! background).
//!
//! * POSIX shells (`sh`, and `bash` in WSL) get the command through `eval`, so
//!   a syntax error (such as an unclosed quote) fails that command instead of
//!   swallowing the marker and the commands after it. Their stdin is
//!   `/dev/null`, so a program that reads input cannot steal the next command.
//! * PowerShell (native on Windows) gets the command and the marker on separate
//!   lines, so a parse error in the command cannot swallow the marker. The exit
//!   code is `0` when the command succeeded, else the program's exit code (or
//!   `1` for a failing cmdlet). PowerShell has no `&&` / `||` before version 7:
//!   use `;` there.

use crate::panels::{self, PanelHub, RunStatus};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

const READY: &str = "__TC_READY__";
const DONE_PREFIX: &str = "__TC_DONE_";

/// Which command language the shell speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellKind {
    /// PowerShell (`pwsh` or `powershell.exe`)
    PowerShell,
    /// `sh` / `bash`
    Posix,
}

#[derive(Default)]
struct Shared {
    /// Runs that were sent to the shell and have not finished, oldest first.
    pending: VecDeque<u32>,
    /// `false` once the shell's output ended (it exited or was killed).
    alive: bool,
}

fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(|error| error.into_inner())
}

/// One open shell.
pub struct Session {
    child: Child,
    stdin: ChildStdin,
    kind: ShellKind,
    shared: Arc<Mutex<Shared>>,
    /// The active script's path when the shell was started (its folder and
    /// environment belong to that script).
    started_for: Option<String>,
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Session {
    /// Starts `command` as a shell. Its stdio is set here. `init` lines are
    /// sent first (e.g. activating a virtualenv); their output is shown.
    pub fn start(
        kind: ShellKind,
        mut command: Command,
        hub: &PanelHub,
        init: &[String],
        started_for: Option<String>,
    ) -> Result<Self, String> {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // The shell (and `wsl.exe` in particular) must not share the controller's
        // console: programs that attach to a console can switch its input mode
        // (for example to raw / virtual-terminal input) and leave the REPL with
        // arrow keys typed as `^[[A` and Enter doing nothing. CREATE_NO_WINDOW
        // gives the child its own hidden console, so the REPL's stays untouched.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command.spawn().map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                format!("The shell program was not found: {error}")
            } else {
                format!("Failed to start the shell: {error}")
            }
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "The shell has no input stream.".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "The shell has no output stream.".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "The shell has no error stream.".to_string())?;

        let shared = Arc::new(Mutex::new(Shared {
            pending: VecDeque::new(),
            alive: true,
        }));
        {
            let hub = hub.clone();
            let shared = Arc::clone(&shared);
            thread::spawn(move || read_stdout(stdout, hub, shared));
        }
        {
            let hub = hub.clone();
            thread::spawn(move || panels::pump(stderr, hub));
        }

        let mut session = Self {
            child,
            stdin,
            kind,
            shared,
            started_for,
        };
        // Everything before this line (a banner, for instance) is dropped.
        session.send_raw(&format!("echo {READY}"))?;
        for line in init {
            session.send_raw(line)?;
        }
        Ok(session)
    }

    pub fn is_alive(&self) -> bool {
        lock(&self.shared).alive
    }

    /// The active script's path at the time the shell was started.
    pub fn started_for(&self) -> Option<&str> {
        self.started_for.as_deref()
    }

    fn send_raw(&mut self, line: &str) -> Result<(), String> {
        self.stdin
            .write_all(format!("{line}\n").as_bytes())
            .and_then(|()| self.stdin.flush())
            .map_err(|error| format!("The shell is not accepting input ({error})."))
    }

    /// Sends `command_line` to the shell as a new run. Returns the run number
    /// and how many earlier commands it has to wait for.
    pub fn run(
        &mut self,
        hub: &PanelHub,
        label: &str,
        command_line: &str,
    ) -> Result<(u32, usize), String> {
        let run_id = hub.with(|store| store.start_run(label));
        let ahead = {
            let mut shared = lock(&self.shared);
            let ahead = shared.pending.len();
            shared.pending.push_back(run_id);
            ahead
        };

        let sent = match self.kind {
            ShellKind::Posix => self.send_raw(&format!(
                "eval {} </dev/null; echo \"{DONE_PREFIX}{run_id}__$?\"",
                posix_quote(command_line)
            )),
            ShellKind::PowerShell => {
                // `$LASTEXITCODE` is cleared first so an old exit code is not
                // reported for a cmdlet; `$?` must be read before anything else.
                self.send_raw("$global:LASTEXITCODE = $null")
                    .and_then(|()| self.send_raw(command_line))
                    .and_then(|()| {
                        self.send_raw(&format!(
                            "$tc = $?; $tcc = $LASTEXITCODE; Write-Output ('{DONE_PREFIX}{run_id}__' + $(if ($tc) {{ 0 }} elseif ($tcc) {{ $tcc }} else {{ 1 }}))"
                        ))
                    })
            }
        };
        if let Err(error) = sent {
            lock(&self.shared).pending.retain(|id| *id != run_id);
            hub.with(|store| store.finish_run(run_id, RunStatus::Failed(error.clone())));
            return Err(error);
        }
        Ok((run_id, ahead))
    }
}

/// Routes the shell's stdout into the panels, ending runs at their markers.
fn read_stdout<R: Read>(stdout: R, hub: PanelHub, shared: Arc<Mutex<Shared>>) {
    let mut reader = BufReader::new(stdout);
    let mut buffer = Vec::new();
    let mut ready = false;
    loop {
        buffer.clear();
        match reader.read_until(b'\n', &mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&buffer).into_owned();
        let text = line.trim_end_matches(['\r', '\n']);

        if !ready {
            ready = text.trim() == READY;
            continue;
        }

        // The marker can follow output that had no trailing newline.
        if let Some(position) = text.find(DONE_PREFIX) {
            if let Some((run_id, code)) = parse_done(text[position..].trim()) {
                let before = &text[..position];
                if !before.is_empty() {
                    hub.with(|store| store.ingest(before));
                }
                lock(&shared).pending.retain(|id| *id != run_id);
                hub.with(|store| store.finish_run(run_id, RunStatus::Exited(code)));
                continue;
            }
        }
        hub.with(|store| store.ingest(&line));
    }

    // The shell is gone: nothing that was waiting for it will finish.
    let abandoned: Vec<u32> = {
        let mut shared = lock(&shared);
        shared.alive = false;
        shared.pending.drain(..).collect()
    };
    for run_id in abandoned {
        hub.with(|store| {
            store.finish_run(run_id, RunStatus::Failed("the shell exited".to_string()))
        });
    }
}

/// `__TC_DONE_<run>__<code>` -> `(run, code)`.
fn parse_done(text: &str) -> Option<(u32, i32)> {
    let rest = text.strip_prefix(DONE_PREFIX)?;
    let (run_id, code) = rest.split_once("__")?;
    Some((run_id.trim().parse().ok()?, code.trim().parse().ok()?))
}

// ---------------------------------------------------------------------------
// Quoting (pure)
// ---------------------------------------------------------------------------

/// `it's` -> `'it'\''s'`: one argument for a POSIX shell.
pub fn posix_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

/// `it's` -> `'it''s'`: one string for PowerShell.
pub fn powershell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

/// Joins the words typed after `/run_command` into one command line.
///
/// The REPL has already split the line into words and removed their quotes,
/// so a word that contains whitespace or a quote is quoted again; every other
/// word (including `&&`, `|`, `;`, `>`, `2>&1`) is kept as typed, so shell
/// syntax works. In PowerShell a quoted first word is a string, so it gets the
/// call operator `&` in front.
pub fn join_command(words: &[String], kind: ShellKind) -> String {
    let line = words
        .iter()
        .map(|word| {
            let needs_quotes = word
                .chars()
                .any(|ch| ch.is_whitespace() || ch == '"' || ch == '\'');
            match (needs_quotes, kind) {
                (false, _) => word.clone(),
                (true, ShellKind::Posix) => posix_quote(word),
                (true, ShellKind::PowerShell) => powershell_quote(word),
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    let first_is_quoted = words.first().is_some_and(|word| {
        word.chars()
            .any(|ch| ch.is_whitespace() || ch == '"' || ch == '\'')
    });
    if kind == ShellKind::PowerShell && first_is_quoted {
        format!("& {line}")
    } else {
        line
    }
}

// ---------------------------------------------------------------------------
// The open sessions
// ---------------------------------------------------------------------------

type Sessions = Vec<(&'static str, Session)>;

/// The open shells, keyed by target (`"native"`, `"wsl"`). They live as long
/// as the controller; nothing is saved between restarts.
static SESSIONS: Mutex<Sessions> = Mutex::new(Vec::new());

/// Runs `f` with the open sessions locked.
pub fn with_sessions<R>(f: impl FnOnce(&mut Sessions) -> R) -> R {
    let mut guard = SESSIONS.lock().unwrap_or_else(|error| error.into_inner());
    f(&mut guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panels::OUTPUT_PANEL;
    use std::time::Duration;

    fn words(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| item.to_string()).collect()
    }

    #[test]
    fn markers_are_parsed() {
        assert_eq!(parse_done("__TC_DONE_7__0"), Some((7, 0)));
        assert_eq!(parse_done("__TC_DONE_12__-1073741819 "), Some((12, -1073741819)));
        assert_eq!(parse_done("__TC_DONE_x__0"), None);
        assert_eq!(parse_done("__TC_DONE_3__"), None);
        assert_eq!(parse_done("hello"), None);
    }

    #[test]
    fn quoting_keeps_shell_syntax_and_protects_arguments() {
        assert_eq!(posix_quote("it's"), "'it'\\''s'");
        assert_eq!(
            join_command(&words(&["cd", "x", "&&", "ls", "|", "head"]), ShellKind::Posix),
            "cd x && ls | head"
        );
        assert_eq!(
            join_command(&words(&["echo", "a b", "it's"]), ShellKind::Posix),
            "echo 'a b' 'it'\\''s'"
        );
        assert_eq!(
            join_command(&words(&["dir", "C:\\My Files"]), ShellKind::PowerShell),
            "dir 'C:\\My Files'"
        );
        assert_eq!(
            join_command(&words(&["C:\\My Files\\a.exe", "-x"]), ShellKind::PowerShell),
            "& 'C:\\My Files\\a.exe' -x"
        );
        assert_eq!(powershell_quote("it's"), "'it''s'");
        assert_eq!(join_command(&words(&["pip", "list"]), ShellKind::PowerShell), "pip list");
    }

    fn test_shell() -> (ShellKind, Command) {
        if cfg!(windows) {
            let mut command = Command::new("powershell.exe");
            command.args(["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "-"]);
            (ShellKind::PowerShell, command)
        } else {
            (ShellKind::Posix, Command::new("sh"))
        }
    }

    fn wait_until_finished(hub: &PanelHub, run_id: u32) -> RunStatus {
        for _ in 0..200 {
            let status = hub.with(|store| store.run_status(run_id));
            if let Some(status) = status.filter(|status| *status != RunStatus::Running) {
                return status;
            }
            thread::sleep(Duration::from_millis(50));
        }
        panic!("run #{run_id} did not finish");
    }

    #[test]
    fn a_session_keeps_state_between_commands_and_reports_exit_codes() {
        let hub = PanelHub::new();
        let (kind, command) = test_shell();
        let mut session = Session::start(kind, command, &hub, &[], None).unwrap();
        assert!(session.is_alive());

        let (set, show, fail) = if cfg!(windows) {
            ("$TC_T = 42", "echo \"value=$TC_T\"", "cmd /c exit 3")
        } else {
            ("TC_T=42", "echo value=$TC_T", "sh -c 'exit 3'")
        };
        session.run(&hub, "set", set).unwrap();
        let (shown, _) = session.run(&hub, "show", show).unwrap();
        assert_eq!(wait_until_finished(&hub, shown), RunStatus::Exited(0));
        let output = hub.with(|store| store.lines_of(OUTPUT_PANEL));
        // The variable set by an earlier command is still there, and no marker leaks.
        assert!(output.iter().any(|line| line == "value=42"), "{output:?}");
        assert!(output.iter().all(|line| !line.contains(DONE_PREFIX)), "{output:?}");

        let (failed, _) = session.run(&hub, "fail", fail).unwrap();
        assert_eq!(wait_until_finished(&hub, failed), RunStatus::Exited(3));
    }
}
