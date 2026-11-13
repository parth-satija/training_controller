//! Plain-Rust functions that are exposed to Python.
//!
//! This file is where you write the *logic* of every function that Python code
//! can call through `import training_controller as tc`. This documentation is
//! the guide for adding new ones.
//!
//! # How the pieces fit together
//!
//! ```text
//! Python script            src/python.rs               src/api.rs
//! -------------            -------------               ----------
//! tc.add(1, 2)   ───────►  py_add(a, b)   ───────────► add(a, b)
//!                          (thin PyO3 wrapper:         (plain Rust: real work,
//!                           converts types and          no PyO3, unit-testable)
//!                           errors)
//! ```
//!
//! * **`api.rs` (this file)** holds ordinary Rust functions. They know nothing
//!   about Python. That keeps them easy to test with `cargo test` and reusable
//!   from the REPL in `main.rs`.
//! * **`python.rs`** holds one small `#[pyfunction]` wrapper per API function.
//!   The wrapper receives Python values, calls the function here, and turns the
//!   result (or error) back into something Python understands.
//! * **`lib.rs`** only wires the two modules together. You normally never edit it.
//!
//! The wrappers are compiled only with the `python` Cargo feature, so the
//! REPL binary never depends on Python.
//!
//! # Adding a new function: step by step
//!
//! 1. **Write the function in this file** (see "Rules" below). It must be `pub`.
//!
//!    ```ignore
//!    /// Adds two numbers.
//!    pub fn add(a: f64, b: f64) -> f64 {
//!        a + b
//!    }
//!    ```
//!
//! 2. **Add a unit test** in the `tests` module at the bottom of this file.
//!
//! 3. **Write the wrapper in `src/python.rs`.** Give it a `py_` prefix to avoid
//!    a name clash with the function in `api`, and use `name = "..."` to choose
//!    the name Python sees:
//!
//!    ```ignore
//!    /// Adds two numbers.
//!    #[pyfunction]
//!    #[pyo3(name = "add", text_signature = "(a, b)")]
//!    fn py_add(a: f64, b: f64) -> f64 {
//!        api::add(a, b)
//!    }
//!    ```
//!
//!    The `///` comment on the wrapper becomes `help(tc.add)` in Python.
//!
//! 4. **Register the wrapper** inside the `#[pymodule]` function in `python.rs`:
//!
//!    ```ignore
//!    m.add_function(wrap_pyfunction!(py_add, m)?)?;
//!    ```
//!
//!    Forgetting this line is the most common mistake: the code compiles but
//!    Python reports `AttributeError: module 'training_controller' has no
//!    attribute 'add'`.
//!
//! 5. **Rebuild with the Python feature**, then try it:
//!
//!    ```text
//!    cargo build --features python     # then use /run_script in the REPL
//!    # or, inside a virtualenv:
//!    maturin develop
//!    ```
//!
//!    ```python
//!    import training_controller as tc
//!    print(tc.add(1.5, 2))   # 3.5
//!    ```
//!
//! # Rules for functions in this file
//!
//! * **No PyO3 types here.** Do not use `Python`, `Bound`, `PyAny`, `PyResult`
//!   and so on. Use ordinary Rust types and put the conversion in the wrapper.
//!   This is what keeps this file testable without a Python installation.
//! * **Wrappers take owned types, API functions borrow.** In `python.rs`,
//!   declare string parameters as `String` (not `&str`), because the build
//!   targets Python's stable ABI for Python 3.9, where `&str` arguments are not
//!   supported. Then call the API function with `&text`. Inside this file,
//!   prefer borrowing (`&str`, `&[f64]`) so callers are not forced to clone.
//! * **Return `Result<T, String>` for anything that can fail** (bad input,
//!   missing file, parse error). Never `panic!`, `unwrap()` or `expect()` on
//!   input that comes from Python: a panic becomes an obscure
//!   `pyo3_runtime.PanicException` instead of a normal Python error. Reserve
//!   `unwrap()` for conditions that are truly impossible.
//! * **Keep functions synchronous and self-contained.** Do not store the
//!   Python interpreter or Python objects in statics. If a function needs shared
//!   state, use a Rust `static` with a `Mutex` / `OnceLock` holding plain Rust
//!   data.
//! * **Document every function** with `///` comments: one summary line, then
//!   details such as units, ranges and error conditions.
//!
//! # Type mapping (Rust ⇄ Python)
//!
//! Wrapper parameter and return types are converted automatically:
//!
//! | Rust type                          | Python type                  |
//! |------------------------------------|------------------------------|
//! | `bool`                             | `bool`                       |
//! | `i8`..`i64`, `u8`..`u64`, `usize`  | `int` (error if out of range)|
//! | `f32`, `f64`                       | `float` (ints are accepted)  |
//! | `String` (`&str` as API parameter) | `str`                        |
//! | `Vec<T>`                           | `list`                       |
//! | `[T; N]`, tuples `(A, B)`          | fixed-size `list` / `tuple`  |
//! | `HashMap<K, V>`, `BTreeMap<K, V>`  | `dict`                       |
//! | `Option<T>`                        | `T` or `None`                |
//! | `()`                               | `None`                       |
//! | `Result<T, E>` (in the wrapper)    | `T`, or raises an exception  |
//!
//! Choose the narrowest type that makes sense. `u32` for a count rejects
//! negative numbers for you; `Vec<f64>` accepts any list of numbers.
//!
//! # Errors become Python exceptions
//!
//! API functions return `Result<T, String>`. The wrapper converts the error
//! into a Python exception with `map_err`, and the wrapper's return type
//! becomes `PyResult<T>`:
//!
//! ```ignore
//! use pyo3::exceptions::PyValueError;
//!
//! #[pyfunction]
//! #[pyo3(name = "parse_lr")]
//! fn py_parse_lr(text: String) -> PyResult<f64> {
//!     api::parse_lr(&text).map_err(PyValueError::new_err)
//! }
//! ```
//!
//! In Python the caller sees an ordinary `ValueError` that they can catch:
//!
//! ```python
//! try:
//!     tc.parse_lr("abc")
//! except ValueError as error:
//!     print(error)
//! ```
//!
//! Pick the exception that matches the problem: `PyValueError` for bad
//! values, `PyTypeError` for wrong types, `PyFileNotFoundError` /
//! `PyOSError` for I/O, `PyRuntimeError` for everything else. They all live in
//! `pyo3::exceptions`.
//!
//! # Optional and keyword arguments
//!
//! Python defaults are declared in the wrapper with `signature`, using
//! `Option<T>` for "may be omitted":
//!
//! ```ignore
//! #[pyfunction]
//! #[pyo3(name = "scale", signature = (value, factor=1.0, offset=None))]
//! fn py_scale(value: f64, factor: f64, offset: Option<f64>) -> f64 {
//!     api::scale(value, factor, offset)
//! }
//! ```
//!
//! ```python
//! tc.scale(2.0)                      # factor defaults to 1.0
//! tc.scale(2.0, factor=3.0)          # keyword argument
//! tc.scale(2.0, 3.0, offset=0.5)
//! ```
//!
//! Without a `signature` attribute every parameter is required, and a trailing
//! `Option<T>` parameter is also optional in current PyO3 versions.
//! Writing the `signature` explicitly is clearer and is the recommended style.
//!
//! # Printing and output
//!
//! `println!` writes to the process's real standard output. In a terminal (and
//! when a script is started with `/run_script`) that is exactly what you want.
//! In environments that replace Python's `sys.stdout` (Jupyter notebooks, IDE
//! consoles, test runners that capture output) text printed from Rust may
//! appear in the terminal instead of in Python's output. When a function
//! produces output for the caller to *use*, return it as a `String` and let
//! the Python code print it.
//!
//! # Long-running functions
//!
//! While Rust code runs, the calling Python thread holds the interpreter lock
//! (the GIL), so other Python threads are blocked. For work that takes more
//! than a moment (training loops, file scans, sleeps), release the lock in
//! the wrapper. In PyO3 0.27 this is `Python::detach` (called `allow_threads`
//! in older versions):
//!
//! ```ignore
//! #[pyfunction]
//! #[pyo3(name = "heavy_work")]
//! fn py_heavy_work(py: Python<'_>, size: usize) -> u64 {
//!     // The closure must not touch any Python objects.
//!     py.detach(|| api::heavy_work(size))
//! }
//! ```
//!
//! `py: Python<'_>` is a special parameter: PyO3 fills it in itself, and it
//! does not appear in the Python signature.
//!
//! # Returning richer data
//!
//! * **Several values**: return a tuple, e.g. `(f64, f64)` becomes
//!   `(float, float)`.
//! * **Structured data**: return a `HashMap<String, f64>` for a `dict`, or
//!   serialize with `serde_json` and return the JSON `String` (Python then calls
//!   `json.loads`). This keeps the API small without defining Python classes.
//! * **Python classes** (`#[pyclass]`) are possible, but they are
//!   wrapper-layer code that belongs in `python.rs`. Ask before introducing
//!   them, since they change the shape of the whole API.
//!
//! # Testing
//!
//! Add unit tests to the `tests` module below; run them with plain
//! `cargo test`. No Python is needed because this file never touches PyO3.
//! Test the success path, the error paths, and edge cases such as empty
//! input.
//!
//! To check a wrapper end to end, write a small script that imports
//! `training_controller`, register it with `/add_scripts`, activate it with
//! `/open_script`, and run it with `/run_script`.
//!
//! # Checklist
//!
//! - [ ] Function is `pub`, documented with `///`, and uses no PyO3 types
//! - [ ] Fallible functions return `Result<T, String>`; no `unwrap()` on input
//! - [ ] Unit test added below
//! - [ ] Wrapper written in `python.rs` (`String` parameters, `map_err` for errors)
//! - [ ] Wrapper registered with `m.add_function(wrap_pyfunction!(...))`
//! - [ ] Rebuilt with `cargo build --features python`

use crate::logdb::{self, LogWriter, NewExperiment};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Builds the line that `print_message` writes.
pub fn format_message(message: &str) -> String {
    format!("[training_controller] {message}")
}

/// Prints a message received from the caller (sample function).
///
/// Exposed to Python as `tc.test(message)` (see `py_test` in `python.rs`).
pub fn print_message(message: &str) {
    println!("{}", format_message(message));
}

// ---------------------------------------------------------------------------
// Training dashboard
// ---------------------------------------------------------------------------
//
// Python calls `init_training_dashboard`, `update_training_dashboard` and
// `update_hero_text`. Each call appends one JSON line (a `TrainingEvent`) to a
// file. The REPL command `/dashboard training` tails that file and draws it.
// Nothing is drawn by these calls: they only record events, so a training
// script never opens a UI by itself.
//
// ```text
// {"event":"init","t":1760000000000,"widgets":[{"name":"loss","kind":"curve"}]}
// {"event":"update","t":1760000000100,"values":[0.52]}
// {"event":"hero","t":1760000000100,"text":"Training at Epoch 1, step 10"}
// ```

/// Environment variable naming the events file. `/run_script` sets it to
/// `training_dashboard.jsonl` next to the active script; when it is missing the
/// file is created in the current working directory.
pub const TRAINING_FILE_ENV: &str = "TC_TRAINING_FILE";

/// File name used next to a script (and in the working directory as a fallback).
pub const TRAINING_FILE_NAME: &str = "training_dashboard.jsonl";

/// How often the running script touches the events file's modified time. The
/// dashboard reads that as "the script is still alive", independent of how
/// often (or rarely) the script calls `update_training_dashboard`.
pub const HEARTBEAT_EVERY: Duration = Duration::from_secs(1);

const TRAINING_NOT_INITIALIZED: &str =
    "The training dashboard is not initialized: call init_training_dashboard(...) first.";

/// How a dashboard variable is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WidgetKind {
    /// Bar graph of the recent history.
    Graph,
    /// Line plot of the full retained history.
    Curve,
    /// Block showing only the latest value (plus change, min and max).
    #[serde(rename = "box")]
    Block,
}

impl WidgetKind {
    /// Parses `graph`, `box` or `curve` (case-insensitive).
    pub fn parse(text: &str) -> Result<Self, String> {
        match text.trim().to_lowercase().as_str() {
            "graph" => Ok(Self::Graph),
            "curve" => Ok(Self::Curve),
            "box" => Ok(Self::Block),
            other => Err(format!(
                "unknown widget type '{other}' (use graph, box or curve)"
            )),
        }
    }

    /// The name Python uses for this type.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Graph => "graph",
            Self::Curve => "curve",
            Self::Block => "box",
        }
    }
}

/// One variable shown on the dashboard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WidgetSpec {
    pub name: String,
    pub kind: WidgetKind,
}

/// One line of the events file. `t` is Unix time in milliseconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TrainingEvent {
    /// A new run: resets everything the dashboard knows.
    Init { t: u64, widgets: Vec<WidgetSpec> },
    /// One value per widget, in `init` order. `None` is a NaN/infinite value.
    Update { t: u64, values: Vec<Option<f64>> },
    /// The bold text in the middle of the dashboard.
    Hero { t: u64, text: String },
}

/// Milliseconds since the Unix epoch (0 if the clock is before it).
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

/// The events file next to `script` (used for `TC_TRAINING_FILE` and by the REPL).
pub fn training_file_for_script(script: &Path) -> Option<PathBuf> {
    script
        .parent()
        .map(|directory| directory.join(TRAINING_FILE_NAME))
}

/// Where this process writes events: `TC_TRAINING_FILE`, else the working directory.
pub fn training_file_path() -> Result<PathBuf, String> {
    match std::env::var_os(TRAINING_FILE_ENV) {
        Some(value) if !value.is_empty() => Ok(PathBuf::from(value)),
        _ => std::env::current_dir()
            .map(|directory| directory.join(TRAINING_FILE_NAME))
            .map_err(|error| format!("Could not determine the working directory: {error}")),
    }
}

/// Turns `name, type, name, type, ...` into widget specs.
///
/// Fails on an empty list, an odd number of arguments, an empty or duplicate
/// name, or a type other than `graph`, `box` or `curve`.
pub fn parse_widget_specs(args: &[String]) -> Result<Vec<WidgetSpec>, String> {
    if args.is_empty() {
        return Err(
            "init_training_dashboard needs at least one (name, type) pair, \
             e.g. init_training_dashboard(\"loss\", \"curve\")."
                .into(),
        );
    }
    if args.len() % 2 != 0 {
        return Err(format!(
            "init_training_dashboard expects (name, type) pairs but got {} arguments.",
            args.len()
        ));
    }

    let mut widgets: Vec<WidgetSpec> = Vec::with_capacity(args.len() / 2);
    for pair in args.chunks(2) {
        let name = pair[0].trim();
        if name.is_empty() {
            return Err("Variable names must not be empty.".into());
        }
        if widgets.iter().any(|widget| widget.name == name) {
            return Err(format!("Variable '{name}' is listed twice."));
        }
        let kind = WidgetKind::parse(&pair[1])
            .map_err(|error| format!("Variable '{name}': {error}"))?;
        widgets.push(WidgetSpec {
            name: name.to_string(),
            kind,
        });
    }
    Ok(widgets)
}

/// Appends training events to the events file. Owns the file handle; the free
/// functions below keep one writer per process.
pub struct TrainingWriter {
    file: File,
    widgets: Vec<WidgetSpec>,
    stop_heartbeat: Arc<AtomicBool>,
}

impl Drop for TrainingWriter {
    fn drop(&mut self) {
        self.stop_heartbeat.store(true, Ordering::Relaxed);
    }
}

/// Touches the file's modified time until `stop` is set (or the file vanishes).
/// Pure Rust and independent of the GIL, so it keeps beating while Python is busy,
/// and it dies with the process.
fn spawn_heartbeat(file: File, stop: Arc<AtomicBool>, every: Duration) {
    let _ = std::thread::Builder::new()
        .name("tc-dashboard-heartbeat".into())
        .spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(every);
                if stop.load(Ordering::Relaxed) || file.set_modified(SystemTime::now()).is_err() {
                    break;
                }
            }
        });
}

impl TrainingWriter {
    /// Starts a new run: truncates `path` and writes the `init` event.
    pub fn create(path: &Path, widgets: Vec<WidgetSpec>) -> Result<Self, String> {
        Self::create_with_heartbeat(path, widgets, HEARTBEAT_EVERY)
    }

    /// [`TrainingWriter::create`] with a custom heartbeat interval (used by tests).
    pub fn create_with_heartbeat(
        path: &Path,
        widgets: Vec<WidgetSpec>,
        heartbeat: Duration,
    ) -> Result<Self, String> {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .map_err(|error| format!("Failed to create {}: {error}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|error| format!("Failed to open {}: {error}", path.display()))?;

        let stop_heartbeat = Arc::new(AtomicBool::new(false));
        if let Ok(handle) = file.try_clone() {
            spawn_heartbeat(handle, Arc::clone(&stop_heartbeat), heartbeat);
        }
        let mut writer = Self {
            file,
            widgets,
            stop_heartbeat,
        };
        let init = TrainingEvent::Init {
            t: now_ms(),
            widgets: writer.widgets.clone(),
        };
        writer.write_event(&init)?;
        Ok(writer)
    }

    /// The variables this run was initialized with.
    pub fn widgets(&self) -> &[WidgetSpec] {
        &self.widgets
    }

    /// Records one value per widget. NaN and infinite values are stored as gaps.
    pub fn update(&mut self, values: &[f64]) -> Result<(), String> {
        if values.len() != self.widgets.len() {
            let names: Vec<&str> = self.widgets.iter().map(|w| w.name.as_str()).collect();
            return Err(format!(
                "update_training_dashboard expected {} values ({}) but got {}.",
                self.widgets.len(),
                names.join(", "),
                values.len()
            ));
        }
        let values = values
            .iter()
            .map(|value| value.is_finite().then_some(*value))
            .collect();
        self.write_event(&TrainingEvent::Update {
            t: now_ms(),
            values,
        })
    }

    /// Replaces the hero text.
    pub fn hero(&mut self, text: &str) -> Result<(), String> {
        self.write_event(&TrainingEvent::Hero {
            t: now_ms(),
            text: text.to_string(),
        })
    }

    fn write_event(&mut self, event: &TrainingEvent) -> Result<(), String> {
        let mut line = serde_json::to_string(event)
            .map_err(|error| format!("Failed to encode a dashboard event: {error}"))?;
        line.push('\n');
        // One write per line, so a reader never sees half of an event for long.
        self.file
            .write_all(line.as_bytes())
            .map_err(|error| format!("Failed to write the dashboard events file: {error}"))
    }
}

static TRAINING: Mutex<Option<TrainingWriter>> = Mutex::new(None);

fn training_writer() -> MutexGuard<'static, Option<TrainingWriter>> {
    // A panic elsewhere must not turn every later call into an error.
    TRAINING.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Declares the dashboard variables and starts a new run.
///
/// `args` alternates variable name and type: `["loss", "curve", "acc", "graph",
/// "lr", "box"]`. Types are `graph`, `box` and `curve`. Calling it again starts a
/// fresh run (the events file is truncated and open dashboards reset).
///
/// This does not open a dashboard; run `/dashboard training` in the REPL.
pub fn init_training_dashboard(args: &[String]) -> Result<(), String> {
    let widgets = parse_widget_specs(args)?;
    let path = training_file_path()?;
    let writer = TrainingWriter::create(&path, widgets)?;
    *training_writer() = Some(writer);
    Ok(())
}

/// Records one value per variable, in the order given to `init_training_dashboard`.
///
/// Errors if the dashboard is not initialized or the number of values differs
/// from the number of variables.
pub fn update_training_dashboard(values: &[f64]) -> Result<(), String> {
    match training_writer().as_mut() {
        Some(writer) => writer.update(values),
        None => Err(TRAINING_NOT_INITIALIZED.into()),
    }
}

/// Sets the bold hero text, e.g. `"Training at Epoch 47, step 4700, Total Loss at 0.0023"`.
///
/// Errors if the dashboard is not initialized.
pub fn update_hero_text(text: &str) -> Result<(), String> {
    match training_writer().as_mut() {
        Some(writer) => writer.hero(text),
        None => Err(TRAINING_NOT_INITIALIZED.into()),
    }
}

// ---------------------------------------------------------------------------
// Experiment log (SQLite)
// ---------------------------------------------------------------------------
//
// The Training Controller creates an experiment (and its ID) for every
// `/run_script` and stores everything in one SQLite database (see `logdb.rs`).
// The script declares its columns with `init_log` and fills them with `log`:
//
// ```python
// exp = tc.init_log("loss", "accuracy", "lr")   # returns the experiment ID
// for epoch in range(10):
//     tc.log(loss, accuracy, lr)                # one value per column
// ```
//
// `/run_script` passes the database path and the experiment ID through the
// `TC_LOG_DB` and `TC_EXPERIMENT_ID` environment variables. A script started
// by hand has neither: `init_log` then writes `logs.db` in the working
// directory and creates a new experiment (status `standalone`).

const LOG_NOT_INITIALIZED: &str = "The log is not initialized: call init_log(...) first.";

static LOG: Mutex<Option<LogWriter>> = Mutex::new(None);

fn log_writer() -> MutexGuard<'static, Option<LogWriter>> {
    // A panic elsewhere must not turn every later call into an error.
    LOG.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Where this process writes the log: `TC_LOG_DB`, else `logs.db` in the working directory.
pub fn log_db_path() -> Result<PathBuf, String> {
    match std::env::var_os(logdb::DB_FILE_ENV) {
        Some(value) if !value.is_empty() => Ok(PathBuf::from(value)),
        _ => std::env::current_dir()
            .map(|directory| directory.join(logdb::DB_FILE_NAME))
            .map_err(|error| format!("Could not determine the working directory: {error}")),
    }
}

/// Parses the experiment ID the controller passed in `TC_EXPERIMENT_ID`.
/// `None` or blank text means "not started by the controller".
pub fn parse_experiment_id(text: Option<&str>) -> Result<Option<i64>, String> {
    match text.map(str::trim) {
        None | Some("") => Ok(None),
        Some(text) => text.parse::<i64>().map(Some).map_err(|_| {
            format!("{} must be an integer but is '{text}'.", logdb::EXPERIMENT_ID_ENV)
        }),
    }
}

fn experiment_id_from_env() -> Result<Option<i64>, String> {
    parse_experiment_id(std::env::var(logdb::EXPERIMENT_ID_ENV).ok().as_deref())
}

/// Declares the log columns and starts an empty log. Returns the experiment ID.
///
/// The experiment ID is the number the controller gave this run among the runs
/// of its script (the first run of a script is 1, the next 2, and so on; every
/// script counts for itself). Calling
/// it again restarts this experiment's log (old rows are discarded). The
/// controller shows logs with `/show_experiments` and `/show_log`.
pub fn init_log(titles: &[String]) -> Result<i64, String> {
    let titles = logdb::validate_titles(titles)?;
    let path = log_db_path()?;
    let conn = logdb::open(&path)?;

    let experiment_id = match experiment_id_from_env()? {
        Some(id) => {
            if !logdb::experiment_exists(&conn, id)? {
                return Err(format!(
                    "Experiment #{id} does not exist in {}.",
                    path.display()
                ));
            }
            id
        }
        None => logdb::create_experiment(
            &conn,
            &NewExperiment {
                script_name: None,
                script_path: None,
                config_name: None,
                args: &[],
                status: logdb::STATUS_STANDALONE,
            },
        )?,
    };

    // The number users see (their ID among this script's runs), not the database ID.
    let number = logdb::experiment_seq(&conn, experiment_id)?.unwrap_or(experiment_id);
    let writer = LogWriter::init(conn, experiment_id, titles)?;
    *log_writer() = Some(writer);
    Ok(number)
}

/// Records one value per column, in the order given to `init_log`.
///
/// Errors if the log is not initialized or the number of values differs from
/// the number of columns. NaN and infinite values are stored as `NULL`.
pub fn log(values: &[f64]) -> Result<(), String> {
    match log_writer().as_mut() {
        Some(writer) => writer.log(values),
        None => Err(LOG_NOT_INITIALIZED.into()),
    }
}

/// The database-wide ID of this run: the one from `init_log`, or the one the
/// controller put in `TC_EXPERIMENT_ID`, or `None` outside a controller run.
/// It is the key of the log tables and what checkpoints record; scripts and
/// users see the per-script number from [`experiment_id`] instead.
fn internal_experiment_id() -> Option<i64> {
    if let Some(writer) = log_writer().as_ref() {
        return Some(writer.experiment_id());
    }
    experiment_id_from_env().ok().flatten()
}

/// The experiment ID of this run: its number among the runs of its script (the
/// same number `/show_experiments`, `/show_log` and `/graph` use), or `None`
/// outside a controller run.
pub fn experiment_id() -> Option<i64> {
    let internal = internal_experiment_id()?;
    let number = log_db_path()
        .ok()
        .and_then(|path| logdb::open(&path).ok())
        .and_then(|conn| logdb::experiment_seq(&conn, internal).ok().flatten());
    Some(number.unwrap_or(internal))
}

// ---------------------------------------------------------------------------
// Checkpoints (see checkpoint.rs)
// ---------------------------------------------------------------------------
//
// ```python
// path = tc.save_checkpoint(buffer.getvalue(), is_best, step=epoch, metrics={"loss": loss})
// resume_from = tc.latest_checkpoint()      # path or None
// ```
//
// `/run_script` passes the store directory in `TC_CHECKPOINT_DIR`; a script
// started by hand writes to `./checkpoints`. What is kept and what is deleted
// is decided by the policy the user set with `/set_checkpoint_config`; each
// call prints `[checkpoint] ...` lines, which show up in the `checkpoint` panel.

use crate::checkpoint::{self, NewCheckpoint, Store};

fn print_checkpoint_outcome(outcome: &checkpoint::SaveOutcome) {
    for line in outcome.summary_lines() {
        println!("{line}");
    }
}

/// Stores a checkpoint given as raw bytes and returns the path of the stored
/// file, or `None` when the policy chose not to store it (`save_every`, or a
/// retention policy that would delete it immediately).
///
/// `data` can be in any format (`torch.save` into a `BytesIO`, `pickle.dumps`,
/// ...): it is stored unchanged. `is_best` marks the best checkpoint so far; the
/// retention policy can protect those. `step` and `metrics` are recorded in the
/// index and shown by `/checkpoints`. NaN and infinite metrics are dropped.
pub fn save_checkpoint(
    data: &[u8],
    is_best: bool,
    step: Option<u64>,
    metrics: std::collections::BTreeMap<String, f64>,
) -> Result<Option<String>, String> {
    let store = Store::from_env()?;
    let outcome = store.save_bytes(
        data,
        NewCheckpoint {
            is_best,
            step,
            metrics,
            experiment: internal_experiment_id(),
        },
    )?;
    print_checkpoint_outcome(&outcome);
    Ok(outcome.path.map(|path| path.display().to_string()))
}

/// Like [`save_checkpoint`], but reads the checkpoint from a file the script
/// already wrote (for example with `torch.save(model, path)`), streaming it so
/// a multi-gigabyte checkpoint never has to fit in memory. With `move_file`
/// the original is deleted once the checkpoint is safely stored.
pub fn save_checkpoint_file(
    path: &str,
    is_best: bool,
    step: Option<u64>,
    metrics: std::collections::BTreeMap<String, f64>,
    move_file: bool,
) -> Result<Option<String>, String> {
    let store = Store::from_env()?;
    let outcome = store.save_file(
        Path::new(path),
        NewCheckpoint {
            is_best,
            step,
            metrics,
            experiment: internal_experiment_id(),
        },
        move_file,
    )?;
    print_checkpoint_outcome(&outcome);
    Ok(outcome.path.map(|path| path.display().to_string()))
}

/// Path of the newest stored checkpoint (any run), or `None` if there is none.
pub fn latest_checkpoint() -> Result<Option<String>, String> {
    let path = Store::from_env()?.latest_path()?;
    Ok(path.map(|path| path.display().to_string()))
}

/// Path of the newest checkpoint that was saved with `is_best=True`, or `None`.
pub fn best_checkpoint() -> Result<Option<String>, String> {
    let path = Store::from_env()?.best_path()?;
    Ok(path.map(|path| path.display().to_string()))
}

/// The directory this process stores checkpoints in.
pub fn checkpoint_dir() -> Result<String, String> {
    Ok(Store::from_env()?.dir().display().to_string())
}

// ---------------------------------------------------------------------------
// Pause (see control.rs)
// ---------------------------------------------------------------------------
//
// ```python
// def pause():
//     stop_requested.set()          # e.g. let the training loop save a checkpoint and return
//
// tc.register_pause(pause)          # `/pause` in the controller now calls pause()
// ```
//
// `/run_script` passes a private control directory in `TC_CONTROL_DIR`. A
// listener thread watches it; `/pause` drops a request file there and the
// listener calls the registered function. `/resume` starts the script again
// with the flags of the last run plus `--resume`.

/// Registers the function that the controller's `/pause` command calls, and
/// starts the listener that waits for the request.
///
/// The function runs on a background thread (not the thread that called
/// `register_pause`), with no arguments, so it should only signal the training
/// code or do work that is safe to run beside it. Registering again replaces
/// the function.
///
/// Returns `true` when `/pause` can reach the function, and `false` when the
/// script was not started by the controller (nothing can send it a request).
pub fn register_pause(handler: std::sync::Arc<dyn Fn() + Send + Sync>) -> Result<bool, String> {
    crate::control::register_pause_handler(handler)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_message_prefixes_the_text() {
        assert_eq!(
            format_message("hello from python"),
            "[training_controller] hello from python"
        );
    }

    #[test]
    fn format_message_keeps_empty_and_unicode_input() {
        assert_eq!(format_message(""), "[training_controller] ");
        assert_eq!(format_message("héllo ✓"), "[training_controller] héllo ✓");
    }

    #[test]
    fn print_message_does_not_panic() {
        print_message("test output");
    }

    // --- training dashboard -------------------------------------------------

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| item.to_string()).collect()
    }

    fn temp_events_path(label: &str) -> PathBuf {
        let unique = now_ms();
        std::env::temp_dir().join(format!(
            "training_controller-api-{label}-{}-{unique}.jsonl",
            std::process::id()
        ))
    }

    fn read_events(path: &Path) -> Vec<TrainingEvent> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn widget_specs_parse_pairs_and_aliases() {
        let specs =
            parse_widget_specs(&strings(&["loss", "curve", " acc ", "GRAPH", "lr", "box"])).unwrap();
        assert_eq!(
            specs,
            vec![
                WidgetSpec { name: "loss".into(), kind: WidgetKind::Curve },
                WidgetSpec { name: "acc".into(), kind: WidgetKind::Graph },
                WidgetSpec { name: "lr".into(), kind: WidgetKind::Block },
            ]
        );
        assert_eq!(WidgetKind::Block.as_str(), "box");
    }

    #[test]
    fn widget_specs_reject_bad_input() {
        assert!(parse_widget_specs(&[]).unwrap_err().contains("at least one"));
        assert!(parse_widget_specs(&strings(&["loss"])).unwrap_err().contains("pairs"));
        assert!(parse_widget_specs(&strings(&["", "curve"])).unwrap_err().contains("empty"));
        assert!(
            parse_widget_specs(&strings(&["a", "curve", "a", "box"]))
                .unwrap_err()
                .contains("twice")
        );
        let error = parse_widget_specs(&strings(&["a", "pie"])).unwrap_err();
        assert!(error.contains("'a'") && error.contains("graph, box or curve"));
    }

    #[test]
    fn events_round_trip_through_json() {
        let event = TrainingEvent::Init {
            t: 5,
            widgets: vec![WidgetSpec { name: "lr".into(), kind: WidgetKind::Block }],
        };
        let text = serde_json::to_string(&event).unwrap();
        assert_eq!(
            text,
            r#"{"event":"init","t":5,"widgets":[{"name":"lr","kind":"box"}]}"#
        );
        assert_eq!(serde_json::from_str::<TrainingEvent>(&text).unwrap(), event);
    }

    #[test]
    fn writer_records_init_updates_and_hero() {
        let path = temp_events_path("writer");
        let widgets = parse_widget_specs(&strings(&["loss", "curve", "lr", "box"])).unwrap();
        let mut writer = TrainingWriter::create(&path, widgets).unwrap();
        writer.update(&[0.5, 0.001]).unwrap();
        writer.update(&[f64::NAN, f64::INFINITY]).unwrap();
        writer.hero("Training at Epoch 1").unwrap();

        let events = read_events(&path);
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], TrainingEvent::Init { widgets, .. } if widgets.len() == 2));
        assert!(matches!(&events[1], TrainingEvent::Update { values, .. } if *values == vec![Some(0.5), Some(0.001)]));
        assert!(matches!(&events[2], TrainingEvent::Update { values, .. } if *values == vec![None, None]));
        assert!(matches!(&events[3], TrainingEvent::Hero { text, .. } if text == "Training at Epoch 1"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn writer_rejects_wrong_value_counts_and_restarts_the_file() {
        let path = temp_events_path("restart");
        let widgets = parse_widget_specs(&strings(&["a", "curve", "b", "graph"])).unwrap();
        let mut writer = TrainingWriter::create(&path, widgets.clone()).unwrap();
        let error = writer.update(&[1.0]).unwrap_err();
        assert!(error.contains("expected 2 values (a, b) but got 1"));
        writer.update(&[1.0, 2.0]).unwrap();
        assert_eq!(read_events(&path).len(), 2);

        // A second run truncates the file.
        let _second = TrainingWriter::create(&path, widgets).unwrap();
        assert_eq!(read_events(&path).len(), 1);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn heartbeat_keeps_the_file_modified_time_fresh_until_the_writer_is_dropped() {
        let path = temp_events_path("beat");
        let widgets = parse_widget_specs(&strings(&["a", "curve"])).unwrap();
        let writer =
            TrainingWriter::create_with_heartbeat(&path, widgets, Duration::from_millis(30)).unwrap();

        // Pretend the file was last touched an hour ago.
        let old = SystemTime::now() - Duration::from_secs(3600);
        OpenOptions::new().write(true).open(&path).unwrap().set_modified(old).unwrap();
        std::thread::sleep(Duration::from_millis(400));
        let touched = fs::metadata(&path).unwrap().modified().unwrap();
        assert!(touched > old + Duration::from_secs(1800), "heartbeat did not touch the file");

        // After the writer is gone the beat stops.
        drop(writer);
        std::thread::sleep(Duration::from_millis(200));
        OpenOptions::new().write(true).open(&path).unwrap().set_modified(old).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        let after = fs::metadata(&path).unwrap().modified().unwrap();
        assert!(after <= old + Duration::from_secs(1), "heartbeat kept running after drop");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn experiment_id_is_parsed_from_the_environment_text() {
        assert_eq!(parse_experiment_id(None), Ok(None));
        assert_eq!(parse_experiment_id(Some("  ")), Ok(None));
        assert_eq!(parse_experiment_id(Some(" 12 ")), Ok(Some(12)));
        let error = parse_experiment_id(Some("abc")).unwrap_err();
        assert!(error.contains("TC_EXPERIMENT_ID") && error.contains("'abc'"));
    }

    #[test]
    fn init_log_rejects_bad_titles_before_touching_the_database() {
        assert!(init_log(&[]).unwrap_err().contains("at least one"));
        assert!(init_log(&strings(&["a", "a"])).unwrap_err().contains("twice"));
    }

    #[test]
    fn script_events_file_sits_next_to_the_script() {
        let script = std::env::current_dir().unwrap().join("train.py");
        assert_eq!(
            training_file_for_script(&script).unwrap(),
            std::env::current_dir().unwrap().join(TRAINING_FILE_NAME)
        );
    }
}
