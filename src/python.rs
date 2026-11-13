//! PyO3 bindings: everything registered in `training_controller` below becomes
//! available from Python as `import training_controller as tc; tc.<name>(...)`.
//!
//! The full guide for adding functions lives at the top of `api.rs`.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::api;

/// Print a string sent from Python.
///
/// Usage: `tc.test("hello")`
#[pyfunction]
#[pyo3(name = "test", text_signature = "(message)")]
fn py_test(message: String) {
    api::print_message(&message);
}

/// Declare the training dashboard variables and start a new run.
///
/// Takes alternating variable names and types; the type is one of "graph"
/// (bar graph), "curve" (line plot) or "box" (latest value block).
///
/// Usage: `tc.init_training_dashboard("loss", "curve", "acc", "graph", "lr", "box")`
///
/// This only records the layout. Open the dashboard from the REPL with
/// `/dashboard training`.
#[pyfunction]
#[pyo3(name = "init_training_dashboard", signature = (*args), text_signature = "(*args)")]
fn py_init_training_dashboard(args: &Bound<'_, PyTuple>) -> PyResult<()> {
    let args: Vec<String> = args.extract().map_err(|_| {
        PyTypeError::new_err(
            "init_training_dashboard expects strings: (name, type, name, type, ...)",
        )
    })?;
    api::init_training_dashboard(&args).map_err(PyValueError::new_err)
}

/// Record one value per variable, in the order given to init_training_dashboard.
///
/// Usage: `tc.update_training_dashboard(0.0023, 0.91, 1e-4)`
///
/// Raises RuntimeError if the dashboard is not initialized or the number of
/// values does not match the number of variables. NaN and infinite values are
/// shown as gaps.
#[pyfunction]
#[pyo3(name = "update_training_dashboard", signature = (*values), text_signature = "(*values)")]
fn py_update_training_dashboard(values: &Bound<'_, PyTuple>) -> PyResult<()> {
    let values: Vec<f64> = values.extract().map_err(|_| {
        PyTypeError::new_err("update_training_dashboard expects numbers: (val_1, val_2, ...)")
    })?;
    api::update_training_dashboard(&values).map_err(PyRuntimeError::new_err)
}

/// Set the bold text shown in the center of the training dashboard.
///
/// Usage: `tc.update_hero_text("Training at Epoch 47, step 4700, Total Loss at 0.0023")`
///
/// Raises RuntimeError if the dashboard is not initialized.
#[pyfunction]
#[pyo3(name = "update_hero_text", text_signature = "(text)")]
fn py_update_hero_text(text: String) -> PyResult<()> {
    api::update_hero_text(&text).map_err(PyRuntimeError::new_err)
}

/// Declare the columns of this run's log and start an empty log.
///
/// Takes one title per column. Returns the experiment ID that the Training
/// Controller assigned to this run.
///
/// Usage: `exp_id = tc.init_log("loss", "accuracy", "lr")`
///
/// Call it once before the training loop, then call `tc.log(...)` every step.
/// Calling it again restarts this experiment's log. Raises ValueError for no
/// titles, an empty title or a duplicate title.
#[pyfunction]
#[pyo3(name = "init_log", signature = (*titles), text_signature = "(*titles)")]
fn py_init_log(titles: &Bound<'_, PyTuple>) -> PyResult<i64> {
    let titles: Vec<String> = titles
        .extract()
        .map_err(|_| PyTypeError::new_err("init_log expects strings: (title_1, title_2, ...)"))?;
    api::init_log(&titles).map_err(PyValueError::new_err)
}

/// Append one row to the log: one value per column, in init_log order.
///
/// Usage: `tc.log(0.0023, 0.91, 1e-4)`
///
/// Every call is one step (0, 1, 2, ...) and is saved to the SQLite database
/// immediately, together with its time in seconds since the first `log` call
/// (the first row has time 0). NaN and infinite values are stored as NULL.
/// Raises RuntimeError if init_log was not called or the number of values does
/// not match the number of titles.
#[pyfunction]
#[pyo3(name = "log", signature = (*values), text_signature = "(*values)")]
fn py_log(values: &Bound<'_, PyTuple>) -> PyResult<()> {
    let values: Vec<f64> = values
        .extract()
        .map_err(|_| PyTypeError::new_err("log expects numbers: (val_1, val_2, ...)"))?;
    api::log(&values).map_err(PyRuntimeError::new_err)
}

/// The experiment ID of this run, or None when the script was not started by
/// the Training Controller and init_log has not been called yet.
///
/// Usage: `print(tc.experiment_id())`
#[pyfunction]
#[pyo3(name = "experiment_id", text_signature = "()")]
fn py_experiment_id() -> Option<i64> {
    api::experiment_id()
}

/// Store a checkpoint and return the path of the stored file.
///
/// `data` is the checkpoint as bytes, in any format you like (for example
/// `torch.save(state, buf)` into an `io.BytesIO` and then `buf.getvalue()`, or
/// `pickle.dumps(state)`); it is stored unchanged. Pass `bytes` (use
/// `bytes(x)` to convert a bytearray).
///
/// `is_best` is True when this is the best checkpoint so far. The retention
/// settings (see `/set_checkpoint_config` in the Training Controller) can
/// protect best checkpoints. `step` and `metrics` (a dict of numbers) are
/// recorded with the checkpoint and shown by `/checkpoints`.
///
/// Returns the stored file's path, or None when the settings chose not to
/// store it (`save_every`, or a retention policy that would delete it
/// immediately). Raises RuntimeError for empty data, a full disk, a failed
/// integrity check or a damaged checkpoint index.
///
/// Usage: `path = tc.save_checkpoint(data, is_best, step=epoch, metrics={"loss": loss})`
#[pyfunction]
#[pyo3(
    name = "save_checkpoint",
    signature = (data, is_best, step=None, metrics=None),
    text_signature = "(data, is_best, step=None, metrics=None)"
)]
fn py_save_checkpoint(
    data: Cow<'_, [u8]>,
    is_best: bool,
    step: Option<u64>,
    metrics: Option<HashMap<String, f64>>,
) -> PyResult<Option<String>> {
    let metrics: BTreeMap<String, f64> = metrics.unwrap_or_default().into_iter().collect();
    api::save_checkpoint(&data, is_best, step, metrics).map_err(PyRuntimeError::new_err)
}

/// Store a checkpoint that is already in a file, and return the stored path.
///
/// Same as `save_checkpoint`, but takes the path of a file (for example one
/// written with `torch.save(model, path)`) and streams it, so a very large
/// checkpoint never has to be held in memory. With `move_file=True` the
/// original file is deleted once the checkpoint is safely stored.
///
/// Usage: `tc.save_checkpoint_file("model.pt", True, step=100, move_file=True)`
#[pyfunction]
#[pyo3(
    name = "save_checkpoint_file",
    signature = (path, is_best, step=None, metrics=None, move_file=false),
    text_signature = "(path, is_best, step=None, metrics=None, move_file=False)"
)]
fn py_save_checkpoint_file(
    path: String,
    is_best: bool,
    step: Option<u64>,
    metrics: Option<HashMap<String, f64>>,
    move_file: bool,
) -> PyResult<Option<String>> {
    let metrics: BTreeMap<String, f64> = metrics.unwrap_or_default().into_iter().collect();
    api::save_checkpoint_file(&path, is_best, step, metrics, move_file)
        .map_err(PyRuntimeError::new_err)
}

/// Path of the newest stored checkpoint (of any run), or None.
///
/// Usage: `resume = tc.latest_checkpoint()`
#[pyfunction]
#[pyo3(name = "latest_checkpoint", text_signature = "()")]
fn py_latest_checkpoint() -> PyResult<Option<String>> {
    api::latest_checkpoint().map_err(PyRuntimeError::new_err)
}

/// Path of the newest checkpoint that was saved with is_best=True, or None.
///
/// Usage: `best = tc.best_checkpoint()`
#[pyfunction]
#[pyo3(name = "best_checkpoint", text_signature = "()")]
fn py_best_checkpoint() -> PyResult<Option<String>> {
    api::best_checkpoint().map_err(PyRuntimeError::new_err)
}

/// The folder this script stores its checkpoints in.
///
/// Usage: `print(tc.checkpoint_dir())`
#[pyfunction]
#[pyo3(name = "checkpoint_dir", text_signature = "()")]
fn py_checkpoint_dir() -> PyResult<String> {
    api::checkpoint_dir().map_err(PyRuntimeError::new_err)
}

/// Register the script's pause method: the function that the Training
/// Controller's `/pause` command calls.
///
/// `callback` takes no arguments. It runs on a background thread while the
/// training code keeps running, so it normally just signals the training code
/// (for example sets a `threading.Event` that the loop checks, saves a
/// checkpoint and returns). Stopping the script is up to the pause method: when
/// the script exits with code 0 after `/pause`, the run is recorded as paused,
/// and `/resume` starts it again with the same flags plus `--resume`.
///
/// Returns True when `/pause` can reach the function, and False when the script
/// was not started by the Training Controller. An exception raised by the
/// function is printed to the `error` panel. Registering again replaces the
/// function.
///
/// Usage: `tc.register_pause(pause)`
#[pyfunction]
#[pyo3(name = "register_pause", text_signature = "(callback)")]
fn py_register_pause(py: Python<'_>, callback: Py<PyAny>) -> PyResult<bool> {
    if !callback.bind(py).is_callable() {
        return Err(PyTypeError::new_err(
            "register_pause expects a function that takes no arguments",
        ));
    }
    api::register_pause(Arc::new(move || {
        Python::attach(|py| {
            if let Err(error) = callback.call0(py) {
                eprintln!("[error] the pause method raised an exception: {error}");
                error.print(py);
            }
        });
    }))
    .map_err(PyRuntimeError::new_err)
}

/// The module name must match `[lib] name` in Cargo.toml.
#[pymodule]
fn training_controller(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_test, m)?)?;
    m.add_function(wrap_pyfunction!(py_init_training_dashboard, m)?)?;
    m.add_function(wrap_pyfunction!(py_update_training_dashboard, m)?)?;
    m.add_function(wrap_pyfunction!(py_update_hero_text, m)?)?;
    m.add_function(wrap_pyfunction!(py_init_log, m)?)?;
    m.add_function(wrap_pyfunction!(py_log, m)?)?;
    m.add_function(wrap_pyfunction!(py_experiment_id, m)?)?;
    m.add_function(wrap_pyfunction!(py_save_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(py_save_checkpoint_file, m)?)?;
    m.add_function(wrap_pyfunction!(py_latest_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(py_best_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(py_checkpoint_dir, m)?)?;
    m.add_function(wrap_pyfunction!(py_register_pause, m)?)?;
    // Register future functions here:
    // m.add_function(wrap_pyfunction!(py_my_function, m)?)?;
    Ok(())
}
