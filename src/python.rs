//! PyO3 bindings: everything registered in `training_controller` below becomes
//! available from Python as `import training_controller as tc; tc.<name>(...)`.

use pyo3::prelude::*;

use crate::api;

/// Print a string sent from Python.
///
/// Usage: `tc.test("hello")`
#[pyfunction(name = "test")]
#[pyo3(text_signature = "(message)")]
fn py_test(message: &str) {
    api::print_message(message);
}

/// The module name must match `[lib] name` in Cargo.toml.
#[pymodule]
fn training_controller(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_test, m)?)?;
    // Register future functions here:
    // m.add_function(wrap_pyfunction!(py_my_function, m)?)?;
    Ok(())
}
