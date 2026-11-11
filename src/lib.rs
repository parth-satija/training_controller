//! Library root for `training_controller`.
//!
//! * [`api`] holds the plain-Rust functions. Add new functionality here.
//! * `python` (feature `python`) exposes those functions to Python as the
//!   `training_controller` module. The REPL binary in `main.rs` is unaffected.

pub mod api;

#[cfg(feature = "python")]
mod python;
