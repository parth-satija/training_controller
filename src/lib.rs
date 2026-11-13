//! Library root for `training_controller`.
//!
//! * [`api`] holds the plain-Rust functions. Add new functionality here.
//! * [`logdb`] stores experiments and training logs in SQLite (used by the REPL
//!   to create experiment IDs and by `api` for `init_log` / `log`).
//! * [`checkpoint`] stores model checkpoints safely (atomic writes, SHA-256,
//!   retention policies); used by `api` for `save_checkpoint` and by the REPL's
//!   `/checkpoints` commands.
//! * [`control`] is the pause protocol between the REPL and a running script
//!   (`/pause`, `tc.register_pause`).
//! * `python` (feature `python`) exposes those functions to Python as the
//!   `training_controller` module. The REPL binary in `main.rs` is unaffected.

pub mod api;
pub mod checkpoint;
pub mod control;
pub mod logdb;

#[cfg(feature = "python")]
mod python;
