//! Plain-Rust functions that are exposed to Python.
//!
//! Keep these free of any PyO3 types so they stay testable and reusable from
//! the REPL. To expose a new function to Python, add it here and then add a
//! thin wrapper plus a `m.add_function(...)` line in `python.rs`.

/// Builds the line that `print_message` writes.
pub fn format_message(message: &str) -> String {
    format!("[training_controller] {message}")
}

/// Prints a message received from the caller (sample function).
pub fn print_message(message: &str) {
    println!("{}", format_message(message));
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
}
