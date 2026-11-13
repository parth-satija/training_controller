//! Live output panels for scripts started with `/run_script`.
//!
//! A script's stdout and stderr are captured line by line and routed into
//! named panels. The REPL stays free; `/panel NAME` opens a live view.
//!
//! # Routing rules (applied to every printed line)
//!
//! * Every line goes to the **`output`** panel, exactly as printed.
//! * A line that starts with a `[tag]` goes to the panel named `tag`, with the
//!   tag (and one following space) removed:
//!   * `[error] ...` -> the **`error`** panel (shown in red)
//!   * `[log] ...`   -> the **`log`** panel
//!   * `[text_example] ...` -> a panel called **`text_example`**, created on
//!     first use. Any tag works.
//! * A line without a tag goes to the **`other`** panel, which therefore never
//!   contains tagged lines.
//!
//! Tags are case-sensitive (`[Error]` is not `[error]`) and may contain letters,
//! digits, `_`, `-` and `.`. `output` and `other` are reserved and are not
//! treated as tags. Each panel keeps its last [`MAX_LINES`] lines.
//!
//! The module is layered like `dashboard.rs`: routing and storage
//! ([`PanelStore`]) and rendering ([`render_panel`]) are pure and tested;
//! [`attach_child`] and [`open_panel`] do the I/O.

use crate::dashboard::{fit, fmt_duration, run_live, Frame};
use std::collections::{BTreeMap, VecDeque};
use std::io::{self, BufRead, BufReader, IsTerminal, Read};
use std::process::Child;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub const OUTPUT_PANEL: &str = "output";
pub const ERROR_PANEL: &str = "error";
pub const LOG_PANEL: &str = "log";
pub const OTHER_PANEL: &str = "other";
const BUILT_IN: [&str; 4] = [OUTPUT_PANEL, ERROR_PANEL, LOG_PANEL, OTHER_PANEL];

/// Lines kept per panel; the oldest are dropped first.
const MAX_LINES: usize = 5_000;
/// Panels created by tags, beyond the built-in ones. Extra tags go to `other`.
const MAX_CUSTOM_PANELS: usize = 50;
const MAX_NAME_LEN: usize = 32;
const MAX_RUNS_KEPT: usize = 50;
const RUNS_SHOWN: usize = 5;
const VIEW_INTERVAL: Duration = Duration::from_millis(100);
const MIN_VIEW_WIDTH: usize = 20;
const MIN_VIEW_HEIGHT: usize = 3;

// ---------------------------------------------------------------------------
// Routing (pure)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineStyle {
    Normal,
    /// Printed in red.
    Error,
    /// Run markers written by the controller itself (dim).
    System,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanelLine {
    pub text: String,
    pub style: LineStyle,
}

fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().count() <= MAX_NAME_LEN
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

pub fn validate_panel_name(name: &str) -> std::result::Result<(), String> {
    if is_valid_name(name) {
        Ok(())
    } else {
        Err(format!(
            "Invalid panel name '{name}'. Use up to {MAX_NAME_LEN} letters, digits, '_', '-' or '.'."
        ))
    }
}

/// Splits a leading `[tag]` off a line, returning the tag and the text after it
/// (minus one separating space). `None` means the line is untagged.
pub fn parse_tag(line: &str) -> Option<(&str, &str)> {
    let rest = line.strip_prefix('[')?;
    let end = rest.find(']')?;
    let tag = &rest[..end];
    if !is_valid_name(tag) || tag == OUTPUT_PANEL || tag == OTHER_PANEL {
        return None;
    }
    let after = &rest[end + 1..];
    Some((tag, after.strip_prefix(' ').unwrap_or(after)))
}

/// Makes a raw line safe to store and draw: drops ANSI escape sequences and
/// control characters, expands tabs, and keeps only the final segment of a
/// carriage-return "progress bar" line.
pub fn sanitize_line(raw: &str) -> String {
    let raw = raw.trim_end_matches(['\r', '\n']);
    let raw = raw.rsplit('\r').next().unwrap_or(raw);

    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        match ch {
            '\x1b' => {
                // CSI sequence: ESC [ parameters, ended by a byte in 0x40..=0x7e.
                if chars.clone().next() == Some('[') {
                    chars.next();
                    for next in chars.by_ref() {
                        if ('\x40'..='\x7e').contains(&next) {
                            break;
                        }
                    }
                } else {
                    chars.next();
                }
            }
            '\t' => out.push_str("    "),
            ch if ch.is_control() => {}
            ch => out.push(ch),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Store (pure)
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Panel {
    lines: VecDeque<PanelLine>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunStatus {
    Running,
    Exited(i32),
    /// Exited with code 0 after `/pause` was acknowledged by the script.
    Paused,
    Killed,
    Failed(String),
}

impl RunStatus {
    fn describe(&self) -> String {
        match self {
            Self::Running => "running".to_string(),
            Self::Exited(0) => "finished (exit 0)".to_string(),
            Self::Exited(code) => format!("exited with code {code}"),
            Self::Paused => "paused".to_string(),
            Self::Killed => "terminated by a signal".to_string(),
            Self::Failed(error) => format!("failed: {error}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunInfo {
    pub id: u32,
    pub label: String,
    pub status: RunStatus,
    started: Instant,
    duration: Option<Duration>,
    /// `/pause` reached the script; a clean exit then counts as paused.
    pause_requested: bool,
}

#[derive(Debug)]
pub struct PanelStore {
    panels: BTreeMap<String, Panel>,
    runs: Vec<RunInfo>,
}

impl Default for PanelStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PanelStore {
    pub fn new() -> Self {
        let panels = BUILT_IN
            .iter()
            .map(|name| (name.to_string(), Panel::default()))
            .collect();
        Self {
            panels,
            runs: Vec::new(),
        }
    }

    fn custom_count(&self) -> usize {
        self.panels
            .keys()
            .filter(|name| !BUILT_IN.contains(&name.as_str()))
            .count()
    }

    fn accepts_tag(&self, tag: &str) -> bool {
        self.panels.contains_key(tag) || self.custom_count() < MAX_CUSTOM_PANELS
    }

    fn push(&mut self, panel: &str, text: String, style: LineStyle) {
        let panel = self.panels.entry(panel.to_string()).or_default();
        if panel.lines.len() >= MAX_LINES {
            panel.lines.pop_front();
        }
        panel.lines.push_back(PanelLine { text, style });
    }

    /// Routes one printed line into the right panels (see the module docs).
    pub fn ingest(&mut self, raw: &str) {
        let text = sanitize_line(raw);
        let routed = parse_tag(&text).map(|(tag, rest)| (tag.to_string(), rest.to_string()));

        match routed {
            Some((tag, rest)) if self.accepts_tag(&tag) => {
                let style = if tag == ERROR_PANEL {
                    LineStyle::Error
                } else {
                    LineStyle::Normal
                };
                self.push(OUTPUT_PANEL, text, style);
                self.push(&tag, rest, style);
            }
            _ => {
                self.push(OUTPUT_PANEL, text.clone(), LineStyle::Normal);
                self.push(OTHER_PANEL, text, LineStyle::Normal);
            }
        }
    }

    pub fn start_run(&mut self, label: &str) -> u32 {
        let id = self.runs.last().map_or(1, |run| run.id + 1);
        self.runs.push(RunInfo {
            id,
            label: label.to_string(),
            status: RunStatus::Running,
            started: Instant::now(),
            duration: None,
            pause_requested: false,
        });
        if self.runs.len() > MAX_RUNS_KEPT {
            self.runs.remove(0);
        }
        self.push(
            OUTPUT_PANEL,
            format!("=== run #{id} started: {label} ==="),
            LineStyle::System,
        );
        id
    }

    /// Records how a run ended and returns the final status. A clean exit
    /// (code 0) of a run whose pause was requested becomes [`RunStatus::Paused`].
    pub fn finish_run(&mut self, id: u32, status: RunStatus) -> RunStatus {
        let mut final_status = status.clone();
        let mut marker = None;
        if let Some(run) = self.runs.iter_mut().find(|run| run.id == id) {
            if run.pause_requested && status == RunStatus::Exited(0) {
                final_status = RunStatus::Paused;
            }
            run.duration = Some(run.started.elapsed());
            marker = Some(format!("=== run #{id} {} ===", final_status.describe()));
            run.status = final_status.clone();
        }
        if let Some(marker) = marker {
            self.push(OUTPUT_PANEL, marker, LineStyle::System);
        }
        final_status
    }

    /// The status of a run, `None` when the run is unknown (or forgotten).
    pub fn run_status(&self, id: u32) -> Option<RunStatus> {
        self.runs
            .iter()
            .find(|run| run.id == id)
            .map(|run| run.status.clone())
    }

    /// Remembers that `/pause` reached a running run. Returns `false` when the
    /// run is unknown or not running.
    pub fn mark_pause_requested(&mut self, id: u32) -> bool {
        match self.runs.iter_mut().find(|run| run.id == id) {
            Some(run) if run.status == RunStatus::Running => {
                run.pause_requested = true;
                true
            }
            _ => false,
        }
    }

    /// Whether `/pause` already reached this run.
    pub fn pause_requested(&self, id: u32) -> bool {
        self.runs
            .iter()
            .find(|run| run.id == id)
            .is_some_and(|run| run.pause_requested)
    }

    pub fn running_count(&self) -> usize {
        self.runs
            .iter()
            .filter(|run| run.status == RunStatus::Running)
            .count()
    }

    pub fn has_panel(&self, name: &str) -> bool {
        self.panels.contains_key(name)
    }

    pub fn line_count(&self, name: &str) -> usize {
        self.panels.get(name).map_or(0, |panel| panel.lines.len())
    }

    /// Built-in panels first, then tag panels alphabetically.
    pub fn panel_names(&self) -> Vec<String> {
        let mut names: Vec<String> = BUILT_IN.iter().map(|name| name.to_string()).collect();
        names.extend(
            self.panels
                .keys()
                .filter(|name| !BUILT_IN.contains(&name.as_str()))
                .cloned(),
        );
        names
    }

    /// Clears one panel (tag panels are removed) or all of them.
    pub fn clear(&mut self, name: Option<&str>) -> std::result::Result<String, String> {
        match name {
            None => {
                for panel in self.panels.values_mut() {
                    panel.lines.clear();
                }
                self.panels
                    .retain(|key, _| BUILT_IN.contains(&key.as_str()));
                Ok("Cleared all panels.".to_string())
            }
            Some(name) => {
                if !self.panels.contains_key(name) {
                    return Err(format!("Panel '{name}' does not exist."));
                }
                if BUILT_IN.contains(&name) {
                    if let Some(panel) = self.panels.get_mut(name) {
                        panel.lines.clear();
                    }
                } else {
                    self.panels.remove(name);
                }
                Ok(format!("Cleared panel '{name}'."))
            }
        }
    }
}

fn plural(count: usize, word: &str) -> String {
    if count == 1 {
        format!("{count} {word}")
    } else {
        format!("{count} {word}s")
    }
}

/// Text for `/show_active_panels`.
pub fn describe_active(store: &PanelStore) -> String {
    let names = store.panel_names();
    let width = names
        .iter()
        .map(|name| name.chars().count())
        .max()
        .unwrap_or(0);

    let running = store.running_count();
    let mut out = if running == 0 {
        "Active panels (no script running):\n".to_string()
    } else {
        format!("Active panels ({} running):\n", plural(running, "script"))
    };
    for name in &names {
        let note = match name.as_str() {
            ERROR_PANEL => "  (red)",
            OTHER_PANEL => "  (untagged prints)",
            OUTPUT_PANEL => "  (everything)",
            _ => "",
        };
        out.push_str(&format!(
            "  {}  {}{note}\n",
            fit(name, width),
            plural(store.line_count(name), "line")
        ));
    }

    if !store.runs.is_empty() {
        out.push_str("Recent runs:\n");
        let skip = store.runs.len().saturating_sub(RUNS_SHOWN);
        for run in &store.runs[skip..] {
            let elapsed = run.duration.unwrap_or_else(|| run.started.elapsed());
            let status = if run.pause_requested && run.status == RunStatus::Running {
                "running, pause requested".to_string()
            } else {
                run.status.describe()
            };
            out.push_str(&format!(
                "  #{} {} - {} ({})\n",
                run.id,
                run.label,
                status,
                fmt_duration(elapsed)
            ));
        }
    }
    out.push_str("Open one with /panel NAME.");
    out
}

// ---------------------------------------------------------------------------
// Rendering (pure)
// ---------------------------------------------------------------------------

fn paint(text: &str, code: &str) -> String {
    if code.is_empty() {
        text.to_string()
    } else {
        format!("\x1b[{code}m{text}\x1b[0m")
    }
}

fn style_code(style: LineStyle) -> &'static str {
    match style {
        LineStyle::Normal => "",
        LineStyle::Error => "31",
        LineStyle::System => "2",
    }
}

/// Hard-wraps `text` into rows of at most `width` characters (never zero rows).
fn wrap_rows(text: &str, width: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return vec![String::new()];
    }
    chars
        .chunks(width.max(1))
        .map(|chunk| chunk.iter().collect())
        .collect()
}

fn pad_frame(mut lines: Vec<String>, height: usize, width: usize) -> Vec<String> {
    while lines.len() < height {
        lines.push(fit("", width));
    }
    lines.truncate(height);
    lines
}

/// Builds exactly `height` lines, each at most `width - 1` visible characters.
/// The view follows the end of the panel, like `tail -f`.
pub fn render_panel(store: &PanelStore, name: &str, width: usize, height: usize) -> Vec<String> {
    let w = width.saturating_sub(1).max(1);
    if width < MIN_VIEW_WIDTH || height < MIN_VIEW_HEIGHT {
        return pad_frame(
            vec![fit(
                &format!(" Terminal too small (need {MIN_VIEW_WIDTH}x{MIN_VIEW_HEIGHT})"),
                w,
            )],
            height,
            w,
        );
    }

    let seen = if store.has_panel(name) {
        ""
    } else {
        "  ·  not seen yet"
    };
    let running = if store.running_count() > 0 {
        "  ·  ● running"
    } else {
        ""
    };
    let header = format!(
        " Panel: {name}  ·  {}{seen}{running}",
        plural(store.line_count(name), "line")
    );

    let body_rows = height - 2;
    let mut rows: Vec<(String, LineStyle)> = Vec::new();
    if let Some(panel) = store.panels.get(name) {
        // Collect wrapped rows from the end until the screen is full.
        for line in panel.lines.iter().rev() {
            for piece in wrap_rows(&line.text, w).into_iter().rev() {
                rows.push((piece, line.style));
            }
            if rows.len() >= body_rows {
                break;
            }
        }
        rows.reverse();
        let skip = rows.len().saturating_sub(body_rows);
        rows.drain(..skip);
    }

    let mut lines = vec![paint(&fit(&header, w), "1;7")];
    if rows.is_empty() {
        lines.push(paint(&fit("  Waiting for output...", w), "2"));
    }
    for (text, style) in rows {
        lines.push(paint(&fit(&text, w), style_code(style)));
    }
    while lines.len() < height - 1 {
        lines.push(fit("", w));
    }
    lines.push(paint(&fit(" q / Esc / Ctrl+C to exit  ·  live", w), "2"));
    pad_frame(lines, height, w)
}

// ---------------------------------------------------------------------------
// Shared handle and I/O
// ---------------------------------------------------------------------------

/// A cheaply cloneable handle on the shared [`PanelStore`], used by the reader
/// threads (writing) and the live view (reading).
#[derive(Clone, Default)]
pub struct PanelHub {
    inner: Arc<Mutex<PanelStore>>,
}

impl PanelHub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs `f` with the store locked. A poisoned lock is recovered because the
    /// store has no invariant that a panicking writer could leave half-done.
    pub fn with<R>(&self, f: impl FnOnce(&mut PanelStore) -> R) -> R {
        let mut guard = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        f(&mut guard)
    }
}

/// Reads lines until the stream ends, routing each into the hub.
fn pump<R: Read>(reader: R, hub: PanelHub) {
    let mut reader = BufReader::new(reader);
    let mut buffer = Vec::new();
    loop {
        buffer.clear();
        match reader.read_until(b'\n', &mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let line = String::from_utf8_lossy(&buffer).into_owned();
                hub.with(|store| store.ingest(&line));
            }
        }
    }
}

/// Starts capturing a spawned script's stdout and stderr into panels and
/// returns immediately with the run number. A background thread records how
/// the script ended once all of its output has been read.
pub fn attach_child(hub: &PanelHub, label: &str, child: Child) -> u32 {
    attach_child_with(hub, label, child, |_| {})
}

/// [`attach_child`] that also calls `on_exit` (on a background thread, after
/// all output has been read) with the final status. The controller uses it to
/// record how an experiment ended.
pub fn attach_child_with(
    hub: &PanelHub,
    label: &str,
    mut child: Child,
    on_exit: impl FnOnce(&RunStatus) + Send + 'static,
) -> u32 {
    let run_id = hub.with(|store| store.start_run(label));

    let mut readers = Vec::new();
    if let Some(stdout) = child.stdout.take() {
        let hub = hub.clone();
        readers.push(thread::spawn(move || pump(stdout, hub)));
    }
    if let Some(stderr) = child.stderr.take() {
        let hub = hub.clone();
        readers.push(thread::spawn(move || pump(stderr, hub)));
    }

    let hub = hub.clone();
    thread::spawn(move || {
        let status = match child.wait() {
            Ok(status) => match status.code() {
                Some(code) => RunStatus::Exited(code),
                None => RunStatus::Killed,
            },
            Err(error) => RunStatus::Failed(error.to_string()),
        };
        for reader in readers {
            let _ = reader.join();
        }
        let final_status = hub.with(|store| store.finish_run(run_id, status));
        on_exit(&final_status);
    });

    run_id
}

/// Opens the live view of one panel until the user presses q, Esc or Ctrl+C.
/// A panel that has not received any line yet can still be opened; it fills in
/// as soon as the script prints something with that tag.
pub fn open_panel(hub: &PanelHub, name: &str) -> std::result::Result<String, String> {
    validate_panel_name(name)?;
    if !io::stdout().is_terminal() {
        return Err("Panels need an interactive terminal.".to_string());
    }

    run_live(|width, height| Frame {
        lines: hub.with(|store| render_panel(store, name, width, height)),
        interval: VIEW_INTERVAL,
    })?;
    Ok(format!("Closed panel '{name}'."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn strip_ansi(text: &str) -> String {
        let mut out = String::new();
        let mut in_escape = false;
        for ch in text.chars() {
            if in_escape {
                if ch.is_ascii_alphabetic() {
                    in_escape = false;
                }
            } else if ch == '\x1b' {
                in_escape = true;
            } else {
                out.push(ch);
            }
        }
        out
    }

    fn texts(store: &PanelStore, panel: &str) -> Vec<String> {
        store.panels[panel]
            .lines
            .iter()
            .map(|line| line.text.clone())
            .collect()
    }

    #[test]
    fn parse_tag_splits_valid_tags_only() {
        assert_eq!(parse_tag("[error] boom"), Some(("error", "boom")));
        assert_eq!(parse_tag("[log]no space"), Some(("log", "no space")));
        assert_eq!(parse_tag("[text_example]   x"), Some(("text_example", "  x")));
        assert_eq!(parse_tag("[error]"), Some(("error", "")));

        assert_eq!(parse_tag("plain"), None);
        assert_eq!(parse_tag(" [error] indented"), None);
        assert_eq!(parse_tag("[] empty"), None);
        assert_eq!(parse_tag("[1/10] progress"), None);
        assert_eq!(parse_tag("[has space] x"), None);
        assert_eq!(parse_tag("[unterminated"), None);
        // Reserved panel names are never tags.
        assert_eq!(parse_tag("[output] x"), None);
        assert_eq!(parse_tag("[other] x"), None);
        assert_eq!(parse_tag(&format!("[{}] x", "a".repeat(40))), None);
    }

    #[test]
    fn sanitize_strips_escapes_controls_and_progress_updates() {
        assert_eq!(sanitize_line("hello\r\n"), "hello");
        assert_eq!(sanitize_line("\x1b[31mred\x1b[0m text"), "red text");
        assert_eq!(sanitize_line("a\tb"), "a    b");
        assert_eq!(sanitize_line("10%\r50%\r100%"), "100%");
        assert_eq!(sanitize_line("bell\x07"), "bell");
        assert_eq!(sanitize_line("héllo ✓"), "héllo ✓");
        assert_eq!(sanitize_line("\x1b"), "");
    }

    #[test]
    fn ingest_routes_lines_to_the_right_panels() {
        let mut store = PanelStore::new();
        store.ingest("plain line\n");
        store.ingest("[error] bad thing\n");
        store.ingest("[log] note\n");
        store.ingest("[text_example] hello\n");
        store.ingest("[Error] wrong case\n");

        // `output` has everything, exactly as printed.
        assert_eq!(
            texts(&store, OUTPUT_PANEL),
            vec![
                "plain line",
                "[error] bad thing",
                "[log] note",
                "[text_example] hello",
                "[Error] wrong case"
            ]
        );
        assert_eq!(texts(&store, ERROR_PANEL), vec!["bad thing"]);
        assert_eq!(texts(&store, LOG_PANEL), vec!["note"]);
        assert_eq!(texts(&store, "text_example"), vec!["hello"]);
        // Tags are case-sensitive: this one is just another custom panel.
        assert_eq!(texts(&store, "Error"), vec!["wrong case"]);
        // `other` holds only untagged lines.
        assert_eq!(texts(&store, OTHER_PANEL), vec!["plain line"]);
    }

    #[test]
    fn error_lines_are_styled_red_in_output_and_error_panels() {
        let mut store = PanelStore::new();
        store.ingest("[error] x");
        store.ingest("[log] y");
        assert_eq!(store.panels[ERROR_PANEL].lines[0].style, LineStyle::Error);
        assert_eq!(store.panels[OUTPUT_PANEL].lines[0].style, LineStyle::Error);
        assert_eq!(store.panels[LOG_PANEL].lines[0].style, LineStyle::Normal);
    }

    #[test]
    fn panels_are_bounded() {
        let mut store = PanelStore::new();
        for index in 0..(MAX_LINES + 10) {
            store.ingest(&format!("line {index}"));
        }
        assert_eq!(store.line_count(OTHER_PANEL), MAX_LINES);
        assert_eq!(texts(&store, OTHER_PANEL)[0], "line 10");

        // Too many distinct tags fall back to `other` instead of growing forever.
        let mut store = PanelStore::new();
        for index in 0..(MAX_CUSTOM_PANELS + 5) {
            store.ingest(&format!("[tag{index}] x"));
        }
        assert_eq!(store.panel_names().len(), BUILT_IN.len() + MAX_CUSTOM_PANELS);
        assert_eq!(store.line_count(OTHER_PANEL), 5);
        // An existing tag keeps working once the limit is reached.
        store.ingest("[tag0] again");
        assert_eq!(store.line_count("tag0"), 2);
    }

    #[test]
    fn runs_are_tracked_and_marked_in_the_output_panel() {
        let mut store = PanelStore::new();
        let first = store.start_run("sample (fast)");
        let second = store.start_run("other");
        assert_eq!((first, second), (1, 2));
        assert_eq!(store.running_count(), 2);

        store.finish_run(first, RunStatus::Exited(0));
        store.finish_run(second, RunStatus::Exited(3));
        store.finish_run(99, RunStatus::Killed); // unknown id is ignored
        assert_eq!(store.running_count(), 0);

        let output = texts(&store, OUTPUT_PANEL);
        assert_eq!(output[0], "=== run #1 started: sample (fast) ===");
        assert!(output.contains(&"=== run #1 finished (exit 0) ===".to_string()));
        assert!(output.contains(&"=== run #2 exited with code 3 ===".to_string()));
        // Markers never appear in the other panels.
        assert_eq!(store.line_count(OTHER_PANEL), 0);
    }

    #[test]
    fn a_clean_exit_after_a_pause_request_is_a_paused_run() {
        let mut store = PanelStore::new();
        let paused = store.start_run("train");
        let normal = store.start_run("train");
        let crashed = store.start_run("train");
        assert_eq!(store.run_status(paused), Some(RunStatus::Running));
        assert!(!store.pause_requested(paused));

        assert!(store.mark_pause_requested(paused));
        assert!(store.mark_pause_requested(crashed));
        assert!(store.pause_requested(paused) && !store.pause_requested(normal));
        assert!(describe_active(&store).contains("#1 train - running, pause requested"));

        assert_eq!(store.finish_run(paused, RunStatus::Exited(0)), RunStatus::Paused);
        // Only a clean exit counts: a crash after /pause stays a failure.
        assert_eq!(store.finish_run(crashed, RunStatus::Exited(1)), RunStatus::Exited(1));
        assert_eq!(store.finish_run(normal, RunStatus::Exited(0)), RunStatus::Exited(0));
        assert_eq!(store.run_status(paused), Some(RunStatus::Paused));
        assert!(texts(&store, OUTPUT_PANEL).contains(&"=== run #1 paused ===".to_string()));
        assert!(describe_active(&store).contains("#1 train - paused"));

        // A finished run cannot be paused any more, and unknown runs are ignored.
        assert!(!store.mark_pause_requested(normal));
        assert!(!store.mark_pause_requested(99));
        assert_eq!(store.run_status(99), None);
        assert_eq!(store.finish_run(99, RunStatus::Killed), RunStatus::Killed);
        assert_eq!(store.running_count(), 0);
    }

    #[test]
    fn describe_active_lists_panels_counts_and_runs() {
        let mut store = PanelStore::new();
        assert!(describe_active(&store).contains("no script running"));

        let run = store.start_run("train");
        store.ingest("[error] bad");
        store.ingest("[metrics] 1");
        store.ingest("[metrics] 2");

        let text = describe_active(&store);
        assert!(text.contains("1 script running"));
        assert!(text.contains("error") && text.contains("1 line"));
        assert!(text.contains("metrics") && text.contains("2 lines"));
        assert!(text.contains("#1 train - running"));

        store.finish_run(run, RunStatus::Exited(0));
        assert!(describe_active(&store).contains("finished (exit 0)"));
    }

    #[test]
    fn clear_empties_built_ins_and_removes_tag_panels() {
        let mut store = PanelStore::new();
        store.ingest("[log] a");
        store.ingest("[mine] b");

        assert_eq!(store.clear(Some("mine")).unwrap(), "Cleared panel 'mine'.");
        assert!(!store.has_panel("mine"));
        assert!(store.clear(Some("mine")).is_err());

        store.clear(Some(LOG_PANEL)).unwrap();
        assert_eq!(store.line_count(LOG_PANEL), 0);
        assert!(store.has_panel(LOG_PANEL));

        store.ingest("[mine] c");
        store.clear(None).unwrap();
        assert!(!store.has_panel("mine"));
        assert_eq!(store.line_count(OUTPUT_PANEL), 0);
        assert_eq!(store.panel_names(), vec!["output", "error", "log", "other"]);
    }

    #[test]
    fn render_panel_follows_the_end_and_fills_the_screen() {
        let mut store = PanelStore::new();
        for index in 0..50 {
            store.ingest(&format!("line {index}"));
        }

        for (width, height) in [(80, 24), (40, 8), (25, 3)] {
            let lines = render_panel(&store, OUTPUT_PANEL, width, height);
            assert_eq!(lines.len(), height);
            for line in &lines {
                assert!(strip_ansi(line).chars().count() < width);
            }
        }

        let text: Vec<String> = render_panel(&store, OUTPUT_PANEL, 80, 10)
            .iter()
            .map(|line| strip_ansi(line))
            .collect();
        assert!(text[0].contains("Panel: output") && text[0].contains("50 lines"));
        assert!(text[8].starts_with("line 49"));
        assert!(text[1].starts_with("line 42"));
        assert!(!text.join("\n").contains("line 41"));
        assert!(text[9].contains("q / Esc"));
    }

    #[test]
    fn render_panel_handles_empty_unknown_wrapped_and_error_lines() {
        let mut store = PanelStore::new();
        let empty: Vec<String> = render_panel(&store, LOG_PANEL, 60, 8)
            .iter()
            .map(|line| strip_ansi(line))
            .collect();
        assert!(empty[1].contains("Waiting for output"));

        let unknown = strip_ansi(&render_panel(&store, "later", 60, 8)[0]);
        assert!(unknown.contains("not seen yet"));

        store.ingest(&"x".repeat(100));
        let wrapped = render_panel(&store, OTHER_PANEL, 41, 8);
        assert_eq!(strip_ansi(&wrapped[1]).trim_end(), "x".repeat(40));
        assert_eq!(strip_ansi(&wrapped[2]).trim_end(), "x".repeat(40));
        assert_eq!(strip_ansi(&wrapped[3]).trim_end(), "x".repeat(20));

        store.ingest("[error] oops");
        let error_view = render_panel(&store, ERROR_PANEL, 60, 8);
        assert!(error_view[1].contains("\x1b[31m"));
        assert_eq!(strip_ansi(&error_view[1]).trim_end(), "oops");

        let tiny = render_panel(&store, OUTPUT_PANEL, 10, 2);
        assert_eq!(tiny.len(), 2);
    }

    #[test]
    fn pump_reads_lines_and_routes_them() {
        let hub = PanelHub::new();
        let input = b"first\n[log] two\r\n[error] three\nno newline at end".to_vec();
        pump(Cursor::new(input), hub.clone());

        hub.with(|store| {
            assert_eq!(texts(store, OTHER_PANEL), vec!["first", "no newline at end"]);
            assert_eq!(texts(store, LOG_PANEL), vec!["two"]);
            assert_eq!(texts(store, ERROR_PANEL), vec!["three"]);
            assert_eq!(store.line_count(OUTPUT_PANEL), 4);
        });
    }

    #[test]
    fn pump_tolerates_invalid_utf8() {
        let hub = PanelHub::new();
        pump(Cursor::new(vec![b'o', b'k', 0xff, b'\n']), hub.clone());
        hub.with(|store| assert_eq!(store.line_count(OTHER_PANEL), 1));
    }

    #[test]
    fn panel_names_are_validated() {
        assert!(validate_panel_name("text_example").is_ok());
        assert!(validate_panel_name("").is_err());
        assert!(validate_panel_name("has space").is_err());
        assert!(validate_panel_name(&"a".repeat(33)).is_err());
    }
}
