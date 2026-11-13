//! `/dashboard training`: a live view of what a Python script reports through
//! `init_training_dashboard`, `update_training_dashboard` and `update_hero_text`
//! (see `api.rs` for the file format).
//!
//! This is a child of the `dashboard` module, so it shares its config
//! (`dashboard.json`: palette, theme, number format, refresh interval, hot
//! reload) and its terminal loop. Layout, top to bottom:
//!
//! ```text
//!  Training dashboard - train.py  ·  training_dashboard.jsonl      ⏱ 00:12:34
//!  4,700 updates · 6 widgets · ● running
//!
//!  ┌─ loss ────────── 0.0023 ─┐ ┌─ val_loss ────── 0.0031 ─┐   curve  (braille line)
//!  │ ⢀⣀⡠⠔⠊⠁                   │ │ ⢀⣀⡠⠔⠊⠁                   │
//!  └─ min 0.0023 ── max 1.9 ──┘ └──────────────────────────┘
//!  ═════════════════════════════════════════════════════════
//!    Training at Epoch 47, step 4700, Total Loss at 0.0023        <- hero (bold, centered)
//!  ═════════════════════════════════════════════════════════
//!  ┌─ accuracy ────── 0.9100 ─┐ ┌─ lr ────────────────────┐   graph (bars) and box
//!  │ ▁▂▃▄▅▆▇█                 │ │         0.0010          │
//!  └──────────────────────────┘ └──────────────────────────┘
//! ```
//!
//! The hero banner sits between the upper and lower rows of widgets (above them
//! when there is only one row). Everything is drawn from the model by pure
//! functions, so the layout is unit-tested without a terminal.

use super::{
    Frame, JsonlTail, Series, Settings, SourceUpdate, blank_frame, fit, fit_right, fmt_duration,
    fmt_num, run_live, style, ConfigWatcher, MIN_HEIGHT, MIN_WIDTH,
};
use serde_json::Value;
use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};
use std::time::Duration;
use training_controller::api::{self, TrainingEvent, WidgetKind, WidgetSpec};

/// The script's heartbeat (see `api::HEARTBEAT_EVERY`) or last event is older than
/// this: the script has finished or died. A few missed beats are tolerated.
const STALE_AFTER_MS: u64 = 4_000;
const GAP: usize = 1;
const MIN_PANEL_W: usize = 24;
const MIN_PANEL_H: usize = 5;
const MAX_PANEL_H: usize = 18;
const HERO_FULL_H: usize = 3;
const DEMO_UPDATES_PER_POLL: usize = 4;
const BRAILLE_BASE: u32 = 0x2800;
/// Bit of the braille dot at column `x` (0..2), row `y` (0..4) inside one cell.
const BRAILLE_BITS: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// No `init_training_dashboard` seen yet.
    Waiting,
    /// The script is alive: events or heartbeats arrived recently.
    Running,
    /// Initialized, but the script has not been heard from for a while.
    Idle,
}

struct Widget {
    name: String,
    kind: WidgetKind,
    series: Series,
    /// The latest value was NaN or infinite.
    invalid: bool,
}

pub struct TrainingModel {
    title: String,
    source_label: String,
    settings: Settings,
    interval_override: Option<Duration>,
    widgets: Vec<Widget>,
    hero: String,
    init_ms: Option<u64>,
    last_ms: Option<u64>,
    /// Modified time of the events file (Unix ms): the script's heartbeat.
    heartbeat_ms: Option<u64>,
    updates: u64,
    source_error: Option<String>,
    config_error: Option<String>,
    data_warning: Option<String>,
}

impl TrainingModel {
    pub fn new(
        title: String,
        source_label: String,
        settings: Settings,
        interval_override: Option<Duration>,
    ) -> Self {
        Self {
            title,
            source_label,
            settings,
            interval_override,
            widgets: Vec::new(),
            hero: String::new(),
            init_ms: None,
            last_ms: None,
            heartbeat_ms: None,
            updates: 0,
            source_error: None,
            config_error: None,
            data_warning: None,
        }
    }

    pub fn interval(&self) -> Duration {
        self.interval_override.unwrap_or(self.settings.interval)
    }

    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Applies new records. Lines that are not valid events are ignored.
    pub fn apply(&mut self, update: SourceUpdate) {
        if update.restarted {
            self.reset();
        }
        for record in update.records {
            if let Ok(event) = serde_json::from_value::<TrainingEvent>(record) {
                self.apply_event(event);
            }
        }
    }

    fn reset(&mut self) {
        self.widgets.clear();
        self.hero.clear();
        self.init_ms = None;
        self.last_ms = None;
        self.updates = 0;
        self.data_warning = None;
    }

    pub fn apply_event(&mut self, event: TrainingEvent) {
        match event {
            TrainingEvent::Init { t, widgets } => {
                self.reset();
                let history = self.settings.history;
                self.widgets = widgets
                    .into_iter()
                    .map(|WidgetSpec { name, kind }| Widget {
                        name,
                        kind,
                        series: Series::new(history),
                        invalid: false,
                    })
                    .collect();
                self.init_ms = Some(t);
                self.last_ms = Some(t);
            }
            TrainingEvent::Update { t, values } => {
                if self.widgets.is_empty() {
                    return;
                }
                if values.len() != self.widgets.len() {
                    self.data_warning = Some(format!(
                        "an update had {} values but the dashboard has {} variables",
                        values.len(),
                        self.widgets.len()
                    ));
                    return;
                }
                for (widget, value) in self.widgets.iter_mut().zip(values) {
                    match value {
                        Some(number) => {
                            widget.series.push(number);
                            widget.invalid = false;
                        }
                        None => widget.invalid = true,
                    }
                }
                self.updates += 1;
                self.last_ms = Some(self.last_ms.map_or(t, |last| last.max(t)));
            }
            TrainingEvent::Hero { t, text } => {
                self.hero = text;
                self.last_ms = Some(self.last_ms.map_or(t, |last| last.max(t)));
            }
        }
    }

    /// Swaps in freshly loaded settings without losing any collected data.
    pub fn apply_settings(&mut self, settings: Settings) {
        for widget in &mut self.widgets {
            widget.series.set_capacity(settings.history);
        }
        self.settings = settings;
        self.config_error = None;
    }

    pub fn set_source_error(&mut self, error: Option<String>) {
        self.source_error = error;
    }

    pub fn set_config_error(&mut self, error: Option<String>) {
        self.config_error = error;
    }

    /// Records the events file's modified time (the script's heartbeat).
    pub fn set_heartbeat(&mut self, modified_ms: Option<u64>) {
        self.heartbeat_ms = modified_ms;
    }

    /// The most recent sign of life: an event or a heartbeat.
    fn last_activity_ms(&self) -> Option<u64> {
        match (self.last_ms, self.heartbeat_ms) {
            (Some(event), Some(beat)) => Some(event.max(beat)),
            (event, beat) => event.or(beat),
        }
    }

    pub fn phase(&self, now_ms: u64) -> Phase {
        match (self.init_ms, self.last_activity_ms()) {
            (Some(_), Some(last)) if now_ms.saturating_sub(last) <= STALE_AFTER_MS => {
                Phase::Running
            }
            (Some(_), _) => Phase::Idle,
            _ => Phase::Waiting,
        }
    }

    /// Time spent training: until now while the script is alive, frozen at its
    /// last sign of life once it is not (so the clock stops when the run ends).
    pub fn elapsed(&self, now_ms: u64) -> Option<Duration> {
        let init = self.init_ms?;
        let end = match self.phase(now_ms) {
            Phase::Running => now_ms,
            _ => self.last_activity_ms().unwrap_or(init),
        };
        Some(Duration::from_millis(end.saturating_sub(init)))
    }

    fn color_of(&self, widget: &Widget, index: usize) -> String {
        self.settings
            .per_metric
            .get(&widget.name)
            .and_then(|meta| meta.color.clone())
            .unwrap_or_else(|| self.settings.palette[index % self.settings.palette.len()].clone())
    }

    fn label_of(&self, widget: &Widget) -> String {
        self.settings
            .per_metric
            .get(&widget.name)
            .and_then(|meta| meta.label.clone())
            .unwrap_or_else(|| widget.name.clone())
    }

    fn number(&self, widget: &Widget, value: f64) -> String {
        let s = &self.settings;
        let meta = s.per_metric.get(&widget.name);
        let decimals = meta.and_then(|meta| meta.decimals).unwrap_or(s.decimals);
        let unit = meta.map(|meta| meta.unit.as_str()).unwrap_or("");
        format!(
            "{}{unit}",
            fmt_num(value, decimals, s.scientific_above, s.scientific_below)
        )
    }

    /// The latest value as text, `NaN` for an invalid one and an empty string before any data.
    fn latest_text(&self, widget: &Widget) -> String {
        if widget.invalid {
            "NaN".to_string()
        } else {
            widget
                .series
                .latest()
                .map(|value| self.number(widget, value))
                .unwrap_or_default()
        }
    }
}

// ---------------------------------------------------------------------------
// Drawing primitives (pure)
// ---------------------------------------------------------------------------

fn dim(text: &str) -> String {
    style(text, "2")
}

fn with_bold(color: &str) -> String {
    format!("1;{color}")
}

/// Centers `text` in exactly `width` characters; clips with an ellipsis if too long.
fn center(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if len > width {
        return fit_right(text, width);
    }
    let left = (width - len) / 2;
    format!("{}{text}{}", " ".repeat(left), " ".repeat(width - len - left))
}

fn bounds(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(*value), hi.max(*value))
        })
}

fn plot(cells: &mut [u8], cols: usize, rows: usize, x: i64, y: i64) {
    if x < 0 || y < 0 {
        return;
    }
    let (col, row) = ((x / 2) as usize, (y / 4) as usize);
    if col >= cols || row >= rows {
        return;
    }
    cells[row * cols + col] |= BRAILLE_BITS[(x % 2) as usize][(y % 4) as usize];
}

/// Bresenham line between two dots.
fn plot_line(cells: &mut [u8], cols: usize, rows: usize, from: (i64, i64), to: (i64, i64)) {
    let (mut x, mut y) = from;
    let dx = (to.0 - x).abs();
    let dy = -(to.1 - y).abs();
    let step_x = if x < to.0 { 1 } else { -1 };
    let step_y = if y < to.1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        plot(cells, cols, rows, x, y);
        if (x, y) == to {
            break;
        }
        let doubled = 2 * error;
        if doubled >= dy {
            error += dy;
            x += step_x;
        }
        if doubled <= dx {
            error += dx;
            y += step_y;
        }
    }
}

/// A line plot using braille dots (2x4 per character): `rows` strings of exactly
/// `cols` characters. The whole history is stretched across the width and scaled
/// between its own minimum and maximum.
fn braille_curve(values: &[f64], cols: usize, rows: usize) -> Vec<String> {
    if cols == 0 || rows == 0 {
        return Vec::new();
    }
    let mut cells = vec![0u8; cols * rows];

    if !values.is_empty() {
        let dots_w = cols * 2;
        let dots_h = rows * 4;
        let (lo, hi) = bounds(values);
        let span = hi - lo;
        let count = values.len();

        let y_of = |value: f64| -> i64 {
            if span <= f64::EPSILON {
                (dots_h / 2) as i64
            } else {
                let t = ((value - lo) / span).clamp(0.0, 1.0);
                (dots_h as i64 - 1) - (t * (dots_h - 1) as f64).round() as i64
            }
        };
        let x_of = |index: usize| -> i64 {
            if count == 1 {
                (dots_w - 1) as i64
            } else {
                ((index * (dots_w - 1) + (count - 1) / 2) / (count - 1)) as i64
            }
        };

        let mut previous: Option<(i64, i64)> = None;
        for (index, value) in values.iter().enumerate() {
            let point = (x_of(index), y_of(*value));
            match previous {
                Some(from) => plot_line(&mut cells, cols, rows, from, point),
                None => plot(&mut cells, cols, rows, point.0, point.1),
            }
            previous = Some(point);
        }
    }

    (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| match cells[row * cols + col] {
                    0 => ' ',
                    bits => char::from_u32(BRAILLE_BASE + u32::from(bits)).unwrap_or(' '),
                })
                .collect()
        })
        .collect()
}

/// A bar graph of the latest `cols` values: `rows` strings of exactly `cols`
/// characters. `glyphs` go from the smallest to the full block; the lowest value
/// in view is always at least one step high.
fn bar_chart(values: &[f64], glyphs: &[char], cols: usize, rows: usize) -> Vec<String> {
    if cols == 0 || rows == 0 || glyphs.is_empty() {
        return Vec::new();
    }
    let window = &values[values.len().saturating_sub(cols)..];
    let levels = glyphs.len();
    let total = rows * levels;
    let (lo, hi) = bounds(window);
    let span = hi - lo;

    let heights: Vec<usize> = window
        .iter()
        .map(|value| {
            if span <= f64::EPSILON {
                (total / 2).max(1)
            } else {
                1 + (((value - lo) / span) * (total - 1) as f64).round() as usize
            }
        })
        .collect();

    (0..rows)
        .map(|row| {
            let from_bottom = rows - 1 - row;
            (0..cols)
                .map(|col| match heights.get(col) {
                    Some(height) => {
                        let fill = height.saturating_sub(from_bottom * levels).min(levels);
                        if fill == 0 { ' ' } else { glyphs[fill - 1] }
                    }
                    None => ' ',
                })
                .collect()
        })
        .collect()
}

fn plan_columns(count: usize, width: usize) -> usize {
    if count == 0 {
        return 1;
    }
    let fits = ((width + GAP) / (MIN_PANEL_W + GAP)).max(1);
    let columns = count.min(fits);
    // Balance the rows: 4 widgets in at most 3 columns become 2 + 2, not 3 + 1.
    let rows = count.div_ceil(columns);
    count.div_ceil(rows)
}

/// Splits `total` columns into `count` panel widths separated by `GAP`.
fn split_widths(total: usize, count: usize) -> Vec<usize> {
    let usable = total.saturating_sub(GAP * count.saturating_sub(1));
    let base = usable / count;
    let extra = usable % count;
    (0..count).map(|i| base + usize::from(i < extra)).collect()
}

fn panel_top(label: &str, value: &str, width: usize, color: &str) -> String {
    let inner = width.saturating_sub(2);
    // "─ " + label + " "  and  " " + value + " ─"
    let value_cost = if value.is_empty() { 0 } else { value.chars().count() + 3 };
    let label_room = inner.saturating_sub(3);
    let label_len = label.chars().count();
    let show_value = value_cost > 0 && label_len + 3 + value_cost <= inner;
    let label = if label_len > label_room {
        fit_right(label, label_room)
    } else {
        label.to_string()
    };
    let used = label.chars().count() + 3 + if show_value { value_cost } else { 0 };
    let fill = inner.saturating_sub(used);

    let mut line = dim("┌─ ");
    line.push_str(&style(&label, &with_bold(color)));
    line.push_str(&dim(" "));
    line.push_str(&dim(&"─".repeat(fill)));
    if show_value {
        line.push_str(&dim(" "));
        line.push_str(&style(value, color));
        line.push_str(&dim(" ─"));
    }
    line.push_str(&dim("┐"));
    line
}

fn panel_bottom(left: &str, right: &str, width: usize) -> String {
    let inner = width.saturating_sub(2);
    let (left, right) = if left.chars().count() + right.chars().count() <= inner {
        (left, right)
    } else {
        ("", "")
    };
    let fill = inner - left.chars().count() - right.chars().count();
    dim(&format!("└{left}{}{right}┘", "─".repeat(fill)))
}

fn framed_row(body: &str) -> String {
    format!("{}{body}{}", dim("│"), dim("│"))
}

// ---------------------------------------------------------------------------
// Rendering (pure)
// ---------------------------------------------------------------------------

impl TrainingModel {
    /// Body rows of a box widget: value, change since the previous update, range.
    fn box_body(&self, widget: &Widget, color: &str, width: usize, height: usize) -> Vec<String> {
        let values = widget.series.to_vec();
        let mut lines: Vec<(String, String)> = vec![(
            self.latest_text(widget),
            with_bold(color),
        )];
        if values.len() >= 2 && !widget.invalid {
            let change = values[values.len() - 1] - values[values.len() - 2];
            let text = if change > 0.0 {
                format!("▲ +{}", self.number(widget, change))
            } else if change < 0.0 {
                format!("▼ -{}", self.number(widget, -change))
            } else {
                "• unchanged".to_string()
            };
            lines.push((text, "2".to_string()));
        }
        if let (Some(min), Some(max)) = (widget.series.min(), widget.series.max()) {
            lines.push((
                format!("min {} · max {}", self.number(widget, min), self.number(widget, max)),
                "2".to_string(),
            ));
        }
        lines.truncate(height);

        let top = (height - lines.len()) / 2;
        let mut rows = vec![" ".repeat(width); top];
        rows.extend(
            lines
                .iter()
                .map(|(text, code)| style(&center(text, width), code)),
        );
        rows.resize(height, " ".repeat(width));
        rows
    }

    fn panel(&self, index: usize, width: usize, height: usize) -> Vec<String> {
        let widget = &self.widgets[index];
        let color = self.color_of(widget, index);
        let inner_w = width - 2;
        let inner_h = height - 2;

        let values = widget.series.to_vec();
        let mut lines = vec![panel_top(
            &self.label_of(widget),
            &self.latest_text(widget),
            width,
            &color,
        )];

        let body: Vec<String> = if values.is_empty() && !widget.invalid {
            let mut rows = vec![" ".repeat(inner_w); inner_h];
            rows[inner_h / 2] = dim(&center("no data yet", inner_w));
            rows
        } else {
            match widget.kind {
                WidgetKind::Curve => braille_curve(&values, inner_w, inner_h)
                    .iter()
                    .map(|row| style(row, &color))
                    .collect(),
                WidgetKind::Graph => {
                    bar_chart(&values, &self.settings.spark_chars, inner_w, inner_h)
                        .iter()
                        .map(|row| style(row, &color))
                        .collect()
                }
                WidgetKind::Block => self.box_body(widget, &color, inner_w, inner_h),
            }
        };
        lines.extend(body.iter().map(|row| framed_row(row)));

        let range = match (widget.kind, widget.series.min(), widget.series.max()) {
            (WidgetKind::Block, _, _) | (_, None, _) | (_, _, None) => None,
            (_, Some(min), Some(max)) => Some((self.number(widget, min), self.number(widget, max))),
        };
        lines.push(match range {
            Some((min, max)) => panel_bottom(&format!("─ min {min} "), &format!(" max {max} ─"), width),
            None => panel_bottom("", "", width),
        });
        lines
    }

    fn panel_row(&self, indices: &[usize], total_width: usize, height: usize) -> Vec<String> {
        let widths = split_widths(total_width, indices.len());
        let panels: Vec<Vec<String>> = indices
            .iter()
            .zip(&widths)
            .map(|(index, width)| self.panel(*index, *width, height))
            .collect();
        (0..height)
            .map(|line| {
                let joined = panels
                    .iter()
                    .map(|panel| panel[line].as_str())
                    .collect::<Vec<_>>()
                    .join(&" ".repeat(GAP));
                format!(" {joined}")
            })
            .collect()
    }

    fn hero_lines(&self, width: usize, height: usize) -> Vec<String> {
        let inner = width.saturating_sub(2);
        let flat: String = self
            .hero
            .chars()
            .map(|ch| if ch.is_control() { ' ' } else { ch })
            .collect();
        let text = flat.trim();
        let line = if text.is_empty() {
            format!(
                " {}",
                style(
                    &center("(waiting for update_hero_text(...))", inner),
                    &self.settings.styles.hint
                )
            )
        } else {
            format!(" {}", style(&center(text, inner), &self.settings.styles.hero))
        };
        if height < HERO_FULL_H {
            return vec![line];
        }
        let rule = format!(
            " {}",
            style(&"═".repeat(inner), &self.settings.styles.subheader)
        );
        vec![rule.clone(), line, rule]
    }

    fn waiting_lines(&self, width: usize) -> Vec<String> {
        let hint = &self.settings.styles.hint;
        vec![
            fit("  Waiting for training data...", width),
            style(
                &fit(
                    "  In your script:  tc.init_training_dashboard(\"loss\", \"curve\", \"lr\", \"box\")",
                    width,
                ),
                hint,
            ),
            style(
                &fit(
                    "  then tc.update_training_dashboard(0.5, 1e-3) and tc.update_hero_text(\"...\")",
                    width,
                ),
                hint,
            ),
            style(&fit(&format!("  Reading {}", self.source_label), width), hint),
        ]
    }

    fn render_body(&self, width: usize, rows: usize, now_ms: u64) -> Vec<String> {
        if self.widgets.is_empty() {
            let _ = now_ms;
            return self.waiting_lines(width);
        }

        let grid_w = width.saturating_sub(2);
        let columns = plan_columns(self.widgets.len(), grid_w);
        let indices: Vec<usize> = (0..self.widgets.len()).collect();
        let chunks: Vec<&[usize]> = indices.chunks(columns).collect();

        let hero_h = if rows >= HERO_FULL_H + MIN_PANEL_H { HERO_FULL_H } else { 1 };
        let mut budget = rows.saturating_sub(hero_h);
        let mut rows_fit = (budget / MIN_PANEL_H).min(chunks.len());
        if rows_fit < chunks.len() && budget > 0 {
            budget -= 1; // room for the "+N more" hint
            rows_fit = (budget / MIN_PANEL_H).min(chunks.len());
        }

        let mut lines: Vec<String> = Vec::new();
        if rows_fit == 0 {
            lines.extend(self.hero_lines(width, hero_h));
            lines.push(style(
                &fit("  Terminal too short for the widgets.", width),
                &self.settings.styles.hint,
            ));
            return lines;
        }

        let row_h = (budget / rows_fit).clamp(MIN_PANEL_H, MAX_PANEL_H);
        let top_rows = rows_fit / 2;
        for chunk in &chunks[..top_rows] {
            lines.extend(self.panel_row(chunk, grid_w, row_h));
        }
        lines.extend(self.hero_lines(width, hero_h));
        for chunk in &chunks[top_rows..rows_fit] {
            lines.extend(self.panel_row(chunk, grid_w, row_h));
        }
        let hidden: usize = chunks[rows_fit..].iter().map(|chunk| chunk.len()).sum();
        if hidden > 0 {
            lines.push(style(
                &fit(&format!("  ... +{hidden} more (enlarge the terminal)"), width),
                &self.settings.styles.hint,
            ));
        }
        lines
    }
}

/// Builds exactly `height` lines, each at most `width - 1` visible characters wide.
/// `now_ms` is passed in (Unix milliseconds) so the output is deterministic in tests.
pub fn render_training_frame(
    model: &TrainingModel,
    width: usize,
    height: usize,
    now_ms: u64,
) -> Vec<String> {
    let s = &model.settings;
    let w = width.saturating_sub(1).max(1);

    if width < MIN_WIDTH || height < MIN_HEIGHT {
        let message = format!(" Terminal too small (need {MIN_WIDTH}x{MIN_HEIGHT})");
        return blank_frame(vec![fit(&message, w)], height, w);
    }

    let mut lines: Vec<String> = Vec::with_capacity(height);
    let phase = model.phase(now_ms);

    if s.show_header {
        let title = s.title.as_deref().unwrap_or(&model.title);
        let clock = match model.elapsed(now_ms) {
            Some(elapsed) => fmt_duration(elapsed),
            None => "--:--:--".to_string(),
        };
        let right = format!(" ⏱ {clock} ");
        let left_w = w.saturating_sub(right.chars().count());
        lines.push(style(
            &format!("{}{right}", fit(&format!(" {title}  ·  {}", model.source_label), left_w)),
            &s.styles.header,
        ));

        let state = match phase {
            Phase::Waiting => "○ waiting for init_training_dashboard",
            Phase::Running => "● running",
            Phase::Idle => "○ idle (script not running)",
        };
        lines.push(style(
            &fit(
                &format!(
                    " {} updates  ·  {} variables  ·  {state}",
                    model.updates,
                    model.widgets.len()
                ),
                w,
            ),
            &s.styles.subheader,
        ));
        lines.push(fit("", w));
    }

    let bottom = usize::from(s.show_status) + usize::from(s.show_footer);
    let body_rows = height.saturating_sub(lines.len() + bottom);
    let mut body = model.render_body(w, body_rows, now_ms);
    body.truncate(body_rows);
    lines.extend(body);

    while lines.len() < height - bottom {
        lines.push(fit("", w));
    }

    if s.show_status {
        let (text, code) = if let Some(error) = &model.config_error {
            (format!(" ! config: {error}"), &s.styles.status_error)
        } else if let Some(error) = &model.source_error {
            (format!(" ! {error}"), &s.styles.status_error)
        } else if let Some(warning) = &model.data_warning {
            (format!(" ! data: {warning}"), &s.styles.status_error)
        } else if phase == Phase::Running {
            (
                format!(" ● live  ·  refresh {}ms", model.interval().as_millis()),
                &s.styles.status_ok,
            )
        } else {
            (
                format!(" ○ script not running  ·  refresh {}ms", model.interval().as_millis()),
                &s.styles.footer,
            )
        };
        lines.push(style(&fit(&text, w), code));
    }
    if s.show_footer {
        lines.push(style(&fit(&s.footer, w), &s.styles.footer));
    }

    blank_frame(lines, height, w)
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

/// Synthetic training run that speaks the real event protocol, so
/// `/dashboard training --demo` shows the design without any Python.
pub struct TrainingDemo {
    step: u64,
    rng: u64,
    started: bool,
}

impl TrainingDemo {
    pub fn new() -> Self {
        Self {
            step: 0,
            rng: 0x9E37_79B9_7F4A_7C15,
            started: false,
        }
    }

    fn noise(&mut self) -> f64 {
        self.rng = self
            .rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
    }

    pub fn poll(&mut self) -> SourceUpdate {
        let mut events: Vec<TrainingEvent> = Vec::new();
        if !self.started {
            self.started = true;
            let spec = |name: &str, kind| WidgetSpec {
                name: name.to_string(),
                kind,
            };
            events.push(TrainingEvent::Init {
                t: api::now_ms(),
                widgets: vec![
                    spec("loss", WidgetKind::Curve),
                    spec("val_loss", WidgetKind::Curve),
                    spec("accuracy", WidgetKind::Graph),
                    spec("grad_norm", WidgetKind::Graph),
                    spec("lr", WidgetKind::Block),
                    spec("samples/s", WidgetKind::Block),
                ],
            });
        }

        let mut loss = 0.0;
        for _ in 0..DEMO_UPDATES_PER_POLL {
            self.step += 1;
            let t = self.step as f64;
            loss = 2.0 * (-t / 150.0).exp() + 0.003 + self.noise() * 0.02;
            let val_loss = loss * 1.1 + 0.002 + self.noise() * 0.01;
            let accuracy = (1.0 - 0.9 * (-t / 200.0).exp() + self.noise() * 0.02).clamp(0.0, 1.0);
            let grad_norm = 1.5 * (-t / 300.0).exp() + 0.1 + self.noise() * 0.1;
            let lr = 1e-3 * 0.5 * (1.0 + (std::f64::consts::PI * (t / 2000.0).min(1.0)).cos());
            let samples = 1200.0 + self.noise() * 80.0;
            events.push(TrainingEvent::Update {
                t: api::now_ms(),
                values: [loss, val_loss, accuracy, grad_norm, lr, samples]
                    .into_iter()
                    .map(Some)
                    .collect(),
            });
        }
        events.push(TrainingEvent::Hero {
            t: api::now_ms(),
            text: format!(
                "Training at Epoch {}, step {}, Total Loss at {loss:.4}",
                self.step / 100 + 1,
                self.step
            ),
        });

        SourceUpdate {
            records: events
                .iter()
                .filter_map(|event| serde_json::to_value(event).ok())
                .collect::<Vec<Value>>(),
            restarted: false,
        }
    }
}

pub enum TrainingSource {
    File(JsonlTail),
    Demo(TrainingDemo),
}

impl TrainingSource {
    pub fn file(path: PathBuf) -> Self {
        Self::File(JsonlTail::from_start(path))
    }

    pub fn demo() -> Self {
        Self::Demo(TrainingDemo::new())
    }

    pub fn label(&self) -> String {
        match self {
            Self::File(tail) => tail.path().display().to_string(),
            Self::Demo(_) => "demo data".to_string(),
        }
    }

    fn poll(&mut self) -> std::result::Result<SourceUpdate, String> {
        match self {
            Self::File(tail) => tail.poll(),
            Self::Demo(demo) => Ok(demo.poll()),
        }
    }

    /// Unix ms of the script's last heartbeat: the events file's modified time.
    fn heartbeat_ms(&self) -> Option<u64> {
        match self {
            Self::File(tail) => std::fs::metadata(tail.path())
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|elapsed| elapsed.as_millis() as u64),
            Self::Demo(_) => Some(api::now_ms()),
        }
    }
}

/// An explicit path wins, then `TC_TRAINING_FILE`, then `training_dashboard.jsonl`
/// next to the active script (where `/run_script` makes Python write it).
pub fn resolve_training_path(
    explicit: Option<&str>,
    active_script_path: Option<&str>,
) -> std::result::Result<PathBuf, String> {
    if let Some(explicit) = explicit {
        let path = PathBuf::from(explicit);
        if !path.is_absolute() {
            return Err("Path must be absolute".into());
        }
        return Ok(path);
    }
    if let Some(value) = std::env::var_os(api::TRAINING_FILE_ENV).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(value));
    }
    let script = active_script_path.ok_or_else(|| {
        "Provide a path, select a script first, or use --demo.".to_string()
    })?;
    api::training_file_for_script(Path::new(script))
        .ok_or_else(|| format!("Could not determine the directory of {script}"))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub struct TrainingConfig {
    /// Used for the header unless the config file sets `title`.
    pub title: String,
    pub source: TrainingSource,
    pub settings: Settings,
    pub watcher: ConfigWatcher,
    /// An interval given in the REPL wins over the config file.
    pub interval_override: Option<Duration>,
}

/// Runs the live training dashboard until the user exits. Returns a short summary for the REPL.
pub fn run_training_dashboard(config: TrainingConfig) -> std::result::Result<String, String> {
    if !io::stdout().is_terminal() {
        return Err("The dashboard needs an interactive terminal.".into());
    }

    let TrainingConfig {
        title,
        mut source,
        settings,
        mut watcher,
        interval_override,
    } = config;
    let mut model = TrainingModel::new(title, source.label(), settings, interval_override);

    run_live(|width, height| {
        if let Some(reloaded) = watcher.check() {
            match reloaded {
                Ok(settings) => model.apply_settings(settings),
                Err(error) => model.set_config_error(Some(error)),
            }
        }
        match source.poll() {
            Ok(update) => {
                model.apply(update);
                model.set_source_error(None);
            }
            Err(error) => model.set_source_error(Some(error)),
        }
        model.set_heartbeat(source.heartbeat_ms());
        Frame {
            lines: render_training_frame(&model, width, height, api::now_ms()),
            interval: model.interval(),
        }
    })?;

    let spent = model
        .elapsed(api::now_ms())
        .map(fmt_duration)
        .unwrap_or_else(|| "00:00:00".to_string());
    Ok(format!(
        "Training dashboard closed (training time {spent}, {} updates).",
        model.updates()
    ))
}

#[cfg(test)]
mod tests;
