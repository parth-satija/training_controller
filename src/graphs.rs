//! Terminal graphs of logged experiments (`/graph`).
//!
//! A *graph* plots one quantity (y) against another (x) for one or more
//! experiments. A quantity is a log column, `time` (seconds since the first
//! `log` call, the default x) or `step`. Several graphs are tiled on one screen
//! (use the arrow keys / PgUp / PgDn to page when they do not all fit), and
//! several experiments in one graph are drawn as separate colored lines.
//!
//! The module follows the layering of `dashboard.rs`:
//!
//! * **Parsing** ([`parse_graph_command`], [`parse_specs`]) is pure and turns
//!   the words typed after `/graph` into [`GraphSpec`]s.
//! * **Model** ([`GraphView`]) holds the plotted points and reads only the new
//!   log rows from SQLite on every refresh, so long runs stay cheap.
//! * **Rendering** ([`render_view`], [`render_tile`]) is pure: model in, lines
//!   out. Lines are drawn with Braille characters (2x4 dots per cell).
//! * **Terminal** ([`run_graphs`]) reuses the live loop of the dashboards.
//!
//! # Command syntax
//!
//! ```text
//! /graph IDS                      every log column against time
//! /graph IDS Y [X]                one graph (X defaults to time)
//! /graph IDS Y1,Y2 [X]            one graph per quantity in the list
//! /graph IDS Y [X] + Y [X] + ...  several graphs ("+" separates them)
//! /graph CONFIG [IDS]             the graphs of a saved graph config
//! ... --save NAME                 also save the graphs as a graph config
//! ```
//!
//! `IDS` is one experiment ID or a comma-separated list (`3,4`): the lines of
//! all those experiments are drawn in the same graph.

use crate::dashboard::{fit, fit_right, run_live_with_keys, Frame};
use crossterm::event::KeyCode;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::io::{self, IsTerminal};
use std::path::Path;
use std::time::Duration;
use training_controller::logdb;

/// The x quantity when none is given.
pub const DEFAULT_X: &str = logdb::QUANTITY_TIME;

const REFRESH: Duration = Duration::from_millis(500);
const MIN_TILE_WIDTH: usize = 24;
const MIN_TILE_HEIGHT: usize = 7;
const TILE_GAP: usize = 1;
/// SGR colors of the lines, in order (cyan, green, yellow, magenta, blue, red, ...).
const PALETTE: [&str; 8] = ["36", "32", "33", "35", "34", "31", "96", "92"];

fn color_for(index: usize) -> &'static str {
    PALETTE[index % PALETTE.len()]
}

// ---------------------------------------------------------------------------
// Specs and command parsing (pure)
// ---------------------------------------------------------------------------

/// One graph: the quantity on the y axis and on the x axis. It does not name
/// an experiment, so a saved list of specs can be applied to any experiment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSpec {
    pub y: String,
    pub x: String,
}

impl GraphSpec {
    pub fn new(y: &str, x: &str) -> Self {
        Self {
            y: y.to_string(),
            x: x.to_string(),
        }
    }
}

/// `loss vs time  +  accuracy vs step`
pub fn format_specs(specs: &[GraphSpec]) -> String {
    specs
        .iter()
        .map(|spec| format!("{} vs {}", spec.y, spec.x))
        .collect::<Vec<_>>()
        .join("  +  ")
}

/// Parses `3`, `#3` or `3,4` into experiment IDs (positive, without repeats).
/// `None` means the text is not an ID list.
pub fn parse_id_list(text: &str) -> Option<Vec<i64>> {
    let mut ids: Vec<i64> = Vec::new();
    for part in text.split(',') {
        let id: i64 = part.trim().trim_start_matches('#').parse().ok()?;
        if id <= 0 {
            return None;
        }
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    if ids.is_empty() { None } else { Some(ids) }
}

fn expand(y: &str, x: &str, out: &mut Vec<GraphSpec>) -> Result<(), String> {
    if x.is_empty() || x.contains(',') {
        return Err(format!("The x quantity must be a single name, got '{x}'."));
    }
    for part in y.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(format!("Empty quantity in '{y}'."));
        }
        out.push(GraphSpec::new(part, x));
    }
    Ok(())
}

/// Parses graph groups `Y [X] [+ Y [X] ...]`. A comma list in `Y` makes one
/// graph per quantity. Returns an empty list for no tokens.
pub fn parse_specs(tokens: &[String]) -> Result<Vec<GraphSpec>, String> {
    let mut specs = Vec::new();
    if tokens.is_empty() {
        return Ok(specs);
    }
    for group in tokens.split(|token| token.as_str() == "+") {
        match group {
            [] => return Err("Empty graph: put a quantity between the '+' signs.".to_string()),
            [y] => expand(y, DEFAULT_X, &mut specs)?,
            [y, x] => expand(y, x, &mut specs)?,
            _ => {
                return Err(format!(
                    "A graph takes a Y and optionally an X quantity, but got: {}. \
                     Separate several graphs with '+'.",
                    group.join(" ")
                ));
            }
        }
    }
    Ok(specs)
}

/// Which graphs a `/graph` command asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// Graphs typed on the command line.
    Explicit(Vec<GraphSpec>),
    /// Only experiment IDs were given: every log column against time.
    AllColumns,
    /// The graphs of the saved graph config with this name.
    Config(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphCommand {
    /// `None`: use the newest experiment.
    pub ids: Option<Vec<i64>>,
    pub selection: Selection,
    /// `--save NAME`
    pub save: Option<String>,
}

pub const GRAPH_USAGE: &str = "Usage: /graph IDS [Y [X]] [+ Y [X] ...] [--save NAME]  or  /graph CONFIG [IDS]";

/// Parses the words after `/graph`. `is_config` tells whether a name is a saved
/// graph config.
pub fn parse_graph_command(
    tokens: &[String],
    is_config: impl Fn(&str) -> bool,
) -> Result<GraphCommand, String> {
    let mut rest: Vec<String> = Vec::new();
    let mut save: Option<String> = None;
    let mut iter = tokens.iter();
    while let Some(token) = iter.next() {
        if token == "--save" {
            let name = iter
                .next()
                .ok_or_else(|| "--save needs a NAME.".to_string())?;
            save = Some(name.clone());
        } else {
            rest.push(token.clone());
        }
    }

    let first = rest.first().ok_or_else(|| GRAPH_USAGE.to_string())?;

    if let Some(ids) = parse_id_list(first) {
        let specs = parse_specs(&rest[1..])?;
        let selection = if specs.is_empty() {
            Selection::AllColumns
        } else {
            Selection::Explicit(specs)
        };
        return Ok(GraphCommand {
            ids: Some(ids),
            selection,
            save,
        });
    }

    if is_config(first) {
        if save.is_some() {
            return Err(format!(
                "'{first}' is already a saved graph config, so there is nothing to save."
            ));
        }
        if rest.len() > 2 {
            return Err(format!(
                "A saved graph config only takes experiment IDs after its name, got: {}.",
                rest[2..].join(" ")
            ));
        }
        let ids = match rest.get(1) {
            Some(text) => Some(
                parse_id_list(text)
                    .ok_or_else(|| format!("'{text}' is not an experiment ID (or a list like 3,4)."))?,
            ),
            None => None,
        };
        return Ok(GraphCommand {
            ids,
            selection: Selection::Config(first.clone()),
            save,
        });
    }

    Err(format!(
        "'{first}' is neither an experiment ID nor a saved graph config. {GRAPH_USAGE}"
    ))
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Pairs the values of two quantities by step (both sorted by step).
pub fn zip_by_step(xs: &[(i64, f64)], ys: &[(i64, f64)]) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(xs.len().min(ys.len()));
    let (mut i, mut j) = (0, 0);
    while i < xs.len() && j < ys.len() {
        match xs[i].0.cmp(&ys[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push((xs[i].1, ys[j].1));
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// The line of one experiment in one graph.
#[derive(Debug, Clone)]
pub struct Track {
    pub experiment_id: i64,
    pub points: Vec<(f64, f64)>,
    last_step: i64,
}

impl Track {
    fn new(experiment_id: i64) -> Self {
        Self {
            experiment_id,
            points: Vec::new(),
            last_step: -1,
        }
    }

    fn reset(&mut self) {
        self.points.clear();
        self.last_step = -1;
    }

    /// Reads the rows logged since the previous poll. If the experiment's log
    /// was restarted (`init_log` called again) the line starts over.
    fn poll(&mut self, conn: &Connection, spec: &GraphSpec) -> Result<(), String> {
        match logdb::max_step(conn, self.experiment_id)? {
            None => {
                self.reset();
                return Ok(());
            }
            Some(max) if max < self.last_step => self.reset(),
            Some(_) => {}
        }
        let xs = logdb::read_quantity(conn, self.experiment_id, &spec.x, self.last_step)?;
        let ys = logdb::read_quantity(conn, self.experiment_id, &spec.y, self.last_step)?;
        // A step is committed as a whole, so both quantities saw the same steps.
        let newest = xs.last().map(|row| row.0).max(ys.last().map(|row| row.0));
        if let Some(newest) = newest {
            self.last_step = newest;
        }
        self.points.extend(zip_by_step(&xs, &ys));
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Plot {
    pub spec: GraphSpec,
    pub tracks: Vec<Track>,
}

impl Plot {
    fn title(&self) -> String {
        let base = format!("{} vs {}", self.spec.y, self.spec.x);
        match self.tracks.as_slice() {
            [only] => format!("{base} · #{}", only.experiment_id),
            _ => base,
        }
    }

    fn render(&self, width: usize, height: usize) -> Vec<String> {
        let series: Vec<TileSeries> = self
            .tracks
            .iter()
            .map(|track| TileSeries {
                label: format!("#{}", track.experiment_id),
                points: &track.points,
            })
            .collect();
        render_tile(&self.title(), &self.spec.x, &series, width, height)
    }
}

/// Everything shown by one `/graph` command.
#[derive(Debug, Clone)]
pub struct GraphView {
    /// Shown in the header, e.g. `config 'train' · experiment #3`.
    pub heading: String,
    pub plots: Vec<Plot>,
    /// Whether any of the experiments is still running.
    pub running: bool,
    pub error: Option<String>,
}

impl GraphView {
    /// Checks that every experiment exists and has every quantity, then
    /// creates one plot per spec with one line per experiment.
    pub fn new(
        conn: &Connection,
        heading: String,
        ids: &[i64],
        specs: Vec<GraphSpec>,
    ) -> Result<Self, String> {
        if ids.is_empty() {
            return Err("No experiment given.".to_string());
        }
        if specs.is_empty() {
            return Err("No graphs to show.".to_string());
        }
        for &id in ids {
            if !logdb::experiment_exists(conn, id)? {
                return Err(format!("Experiment #{id} does not exist."));
            }
        }
        for spec in &specs {
            for &id in ids {
                // Reading "after the last step" is empty but fails for unknown quantities.
                logdb::read_quantity(conn, id, &spec.y, i64::MAX)?;
                logdb::read_quantity(conn, id, &spec.x, i64::MAX)?;
            }
        }
        let plots = specs
            .into_iter()
            .map(|spec| Plot {
                tracks: ids.iter().map(|&id| Track::new(id)).collect(),
                spec,
            })
            .collect();
        Ok(Self {
            heading,
            plots,
            running: false,
            error: None,
        })
    }

    fn experiment_ids(&self) -> Vec<i64> {
        let mut ids: Vec<i64> = Vec::new();
        for plot in &self.plots {
            for track in &plot.tracks {
                if !ids.contains(&track.experiment_id) {
                    ids.push(track.experiment_id);
                }
            }
        }
        ids
    }

    /// Pulls new rows into every line. Problems are kept in `error` (shown in
    /// the footer) instead of closing the view.
    pub fn poll(&mut self, conn: &Connection) {
        self.error = None;
        for plot in &mut self.plots {
            for track in &mut plot.tracks {
                if let Err(error) = track.poll(conn, &plot.spec) {
                    self.error = Some(error);
                }
            }
        }
        let mut running = false;
        for id in self.experiment_ids() {
            match logdb::experiment_status(conn, id) {
                Ok(Some(status)) if status == logdb::STATUS_RUNNING => running = true,
                Ok(_) => {}
                Err(error) => self.error = Some(error),
            }
        }
        self.running = running;
    }
}

// ---------------------------------------------------------------------------
// Rendering: one graph (pure)
// ---------------------------------------------------------------------------

/// The points of one line and its legend label.
pub struct TileSeries<'a> {
    pub label: String,
    pub points: &'a [(f64, f64)],
}

/// Braille dot bits: `BRAILLE[dot column][dot row]`.
const BRAILLE: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];

/// A grid of Braille cells. Every cell remembers the line that drew in it last.
struct Canvas {
    cols: usize,
    rows: usize,
    mask: Vec<u8>,
    color: Vec<u8>,
}

impl Canvas {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            mask: vec![0; cols * rows],
            color: vec![0; cols * rows],
        }
    }

    fn dot_width(&self) -> i64 {
        (self.cols * 2) as i64
    }

    fn dot_height(&self) -> i64 {
        (self.rows * 4) as i64
    }

    fn set(&mut self, x: i64, y: i64, series: usize) {
        if x < 0 || y < 0 || x >= self.dot_width() || y >= self.dot_height() {
            return;
        }
        let cell = (y / 4) as usize * self.cols + (x / 2) as usize;
        self.mask[cell] |= BRAILLE[(x % 2) as usize][(y % 4) as usize];
        self.color[cell] = (series % PALETTE.len()) as u8 + 1;
    }

    /// Bresenham line between two dots.
    fn line(&mut self, from: (i64, i64), to: (i64, i64), series: usize) {
        let (mut x, mut y) = from;
        let dx = (to.0 - x).abs();
        let dy = -(to.1 - y).abs();
        let sx = if x < to.0 { 1 } else { -1 };
        let sy = if y < to.1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            self.set(x, y, series);
            if x == to.0 && y == to.1 {
                break;
            }
            let doubled = 2 * error;
            if doubled >= dy {
                error += dy;
                x += sx;
            }
            if doubled <= dx {
                error += dx;
                y += sy;
            }
        }
    }

    /// One screen row, with the colors of the lines.
    fn row_text(&self, row: usize) -> String {
        let mut text = String::new();
        for col in 0..self.cols {
            let cell = row * self.cols + col;
            let mask = self.mask[cell];
            if mask == 0 {
                text.push(' ');
                continue;
            }
            let glyph = char::from_u32(0x2800 + u32::from(mask)).unwrap_or(' ');
            let color = PALETTE[usize::from(self.color[cell]).saturating_sub(1) % PALETTE.len()];
            text.push_str(&format!("\x1b[{color}m{glyph}\x1b[0m"));
        }
        text
    }
}

/// Compact axis label: plain for ordinary magnitudes, scientific for tiny/huge.
pub fn fmt_axis(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs();
    if !(1e-3..1e5).contains(&magnitude) {
        return format!("{value:.2e}");
    }
    let decimals = if magnitude >= 100.0 { 1 } else { 3 };
    let text = format!("{value:.decimals$}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

fn bounds(series: &[TileSeries]) -> Option<(f64, f64, f64, f64)> {
    let mut found: Option<(f64, f64, f64, f64)> = None;
    for &(x, y) in series.iter().flat_map(|line| line.points.iter()) {
        found = Some(match found {
            None => (x, x, y, y),
            Some((x_min, x_max, y_min, y_max)) => {
                (x_min.min(x), x_max.max(x), y_min.min(y), y_max.max(y))
            }
        });
    }
    found
}

/// Gives a flat range some height so a constant line is drawn in the middle.
fn widen(min: f64, max: f64) -> (f64, f64) {
    let scale = min.abs().max(max.abs()).max(1.0);
    if max - min > f64::EPSILON * scale {
        return (min, max);
    }
    let pad = if min == 0.0 && max == 0.0 {
        1.0
    } else {
        min.abs().max(max.abs()) * 0.1
    };
    (min - pad, max + pad)
}

/// Maps `value` from `[min, max]` to `0..=span`.
fn scale_to(value: f64, min: f64, max: f64, span: i64) -> i64 {
    let fraction = ((value - min) / (max - min)).clamp(0.0, 1.0);
    (fraction * span as f64).round() as i64
}

fn pad_lines(mut lines: Vec<String>, height: usize, width: usize) -> Vec<String> {
    while lines.len() < height {
        lines.push(fit("", width));
    }
    lines.truncate(height);
    lines
}

/// Draws one graph into exactly `height` lines of exactly `width` visible
/// characters: a title, the plot with a y axis, an x axis with its labels, and
/// a legend when there is more than one line.
pub fn render_tile(
    title: &str,
    x_name: &str,
    series: &[TileSeries],
    width: usize,
    height: usize,
) -> Vec<String> {
    if width < MIN_TILE_WIDTH || height < MIN_TILE_HEIGHT {
        return pad_lines(vec![fit(" too small", width)], height, width);
    }

    let legend_rows = usize::from(series.len() > 1);
    let plot_rows = height - 3 - legend_rows;
    let data_bounds = bounds(series);
    let (x_min, x_max, y_min, y_max) = match data_bounds {
        Some((x0, x1, y0, y1)) => {
            let (x0, x1) = widen(x0, x1);
            let (y0, y1) = widen(y0, y1);
            (x0, x1, y0, y1)
        }
        None => (0.0, 1.0, 0.0, 1.0),
    };

    let y_labels: [String; 3] = if data_bounds.is_some() {
        [
            fmt_axis(y_max),
            fmt_axis((y_min + y_max) / 2.0),
            fmt_axis(y_min),
        ]
    } else {
        [String::new(), String::new(), String::new()]
    };
    let label_width = y_labels
        .iter()
        .map(|label| label.chars().count())
        .max()
        .unwrap_or(1)
        .clamp(3, 10);
    let plot_cols = width - label_width - 1;

    let mut canvas = Canvas::new(plot_cols, plot_rows);
    if data_bounds.is_some() {
        let (dot_w, dot_h) = (canvas.dot_width(), canvas.dot_height());
        for (index, line) in series.iter().enumerate() {
            let dots: Vec<(i64, i64)> = line
                .points
                .iter()
                .map(|&(x, y)| {
                    (
                        scale_to(x, x_min, x_max, dot_w - 1),
                        (dot_h - 1) - scale_to(y, y_min, y_max, dot_h - 1),
                    )
                })
                .collect();
            match dots.as_slice() {
                [] => {}
                [only] => canvas.set(only.0, only.1, index),
                _ => {
                    for pair in dots.windows(2) {
                        canvas.line(pair[0], pair[1], index);
                    }
                }
            }
        }
    }

    let mut lines: Vec<String> = Vec::with_capacity(height);
    lines.push(format!("\x1b[1m{}\x1b[0m", fit(&format!(" {title}"), width)));

    let waiting_row = plot_rows / 2;
    for row in 0..plot_rows {
        let (label, tick) = if row == 0 {
            (y_labels[0].as_str(), '┤')
        } else if row == plot_rows / 2 {
            (y_labels[1].as_str(), '┤')
        } else if row == plot_rows - 1 {
            (y_labels[2].as_str(), '┤')
        } else {
            ("", '│')
        };
        let mut line = format!("{}{tick}", fit_right(label, label_width));
        if data_bounds.is_none() {
            if row == waiting_row {
                line.push_str(&format!(
                    "\x1b[2m{}\x1b[0m",
                    fit(" waiting for data...", plot_cols)
                ));
            } else {
                line.push_str(&" ".repeat(plot_cols));
            }
        } else {
            line.push_str(&canvas.row_text(row));
        }
        lines.push(line);
    }

    lines.push(format!(
        "{}└{}",
        " ".repeat(label_width),
        "─".repeat(plot_cols)
    ));

    let mut axis: Vec<char> = vec![' '; plot_cols];
    if data_bounds.is_some() {
        let left: Vec<char> = fmt_axis(x_min).chars().collect();
        let right: Vec<char> = fmt_axis(x_max).chars().collect();
        let name: Vec<char> = x_name.chars().collect();
        if left.len() + right.len() < plot_cols {
            axis[..left.len()].copy_from_slice(&left);
            axis[plot_cols - right.len()..].copy_from_slice(&right);
            let start = (plot_cols.saturating_sub(name.len())) / 2;
            if start > left.len() && start + name.len() + 1 < plot_cols - right.len() {
                axis[start..start + name.len()].copy_from_slice(&name);
            }
        }
    }
    lines.push(format!(
        "{}{}",
        " ".repeat(label_width + 1),
        axis.into_iter().collect::<String>()
    ));

    if legend_rows == 1 {
        let mut legend = String::from(" ");
        let mut used = 1;
        for (index, line) in series.iter().enumerate() {
            let separator = if index > 0 { 2 } else { 0 };
            let needed = separator + 2 + line.label.chars().count();
            if used + needed > width {
                break;
            }
            if index > 0 {
                legend.push_str("  ");
            }
            legend.push_str(&format!("\x1b[{}m●\x1b[0m {}", color_for(index), line.label));
            used += needed;
        }
        legend.push_str(&" ".repeat(width - used));
        lines.push(legend);
    }

    pad_lines(lines, height, width)
}

// ---------------------------------------------------------------------------
// Rendering: the screen (pure)
// ---------------------------------------------------------------------------

/// Splits `total` into `parts` sizes with `gap` between them; the first sizes
/// get the remainder.
pub fn split_even(total: usize, parts: usize, gap: usize) -> Vec<usize> {
    if parts == 0 {
        return Vec::new();
    }
    let available = total.saturating_sub(gap * (parts - 1));
    let base = available / parts;
    let extra = available % parts;
    (0..parts).map(|i| base + usize::from(i < extra)).collect()
}

/// `(columns, graphs per page)` for `count` graphs on a `width` x `height`
/// area. Prefers a roughly square grid and never makes a tile smaller than
/// the minimum; what does not fit goes to further pages.
pub fn page_layout(count: usize, width: usize, height: usize) -> (usize, usize) {
    let count = count.max(1);
    let max_cols = ((width + TILE_GAP) / (MIN_TILE_WIDTH + TILE_GAP)).max(1);
    let max_rows = (height / MIN_TILE_HEIGHT).max(1);
    let ideal_cols = (count as f64).sqrt().ceil() as usize;
    let cols = ideal_cols.min(max_cols).max(1);
    let rows = count.div_ceil(cols).min(max_rows);
    (cols, cols * rows)
}

/// Builds exactly `height` lines of at most `width - 1` visible characters.
/// Returns the lines, the page actually shown (the request is clamped) and the
/// number of pages.
pub fn render_view(
    view: &GraphView,
    page: usize,
    width: usize,
    height: usize,
) -> (Vec<String>, usize, usize) {
    let w = width.saturating_sub(1).max(1);
    if w < MIN_TILE_WIDTH || height < MIN_TILE_HEIGHT + 2 {
        let message = format!(
            " Terminal too small (need {}x{})",
            MIN_TILE_WIDTH + 1,
            MIN_TILE_HEIGHT + 2
        );
        return (pad_lines(vec![fit(&message, w)], height, w), 0, 1);
    }

    let body_height = height - 2;
    let total = view.plots.len();
    let (cols, per_page) = page_layout(total, w, body_height);
    let pages = total.div_ceil(per_page).max(1);
    let page = page.min(pages - 1);
    let start = page * per_page;
    let items = &view.plots[start.min(total)..(start + per_page).min(total)];

    let mut lines: Vec<String> = Vec::with_capacity(height);
    lines.push(format!(
        "\x1b[1;7m{}\x1b[0m",
        fit(
            &format!(" Graphs  ·  {}  ·  {}", view.heading, plural_graphs(total)),
            w
        )
    ));

    let row_count = items.len().div_ceil(cols);
    let heights = split_even(body_height, row_count, 0);
    for (row, tile_height) in heights.iter().enumerate() {
        let row_items = &items[row * cols..((row + 1) * cols).min(items.len())];
        let widths = split_even(w, row_items.len(), TILE_GAP);
        let tiles: Vec<Vec<String>> = row_items
            .iter()
            .zip(&widths)
            .map(|(plot, tile_width)| plot.render(*tile_width, *tile_height))
            .collect();
        let gap = " ".repeat(TILE_GAP);
        for line in 0..*tile_height {
            let parts: Vec<&str> = tiles.iter().map(|tile| tile[line].as_str()).collect();
            lines.push(parts.join(gap.as_str()));
        }
    }
    while lines.len() < height - 1 {
        lines.push(fit("", w));
    }

    let paging = if pages > 1 {
        format!("  ·  ←/→ page {}/{}", page + 1, pages)
    } else {
        String::new()
    };
    let (status, code) = if let Some(error) = &view.error {
        (format!("! {error}"), "31")
    } else if view.running {
        ("● live".to_string(), "2")
    } else {
        ("○ finished".to_string(), "2")
    };
    lines.push(format!(
        "\x1b[{code}m{}\x1b[0m",
        fit(&format!(" q / Esc exit{paging}  ·  {status}"), w)
    ));

    (pad_lines(lines, height, w), page, pages)
}

fn plural_graphs(count: usize) -> String {
    if count == 1 {
        "1 graph".to_string()
    } else {
        format!("{count} graphs")
    }
}

// ---------------------------------------------------------------------------
// Terminal
// ---------------------------------------------------------------------------

/// Shows the graphs live until the user presses q, Esc or Ctrl+C. New log rows
/// appear while the experiment is still running.
pub fn run_graphs(db_path: &Path, mut view: GraphView) -> Result<String, String> {
    if !io::stdout().is_terminal() {
        return Err("Graphs need an interactive terminal.".to_string());
    }
    let conn = logdb::open(db_path)?;
    let page = Cell::new(0usize);
    let graph_count = view.plots.len();

    run_live_with_keys(
        |width, height| {
            view.poll(&conn);
            let (lines, shown, _pages) = render_view(&view, page.get(), width, height);
            page.set(shown);
            Frame {
                lines,
                interval: REFRESH,
            }
        },
        |key| match key {
            KeyCode::Left | KeyCode::PageUp | KeyCode::Char('p') => {
                page.set(page.get().saturating_sub(1));
            }
            KeyCode::Right | KeyCode::PageDown | KeyCode::Char('n') => {
                page.set(page.get() + 1);
            }
            KeyCode::Home => page.set(0),
            _ => {}
        },
    )?;

    Ok(format!("Closed the graph view ({}).", plural_graphs(graph_count)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use training_controller::logdb::{
        create_experiment, init_schema, LogWriter, NewExperiment, STATUS_RUNNING,
    };

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| item.to_string()).collect()
    }

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

    fn visible(text: &str) -> usize {
        strip_ansi(text).chars().count()
    }

    fn memory_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn
    }

    fn new_experiment(conn: &Connection) -> i64 {
        create_experiment(
            conn,
            &NewExperiment {
                script_name: Some("train"),
                script_path: None,
                config_name: None,
                args: &[],
                status: STATUS_RUNNING,
            },
        )
        .unwrap()
    }

    // --- parsing ------------------------------------------------------------

    #[test]
    fn id_lists_are_parsed() {
        assert_eq!(parse_id_list("3"), Some(vec![3]));
        assert_eq!(parse_id_list("#3"), Some(vec![3]));
        assert_eq!(parse_id_list("3,4, 5"), Some(vec![3, 4, 5]));
        assert_eq!(parse_id_list("3,3,4"), Some(vec![3, 4]));
        assert_eq!(parse_id_list("0"), None);
        assert_eq!(parse_id_list("-2"), None);
        assert_eq!(parse_id_list("loss"), None);
        assert_eq!(parse_id_list("3,loss"), None);
        assert_eq!(parse_id_list(""), None);
    }

    #[test]
    fn specs_default_to_time_and_expand_lists() {
        assert_eq!(parse_specs(&[]).unwrap(), vec![]);
        assert_eq!(
            parse_specs(&strings(&["loss"])).unwrap(),
            vec![GraphSpec::new("loss", "time")]
        );
        assert_eq!(
            parse_specs(&strings(&["acc", "step"])).unwrap(),
            vec![GraphSpec::new("acc", "step")]
        );
        assert_eq!(
            parse_specs(&strings(&["loss,acc", "step"])).unwrap(),
            vec![GraphSpec::new("loss", "step"), GraphSpec::new("acc", "step")]
        );
        assert_eq!(
            parse_specs(&strings(&["loss", "+", "acc", "step", "+", "lr"])).unwrap(),
            vec![
                GraphSpec::new("loss", "time"),
                GraphSpec::new("acc", "step"),
                GraphSpec::new("lr", "time"),
            ]
        );
    }

    #[test]
    fn bad_specs_are_rejected() {
        assert!(parse_specs(&strings(&["+"])).unwrap_err().contains("Empty graph"));
        assert!(parse_specs(&strings(&["loss", "+", "+", "acc"])).is_err());
        assert!(parse_specs(&strings(&["a", "b", "c"])).unwrap_err().contains("'+'"));
        assert!(parse_specs(&strings(&["a,,b"])).unwrap_err().contains("Empty quantity"));
        assert!(parse_specs(&strings(&["a", "b,c"])).unwrap_err().contains("single name"));
    }

    #[test]
    fn graph_commands_are_parsed() {
        let is_config = |name: &str| name == "train";

        let all = parse_graph_command(&strings(&["3"]), is_config).unwrap();
        assert_eq!(all.ids, Some(vec![3]));
        assert_eq!(all.selection, Selection::AllColumns);

        let explicit =
            parse_graph_command(&strings(&["3,4", "loss", "acc", "+", "lr"]), is_config).unwrap();
        assert_eq!(explicit.ids, Some(vec![3, 4]));
        assert_eq!(
            explicit.selection,
            Selection::Explicit(vec![
                GraphSpec::new("loss", "acc"),
                GraphSpec::new("lr", "time")
            ])
        );

        let saved =
            parse_graph_command(&strings(&["3", "loss", "--save", "mine"]), is_config).unwrap();
        assert_eq!(saved.save.as_deref(), Some("mine"));
        let saved_first =
            parse_graph_command(&strings(&["--save", "mine", "3", "loss"]), is_config).unwrap();
        assert_eq!(saved_first.ids, Some(vec![3]));

        let config = parse_graph_command(&strings(&["train"]), is_config).unwrap();
        assert_eq!(config.ids, None);
        assert_eq!(config.selection, Selection::Config("train".into()));
        let config_ids = parse_graph_command(&strings(&["train", "5,6"]), is_config).unwrap();
        assert_eq!(config_ids.ids, Some(vec![5, 6]));
    }

    #[test]
    fn graph_command_errors_are_helpful() {
        let is_config = |name: &str| name == "train";
        assert!(parse_graph_command(&[], is_config).unwrap_err().contains("Usage"));
        assert!(
            parse_graph_command(&strings(&["nope"]), is_config)
                .unwrap_err()
                .contains("neither an experiment ID nor a saved graph config")
        );
        assert!(
            parse_graph_command(&strings(&["3", "--save"]), is_config)
                .unwrap_err()
                .contains("needs a NAME")
        );
        assert!(
            parse_graph_command(&strings(&["train", "--save", "x"]), is_config)
                .unwrap_err()
                .contains("nothing to save")
        );
        assert!(parse_graph_command(&strings(&["train", "loss"]), is_config).is_err());
        assert!(parse_graph_command(&strings(&["train", "3", "4"]), is_config).is_err());
    }

    #[test]
    fn specs_are_formatted_for_the_repl() {
        assert_eq!(
            format_specs(&[GraphSpec::new("loss", "time"), GraphSpec::new("acc", "step")]),
            "loss vs time  +  acc vs step"
        );
    }

    // --- model --------------------------------------------------------------

    #[test]
    fn zip_pairs_values_of_equal_steps() {
        let xs = [(0, 0.0), (1, 1.0), (3, 3.0)];
        let ys = [(1, 10.0), (2, 20.0), (3, 30.0)];
        assert_eq!(zip_by_step(&xs, &ys), vec![(1.0, 10.0), (3.0, 30.0)]);
        assert!(zip_by_step(&[], &ys).is_empty());
    }

    #[test]
    fn view_reads_only_new_rows_and_restarts_with_the_log() {
        let conn = memory_db();
        let id = new_experiment(&conn);
        let mut writer = LogWriter::init(conn, id, strings(&["loss", "acc"])).unwrap();
        writer.log(&[1.0, 0.1]).unwrap();
        writer.log(&[0.5, 0.2]).unwrap();

        let mut view = GraphView::new(
            writer.connection(),
            "test".into(),
            &[id],
            vec![GraphSpec::new("loss", "step"), GraphSpec::new("acc", "loss")],
        )
        .unwrap();
        view.poll(writer.connection());
        assert!(view.running);
        assert_eq!(view.plots[0].tracks[0].points, vec![(0.0, 1.0), (1.0, 0.5)]);
        assert_eq!(view.plots[1].tracks[0].points, vec![(1.0, 0.1), (0.5, 0.2)]);

        writer.log(&[0.25, 0.3]).unwrap();
        view.poll(writer.connection());
        assert_eq!(view.plots[0].tracks[0].points.len(), 3);
        assert_eq!(view.plots[0].tracks[0].points[2], (2.0, 0.25));
        // Polling again without new rows changes nothing.
        view.poll(writer.connection());
        assert_eq!(view.plots[0].tracks[0].points.len(), 3);

        // Finishing the experiment stops the "live" status.
        logdb::finish_experiment(writer.connection(), id, "finished", Some(0)).unwrap();
        view.poll(writer.connection());
        assert!(!view.running);

        // init_log again restarts the line.
        let conn = writer.into_connection();
        let mut writer = LogWriter::init(conn, id, strings(&["loss", "acc"])).unwrap();
        view.poll(writer.connection());
        assert!(view.plots[0].tracks[0].points.is_empty());
        writer.log(&[9.0, 9.0]).unwrap();
        view.poll(writer.connection());
        assert_eq!(view.plots[0].tracks[0].points, vec![(0.0, 9.0)]);
        assert!(view.error.is_none());
    }

    #[test]
    fn view_overlays_experiments_and_validates_quantities() {
        let conn = memory_db();
        let first = new_experiment(&conn);
        let second = new_experiment(&conn);
        let mut one = LogWriter::init(conn, first, strings(&["loss"])).unwrap();
        one.log(&[1.0]).unwrap();
        let conn_ref = one.connection();
        // Experiment two logs through the same connection for the test.
        conn_ref
            .execute(
                "INSERT INTO log_columns (experiment_id, idx, title) VALUES (?1, 0, 'loss')",
                [second],
            )
            .unwrap();

        let view = GraphView::new(
            conn_ref,
            "two runs".into(),
            &[first, second],
            vec![GraphSpec::new("loss", "time")],
        )
        .unwrap();
        assert_eq!(view.plots.len(), 1);
        assert_eq!(view.plots[0].tracks.len(), 2);
        assert_eq!(view.plots[0].title(), "loss vs time");

        let error = GraphView::new(conn_ref, "x".into(), &[first], vec![GraphSpec::new("nope", "time")])
            .unwrap_err();
        assert!(error.contains("no quantity 'nope'") && error.contains("loss, time, step"));
        let error = GraphView::new(conn_ref, "x".into(), &[99], vec![GraphSpec::new("loss", "time")])
            .unwrap_err();
        assert!(error.contains("#99 does not exist"));
        assert!(GraphView::new(conn_ref, "x".into(), &[], vec![]).is_err());
    }

    // --- rendering ----------------------------------------------------------

    #[test]
    fn axis_labels_are_compact() {
        assert_eq!(fmt_axis(0.0), "0");
        assert_eq!(fmt_axis(0.5), "0.5");
        assert_eq!(fmt_axis(12.0), "12");
        assert_eq!(fmt_axis(-1.25), "-1.25");
        assert_eq!(fmt_axis(123.45), "123.5");
        assert_eq!(fmt_axis(0.0001), "1.00e-4");
        assert_eq!(fmt_axis(250000.0), "2.50e5");
    }

    #[test]
    fn ranges_scale_and_widen() {
        assert_eq!(widen(0.0, 1.0), (0.0, 1.0));
        assert_eq!(widen(0.0, 0.0), (-1.0, 1.0));
        let (low, high) = widen(5.0, 5.0);
        assert!(low < 5.0 && high > 5.0);
        assert_eq!(scale_to(0.0, 0.0, 10.0, 20), 0);
        assert_eq!(scale_to(10.0, 0.0, 10.0, 20), 20);
        assert_eq!(scale_to(5.0, 0.0, 10.0, 20), 10);
        assert_eq!(scale_to(99.0, 0.0, 10.0, 20), 20);
    }

    #[test]
    fn tiles_have_exact_dimensions_and_content() {
        let points: Vec<(f64, f64)> = (0..50).map(|i| (i as f64, (i as f64 / 5.0).sin())).collect();
        let series = [TileSeries { label: "#1".into(), points: &points }];

        for (width, height) in [(24, 7), (40, 10), (80, 24)] {
            let lines = render_tile("sin vs time · #1", "time", &series, width, height);
            assert_eq!(lines.len(), height);
            for line in &lines {
                assert_eq!(visible(line), width, "line {:?}", strip_ansi(line));
            }
        }

        let text: Vec<String> = render_tile("sin vs time · #1", "time", &series, 60, 12)
            .iter()
            .map(|line| strip_ansi(line))
            .collect();
        assert!(text[0].contains("sin vs time · #1"));
        assert!(text.iter().any(|line| line.contains('└') && line.contains('─')));
        assert!(text.iter().any(|line| line.chars().any(|ch| ('\u{2801}'..='\u{28ff}').contains(&ch))));
        let x_axis = text.last().unwrap();
        assert!(x_axis.contains('0') && x_axis.contains("49") && x_axis.contains("time"));
        // The y axis shows the data range.
        assert!(text[1].contains('1'));
    }

    #[test]
    fn tiles_handle_no_data_flat_data_single_points_and_legends() {
        let none: [TileSeries; 0] = [];
        let text = render_tile("t", "time", &none, 40, 10)
            .iter()
            .map(|line| strip_ansi(line))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("waiting for data"));

        let flat = [(0.0, 2.0), (1.0, 2.0), (2.0, 2.0)];
        let lines = render_tile("t", "x", &[TileSeries { label: "a".into(), points: &flat }], 40, 10);
        assert!(lines.iter().any(|line| {
            strip_ansi(line)
                .chars()
                .any(|ch| ('\u{2801}'..='\u{28ff}').contains(&ch))
        }));
        assert_eq!(lines.len(), 10);

        let single = [(1.0, 1.0)];
        let lines = render_tile("t", "x", &[TileSeries { label: "a".into(), points: &single }], 40, 10);
        assert!(lines.iter().any(|line| line.contains("\x1b[36m")));

        let other = [(0.0, 0.0), (1.0, 1.0)];
        let two = [
            TileSeries { label: "#3".into(), points: &flat },
            TileSeries { label: "#4".into(), points: &other },
        ];
        let lines = render_tile("two", "x", &two, 40, 10);
        let legend = strip_ansi(lines.last().unwrap());
        assert!(legend.contains("● #3") && legend.contains("● #4"));
        // Both colors are used.
        let joined = lines.join("\n");
        assert!(joined.contains("\x1b[36m") && joined.contains("\x1b[32m"));

        let tiny = render_tile("t", "x", &two, 10, 3);
        assert_eq!(tiny.len(), 3);
        assert!(strip_ansi(&tiny[0]).contains("too small"));
    }

    #[test]
    fn lines_connect_distant_points() {
        let mut canvas = Canvas::new(10, 3);
        canvas.line((0, 0), (19, 11), 0);
        let dots: u32 = canvas.mask.iter().map(|mask| mask.count_ones()).sum();
        assert!(dots >= 20, "only {dots} dots");
        // Out-of-range dots are ignored.
        canvas.set(-1, 0, 0);
        canvas.set(0, 99, 0);
    }

    #[test]
    fn layout_splits_and_pages() {
        assert_eq!(split_even(10, 3, 0), vec![4, 3, 3]);
        assert_eq!(split_even(20, 2, 1), vec![10, 9]);
        assert!(split_even(5, 0, 1).is_empty());

        assert_eq!(page_layout(1, 100, 30), (1, 1));
        assert_eq!(page_layout(2, 100, 30), (2, 2));
        assert_eq!(page_layout(4, 100, 30), (2, 4));
        assert_eq!(page_layout(6, 100, 30).0, 3);
        // A narrow terminal gets fewer columns, a short one fewer rows.
        assert_eq!(page_layout(6, 30, 30), (1, 4));
        assert_eq!(page_layout(6, 100, 8), (3, 3));
    }

    fn view_with(count: usize) -> GraphView {
        let conn = memory_db();
        let id = new_experiment(&conn);
        let titles: Vec<String> = (0..count).map(|i| format!("q{i}")).collect();
        let mut writer = LogWriter::init(conn, id, titles.clone()).unwrap();
        for step in 0..20 {
            let values: Vec<f64> = (0..count).map(|i| (step * (i + 1)) as f64).collect();
            writer.log(&values).unwrap();
        }
        let specs = titles.iter().map(|title| GraphSpec::new(title, "step")).collect();
        let mut view = GraphView::new(writer.connection(), "exp #1".into(), &[id], specs).unwrap();
        view.poll(writer.connection());
        view
    }

    #[test]
    fn screens_fill_the_terminal_and_page() {
        let view = view_with(1);
        for (width, height) in [(80, 24), (50, 12), (30, 9)] {
            let (lines, page, pages) = render_view(&view, 0, width, height);
            assert_eq!(lines.len(), height);
            assert_eq!((page, pages), (0, 1));
            for line in &lines {
                assert!(visible(line) < width, "{:?}", strip_ansi(line));
            }
        }

        let view = view_with(4);
        let (lines, _, pages) = render_view(&view, 0, 120, 30);
        assert_eq!(pages, 1);
        assert_eq!(lines.len(), 30);
        let text: String = lines.iter().map(|line| strip_ansi(line)).collect::<Vec<_>>().join("\n");
        for name in ["q0 vs step", "q1 vs step", "q2 vs step", "q3 vs step"] {
            assert!(text.contains(name), "missing {name}");
        }
        assert!(text.contains("4 graphs") && text.contains("exp #1"));
        // Two tiles share a line.
        assert!(strip_ansi(&lines[1]).matches("vs step").count() == 2);

        // Too many graphs for a small terminal go to more pages.
        let view = view_with(9);
        let (lines, page, pages) = render_view(&view, 0, 60, 16);
        assert!(pages > 1, "pages = {pages}");
        assert_eq!(page, 0);
        let footer = strip_ansi(lines.last().unwrap());
        assert!(footer.contains(&format!("page 1/{pages}")), "{footer}");
        let (_, clamped, _) = render_view(&view, 99, 60, 16);
        assert_eq!(clamped, pages - 1);
    }

    #[test]
    fn odd_graph_counts_stretch_the_last_row() {
        let view = view_with(3);
        let (lines, _, pages) = render_view(&view, 0, 100, 24);
        assert_eq!(pages, 1);
        for line in &lines {
            assert!(visible(line) < 100);
        }
        // The third graph owns a full-width row.
        let titles: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| strip_ansi(line).contains("vs step"))
            .map(|(index, _)| index)
            .collect();
        assert_eq!(titles.len(), 2, "one title row per tile row: {titles:?}");
    }

    #[test]
    fn small_terminals_and_errors_are_reported() {
        let mut view = view_with(1);
        let (lines, _, pages) = render_view(&view, 0, 20, 5);
        assert_eq!(lines.len(), 5);
        assert_eq!(pages, 1);
        assert!(strip_ansi(&lines[0]).contains("Terminal too small"));

        view.error = Some("boom".into());
        let (lines, _, _) = render_view(&view, 0, 80, 24);
        assert!(strip_ansi(lines.last().unwrap()).contains("! boom"));

        view.error = None;
        view.running = true;
        let (lines, _, _) = render_view(&view, 0, 80, 24);
        assert!(strip_ansi(lines.last().unwrap()).contains("● live"));
    }
}
