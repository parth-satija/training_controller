//! Live, flicker-free terminal dashboards for the REPL.
//!
//! The module is split into small, independently testable layers:
//!
//! * **Config** ([`DashboardFile`] -> [`Settings`], [`ConfigWatcher`]) is the
//!   user-editable `dashboard.json`, validated and hot-reloaded while running.
//! * **Sources** ([`JsonlTail`], [`DemoFeed`]) produce JSON records.
//! * **Model** ([`MetricStore`], [`DashboardModel`]) turns records into series.
//! * **Rendering** ([`render_frame`], [`sparkline`]) is pure: model in, lines out.
//! * **Terminal** ([`run_live`]) owns the alternate screen and redraws in place.
//!
//! `/dashboard training` is a second view ([`training`]) fed by the Python API
//! (`init_training_dashboard`, `update_training_dashboard`, `update_hero_text`).
//!
//! A training script only has to append JSON lines to a file, for example
//! `{"step": 10, "loss": 0.42, "accuracy": 0.81}`. Every numeric field becomes
//! a live series. Everything about how it looks is controlled from the config
//! file, never from the REPL.

use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, File};
use std::io::{self, IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

mod training;
pub use training::{
    TrainingConfig, TrainingSource, resolve_training_path, run_training_dashboard,
};

pub const CONFIG_ENV_VAR: &str = "TC_DASHBOARD_CONFIG";
pub const CONFIG_FILE_NAME: &str = "dashboard.json";

const DEFAULT_INTERVAL_MS: u64 = 500;
const DEFAULT_HISTORY: usize = 600;
const MIN_INTERVAL_MS: u64 = 50;
const MAX_INTERVAL_MS: u64 = 60_000;
const MAX_HISTORY: usize = 100_000;
const MAX_DECIMALS: usize = 12;
const BACKFILL_BYTES: u64 = 256 * 1024;
const MIN_WIDTH: usize = 30;
const MIN_HEIGHT: usize = 8;
const MIN_TREND_WIDTH: usize = 4;
const DEMO_RECORDS_PER_POLL: usize = 4;
const DEFAULT_METRICS_FILE: &str = "metrics.jsonl";
const DEFAULT_SPARKLINE: &str = "▁▂▃▄▅▆▇█";
const DEFAULT_FOOTER: &str = " q / Esc / Ctrl+C to exit";

// ---------------------------------------------------------------------------
// Config file (serde layer)
// ---------------------------------------------------------------------------

/// The on-disk `dashboard.json`. Every key is optional; unknown keys are
/// rejected so typos are reported instead of silently ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DashboardFile {
    /// Overrides the header title.
    pub title: Option<String>,
    pub interval_ms: u64,
    /// Values kept per metric.
    pub history: usize,
    pub source: SourceFile,
    pub layout: LayoutFile,
    pub format: FormatFile,
    pub theme: ThemeFile,
    pub metrics: MetricsFile,
}

impl Default for DashboardFile {
    fn default() -> Self {
        Self {
            title: None,
            interval_ms: DEFAULT_INTERVAL_MS,
            history: DEFAULT_HISTORY,
            source: SourceFile::default(),
            layout: LayoutFile::default(),
            format: FormatFile::default(),
            theme: ThemeFile::default(),
            metrics: MetricsFile::default(),
        }
    }
}

/// Where data comes from when no path or `--demo` is given in the REPL.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SourceFile {
    /// Absolute path of a JSON-lines metrics file.
    pub path: Option<String>,
    pub demo: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LayoutFile {
    pub show_header: bool,
    pub show_status: bool,
    pub show_footer: bool,
    /// Ordered subset of: trend, latest, min, max, mean.
    pub columns: Vec<String>,
    /// Glyphs from lowest to highest, at least two (e.g. " .:-=+*#%@" for ASCII).
    pub sparkline: String,
    /// Width of the numeric columns.
    pub column_width: usize,
    pub name_width_max: usize,
    pub footer: String,
}

impl Default for LayoutFile {
    fn default() -> Self {
        Self {
            show_header: true,
            show_status: true,
            show_footer: true,
            columns: ["trend", "latest", "min", "max"]
                .iter()
                .map(|name| name.to_string())
                .collect(),
            sparkline: DEFAULT_SPARKLINE.to_string(),
            column_width: 10,
            name_width_max: 16,
            footer: DEFAULT_FOOTER.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FormatFile {
    pub decimals: usize,
    /// Values at or above this use scientific notation.
    pub scientific_above: f64,
    /// Non-zero values below this use scientific notation.
    pub scientific_below: f64,
}

impl Default for FormatFile {
    fn default() -> Self {
        Self {
            decimals: 4,
            scientific_above: 10_000.0,
            scientific_below: 0.001,
        }
    }
}

/// Colors are a name (`red`, `bright_blue`, `gray`, `default`), a 256-color
/// index (`0`-`255`) or `#rrggbb`. Unspecified attributes are off.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StyleFile {
    pub fg: Option<String>,
    pub bg: Option<String>,
    pub bold: bool,
    pub dim: bool,
    pub underline: bool,
    pub reverse: bool,
}

impl StyleFile {
    fn new(fg: Option<&str>, bold: bool, dim: bool, reverse: bool) -> Self {
        Self {
            fg: fg.map(str::to_string),
            bg: None,
            bold,
            dim,
            underline: false,
            reverse,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThemeFile {
    pub header: StyleFile,
    pub subheader: StyleFile,
    pub column_header: StyleFile,
    pub status_ok: StyleFile,
    pub status_error: StyleFile,
    pub footer: StyleFile,
    pub hint: StyleFile,
    /// The bold line in the middle of the training dashboard.
    pub hero: StyleFile,
    /// Series colors, assigned in display order.
    pub palette: Vec<String>,
}

impl Default for ThemeFile {
    fn default() -> Self {
        Self {
            header: StyleFile::new(None, true, false, true),
            subheader: StyleFile::new(None, false, true, false),
            column_header: StyleFile::new(None, false, true, false),
            status_ok: StyleFile::new(Some("green"), false, false, false),
            status_error: StyleFile::new(Some("red"), false, false, false),
            footer: StyleFile::new(None, false, true, false),
            hint: StyleFile::new(None, false, true, false),
            hero: StyleFile::new(None, true, false, false),
            palette: ["cyan", "green", "yellow", "magenta", "blue", "red"]
                .iter()
                .map(|name| name.to_string())
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetricsFile {
    /// Metrics shown first, in this order.
    pub order: Vec<String>,
    /// Metrics never shown (their history is still kept).
    pub hide: Vec<String>,
    /// Show only the metrics listed in `order`.
    pub only_ordered: bool,
    /// Numeric fields treated as progress (shown in the header, not plotted).
    pub index_keys: Vec<String>,
    /// Fields that are ignored completely.
    pub ignore_keys: Vec<String>,
    pub per_metric: BTreeMap<String, MetricOverride>,
}

impl Default for MetricsFile {
    fn default() -> Self {
        Self {
            order: Vec::new(),
            hide: Vec::new(),
            only_ordered: false,
            index_keys: ["step", "epoch", "iteration", "iter"]
                .iter()
                .map(|name| name.to_string())
                .collect(),
            ignore_keys: ["time", "timestamp"]
                .iter()
                .map(|name| name.to_string())
                .collect(),
            per_metric: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetricOverride {
    pub label: Option<String>,
    pub color: Option<String>,
    pub decimals: Option<usize>,
    /// Suffix appended to the numeric columns, e.g. "%" or "s".
    pub unit: Option<String>,
}

// ---------------------------------------------------------------------------
// Settings (validated, ready to render)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Column {
    Trend,
    Latest,
    Min,
    Max,
    Mean,
}

impl Column {
    fn parse(name: &str) -> std::result::Result<Self, String> {
        match name.trim().to_lowercase().as_str() {
            "trend" => Ok(Self::Trend),
            "latest" => Ok(Self::Latest),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "mean" => Ok(Self::Mean),
            other => Err(format!(
                "layout.columns: unknown column '{other}' (use trend, latest, min, max, mean)"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Trend => "trend",
            Self::Latest => "latest",
            Self::Min => "min",
            Self::Max => "max",
            Self::Mean => "mean",
        }
    }
}

#[derive(Debug, Clone)]
struct MetricStyle {
    label: Option<String>,
    color: Option<String>,
    decimals: Option<usize>,
    unit: String,
}

/// SGR parameter strings, e.g. `"1;7"`; empty means unstyled.
#[derive(Debug, Clone)]
struct Styles {
    header: String,
    subheader: String,
    column_header: String,
    status_ok: String,
    status_error: String,
    footer: String,
    hint: String,
    hero: String,
}

#[derive(Debug, Clone)]
pub struct Settings {
    title: Option<String>,
    interval: Duration,
    history: usize,
    show_header: bool,
    show_status: bool,
    show_footer: bool,
    columns: Vec<Column>,
    spark_chars: Vec<char>,
    column_width: usize,
    name_width_max: usize,
    footer: String,
    decimals: usize,
    scientific_above: f64,
    scientific_below: f64,
    styles: Styles,
    palette: Vec<String>,
    order: Vec<String>,
    hide: Vec<String>,
    only_ordered: bool,
    index_keys: Vec<String>,
    ignore_keys: Vec<String>,
    per_metric: BTreeMap<String, MetricStyle>,
    source_path: Option<PathBuf>,
    source_demo: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self::compile(&DashboardFile::default()).expect("built-in defaults are valid")
    }
}

impl Settings {
    pub fn compile(file: &DashboardFile) -> std::result::Result<Self, String> {
        let interval = validate_interval_ms(file.interval_ms)?;
        if !(1..=MAX_HISTORY).contains(&file.history) {
            return Err(format!("history must be between 1 and {MAX_HISTORY}."));
        }

        let layout = &file.layout;
        let mut columns: Vec<Column> = Vec::new();
        for name in &layout.columns {
            let column = Column::parse(name)?;
            if columns.contains(&column) {
                return Err(format!("layout.columns lists '{}' twice.", column.label()));
            }
            columns.push(column);
        }
        let spark_chars: Vec<char> = layout.sparkline.chars().collect();
        if spark_chars.len() < 2 {
            return Err("layout.sparkline needs at least two characters.".into());
        }
        if !(4..=24).contains(&layout.column_width) {
            return Err("layout.column_width must be between 4 and 24.".into());
        }
        if !(1..=64).contains(&layout.name_width_max) {
            return Err("layout.name_width_max must be between 1 and 64.".into());
        }

        let format = &file.format;
        if format.decimals > MAX_DECIMALS {
            return Err(format!("format.decimals must be at most {MAX_DECIMALS}."));
        }
        if !format.scientific_above.is_finite() || format.scientific_above <= 0.0 {
            return Err("format.scientific_above must be a positive number.".into());
        }
        if !format.scientific_below.is_finite() || format.scientific_below < 0.0 {
            return Err("format.scientific_below must be zero or a positive number.".into());
        }

        let theme = &file.theme;
        let styles = Styles {
            header: compile_style(&theme.header, "theme.header")?,
            subheader: compile_style(&theme.subheader, "theme.subheader")?,
            column_header: compile_style(&theme.column_header, "theme.column_header")?,
            status_ok: compile_style(&theme.status_ok, "theme.status_ok")?,
            status_error: compile_style(&theme.status_error, "theme.status_error")?,
            footer: compile_style(&theme.footer, "theme.footer")?,
            hint: compile_style(&theme.hint, "theme.hint")?,
            hero: compile_style(&theme.hero, "theme.hero")?,
        };
        if theme.palette.is_empty() {
            return Err("theme.palette needs at least one color.".into());
        }
        let palette = theme
            .palette
            .iter()
            .map(|color| color_code(color, false).map_err(|error| format!("theme.palette: {error}")))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut per_metric = BTreeMap::new();
        for (name, item) in &file.metrics.per_metric {
            if item.decimals.is_some_and(|decimals| decimals > MAX_DECIMALS) {
                return Err(format!(
                    "metrics.per_metric.{name}.decimals must be at most {MAX_DECIMALS}."
                ));
            }
            let color = item
                .color
                .as_deref()
                .map(|color| color_code(color, false))
                .transpose()
                .map_err(|error| format!("metrics.per_metric.{name}.color: {error}"))?;
            per_metric.insert(
                name.clone(),
                MetricStyle {
                    label: item.label.clone(),
                    color,
                    decimals: item.decimals,
                    unit: item.unit.clone().unwrap_or_default(),
                },
            );
        }

        let mut order: Vec<String> = Vec::new();
        for name in &file.metrics.order {
            if !order.contains(name) {
                order.push(name.clone());
            }
        }

        let source_path = match file.source.path.as_deref() {
            Some(path) if !Path::new(path).is_absolute() => {
                return Err("source.path must be absolute.".into());
            }
            Some(path) => Some(PathBuf::from(path)),
            None => None,
        };

        Ok(Self {
            title: file.title.clone(),
            interval,
            history: file.history,
            show_header: layout.show_header,
            show_status: layout.show_status,
            show_footer: layout.show_footer,
            columns,
            spark_chars,
            column_width: layout.column_width,
            name_width_max: layout.name_width_max,
            footer: layout.footer.clone(),
            decimals: format.decimals,
            scientific_above: format.scientific_above,
            scientific_below: format.scientific_below,
            styles,
            palette,
            order,
            hide: file.metrics.hide.clone(),
            only_ordered: file.metrics.only_ordered,
            index_keys: file.metrics.index_keys.clone(),
            ignore_keys: file.metrics.ignore_keys.clone(),
            per_metric,
            source_path,
            source_demo: file.source.demo,
        })
    }
}

fn validate_interval_ms(millis: u64) -> std::result::Result<Duration, String> {
    if !(MIN_INTERVAL_MS..=MAX_INTERVAL_MS).contains(&millis) {
        return Err(format!(
            "Interval must be between {MIN_INTERVAL_MS} and {MAX_INTERVAL_MS} ms."
        ));
    }
    Ok(Duration::from_millis(millis))
}

/// Converts a color spec into an SGR parameter string.
fn color_code(spec: &str, background: bool) -> std::result::Result<String, String> {
    const NAMES: [&str; 8] = [
        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
    ];
    let spec = spec.trim().to_lowercase();
    let base: usize = if background { 40 } else { 30 };
    let bright: usize = if background { 100 } else { 90 };
    let extended = if background { 48 } else { 38 };

    if spec == "default" {
        return Ok((base + 9).to_string());
    }
    if spec == "gray" || spec == "grey" {
        return Ok(bright.to_string());
    }
    if let Some(index) = NAMES.iter().position(|name| *name == spec) {
        return Ok((base + index).to_string());
    }
    if let Some(name) = spec.strip_prefix("bright_") {
        if let Some(index) = NAMES.iter().position(|candidate| *candidate == name) {
            return Ok((bright + index).to_string());
        }
    }
    if let Some(hex) = spec.strip_prefix('#') {
        if hex.len() == 6 && hex.chars().all(|ch| ch.is_ascii_hexdigit()) {
            let channel = |range: std::ops::Range<usize>| u8::from_str_radix(&hex[range], 16);
            if let (Ok(r), Ok(g), Ok(b)) = (channel(0..2), channel(2..4), channel(4..6)) {
                return Ok(format!("{extended};2;{r};{g};{b}"));
            }
        }
    }
    if let Ok(index) = spec.parse::<u8>() {
        return Ok(format!("{extended};5;{index}"));
    }
    Err(format!(
        "unknown color '{spec}' (use a name, bright_<name>, gray, 0-255 or #rrggbb)"
    ))
}

fn compile_style(style: &StyleFile, field: &str) -> std::result::Result<String, String> {
    let mut parts: Vec<String> = Vec::new();
    if style.bold {
        parts.push("1".into());
    }
    if style.dim {
        parts.push("2".into());
    }
    if style.underline {
        parts.push("4".into());
    }
    if style.reverse {
        parts.push("7".into());
    }
    if let Some(fg) = &style.fg {
        parts.push(color_code(fg, false).map_err(|error| format!("{field}.fg: {error}"))?);
    }
    if let Some(bg) = &style.bg {
        parts.push(color_code(bg, true).map_err(|error| format!("{field}.bg: {error}"))?);
    }
    Ok(parts.join(";"))
}

// ---------------------------------------------------------------------------
// Config loading and hot reload
// ---------------------------------------------------------------------------

/// `TC_DASHBOARD_CONFIG` if set, otherwise `dashboard.json` next to `tc.json`.
pub fn config_path(app_dir: &Path) -> PathBuf {
    match std::env::var_os(CONFIG_ENV_VAR) {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => app_dir.join(CONFIG_FILE_NAME),
    }
}

/// Writes a complete default config the first time, so every option is discoverable.
pub fn ensure_config_file(path: &Path) -> std::result::Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create {}: {error}", parent.display()))?;
    }
    let contents = serde_json::to_vec_pretty(&DashboardFile::default())
        .map_err(|error| format!("Failed to serialize the default dashboard config: {error}"))?;
    fs::write(path, contents)
        .map_err(|error| format!("Failed to write {}: {error}", path.display()))
}

pub fn load_settings(path: &Path) -> std::result::Result<Settings, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    let file: DashboardFile = serde_json::from_str(&contents)
        .map_err(|error| format!("Failed to parse {}: {error}", path.display()))?;
    Settings::compile(&file)
}

/// Creates the config if needed, loads it, and returns a watcher for live edits.
pub fn open_settings(path: &Path) -> std::result::Result<(Settings, ConfigWatcher), String> {
    ensure_config_file(path)?;
    let settings = load_settings(path)?;
    Ok((settings, ConfigWatcher::primed(path.to_path_buf())))
}

/// Detects edits to the config file by modification time and size.
pub struct ConfigWatcher {
    path: PathBuf,
    stamp: Option<(SystemTime, u64)>,
}

impl ConfigWatcher {
    fn stamp_of(path: &Path) -> Option<(SystemTime, u64)> {
        let metadata = fs::metadata(path).ok()?;
        Some((metadata.modified().ok()?, metadata.len()))
    }

    pub fn primed(path: PathBuf) -> Self {
        let stamp = Self::stamp_of(&path);
        Self { path, stamp }
    }

    /// `None` when nothing changed; otherwise the result of reloading the file.
    pub fn check(&mut self) -> Option<std::result::Result<Settings, String>> {
        let stamp = Self::stamp_of(&self.path)?;
        if self.stamp == Some(stamp) {
            return None;
        }
        self.stamp = Some(stamp);
        Some(load_settings(&self.path))
    }
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

/// New records produced by a source since the previous poll.
#[derive(Debug, Default)]
pub struct SourceUpdate {
    pub records: Vec<Value>,
    /// The source started over (for example the log file was truncated).
    pub restarted: bool,
}

/// Follows a JSON-lines file, yielding only complete lines appended since the last poll.
pub struct JsonlTail {
    path: PathBuf,
    offset: u64,
    primed: bool,
}

impl JsonlTail {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            offset: 0,
            primed: false,
        }
    }

    /// Like [`JsonlTail::new`], but always reads from the first byte instead of
    /// only the last `BACKFILL_BYTES`. The training view needs the `init` line.
    pub fn from_start(path: PathBuf) -> Self {
        Self {
            path,
            offset: 0,
            primed: true,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn poll(&mut self) -> std::result::Result<SourceUpdate, String> {
        let mut update = SourceUpdate::default();

        let mut file = match File::open(&self.path) {
            Ok(file) => file,
            // The training run has not created its log yet; keep waiting.
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(update),
            Err(error) => {
                return Err(format!("Failed to open {}: {error}", self.path.display()));
            }
        };
        let len = file
            .metadata()
            .map_err(|error| format!("Failed to stat {}: {error}", self.path.display()))?
            .len();

        if len < self.offset {
            self.offset = 0;
            update.restarted = true;
        }

        let mut start = self.offset;
        let mut drop_partial_line = false;
        if !self.primed {
            self.primed = true;
            if len > BACKFILL_BYTES {
                start = len - BACKFILL_BYTES;
                drop_partial_line = true;
            }
        }
        if len <= start {
            return Ok(update);
        }

        file.seek(SeekFrom::Start(start))
            .map_err(|error| format!("Failed to seek {}: {error}", self.path.display()))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|error| format!("Failed to read {}: {error}", self.path.display()))?;

        // Only consume complete lines; a half-written line is picked up next poll.
        let Some(last_newline) = bytes.iter().rposition(|byte| *byte == b'\n') else {
            return Ok(update);
        };
        self.offset = start + last_newline as u64 + 1;

        let first = if drop_partial_line {
            bytes
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |index| index + 1)
        } else {
            0
        };
        if first > last_newline {
            return Ok(update);
        }

        update.records = String::from_utf8_lossy(&bytes[first..=last_newline])
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .collect();
        Ok(update)
    }
}

/// Deterministic synthetic training run, useful for trying the dashboard out.
pub struct DemoFeed {
    step: u64,
    rng: u64,
}

impl DemoFeed {
    pub fn new() -> Self {
        Self {
            step: 0,
            rng: 0x2545_F491_4F6C_DD1D,
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
        let mut records = Vec::with_capacity(DEMO_RECORDS_PER_POLL);
        for _ in 0..DEMO_RECORDS_PER_POLL {
            self.step += 1;
            let t = self.step as f64;
            let loss = 2.0 * (-t / 150.0).exp() + 0.05 + self.noise() * 0.08;
            let val_loss = loss * 1.1 + 0.03 + self.noise() * 0.05;
            let accuracy =
                (1.0 - 0.9 * (-t / 200.0).exp() + self.noise() * 0.02).clamp(0.0, 1.0);
            let lr = 1e-3 * 0.5 * (1.0 + (std::f64::consts::PI * (t / 2000.0).min(1.0)).cos());
            records.push(serde_json::json!({
                "step": self.step,
                "loss": loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "lr": lr,
            }));
        }
        SourceUpdate {
            records,
            restarted: false,
        }
    }
}

pub enum DashboardSource {
    Jsonl(JsonlTail),
    Demo(DemoFeed),
}

impl DashboardSource {
    pub fn jsonl(path: PathBuf) -> Self {
        Self::Jsonl(JsonlTail::new(path))
    }

    pub fn demo() -> Self {
        Self::Demo(DemoFeed::new())
    }

    pub fn label(&self) -> String {
        match self {
            Self::Jsonl(tail) => tail.path().display().to_string(),
            Self::Demo(_) => "demo data".to_string(),
        }
    }

    pub fn poll(&mut self) -> std::result::Result<SourceUpdate, String> {
        match self {
            Self::Jsonl(tail) => tail.poll(),
            Self::Demo(feed) => Ok(feed.poll()),
        }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A bounded history of values for one metric.
#[derive(Debug)]
pub struct Series {
    values: VecDeque<f64>,
    capacity: usize,
}

impl Series {
    pub fn new(capacity: usize) -> Self {
        Self {
            values: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
        while self.values.len() > self.capacity {
            self.values.pop_front();
        }
    }

    pub fn push(&mut self, value: f64) {
        if self.values.len() >= self.capacity {
            self.values.pop_front();
        }
        self.values.push_back(value);
    }

    pub fn latest(&self) -> Option<f64> {
        self.values.back().copied()
    }

    pub fn min(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::min)
    }

    pub fn max(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::max)
    }

    pub fn mean(&self) -> Option<f64> {
        if self.values.is_empty() {
            None
        } else {
            Some(self.values.iter().sum::<f64>() / self.values.len() as f64)
        }
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.values.iter().copied().collect()
    }
}

#[derive(Debug)]
pub struct MetricStore {
    series: BTreeMap<String, Series>,
    progress: Option<(String, f64)>,
    history: usize,
}

impl MetricStore {
    pub fn new(history: usize) -> Self {
        Self {
            series: BTreeMap::new(),
            progress: None,
            history,
        }
    }

    pub fn set_history(&mut self, history: usize) {
        self.history = history;
        for series in self.series.values_mut() {
            series.set_capacity(history);
        }
    }

    pub fn clear(&mut self) {
        self.series.clear();
        self.progress = None;
    }

    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    pub fn series(&self) -> &BTreeMap<String, Series> {
        &self.series
    }

    pub fn progress(&self) -> Option<&(String, f64)> {
        self.progress.as_ref()
    }

    /// Adds every finite numeric field of a JSON object. Returns how many values were stored.
    pub fn ingest(&mut self, record: &Value, settings: &Settings) -> usize {
        let Some(object) = record.as_object() else {
            return 0;
        };

        let mut stored = 0;
        for (key, value) in object {
            let Some(number) = value.as_f64().filter(|number| number.is_finite()) else {
                continue;
            };
            if settings.index_keys.iter().any(|candidate| candidate == key) {
                self.progress = Some((key.clone(), number));
                continue;
            }
            if settings.ignore_keys.iter().any(|candidate| candidate == key) {
                continue;
            }
            let history = self.history;
            self.series
                .entry(key.clone())
                .or_insert_with(|| Series::new(history))
                .push(number);
            stored += 1;
        }
        stored
    }
}

pub struct DashboardModel {
    title: String,
    source_label: String,
    settings: Settings,
    interval_override: Option<Duration>,
    started: Instant,
    records: u64,
    store: MetricStore,
    source_error: Option<String>,
    config_error: Option<String>,
}

impl DashboardModel {
    pub fn new(
        title: String,
        source_label: String,
        settings: Settings,
        interval_override: Option<Duration>,
    ) -> Self {
        let store = MetricStore::new(settings.history);
        Self {
            title,
            source_label,
            settings,
            interval_override,
            started: Instant::now(),
            records: 0,
            store,
            source_error: None,
            config_error: None,
        }
    }

    pub fn interval(&self) -> Duration {
        self.interval_override.unwrap_or(self.settings.interval)
    }

    pub fn apply(&mut self, update: SourceUpdate) {
        if update.restarted {
            self.store.clear();
            self.records = 0;
        }
        for record in &update.records {
            self.store.ingest(record, &self.settings);
        }
        self.records += update.records.len() as u64;
    }

    /// Swaps in freshly loaded settings without losing any collected data.
    pub fn apply_settings(&mut self, settings: Settings) {
        self.store.set_history(settings.history);
        self.settings = settings;
        self.config_error = None;
    }

    pub fn set_source_error(&mut self, error: Option<String>) {
        self.source_error = error;
    }

    pub fn set_config_error(&mut self, error: Option<String>) {
        self.config_error = error;
    }

    pub fn records(&self) -> u64 {
        self.records
    }

    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Rendering (pure)
// ---------------------------------------------------------------------------

fn style(text: &str, code: &str) -> String {
    if code.is_empty() {
        text.to_string()
    } else {
        format!("\x1b[{code}m{text}\x1b[0m")
    }
}

/// Truncates or pads `text` to exactly `width` characters.
pub fn fit(text: &str, width: usize) -> String {
    let mut out: String = text.chars().take(width).collect();
    let len = out.chars().count();
    out.push_str(&" ".repeat(width - len));
    out
}

/// Right-aligns `text`; if it does not fit, it is clipped and marked with an ellipsis.
pub fn fit_right(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if len <= width {
        format!("{}{text}", " ".repeat(width - len))
    } else if width == 0 {
        String::new()
    } else {
        let mut clipped: String = text.chars().take(width - 1).collect();
        clipped.push('…');
        clipped
    }
}

pub fn fmt_num(value: f64, decimals: usize, scientific_above: f64, scientific_below: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else if value.abs() >= scientific_above || value.abs() < scientific_below {
        format!("{value:.2e}")
    } else {
        format!("{value:.decimals$}")
    }
}

pub fn fmt_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    format!("{:02}:{:02}:{:02}", secs / 3600, (secs / 60) % 60, secs % 60)
}

/// Renders the most recent `width` values using the given glyphs (lowest to highest).
pub fn sparkline(values: &[f64], glyphs: &[char], width: usize) -> String {
    if width == 0 || values.is_empty() || glyphs.is_empty() {
        return String::new();
    }

    let window = &values[values.len().saturating_sub(width)..];
    let (min, max) = window
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(*value), hi.max(*value))
        });
    let span = max - min;
    let top = glyphs.len() - 1;

    window
        .iter()
        .map(|value| {
            if span <= f64::EPSILON {
                glyphs[glyphs.len() / 2]
            } else {
                let index = (((value - min) / span) * top as f64).round() as usize;
                glyphs[index.min(top)]
            }
        })
        .collect()
}

fn blank_frame(lines: Vec<String>, height: usize, width: usize) -> Vec<String> {
    let mut lines = lines;
    while lines.len() < height {
        lines.push(fit("", width));
    }
    lines.truncate(height);
    lines
}

/// Series to display, honoring `order`, `hide` and `only_ordered`.
fn visible_series<'a>(
    store: &'a MetricStore,
    settings: &Settings,
) -> Vec<(&'a String, &'a Series)> {
    let hidden = |name: &str| settings.hide.iter().any(|candidate| candidate == name);

    let mut visible: Vec<(&String, &Series)> = settings
        .order
        .iter()
        .filter(|name| !hidden(name))
        .filter_map(|name| store.series().get_key_value(name))
        .collect();

    if !settings.only_ordered {
        for (name, series) in store.series() {
            if !settings.order.contains(name) && !hidden(name) {
                visible.push((name, series));
            }
        }
    }
    visible
}

fn render_body(model: &DashboardModel, w: usize, rows: usize) -> Vec<String> {
    let s = &model.settings;

    if model.store.is_empty() {
        return vec![
            fit("  Waiting for metrics...", w),
            style(
                &fit("  Append JSON lines such as {\"step\": 1, \"loss\": 0.5}", w),
                &s.styles.hint,
            ),
        ];
    }

    let visible = visible_series(&model.store, s);
    if visible.is_empty() {
        return vec![fit("  No visible metrics (see metrics.order / metrics.hide).", w)];
    }

    let label_of = |name: &str| {
        s.per_metric
            .get(name)
            .and_then(|meta| meta.label.clone())
            .unwrap_or_else(|| name.to_string())
    };
    let name_width = visible
        .iter()
        .map(|(name, _)| label_of(name).chars().count())
        .max()
        .unwrap_or(0)
        .min(s.name_width_max);

    let has_trend = s.columns.contains(&Column::Trend);
    let numeric_columns = s.columns.iter().filter(|c| **c != Column::Trend).count();
    let base = 1 + name_width + numeric_columns * (s.column_width + 1) + usize::from(has_trend);
    let needed = base + if has_trend { MIN_TREND_WIDTH } else { 0 };
    if w < needed {
        return vec![fit(
            &format!("  Too narrow for the configured columns (need {})", needed + 1),
            w,
        )];
    }
    let spark_width = w - base;

    let mut header = format!(" {}", fit("metric", name_width));
    for column in &s.columns {
        match column {
            Column::Trend => header.push_str(&format!(" {}", fit("trend", spark_width))),
            other => header.push_str(&format!(" {}", fit_right(other.label(), s.column_width))),
        }
    }

    let mut lines = vec![style(&fit(&header, w), &s.styles.column_header)];

    let rows_available = rows.saturating_sub(1);
    let total = visible.len();
    let truncated = total > rows_available;
    let shown = if truncated {
        rows_available.saturating_sub(1)
    } else {
        total
    };

    for (index, (name, series)) in visible.iter().take(shown).enumerate() {
        let meta = s.per_metric.get(name.as_str());
        let color = meta
            .and_then(|meta| meta.color.clone())
            .unwrap_or_else(|| s.palette[index % s.palette.len()].clone());
        let decimals = meta.and_then(|meta| meta.decimals).unwrap_or(s.decimals);
        let unit = meta.map(|meta| meta.unit.as_str()).unwrap_or("");
        let values = series.to_vec();

        let mut row = format!(" {}", fit(&label_of(name), name_width));
        for column in &s.columns {
            let stat = match column {
                Column::Trend => {
                    let spark = sparkline(&values, &s.spark_chars, spark_width);
                    row.push_str(&format!(" {}", style(&fit(&spark, spark_width), &color)));
                    continue;
                }
                Column::Latest => series.latest(),
                Column::Min => series.min(),
                Column::Max => series.max(),
                Column::Mean => series.mean(),
            };
            let text = format!(
                "{}{unit}",
                fmt_num(
                    stat.unwrap_or(0.0),
                    decimals,
                    s.scientific_above,
                    s.scientific_below
                )
            );
            row.push_str(&format!(" {}", fit_right(&text, s.column_width)));
        }
        lines.push(row);
    }
    if truncated {
        lines.push(style(
            &fit(&format!("  ... +{} more", total - shown), w),
            &s.styles.hint,
        ));
    }
    lines
}

/// Builds exactly `height` lines, each at most `width - 1` visible characters wide.
pub fn render_frame(model: &DashboardModel, width: usize, height: usize) -> Vec<String> {
    let s = &model.settings;
    let w = width.saturating_sub(1).max(1);

    if width < MIN_WIDTH || height < MIN_HEIGHT {
        let message = format!(" Terminal too small (need {MIN_WIDTH}x{MIN_HEIGHT})");
        return blank_frame(vec![fit(&message, w)], height, w);
    }

    let mut lines: Vec<String> = Vec::with_capacity(height);

    if s.show_header {
        let title = s.title.as_deref().unwrap_or(&model.title);
        lines.push(style(
            &fit(&format!(" {title}  ·  {}", model.source_label), w),
            &s.styles.header,
        ));

        let progress = model
            .store
            .progress()
            .map(|(key, value)| {
                format!(
                    "  ·  {key} {}",
                    fmt_num(*value, s.decimals, s.scientific_above, s.scientific_below)
                )
            })
            .unwrap_or_default();
        lines.push(style(
            &fit(
                &format!(
                    " up {}  ·  {} records{progress}",
                    fmt_duration(model.uptime()),
                    model.records()
                ),
                w,
            ),
            &s.styles.subheader,
        ));
        lines.push(fit("", w));
    }

    let bottom = usize::from(s.show_status) + usize::from(s.show_footer);
    let body_rows = height.saturating_sub(lines.len() + bottom);
    let mut body = render_body(model, w, body_rows);
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
        } else {
            (
                format!(" ● live  ·  refresh {}ms", model.interval().as_millis()),
                &s.styles.status_ok,
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
// Terminal handling
// ---------------------------------------------------------------------------

/// Enters the alternate screen in raw mode and restores the terminal on drop,
/// including when the dashboard exits through an error or a panic.
struct ScreenGuard;

impl ScreenGuard {
    fn enter() -> std::result::Result<Self, String> {
        terminal::enable_raw_mode()
            .map_err(|error| format!("Failed to enable raw mode: {error}"))?;
        if let Err(error) = execute!(io::stdout(), EnterAlternateScreen, Hide) {
            let _ = terminal::disable_raw_mode();
            return Err(format!("Failed to enter the alternate screen: {error}"));
        }
        Ok(Self)
    }
}

impl Drop for ScreenGuard {
    fn drop(&mut self) {
        let _ = execute!(io::stdout(), Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

enum Input {
    Timeout,
    Resize,
    Quit,
    /// Any other key press (used by views that react to keys, e.g. paging).
    Key(KeyCode),
}

/// Waits up to `interval` for input, so a keypress never shifts the refresh cadence.
fn wait_for_input(interval: Duration) -> std::result::Result<Input, String> {
    let deadline = Instant::now() + interval;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(Input::Timeout);
        }
        let ready = event::poll(remaining)
            .map_err(|error| format!("Failed to poll terminal events: {error}"))?;
        if !ready {
            return Ok(Input::Timeout);
        }
        match event::read().map_err(|error| format!("Failed to read terminal event: {error}"))? {
            Event::Resize(_, _) => return Ok(Input::Resize),
            Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => return Ok(Input::Quit),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Ok(Input::Quit);
                }
                code => return Ok(Input::Key(code)),
            },
            _ => {}
        }
    }
}

/// Paints a frame in place: no full clear, one write, wrapped in a synchronized
/// update so terminals that support it never show a half-drawn frame.
fn draw(out: &mut impl Write, lines: &[String], clear: bool) -> io::Result<()> {
    let mut buffer = String::from("\x1b[?2026h");
    if clear {
        buffer.push_str("\x1b[2J");
    }
    for (row, line) in lines.iter().enumerate() {
        buffer.push_str(&format!("\x1b[{};1H{line}\x1b[K", row + 1));
    }
    buffer.push_str("\x1b[J\x1b[?2026l");
    out.write_all(buffer.as_bytes())?;
    out.flush()
}

/// One repaint plus how long to wait before the next one.
pub struct Frame {
    pub lines: Vec<String>,
    pub interval: Duration,
}

/// Generic live loop: calls `frame(width, height)`, paints it, and waits for the
/// interval it asks for. Returns when the user presses q, Esc or Ctrl+C.
pub fn run_live(
    frame: impl FnMut(usize, usize) -> Frame,
) -> std::result::Result<(), String> {
    run_live_with_keys(frame, |_| {})
}

/// [`run_live`] that also reports every key other than q / Esc / Ctrl+C to
/// `on_key`, then repaints at once so the view reacts immediately.
pub fn run_live_with_keys(
    mut frame: impl FnMut(usize, usize) -> Frame,
    mut on_key: impl FnMut(KeyCode),
) -> std::result::Result<(), String> {
    let _guard = ScreenGuard::enter()?;
    let mut out = io::stdout();
    let mut clear = true;

    loop {
        let (columns, rows) = terminal::size()
            .map_err(|error| format!("Failed to read terminal size: {error}"))?;
        let next = frame(columns as usize, rows as usize);
        draw(&mut out, &next.lines, clear)
            .map_err(|error| format!("Failed to draw the dashboard: {error}"))?;
        clear = false;

        match wait_for_input(next.interval)? {
            Input::Quit => return Ok(()),
            Input::Resize => clear = true,
            Input::Key(code) => on_key(code),
            Input::Timeout => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

pub struct DashboardConfig {
    /// Used for the header unless the config file sets `title`.
    pub title: String,
    pub source: DashboardSource,
    pub settings: Settings,
    pub watcher: ConfigWatcher,
    /// An interval given in the REPL wins over the config file.
    pub interval_override: Option<Duration>,
}

impl DashboardConfig {
    pub fn new(
        title: String,
        source: DashboardSource,
        settings: Settings,
        watcher: ConfigWatcher,
        interval_override: Option<Duration>,
    ) -> Self {
        Self {
            title,
            source,
            settings,
            watcher,
            interval_override,
        }
    }
}

pub fn parse_interval(text: &str) -> std::result::Result<Duration, String> {
    let millis = text
        .trim()
        .parse::<u64>()
        .map_err(|_| "Interval must be a whole number of milliseconds.".to_string())?;
    validate_interval_ms(millis)
}

/// Uses the explicit path if given, otherwise `metrics.jsonl` next to the active script.
pub fn resolve_metrics_path(
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

    let script = active_script_path.ok_or_else(|| {
        "Provide a metrics path, select a script first, set source in the dashboard config, or use --demo."
            .to_string()
    })?;
    Path::new(script)
        .parent()
        .map(|directory| directory.join(DEFAULT_METRICS_FILE))
        .ok_or_else(|| format!("Could not determine the directory of {script}"))
}

/// REPL arguments win, then the config file's `source`, then the active script's folder.
pub fn resolve_source(
    cli_demo: bool,
    cli_path: Option<&str>,
    settings: &Settings,
    active_script_path: Option<&str>,
) -> std::result::Result<DashboardSource, String> {
    if cli_demo || (cli_path.is_none() && settings.source_demo) {
        return Ok(DashboardSource::demo());
    }
    if cli_path.is_none() {
        if let Some(path) = &settings.source_path {
            return Ok(DashboardSource::jsonl(path.clone()));
        }
    }
    resolve_metrics_path(cli_path, active_script_path).map(DashboardSource::jsonl)
}

/// Runs a live dashboard until the user exits. Returns a short summary for the REPL.
pub fn run_dashboard(config: DashboardConfig) -> std::result::Result<String, String> {
    if !io::stdout().is_terminal() {
        return Err("The dashboard needs an interactive terminal.".into());
    }

    let DashboardConfig {
        title,
        mut source,
        settings,
        mut watcher,
        interval_override,
    } = config;
    let mut model = DashboardModel::new(title, source.label(), settings, interval_override);

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
        Frame {
            lines: render_frame(&model, width, height),
            interval: model.interval(),
        }
    })?;

    Ok(format!(
        "Dashboard closed after {} ({} records).",
        fmt_duration(model.uptime()),
        model.records()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    fn temp_path(label: &str) -> PathBuf {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "training_controller-{label}-{}-{unique_id}.json",
            std::process::id()
        ))
    }

    fn settings_from(value: serde_json::Value) -> Settings {
        let file: DashboardFile = serde_json::from_value(value).unwrap();
        Settings::compile(&file).unwrap()
    }

    fn demo_model(settings: Settings) -> DashboardModel {
        let mut model = DashboardModel::new("test".into(), "demo data".into(), settings, None);
        model.apply(DemoFeed::new().poll());
        model
    }

    fn frame_text(model: &DashboardModel, width: usize, height: usize) -> String {
        render_frame(model, width, height)
            .iter()
            .map(|line| strip_ansi(line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    const GLYPHS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    #[test]
    fn sparkline_scales_and_handles_flat_and_empty_input() {
        assert_eq!(sparkline(&[], &GLYPHS, 5), "");
        assert_eq!(sparkline(&[1.0, 2.0], &GLYPHS, 0), "");
        assert_eq!(sparkline(&[0.0, 1.0], &GLYPHS, 2), "▁█");
        assert_eq!(sparkline(&[2.0, 2.0, 2.0], &GLYPHS, 3), "▅▅▅");
        assert_eq!(sparkline(&[0.0, 5.0, 0.0, 1.0], &GLYPHS, 2).chars().count(), 2);
        assert_eq!(sparkline(&[0.0, 1.0], &['.', '#'], 2), ".#");
    }

    #[test]
    fn fit_helpers_truncate_pad_and_clip() {
        assert_eq!(fit("abcdef", 3), "abc");
        assert_eq!(fit("ab", 4), "ab  ");
        assert_eq!(fit_right("ab", 4), "  ab");
        assert_eq!(fit_right("abcdef", 4), "abc…");
    }

    #[test]
    fn series_is_bounded_and_tracks_statistics() {
        let mut series = Series::new(3);
        for value in [5.0, 1.0, 3.0, 4.0] {
            series.push(value);
        }
        assert_eq!(series.to_vec(), vec![1.0, 3.0, 4.0]);
        assert_eq!(series.latest(), Some(4.0));
        assert_eq!(series.min(), Some(1.0));
        assert_eq!(series.max(), Some(4.0));
        assert!((series.mean().unwrap() - 8.0 / 3.0).abs() < 1e-9);

        series.set_capacity(1);
        assert_eq!(series.to_vec(), vec![4.0]);
    }

    #[test]
    fn ingest_respects_configured_keys() {
        let mut store = MetricStore::new(10);
        let settings = Settings::default();
        let stored = store.ingest(
            &serde_json::json!({"step": 7, "loss": 0.5, "note": "x", "timestamp": 1}),
            &settings,
        );
        assert_eq!(stored, 1);
        assert_eq!(store.progress(), Some(&("step".to_string(), 7.0)));
        assert_eq!(store.ingest(&serde_json::json!([1, 2]), &settings), 0);

        let custom = settings_from(serde_json::json!({
            "metrics": {"index_keys": ["global_step"], "ignore_keys": ["noise"]}
        }));
        let mut store = MetricStore::new(10);
        store.ingest(
            &serde_json::json!({"global_step": 3, "step": 9, "noise": 1.0, "loss": 1.0}),
            &custom,
        );
        assert_eq!(store.progress(), Some(&("global_step".to_string(), 3.0)));
        assert!(store.series().contains_key("step"));
        assert!(!store.series().contains_key("noise"));
    }

    #[test]
    fn default_config_round_trips_and_compiles() {
        let file = DashboardFile::default();
        let text = serde_json::to_string_pretty(&file).unwrap();
        let parsed: DashboardFile = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed, file);
        assert!(Settings::compile(&parsed).is_ok());

        let partial: DashboardFile = serde_json::from_str("{\"interval_ms\": 250}").unwrap();
        assert_eq!(partial.history, DEFAULT_HISTORY);
    }

    #[test]
    fn config_rejects_typos_and_invalid_values() {
        assert!(serde_json::from_str::<DashboardFile>("{\"intervall_ms\": 5}").is_err());

        let bad = |value: serde_json::Value| {
            let file: DashboardFile = serde_json::from_value(value).unwrap();
            Settings::compile(&file).unwrap_err()
        };
        assert!(bad(serde_json::json!({"interval_ms": 1})).contains("Interval"));
        assert!(bad(serde_json::json!({"history": 0})).contains("history"));
        assert!(bad(serde_json::json!({"layout": {"columns": ["nope"]}})).contains("unknown column"));
        assert!(bad(serde_json::json!({"layout": {"columns": ["min", "min"]}})).contains("twice"));
        assert!(bad(serde_json::json!({"layout": {"sparkline": "x"}})).contains("sparkline"));
        assert!(bad(serde_json::json!({"theme": {"palette": ["chartreuse"]}})).contains("palette"));
        assert!(bad(serde_json::json!({"theme": {"header": {"fg": "#12"}}})).contains("theme.header.fg"));
        assert!(bad(serde_json::json!({"source": {"path": "relative.jsonl"}})).contains("absolute"));
    }

    #[test]
    fn color_codes_cover_names_indexes_and_truecolor() {
        assert_eq!(color_code("red", false).unwrap(), "31");
        assert_eq!(color_code("red", true).unwrap(), "41");
        assert_eq!(color_code("bright_blue", false).unwrap(), "94");
        assert_eq!(color_code("gray", false).unwrap(), "90");
        assert_eq!(color_code("default", false).unwrap(), "39");
        assert_eq!(color_code("208", false).unwrap(), "38;5;208");
        assert_eq!(color_code("#ff8000", true).unwrap(), "48;2;255;128;0");
        assert!(color_code("nope", false).is_err());
    }

    #[test]
    fn tail_reads_only_complete_lines_and_detects_restart() {
        let path = temp_path("tail");
        let mut tail = JsonlTail::new(path.clone());
        assert!(tail.poll().unwrap().records.is_empty());

        let mut file = File::create(&path).unwrap();
        write!(file, "{{\"loss\": 1.0}}\n{{\"loss\": 0.9").unwrap();
        file.flush().unwrap();
        let update = tail.poll().unwrap();
        assert_eq!(update.records.len(), 1);
        assert!(!update.restarted);

        write!(file, "}}\nnot json\n").unwrap();
        file.flush().unwrap();
        let update = tail.poll().unwrap();
        assert_eq!(update.records.len(), 1);
        assert_eq!(update.records[0]["loss"], 0.9);

        drop(file);
        fs::write(&path, "{\"loss\": 5.0}\n").unwrap();
        let update = tail.poll().unwrap();
        assert!(update.restarted);
        assert_eq!(update.records.len(), 1);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn watcher_reports_edits_and_invalid_edits_once() {
        let path = temp_path("watch");
        let (settings, mut watcher) = open_settings(&path).unwrap();
        assert_eq!(settings.interval, Duration::from_millis(DEFAULT_INTERVAL_MS));
        assert!(watcher.check().is_none());

        fs::write(&path, "{\"interval_ms\": 250}").unwrap();
        let reloaded = watcher.check().unwrap().unwrap();
        assert_eq!(reloaded.interval, Duration::from_millis(250));
        assert!(watcher.check().is_none());

        fs::write(&path, "{bad").unwrap();
        assert!(watcher.check().unwrap().is_err());
        assert!(watcher.check().is_none());

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn model_keeps_data_when_settings_reload() {
        let mut model = demo_model(Settings::default());
        let before = model.store.series()["loss"].to_vec().len();
        model.set_config_error(Some("old".into()));
        model.apply_settings(settings_from(serde_json::json!({"history": 2})));
        assert!(model.config_error.is_none());
        assert_eq!(model.store.series()["loss"].to_vec().len(), 2usize.min(before));
    }

    #[test]
    fn render_frame_fills_the_screen_without_overflowing() {
        let model = demo_model(Settings::default());

        for (width, height) in [(80, 24), (120, 10), (60, 8), (40, 5)] {
            let lines = render_frame(&model, width, height);
            assert_eq!(lines.len(), height);
            for line in &lines {
                assert!(strip_ansi(line).chars().count() < width.max(2));
            }
        }

        let text = frame_text(&model, 80, 24);
        assert!(text.contains("loss"));
        assert!(text.contains("accuracy"));
        assert!(text.contains("step"));
    }

    #[test]
    fn render_frame_handles_empty_model_and_errors() {
        let mut model = DashboardModel::new(
            "test".into(),
            "file.jsonl".into(),
            Settings::default(),
            None,
        );
        assert!(frame_text(&model, 80, 24).contains("Waiting for metrics"));

        model.set_source_error(Some("boom".into()));
        assert!(frame_text(&model, 80, 24).contains("! boom"));

        model.set_config_error(Some("bad json".into()));
        assert!(frame_text(&model, 80, 24).contains("! config: bad json"));
    }

    #[test]
    fn layout_and_metric_options_change_the_output() {
        let settings = settings_from(serde_json::json!({
            "title": "My run",
            "layout": {
                "columns": ["latest", "trend", "mean"],
                "show_footer": false,
                "sparkline": " .:-=+*#%@"
            },
            "metrics": {
                "order": ["accuracy"],
                "hide": ["lr"],
                "per_metric": {"loss": {"label": "Train loss", "unit": "s", "decimals": 2}}
            }
        }));
        let model = demo_model(settings);
        let text = frame_text(&model, 100, 24);

        assert!(text.contains("My run"));
        assert!(text.contains("Train loss"));
        assert!(!text.contains("\n lr "));
        assert!(!text.contains("q / Esc"));
        let header = text.lines().find(|line| line.contains("metric")).unwrap();
        assert!(header.find("latest").unwrap() < header.find("trend").unwrap());
        assert!(header.contains("mean"));
        assert!(!header.contains("max"));

        let rows: Vec<&str> = text.lines().filter(|line| line.starts_with(' ')).collect();
        let first_metric = rows
            .iter()
            .position(|line| line.contains("metric"))
            .map(|index| rows[index + 1])
            .unwrap();
        assert!(first_metric.contains("accuracy"));
    }

    #[test]
    fn only_ordered_and_narrow_columns_are_handled() {
        let settings = settings_from(serde_json::json!({
            "metrics": {"order": ["loss"], "only_ordered": true}
        }));
        let model = demo_model(settings);
        let text = frame_text(&model, 80, 24);
        assert!(text.contains("loss"));
        assert!(!text.contains("val_loss"));

        let wide = settings_from(serde_json::json!({
            "layout": {"columns": ["trend", "latest", "min", "max", "mean"], "column_width": 12}
        }));
        let model = demo_model(wide);
        assert!(frame_text(&model, 60, 12).contains("Too narrow"));
    }

    #[test]
    fn parse_interval_validates_range() {
        assert_eq!(parse_interval("500").unwrap(), Duration::from_millis(500));
        assert!(parse_interval("10").is_err());
        assert!(parse_interval("abc").is_err());
        assert!(parse_interval("999999").is_err());
    }

    #[test]
    fn source_resolution_follows_precedence() {
        let cwd = std::env::current_dir().unwrap();
        let absolute = cwd.join("m.jsonl").to_string_lossy().into_owned();
        let script = cwd.join("train.py").to_string_lossy().into_owned();
        let default_settings = Settings::default();

        assert!(matches!(
            resolve_source(true, None, &default_settings, None).unwrap(),
            DashboardSource::Demo(_)
        ));
        assert!(resolve_source(false, None, &default_settings, None).is_err());
        assert!(resolve_source(false, Some("relative.jsonl"), &default_settings, None).is_err());
        let from_script = resolve_source(false, None, &default_settings, Some(&script)).unwrap();
        assert_eq!(
            from_script.label(),
            cwd.join(DEFAULT_METRICS_FILE).display().to_string()
        );

        let configured = settings_from(serde_json::json!({"source": {"path": absolute}}));
        assert_eq!(
            resolve_source(false, None, &configured, Some(&script))
                .unwrap()
                .label(),
            absolute
        );
        // A path typed in the REPL beats the config file.
        let other = cwd.join("other.jsonl").to_string_lossy().into_owned();
        assert_eq!(
            resolve_source(false, Some(&other), &configured, None)
                .unwrap()
                .label(),
            other
        );

        let demo_configured = settings_from(serde_json::json!({"source": {"demo": true}}));
        assert!(matches!(
            resolve_source(false, None, &demo_configured, None).unwrap(),
            DashboardSource::Demo(_)
        ));
    }
}
