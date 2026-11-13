//! Checkpoint storage: safe writes, an index, retention policies and management.
//!
//! This module is plain Rust. The Python API (`tc.save_checkpoint(...)`, see
//! `api.rs` / `python.rs`) and the REPL commands (`/checkpoints`, ...) both use
//! it, so they always agree on the on-disk format.
//!
//! # Layout
//!
//! One *store* is one directory (the REPL uses `<app dir>/checkpoints/<script>/`
//! and passes it to the script in `TC_CHECKPOINT_DIR`):
//!
//! ```text
//! checkpoint_000001.ckpt   the checkpoint data (opaque bytes, any format)
//! checkpoint_000002.ckpt
//! index.json               id, step, best/pinned flags, size, SHA-256, metrics, ...
//! index.json.bak           the previous index (for manual recovery)
//! policy.json              the retention / storage settings (see [`Policy`])
//! ```
//!
//! # Safety guarantees
//!
//! * **Atomic writes.** Data is written to a temporary file, flushed to disk
//!   (`fsync`), read back and checked against its SHA-256 (policy `verify`),
//!   and only then renamed into place. A crash or full disk never leaves a
//!   half-written checkpoint under a real name.
//! * **Index first, files second.** The index is replaced atomically before
//!   any file is deleted, so the index never points at something that was
//!   already removed. Leftovers can only be orphan files, which
//!   [`Store::verify`] reports.
//! * **Nothing is deleted by default.** A fresh store keeps every checkpoint.
//!   Deletion only happens through the retention rules the user configured,
//!   or an explicit remove.
//! * **Protected checkpoints.** Pinned checkpoints are never pruned, and
//!   never removed without `force`.
//! * **Runs are isolated.** Retention rules apply per run (experiment), so a
//!   new run cannot delete the checkpoints of an earlier one. Only the
//!   optional total size cap spans runs (see [`Policy::max_total_mb`]).
//! * **Concurrency.** Several processes (two scripts at once) can use one
//!   store: index updates are serialized with a lock file that is only held
//!   for the quick commit step, never while data is being written.
//! * **No path tricks.** File names come from the policy prefix/extension
//!   (validated) and an id; names read from a hand-edited index are rejected
//!   unless they are plain file names inside the store directory.
//! * **A damaged index is never overwritten.** If `index.json` cannot be
//!   parsed, every operation fails with a clear message and changes nothing.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub type Result<T> = std::result::Result<T, String>;

/// Environment variable naming the store directory. `/run_script` sets it.
pub const DIR_ENV: &str = "TC_CHECKPOINT_DIR";
/// Used when [`DIR_ENV`] is missing (a script started by hand): `./checkpoints`.
pub const DEFAULT_DIR: &str = "checkpoints";

const INDEX_FILE: &str = "index.json";
const INDEX_BACKUP: &str = "index.json.bak";
const POLICY_FILE: &str = "policy.json";
const LOCK_FILE: &str = ".lock";
const TMP_SUFFIX: &str = ".tmp";
const BUFFER_SIZE: usize = 1 << 20;
const LOCK_TIMEOUT: Duration = Duration::from_secs(10);
const LOCK_STALE_AFTER: Duration = Duration::from_secs(60);
const TMP_STALE_AFTER: Duration = Duration::from_secs(3600);
const MIB: u64 = 1024 * 1024;

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// Presets for [`Policy::apply_preset`]: `(name, description)`.
pub const PRESETS: [(&str, &str); 4] = [
    ("all", "keep every checkpoint"),
    ("best_only", "keep only the best checkpoint of each run"),
    ("last_and_best", "keep the last 2 checkpoints and the best one of each run"),
    (
        "milestones",
        "keep the last 2, the best, and every 1000th step of each run",
    ),
];

/// The names accepted by [`Policy::set`].
pub const POLICY_KEYS: [&str; 9] = [
    "keep_last",
    "keep_best",
    "keep_first",
    "keep_every",
    "max_total_mb",
    "save_every",
    "verify",
    "name_prefix",
    "extension",
];

/// Storage and retention settings, stored as `policy.json` in the store.
///
/// The retention rules (`keep_*`) decide which checkpoints *survive* pruning,
/// per run. A checkpoint is kept if **any** rule keeps it. With
/// `keep_last = all` (the default) nothing is pruned by these rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Policy {
    /// Keep the newest N checkpoints of each run. `None` = keep all, `0` = none.
    pub keep_last: Option<usize>,
    /// Keep the newest N checkpoints of each run that were saved with `is_best=True`.
    pub keep_best: usize,
    /// Always keep the first checkpoint of each run.
    pub keep_first: bool,
    /// Keep every checkpoint whose `step` is a multiple of N (milestones).
    pub keep_every: Option<u64>,
    /// Size cap in MiB for the whole store. When exceeded, the oldest checkpoints
    /// that are not pinned, not a kept best one and not the newest overall are
    /// deleted, across runs. `None` = no cap.
    pub max_total_mb: Option<u64>,
    /// Store only every Nth `save_checkpoint` call; best checkpoints are always stored.
    pub save_every: u64,
    /// Read every checkpoint back and check its SHA-256 before committing it.
    pub verify: bool,
    /// File name prefix: `<prefix>_<id>.<extension>`.
    pub name_prefix: String,
    /// File extension without the dot.
    pub extension: String,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            keep_last: None,
            keep_best: 1,
            keep_first: false,
            keep_every: None,
            max_total_mb: None,
            save_every: 1,
            verify: true,
            name_prefix: "checkpoint".to_string(),
            extension: "ckpt".to_string(),
        }
    }
}

fn is_safe_component(value: &str, allow_punctuation: bool) -> bool {
    let starts_alphanumeric = value
        .chars()
        .next()
        .is_some_and(|first| first.is_ascii_alphanumeric());
    starts_alphanumeric
        && value.chars().count() <= 32
        && value.chars().all(|ch| {
            ch.is_ascii_alphanumeric() || (allow_punctuation && matches!(ch, '_' | '-' | '.'))
        })
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Ok(true),
        "false" | "no" | "off" | "0" => Ok(false),
        _ => Err(format!("'{value}' is not a boolean (use true or false).")),
    }
}

fn parse_number(value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|_| format!("'{value}' is not a non-negative whole number."))
}

/// A number, or one of `words` meaning "no value".
fn parse_optional(value: &str, words: &[&str]) -> Result<Option<u64>> {
    if words.iter().any(|word| value.eq_ignore_ascii_case(word)) {
        Ok(None)
    } else {
        parse_number(value).map(Some)
    }
}

fn show_optional<T: ToString>(value: Option<T>, none: &str) -> String {
    value.map_or_else(|| none.to_string(), |value| value.to_string())
}

impl Policy {
    /// Rejects settings that would be unsafe or meaningless.
    pub fn validate(&self) -> Result<()> {
        if self.keep_every == Some(0) {
            return Err("keep_every must be at least 1 (use 'off' to disable it).".to_string());
        }
        if self.max_total_mb == Some(0) {
            return Err("max_total_mb must be at least 1 (use 'off' to disable it).".to_string());
        }
        if self.save_every == 0 {
            return Err("save_every must be at least 1.".to_string());
        }
        if !is_safe_component(&self.name_prefix, true) {
            return Err(format!(
                "Invalid name_prefix '{}': use up to 32 letters, digits, '_', '-' or '.', starting with a letter or digit.",
                self.name_prefix
            ));
        }
        if !is_safe_component(&self.extension, false) {
            return Err(format!(
                "Invalid extension '{}': use up to 32 letters or digits, without a dot.",
                self.extension
            ));
        }
        Ok(())
    }

    /// Changes one setting by name (see [`POLICY_KEYS`]) and validates the result.
    ///
    /// * `keep_last`: a number, or `all` / `unlimited`
    /// * `keep_every`, `max_total_mb`: a number, or `off` / `none`
    /// * `keep_best`, `save_every`: a number
    /// * `keep_first`, `verify`: `true` or `false`
    /// * `name_prefix`, `extension`: text
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let value = value.trim();
        match key {
            "keep_last" => {
                self.keep_last = parse_optional(value, &["all", "unlimited"])?
                    .map(|number| usize::try_from(number).map_err(|_| "keep_last is too large.".to_string()))
                    .transpose()?;
            }
            "keep_best" => {
                self.keep_best =
                    usize::try_from(parse_number(value)?).map_err(|_| "keep_best is too large.".to_string())?;
            }
            "keep_first" => self.keep_first = parse_bool(value)?,
            "keep_every" => self.keep_every = parse_optional(value, &["off", "none", "unlimited"])?,
            "max_total_mb" => self.max_total_mb = parse_optional(value, &["off", "none", "unlimited"])?,
            "save_every" => self.save_every = parse_number(value)?,
            "verify" => self.verify = parse_bool(value)?,
            "name_prefix" => self.name_prefix = value.to_string(),
            "extension" => self.extension = value.trim_start_matches('.').to_string(),
            other => {
                return Err(format!(
                    "Unknown setting '{other}'. Valid settings: {}.",
                    POLICY_KEYS.join(", ")
                ));
            }
        }
        self.validate()
    }

    /// Replaces the retention rules with a named preset (see [`PRESETS`]).
    /// `save_every`, `verify` and the file naming are left unchanged.
    pub fn apply_preset(&mut self, name: &str) -> Result<()> {
        let (keep_last, keep_every) = match name {
            "all" => (None, None),
            "best_only" => (Some(0), None),
            "last_and_best" => (Some(2), None),
            "milestones" => (Some(2), Some(1000)),
            other => {
                let names: Vec<&str> = PRESETS.iter().map(|(preset, _)| *preset).collect();
                return Err(format!(
                    "Unknown preset '{other}'. Available: {}.",
                    names.join(", ")
                ));
            }
        };
        self.keep_last = keep_last;
        self.keep_best = 1;
        self.keep_first = false;
        self.keep_every = keep_every;
        self.max_total_mb = None;
        Ok(())
    }

    /// `(key, current value, explanation)` for every setting, in display order.
    pub fn describe(&self) -> Vec<(&'static str, String, &'static str)> {
        let yes_no = |value: bool| {
            if value {
                "true".to_string()
            } else {
                "false".to_string()
            }
        };
        vec![
            (
                "keep_last",
                show_optional(self.keep_last, "all"),
                "newest checkpoints kept per run (all = no limit, 0 = none)",
            ),
            (
                "keep_best",
                self.keep_best.to_string(),
                "newest checkpoints saved as best kept per run",
            ),
            (
                "keep_first",
                yes_no(self.keep_first),
                "always keep the first checkpoint of each run",
            ),
            (
                "keep_every",
                show_optional(self.keep_every, "off"),
                "keep every checkpoint whose step is a multiple of N",
            ),
            (
                "max_total_mb",
                show_optional(self.max_total_mb, "off"),
                "size cap for the whole store; oldest unprotected go first",
            ),
            (
                "save_every",
                self.save_every.to_string(),
                "store every Nth save call (best ones are always stored)",
            ),
            (
                "verify",
                yes_no(self.verify),
                "read each checkpoint back and check its SHA-256 before committing",
            ),
            ("name_prefix", self.name_prefix.clone(), "file name prefix"),
            ("extension", self.extension.clone(), "file extension"),
        ]
    }
}

// ---------------------------------------------------------------------------
// Index entries and the pruning plan (pure)
// ---------------------------------------------------------------------------

/// One stored checkpoint, as recorded in `index.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub id: u64,
    /// Plain file name inside the store directory.
    pub file: String,
    #[serde(default)]
    pub step: Option<u64>,
    #[serde(default)]
    pub best: bool,
    #[serde(default)]
    pub pinned: bool,
    pub size: u64,
    pub sha256: String,
    pub created_unix: u64,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    /// The experiment (run) that saved it, when known.
    #[serde(default)]
    pub experiment: Option<i64>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Index {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    next_id: u64,
    #[serde(default)]
    calls: u64,
    #[serde(default)]
    checkpoints: Vec<Entry>,
}

/// What [`plan_prune`] decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrunePlan {
    /// Ids to delete, ascending.
    pub remove: Vec<u64>,
    /// The size cap could not be met because the rest is protected.
    pub still_over_cap: bool,
}

/// Decides which checkpoints a policy deletes. Pure: no I/O.
///
/// Retention rules apply per run (`experiment`); the size cap applies to the
/// whole list, oldest first, and never removes pinned checkpoints, the kept
/// best ones, or the newest checkpoint overall.
pub fn plan_prune(entries: &[Entry], policy: &Policy) -> PrunePlan {
    let mut runs: BTreeMap<Option<i64>, Vec<&Entry>> = BTreeMap::new();
    for entry in entries {
        runs.entry(entry.experiment).or_default().push(entry);
    }

    let mut remove: HashSet<u64> = HashSet::new();
    let mut protected: HashSet<u64> = HashSet::new();

    for run in runs.values_mut() {
        run.sort_by_key(|entry| entry.id);

        let best: HashSet<u64> = run
            .iter()
            .rev()
            .filter(|entry| entry.best)
            .take(policy.keep_best)
            .map(|entry| entry.id)
            .collect();
        protected.extend(best.iter().copied());
        protected.extend(run.iter().filter(|entry| entry.pinned).map(|entry| entry.id));

        let Some(keep_last) = policy.keep_last else {
            continue; // keep everything
        };
        let newest: HashSet<u64> = run.iter().rev().take(keep_last).map(|entry| entry.id).collect();
        let first = if policy.keep_first {
            run.first().map(|entry| entry.id)
        } else {
            None
        };
        for entry in run.iter() {
            let milestone = match (policy.keep_every, entry.step) {
                (Some(every), Some(step)) => every > 0 && step % every == 0,
                _ => false,
            };
            let keep = entry.pinned
                || best.contains(&entry.id)
                || newest.contains(&entry.id)
                || first == Some(entry.id)
                || milestone;
            if !keep {
                remove.insert(entry.id);
            }
        }
    }

    let mut still_over_cap = false;
    if let Some(cap_mb) = policy.max_total_mb {
        let cap = cap_mb.saturating_mul(MIB);
        let mut oldest_first: Vec<&Entry> = entries.iter().collect();
        oldest_first.sort_by_key(|entry| entry.id);
        let newest_overall = oldest_first.last().map(|entry| entry.id);

        let mut total: u64 = oldest_first
            .iter()
            .filter(|entry| !remove.contains(&entry.id))
            .map(|entry| entry.size)
            .sum();
        for entry in &oldest_first {
            if total <= cap {
                break;
            }
            if remove.contains(&entry.id)
                || protected.contains(&entry.id)
                || Some(entry.id) == newest_overall
            {
                continue;
            }
            remove.insert(entry.id);
            total = total.saturating_sub(entry.size);
        }
        still_over_cap = total > cap;
    }

    let mut remove: Vec<u64> = remove.into_iter().collect();
    remove.sort_unstable();
    PrunePlan {
        remove,
        still_over_cap,
    }
}

/// Finds an entry by id (`7` or `#7`), or by `latest` / `best`.
pub fn resolve_entry<'a>(entries: &'a [Entry], spec: &str) -> Result<&'a Entry> {
    let spec = spec.trim().trim_start_matches('#');
    let found = match spec.to_ascii_lowercase().as_str() {
        "latest" => entries.iter().max_by_key(|entry| entry.id),
        "best" => entries
            .iter()
            .filter(|entry| entry.best)
            .max_by_key(|entry| entry.id),
        other => {
            let id: u64 = other.parse().map_err(|_| {
                format!("'{spec}' is not a checkpoint id (use a number, 'latest' or 'best').")
            })?;
            entries.iter().find(|entry| entry.id == id)
        }
    };
    found.ok_or_else(|| format!("No checkpoint matches '{spec}'."))
}

/// `1536` -> `1.5 KB`.
pub fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// A folder name for a script's store: unsafe characters become `_`.
pub fn dir_name_for(script_name: &str) -> String {
    let cleaned: String = script_name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let trimmed: String = cleaned.trim_start_matches('.').chars().take(64).collect();
    if trimmed.is_empty() {
        "script".to_string()
    } else {
        trimmed
    }
}

fn describe_entry(entry: &Entry) -> String {
    let mut parts = Vec::new();
    if let Some(step) = entry.step {
        parts.push(format!("step {step}"));
    }
    if entry.best {
        parts.push("best".to_string());
    }
    parts.push(format_size(entry.size));
    parts.join(", ")
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// What the caller supplies with a checkpoint.
#[derive(Debug, Clone, Default)]
pub struct NewCheckpoint {
    /// Saved because it is the best so far.
    pub is_best: bool,
    /// Training step or epoch.
    pub step: Option<u64>,
    /// Metrics at that step. NaN and infinite values are dropped.
    pub metrics: BTreeMap<String, f64>,
    /// The experiment (run) saving it.
    pub experiment: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// Not stored because `save_every` is greater than 1 and this was not a best checkpoint.
    Interval(u64),
    /// Not stored because the retention policy would delete it immediately.
    Policy,
}

#[derive(Debug, Clone)]
pub struct SaveOutcome {
    /// The stored checkpoint, or `None` if it was skipped.
    pub entry: Option<Entry>,
    pub path: Option<PathBuf>,
    /// Older checkpoints deleted by the retention policy.
    pub removed: Vec<Entry>,
    pub warnings: Vec<String>,
    pub skipped: Option<SkipReason>,
}

impl SaveOutcome {
    fn skipped(reason: SkipReason, warnings: Vec<String>) -> Self {
        Self {
            entry: None,
            path: None,
            removed: Vec::new(),
            warnings,
            skipped: Some(reason),
        }
    }

    /// Lines for the script's output. They start with `[checkpoint]`, so the
    /// REPL shows them in their own `checkpoint` panel.
    pub fn summary_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(entry) = &self.entry {
            let location = self
                .path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default();
            lines.push(format!(
                "[checkpoint] saved #{} ({}) -> {location}",
                entry.id,
                describe_entry(entry)
            ));
        }
        match &self.skipped {
            Some(SkipReason::Interval(every)) => lines.push(format!(
                "[checkpoint] skipped: only every {every}th save is stored (save_every); best checkpoints are always stored"
            )),
            Some(SkipReason::Policy) => lines.push(
                "[checkpoint] not stored: the retention policy would delete it immediately (it is not best and keep_last is too low)"
                    .to_string(),
            ),
            None => {}
        }
        for entry in &self.removed {
            lines.push(format!(
                "[checkpoint] removed #{} ({}) by the retention policy",
                entry.id,
                describe_entry(entry)
            ));
        }
        for warning in &self.warnings {
            lines.push(format!("[checkpoint] warning: {warning}"));
        }
        lines
    }
}

#[derive(Debug, Clone)]
pub struct PruneReport {
    pub removed: Vec<Entry>,
    pub still_over_cap: bool,
    pub warnings: Vec<String>,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub checked: usize,
    /// Missing, resized or modified checkpoints.
    pub problems: Vec<String>,
    /// Files in the store that no index entry refers to.
    pub orphans: Vec<String>,
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hash_file(path: &Path) -> io::Result<(u64, String)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; BUFFER_SIZE];
    let mut size = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size += read as u64;
    }
    Ok((size, hex(hasher.finalize())))
}

/// Writes `bytes` to a temporary file next to `path`, flushes it to disk and
/// renames it over `path`.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid path {}", path.display()))?;
    let temporary = path.with_file_name(format!("{file_name}.{}{TMP_SUFFIX}", std::process::id()));

    let written = (|| -> io::Result<()> {
        let mut file = File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)
    })();
    if let Err(error) = written {
        let _ = fs::remove_file(&temporary);
        return Err(format!("Failed to write {}: {error}", path.display()));
    }
    Ok(())
}

fn is_plain_file_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', ':', '\0'])
}

fn is_internal_file(name: &str) -> bool {
    matches!(name, INDEX_FILE | INDEX_BACKUP | POLICY_FILE | LOCK_FILE) || name.ends_with(TMP_SUFFIX)
}

/// A lock file held while the index is read, changed and written.
struct DirLock {
    path: PathBuf,
}

impl DirLock {
    fn acquire(directory: &Path) -> Result<Self> {
        let path = directory.join(LOCK_FILE);
        let deadline = Instant::now() + LOCK_TIMEOUT;
        loop {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let _ = writeln!(file, "{}", std::process::id());
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    // A crashed process can leave its lock behind.
                    let stale = fs::metadata(&path)
                        .and_then(|meta| meta.modified())
                        .ok()
                        .and_then(|modified| modified.elapsed().ok())
                        .is_some_and(|age| age > LOCK_STALE_AFTER);
                    if stale {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return Err(format!(
                            "Timed out waiting for the checkpoint store lock ({}). If no other script is saving, delete that file.",
                            path.display()
                        ));
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                Err(error) => {
                    return Err(format!("Failed to create {}: {error}", path.display()));
                }
            }
        }
    }
}

impl Drop for DirLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Data written to a temporary file, not yet part of the store.
struct Staged {
    temporary: PathBuf,
    size: u64,
    sha256: String,
}

fn next_id(index: &Index) -> u64 {
    let highest = index.checkpoints.iter().map(|entry| entry.id).max().unwrap_or(0);
    index.next_id.max(highest + 1).max(1)
}

fn finite_metrics(metrics: BTreeMap<String, f64>, warnings: &mut Vec<String>) -> BTreeMap<String, f64> {
    let total = metrics.len();
    let kept: BTreeMap<String, f64> = metrics
        .into_iter()
        .filter(|(_, value)| value.is_finite())
        .collect();
    if kept.len() != total {
        warnings.push(format!(
            "{} metric(s) were NaN or infinite and were not recorded",
            total - kept.len()
        ));
    }
    kept
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// A checkpoint directory. Cheap to create; every method re-reads the files, so
/// several processes can share one store.
#[derive(Debug, Clone)]
pub struct Store {
    dir: PathBuf,
}

impl Store {
    /// Opens (and creates) the store directory and removes stale temporary files.
    pub fn open(directory: impl Into<PathBuf>) -> Result<Self> {
        let dir = directory.into();
        fs::create_dir_all(&dir).map_err(|error| {
            format!("Cannot create the checkpoint directory {}: {error}", dir.display())
        })?;
        let store = Self { dir };
        store.remove_stale_temporary_files();
        Ok(store)
    }

    /// The store named by `TC_CHECKPOINT_DIR`, else `./checkpoints`.
    pub fn from_env() -> Result<Self> {
        let directory = std::env::var_os(DIR_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_DIR));
        Self::open(directory)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn remove_stale_temporary_files(&self) {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return;
        };
        for item in entries.flatten() {
            let path = item.path();
            let is_temporary = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(TMP_SUFFIX));
            let stale = fs::metadata(&path)
                .and_then(|meta| meta.modified())
                .ok()
                .and_then(|modified| modified.elapsed().ok())
                .is_some_and(|age| age > TMP_STALE_AFTER);
            if is_temporary && stale {
                let _ = fs::remove_file(path);
            }
        }
    }

    fn lock(&self) -> Result<DirLock> {
        DirLock::acquire(&self.dir)
    }

    fn entry_path(&self, entry: &Entry) -> Result<PathBuf> {
        if !is_plain_file_name(&entry.file) {
            return Err(format!(
                "Checkpoint #{} has an unsafe file name '{}' in {INDEX_FILE}; refusing to use it.",
                entry.id, entry.file
            ));
        }
        Ok(self.dir.join(&entry.file))
    }

    // -- policy ---------------------------------------------------------------

    /// The saved policy, or the defaults when none was saved.
    pub fn policy(&self) -> Result<Policy> {
        let path = self.dir.join(POLICY_FILE);
        match fs::read(&path) {
            Ok(bytes) => {
                let policy: Policy = serde_json::from_slice(&bytes)
                    .map_err(|error| format!("{} is invalid: {error}", path.display()))?;
                policy.validate()?;
                Ok(policy)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Policy::default()),
            Err(error) => Err(format!("Failed to read {}: {error}", path.display())),
        }
    }

    pub fn set_policy(&self, policy: &Policy) -> Result<()> {
        policy.validate()?;
        let bytes = serde_json::to_vec_pretty(policy)
            .map_err(|error| format!("Failed to encode the policy: {error}"))?;
        atomic_write(&self.dir.join(POLICY_FILE), &bytes)
    }

    /// Back to the defaults (keep everything). Existing checkpoints are untouched.
    pub fn reset_policy(&self) -> Result<()> {
        match fs::remove_file(self.dir.join(POLICY_FILE)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(format!("Failed to remove the policy file: {error}")),
        }
    }

    // -- index ----------------------------------------------------------------

    fn read_index(&self) -> Result<Index> {
        let path = self.dir.join(INDEX_FILE);
        match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<Index>(&bytes).map_err(|error| {
                format!(
                    "{} is damaged ({error}). Nothing was changed; the previous copy may still be {INDEX_BACKUP} in the same folder.",
                    path.display()
                )
            }),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Index {
                version: 1,
                next_id: 1,
                ..Index::default()
            }),
            Err(error) => Err(format!("Failed to read {}: {error}", path.display())),
        }
    }

    fn write_index(&self, index: &Index) -> Result<()> {
        let path = self.dir.join(INDEX_FILE);
        if path.exists() {
            // Best effort: keep the last good index for manual recovery.
            let _ = fs::copy(&path, self.dir.join(INDEX_BACKUP));
        }
        let bytes = serde_json::to_vec_pretty(index)
            .map_err(|error| format!("Failed to encode the index: {error}"))?;
        atomic_write(&path, &bytes)
    }

    /// All checkpoints, oldest first.
    pub fn list(&self) -> Result<Vec<Entry>> {
        let mut entries = self.read_index()?.checkpoints;
        entries.sort_by_key(|entry| entry.id);
        Ok(entries)
    }

    /// Finds one checkpoint by id, `latest` or `best` (see [`resolve_entry`]).
    pub fn find(&self, spec: &str) -> Result<Entry> {
        let entries = self.list()?;
        resolve_entry(&entries, spec).cloned()
    }

    pub fn latest_path(&self) -> Result<Option<PathBuf>> {
        let entries = self.list()?;
        match entries.iter().max_by_key(|entry| entry.id) {
            Some(entry) => self.entry_path(entry).map(Some),
            None => Ok(None),
        }
    }

    pub fn best_path(&self) -> Result<Option<PathBuf>> {
        let entries = self.list()?;
        match entries.iter().filter(|entry| entry.best).max_by_key(|entry| entry.id) {
            Some(entry) => self.entry_path(entry).map(Some),
            None => Ok(None),
        }
    }

    // -- saving ---------------------------------------------------------------

    /// Stores `data` as a new checkpoint.
    pub fn save_bytes(&self, data: &[u8], meta: NewCheckpoint) -> Result<SaveOutcome> {
        self.save_reader(data, meta)
    }

    /// Stores the contents of the file at `source`, streaming it (no need to
    /// hold a large checkpoint in memory). With `move_source`, the source file
    /// is deleted once the checkpoint is safely stored.
    pub fn save_file(&self, source: &Path, meta: NewCheckpoint, move_source: bool) -> Result<SaveOutcome> {
        let file = File::open(source)
            .map_err(|error| format!("Cannot open {}: {error}", source.display()))?;
        let outcome = self.save_reader(file, meta)?;
        if move_source && outcome.entry.is_some() {
            if let Err(error) = fs::remove_file(source) {
                let mut outcome = outcome;
                outcome.warnings.push(format!(
                    "the checkpoint was stored but {} could not be deleted: {error}",
                    source.display()
                ));
                return Ok(outcome);
            }
        }
        Ok(outcome)
    }

    fn save_reader<R: Read>(&self, reader: R, meta: NewCheckpoint) -> Result<SaveOutcome> {
        let policy = self.policy()?;
        if policy.save_every > 1 && self.should_skip(&policy, meta.is_best)? {
            return Ok(SaveOutcome::skipped(SkipReason::Interval(policy.save_every), Vec::new()));
        }
        // The slow part (writing the data) happens without holding the lock.
        let staged = self.stage(reader, policy.verify)?;
        let result = self.commit(&staged, meta);
        // After a successful commit the temporary file has been renamed away.
        let _ = fs::remove_file(&staged.temporary);
        result
    }

    fn should_skip(&self, policy: &Policy, is_best: bool) -> Result<bool> {
        let _lock = self.lock()?;
        let mut index = self.read_index()?;
        index.calls += 1;
        let skip = !is_best && index.calls % policy.save_every != 0;
        self.write_index(&index)?;
        Ok(skip)
    }

    fn stage<R: Read>(&self, mut reader: R, verify: bool) -> Result<Staged> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0);
        let temporary = self
            .dir
            .join(format!("incoming.{}.{unique}{TMP_SUFFIX}", std::process::id()));

        let written = (|| -> io::Result<(u64, String)> {
            let mut file = File::create(&temporary)?;
            let mut hasher = Sha256::new();
            let mut buffer = vec![0u8; BUFFER_SIZE];
            let mut size = 0u64;
            loop {
                let read = reader.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
                file.write_all(&buffer[..read])?;
                size += read as u64;
            }
            file.sync_all()?;
            Ok((size, hex(hasher.finalize())))
        })();

        let (size, sha256) = match written {
            Ok(result) => result,
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                return Err(format!("Failed to write the checkpoint data (disk full?): {error}"));
            }
        };
        if size == 0 {
            let _ = fs::remove_file(&temporary);
            return Err("Refusing to store an empty checkpoint.".to_string());
        }
        if verify {
            match hash_file(&temporary) {
                Ok((read_size, read_sha)) if read_size == size && read_sha == sha256 => {}
                Ok(_) => {
                    let _ = fs::remove_file(&temporary);
                    return Err(
                        "Verification failed: the data read back from disk does not match what was written."
                            .to_string(),
                    );
                }
                Err(error) => {
                    let _ = fs::remove_file(&temporary);
                    return Err(format!("Verification failed: could not read the data back: {error}"));
                }
            }
        }
        Ok(Staged {
            temporary,
            size,
            sha256,
        })
    }

    fn commit(&self, staged: &Staged, meta: NewCheckpoint) -> Result<SaveOutcome> {
        let _lock = self.lock()?;
        let policy = self.policy()?;
        let mut index = self.read_index()?;
        let mut warnings = Vec::new();

        let id = next_id(&index);
        let file = format!("{}_{id:06}.{}", policy.name_prefix, policy.extension);
        let entry = Entry {
            id,
            file: file.clone(),
            step: meta.step,
            best: meta.is_best,
            pinned: false,
            size: staged.size,
            sha256: staged.sha256.clone(),
            created_unix: now_unix(),
            metrics: finite_metrics(meta.metrics, &mut warnings),
            experiment: meta.experiment,
        };

        let mut candidates = index.checkpoints.clone();
        candidates.push(entry.clone());
        let plan = plan_prune(&candidates, &policy);
        if plan.remove.contains(&id) {
            // The policy would delete it right away: do not store it at all.
            return Ok(SaveOutcome::skipped(SkipReason::Policy, warnings));
        }

        let final_path = self.dir.join(&file);
        if final_path.exists() {
            return Err(format!(
                "{} already exists; refusing to overwrite it.",
                final_path.display()
            ));
        }
        fs::rename(&staged.temporary, &final_path)
            .map_err(|error| format!("Failed to move the checkpoint into place: {error}"))?;

        let removed: Vec<Entry> = candidates
            .iter()
            .filter(|candidate| plan.remove.contains(&candidate.id))
            .cloned()
            .collect();
        index.version = 1;
        index.next_id = id + 1;
        index.checkpoints = candidates
            .into_iter()
            .filter(|candidate| !plan.remove.contains(&candidate.id))
            .collect();

        if let Err(error) = self.write_index(&index) {
            let _ = fs::remove_file(&final_path); // roll back
            return Err(error);
        }
        // The index is durable; now delete the files it no longer lists.
        for old in &removed {
            self.delete_file(old, &mut warnings);
        }
        if plan.still_over_cap {
            warnings.push(
                "the size cap (max_total_mb) is exceeded but everything left is protected (pinned, best or newest)"
                    .to_string(),
            );
        }

        Ok(SaveOutcome {
            entry: Some(entry),
            path: Some(final_path),
            removed,
            warnings,
            skipped: None,
        })
    }

    fn delete_file(&self, entry: &Entry, warnings: &mut Vec<String>) {
        match self.entry_path(entry) {
            Ok(path) => {
                if let Err(error) = fs::remove_file(&path) {
                    if error.kind() != io::ErrorKind::NotFound {
                        warnings.push(format!("could not delete {}: {error}", path.display()));
                    }
                }
            }
            Err(message) => warnings.push(message),
        }
    }

    // -- management -----------------------------------------------------------

    /// Pins or unpins a checkpoint. Pinned checkpoints are never pruned.
    pub fn set_pinned(&self, id: u64, pinned: bool) -> Result<Entry> {
        let _lock = self.lock()?;
        let mut index = self.read_index()?;
        let entry = index
            .checkpoints
            .iter_mut()
            .find(|entry| entry.id == id)
            .ok_or_else(|| format!("No checkpoint #{id}."))?;
        entry.pinned = pinned;
        let updated = entry.clone();
        self.write_index(&index)?;
        Ok(updated)
    }

    /// Deletes one checkpoint. A pinned one needs `force`.
    pub fn remove(&self, id: u64, force: bool) -> Result<(Entry, Vec<String>)> {
        let _lock = self.lock()?;
        let mut index = self.read_index()?;
        let position = index
            .checkpoints
            .iter()
            .position(|entry| entry.id == id)
            .ok_or_else(|| format!("No checkpoint #{id}."))?;
        if index.checkpoints[position].pinned && !force {
            return Err(format!("Checkpoint #{id} is pinned. Unpin it first."));
        }
        let entry = index.checkpoints.remove(position);
        self.write_index(&index)?;
        let mut warnings = Vec::new();
        self.delete_file(&entry, &mut warnings);
        Ok((entry, warnings))
    }

    /// Applies the retention policy to the existing checkpoints now.
    pub fn prune(&self, dry_run: bool) -> Result<PruneReport> {
        let _lock = self.lock()?;
        let policy = self.policy()?;
        let mut index = self.read_index()?;
        let plan = plan_prune(&index.checkpoints, &policy);
        let removed: Vec<Entry> = index
            .checkpoints
            .iter()
            .filter(|entry| plan.remove.contains(&entry.id))
            .cloned()
            .collect();

        let mut warnings = Vec::new();
        if !dry_run && !removed.is_empty() {
            index.checkpoints.retain(|entry| !plan.remove.contains(&entry.id));
            self.write_index(&index)?;
            for entry in &removed {
                self.delete_file(entry, &mut warnings);
            }
        }
        Ok(PruneReport {
            removed,
            still_over_cap: plan.still_over_cap,
            warnings,
            dry_run,
        })
    }

    /// Checks every checkpoint against its recorded size and SHA-256, and looks
    /// for files the index does not know. Changes nothing.
    pub fn verify(&self) -> Result<VerifyReport> {
        let entries = self.list()?;
        let mut problems = Vec::new();
        for entry in &entries {
            let path = match self.entry_path(entry) {
                Ok(path) => path,
                Err(message) => {
                    problems.push(message);
                    continue;
                }
            };
            match hash_file(&path) {
                Ok((size, _)) if size != entry.size => problems.push(format!(
                    "#{} {}: size is {size} bytes but {} were recorded",
                    entry.id, entry.file, entry.size
                )),
                Ok((_, sha256)) if sha256 != entry.sha256 => problems.push(format!(
                    "#{} {}: SHA-256 mismatch (the file was modified or damaged)",
                    entry.id, entry.file
                )),
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    problems.push(format!("#{} {}: file is missing", entry.id, entry.file));
                }
                Err(error) => {
                    problems.push(format!("#{} {}: cannot be read ({error})", entry.id, entry.file));
                }
            }
        }

        let known: HashSet<&str> = entries.iter().map(|entry| entry.file.as_str()).collect();
        let mut orphans = Vec::new();
        if let Ok(items) = fs::read_dir(&self.dir) {
            for item in items.flatten() {
                if !item.path().is_file() {
                    continue;
                }
                if let Some(name) = item.file_name().to_str() {
                    if !is_internal_file(name) && !known.contains(name) {
                        orphans.push(name.to_string());
                    }
                }
            }
        }
        orphans.sort();
        Ok(VerifyReport {
            checked: entries.len(),
            problems,
            orphans,
        })
    }

    /// Copies a checkpoint out of the store. If `destination` is an existing
    /// directory the original file name is used. Never overwrites, and refuses
    /// to export data that no longer matches its recorded SHA-256.
    pub fn export(&self, spec: &str, destination: &Path) -> Result<PathBuf> {
        let entry = self.find(spec)?;
        let source = self.entry_path(&entry)?;
        let target = if destination.is_dir() {
            destination.join(&entry.file)
        } else {
            destination.to_path_buf()
        };

        let (size, sha256) = hash_file(&source)
            .map_err(|error| format!("Cannot read {}: {error}", source.display()))?;
        if size != entry.size || sha256 != entry.sha256 {
            return Err(format!(
                "Checkpoint #{} does not match its recorded checksum; export aborted. Run /verify_checkpoints.",
                entry.id
            ));
        }

        let mut reader = File::open(&source)
            .map_err(|error| format!("Cannot read {}: {error}", source.display()))?;
        let mut writer = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)
            .map_err(|error| {
                if error.kind() == io::ErrorKind::AlreadyExists {
                    format!("{} already exists; refusing to overwrite it.", target.display())
                } else {
                    format!("Cannot create {}: {error}", target.display())
                }
            })?;
        let copied = io::copy(&mut reader, &mut writer).and_then(|_| writer.sync_all());
        if let Err(error) = copied {
            let _ = fs::remove_file(&target);
            return Err(format!("Failed to copy the checkpoint: {error}"));
        }
        Ok(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "tc-checkpoints-{tag}-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn meta(best: bool, step: u64) -> NewCheckpoint {
        NewCheckpoint {
            is_best: best,
            step: Some(step),
            metrics: BTreeMap::new(),
            experiment: Some(1),
        }
    }

    fn entry(id: u64, run: i64, step: u64, best: bool, size: u64) -> Entry {
        Entry {
            id,
            file: format!("checkpoint_{id:06}.ckpt"),
            step: Some(step),
            best,
            pinned: false,
            size,
            sha256: String::new(),
            created_unix: 0,
            metrics: BTreeMap::new(),
            experiment: Some(run),
        }
    }

    fn ids(store: &Store) -> Vec<u64> {
        store.list().unwrap().iter().map(|entry| entry.id).collect()
    }

    fn files_with_extension(directory: &Path, extension: &str) -> usize {
        fs::read_dir(directory)
            .unwrap()
            .flatten()
            .filter(|item| item.path().extension().is_some_and(|ext| ext == extension))
            .count()
    }

    // --- policy ------------------------------------------------------------

    #[test]
    fn policy_defaults_keep_everything_and_verify() {
        let policy = Policy::default();
        assert_eq!(policy.keep_last, None);
        assert!(policy.verify && policy.validate().is_ok());
        assert!(plan_prune(&[entry(1, 1, 1, false, 5), entry(2, 1, 2, false, 5)], &policy)
            .remove
            .is_empty());
    }

    #[test]
    fn policy_set_parses_every_key_and_rejects_bad_values() {
        let mut policy = Policy::default();
        policy.set("keep_last", "3").unwrap();
        policy.set("keep_best", "2").unwrap();
        policy.set("keep_first", "yes").unwrap();
        policy.set("keep_every", "500").unwrap();
        policy.set("max_total_mb", "2048").unwrap();
        policy.set("save_every", "4").unwrap();
        policy.set("verify", "off").unwrap();
        policy.set("name_prefix", "model").unwrap();
        policy.set("extension", ".pt").unwrap();
        assert_eq!(policy.keep_last, Some(3));
        assert_eq!(policy.keep_best, 2);
        assert!(policy.keep_first && !policy.verify);
        assert_eq!(policy.keep_every, Some(500));
        assert_eq!(policy.max_total_mb, Some(2048));
        assert_eq!(policy.save_every, 4);
        assert_eq!((policy.name_prefix.as_str(), policy.extension.as_str()), ("model", "pt"));

        policy.set("keep_last", "all").unwrap();
        policy.set("keep_every", "off").unwrap();
        policy.set("max_total_mb", "none").unwrap();
        assert_eq!((policy.keep_last, policy.keep_every, policy.max_total_mb), (None, None, None));
        policy.set("keep_last", "0").unwrap();
        assert_eq!(policy.keep_last, Some(0));

        assert!(policy.set("keep_last", "-1").is_err());
        assert!(policy.set("keep_every", "0").is_err());
        assert!(policy.set("save_every", "0").is_err());
        assert!(policy.set("max_total_mb", "0").is_err());
        assert!(policy.set("verify", "maybe").is_err());
        assert!(policy.set("name_prefix", "../evil").is_err());
        assert!(policy.set("name_prefix", "").is_err());
        assert!(policy.set("extension", "a/b").is_err());
        let error = policy.set("nope", "1").unwrap_err();
        assert!(error.contains("keep_last") && error.contains("extension"));
    }

    #[test]
    fn presets_replace_only_the_retention_rules() {
        let mut policy = Policy::default();
        policy.set("save_every", "5").unwrap();
        policy.set("max_total_mb", "100").unwrap();

        policy.apply_preset("last_and_best").unwrap();
        assert_eq!((policy.keep_last, policy.keep_best), (Some(2), 1));
        assert_eq!(policy.max_total_mb, None);
        assert_eq!(policy.save_every, 5);

        policy.apply_preset("best_only").unwrap();
        assert_eq!(policy.keep_last, Some(0));
        policy.apply_preset("milestones").unwrap();
        assert_eq!(policy.keep_every, Some(1000));
        policy.apply_preset("all").unwrap();
        assert_eq!((policy.keep_last, policy.keep_every), (None, None));
        assert!(policy.apply_preset("nope").unwrap_err().contains("last_and_best"));
        assert_eq!(policy.describe().len(), POLICY_KEYS.len());
    }

    // --- pruning plan --------------------------------------------------------

    #[test]
    fn plan_keeps_last_n_and_best_per_run() {
        let entries: Vec<Entry> = (1..=6).map(|id| entry(id, 1, id, id == 2 || id == 4, 10)).collect();
        let mut policy = Policy::default();
        policy.apply_preset("last_and_best").unwrap();
        // Newest two (5, 6) plus the newest best (4).
        assert_eq!(plan_prune(&entries, &policy).remove, vec![1, 2, 3]);

        policy.keep_best = 2;
        assert_eq!(plan_prune(&entries, &policy).remove, vec![1, 3]);
        policy.keep_best = 0;
        assert_eq!(plan_prune(&entries, &policy).remove, vec![1, 2, 3, 4]);
    }

    #[test]
    fn plan_best_only_keeps_just_the_best() {
        let entries = vec![
            entry(1, 1, 1, true, 10),
            entry(2, 1, 2, false, 10),
            entry(3, 1, 3, true, 10),
            entry(4, 1, 4, false, 10),
        ];
        let mut policy = Policy::default();
        policy.apply_preset("best_only").unwrap();
        assert_eq!(plan_prune(&entries, &policy).remove, vec![1, 2, 4]);
    }

    #[test]
    fn plan_first_milestones_and_pins_are_protected() {
        let mut entries: Vec<Entry> = (1..=6).map(|id| entry(id, 1, id * 100, false, 10)).collect();
        entries[2].pinned = true; // id 3
        let mut policy = Policy::default();
        policy.keep_last = Some(1);
        policy.keep_best = 0;
        policy.keep_first = true;
        policy.keep_every = Some(200);
        // Kept: 6 (newest), 1 (first), 2/4 (steps 200/400), 3 (pinned) -> only 5 goes.
        assert_eq!(plan_prune(&entries, &policy).remove, vec![5]);
    }

    #[test]
    fn plan_applies_retention_separately_for_each_run() {
        let entries = vec![
            entry(1, 1, 1, false, 10),
            entry(2, 1, 2, false, 10),
            entry(3, 2, 1, false, 10),
            entry(4, 2, 2, false, 10),
        ];
        let mut policy = Policy::default();
        policy.keep_last = Some(1);
        // Run 1 keeps #2 and run 2 keeps #4: a new run never deletes an older run's newest.
        assert_eq!(plan_prune(&entries, &policy).remove, vec![1, 3]);
    }

    #[test]
    fn plan_size_cap_removes_oldest_unprotected_across_runs() {
        let size = 10 * MIB;
        let entries: Vec<Entry> = (1..=5).map(|id| entry(id, if id <= 2 { 1 } else { 2 }, id, false, size)).collect();
        let mut policy = Policy::default();
        policy.max_total_mb = Some(25);
        let plan = plan_prune(&entries, &policy);
        assert_eq!(plan.remove, vec![1, 2, 3]);
        assert!(!plan.still_over_cap);

        // Pinned, best and the newest checkpoint cannot be removed by the cap.
        let mut entries = entries;
        entries[0].pinned = true;
        entries[1].best = true;
        let plan = plan_prune(&entries, &policy);
        assert_eq!(plan.remove, vec![3, 4]);
        assert!(plan.still_over_cap);
    }

    // --- store: saving and retention ----------------------------------------

    #[test]
    fn save_stores_data_index_and_verifies_checksums() {
        let dir = TempDir::new("save");
        let store = Store::open(dir.path()).unwrap();
        let mut metrics = BTreeMap::new();
        metrics.insert("loss".to_string(), 0.25);
        metrics.insert("bad".to_string(), f64::NAN);

        let outcome = store
            .save_bytes(
                b"hello checkpoint",
                NewCheckpoint { is_best: true, step: Some(7), metrics, experiment: Some(3) },
            )
            .unwrap();
        let entry = outcome.entry.clone().unwrap();
        assert_eq!((entry.id, entry.step, entry.best, entry.experiment), (1, Some(7), true, Some(3)));
        assert_eq!(entry.size, 16);
        assert_eq!(entry.metrics.len(), 1);
        assert!(outcome.warnings.iter().any(|w| w.contains("NaN")));
        assert_eq!(fs::read(outcome.path.as_ref().unwrap()).unwrap(), b"hello checkpoint");
        assert_eq!(entry.file, "checkpoint_000001.ckpt");

        let report = store.verify().unwrap();
        assert_eq!(report.checked, 1);
        assert!(report.problems.is_empty() && report.orphans.is_empty());
        assert_eq!(files_with_extension(dir.path(), "tmp"), 0);

        let lines = outcome.summary_lines();
        assert!(lines[0].starts_with("[checkpoint] saved #1 (step 7, best, 16 B)"));
    }

    #[test]
    fn save_file_streams_and_can_move_the_source() {
        let dir = TempDir::new("savefile");
        let source_dir = TempDir::new("savefile-src");
        let store = Store::open(dir.path()).unwrap();

        let source = source_dir.path().join("model.pt");
        fs::write(&source, vec![7u8; 3 * 1024 * 1024 + 5]).unwrap();
        let outcome = store.save_file(&source, meta(false, 1), false).unwrap();
        assert_eq!(outcome.entry.unwrap().size, 3 * 1024 * 1024 + 5);
        assert!(source.exists());

        store.save_file(&source, meta(false, 2), true).unwrap();
        assert!(!source.exists());
        assert!(store.save_file(&source, meta(false, 3), false).is_err());
        assert_eq!(ids(&store), vec![1, 2]);
    }

    #[test]
    fn empty_data_is_refused_and_leaves_nothing_behind() {
        let dir = TempDir::new("empty");
        let store = Store::open(dir.path()).unwrap();
        assert!(store.save_bytes(b"", meta(false, 1)).unwrap_err().contains("empty"));
        assert!(store.list().unwrap().is_empty());
        assert_eq!(files_with_extension(dir.path(), "tmp"), 0);
    }

    #[test]
    fn last_and_best_policy_prunes_files_and_index_together() {
        let dir = TempDir::new("retention");
        let store = Store::open(dir.path()).unwrap();
        let mut policy = Policy::default();
        policy.apply_preset("last_and_best").unwrap();
        store.set_policy(&policy).unwrap();

        let mut last = None;
        for step in 1..=5u64 {
            last = Some(store.save_bytes(format!("data{step}").as_bytes(), meta(step == 2, step)).unwrap());
        }
        assert_eq!(ids(&store), vec![2, 4, 5]);
        assert_eq!(files_with_extension(dir.path(), "ckpt"), 3);
        assert!(!dir.path().join("checkpoint_000001.ckpt").exists());
        assert!(dir.path().join("checkpoint_000002.ckpt").exists());
        assert_eq!(last.unwrap().removed.iter().map(|e| e.id).collect::<Vec<_>>(), vec![3]);
        assert!(store.verify().unwrap().problems.is_empty());
    }

    #[test]
    fn a_new_run_never_deletes_an_older_runs_checkpoints() {
        let dir = TempDir::new("runs");
        let store = Store::open(dir.path()).unwrap();
        let mut policy = Policy::default();
        policy.keep_last = Some(1);
        store.set_policy(&policy).unwrap();

        for step in 1..=3u64 {
            store.save_bytes(b"run one", NewCheckpoint { step: Some(step), experiment: Some(1), ..Default::default() }).unwrap();
        }
        for step in 1..=3u64 {
            store.save_bytes(b"run two", NewCheckpoint { step: Some(step), experiment: Some(2), ..Default::default() }).unwrap();
        }
        let entries = store.list().unwrap();
        assert_eq!(entries.iter().map(|e| (e.experiment, e.step)).collect::<Vec<_>>(), vec![(Some(1), Some(3)), (Some(2), Some(3))]);
    }

    #[test]
    fn best_only_policy_never_stores_a_non_best_checkpoint() {
        let dir = TempDir::new("bestonly");
        let store = Store::open(dir.path()).unwrap();
        let mut policy = Policy::default();
        policy.apply_preset("best_only").unwrap();
        store.set_policy(&policy).unwrap();

        assert!(store.save_bytes(b"one", meta(true, 1)).unwrap().entry.is_some());
        let skipped = store.save_bytes(b"two", meta(false, 2)).unwrap();
        assert!(skipped.entry.is_none() && skipped.skipped == Some(SkipReason::Policy));
        assert!(skipped.summary_lines()[0].contains("not stored"));
        assert_eq!(files_with_extension(dir.path(), "ckpt"), 1);
        assert_eq!(files_with_extension(dir.path(), "tmp"), 0);

        // A new best replaces the old one; ids keep counting without gaps from skipped saves.
        let third = store.save_bytes(b"three", meta(true, 3)).unwrap();
        assert_eq!(third.entry.unwrap().id, 2);
        assert_eq!(third.removed.len(), 1);
        assert_eq!(ids(&store), vec![2]);
        assert_eq!(files_with_extension(dir.path(), "ckpt"), 1);
    }

    #[test]
    fn save_every_throttles_but_always_stores_best() {
        let dir = TempDir::new("every");
        let store = Store::open(dir.path()).unwrap();
        let mut policy = Policy::default();
        policy.set("save_every", "3").unwrap();
        store.set_policy(&policy).unwrap();

        let mut skipped = 0;
        for step in 1..=6u64 {
            let outcome = store.save_bytes(b"data", meta(false, step)).unwrap();
            if outcome.skipped == Some(SkipReason::Interval(3)) {
                skipped += 1;
            }
        }
        assert_eq!((skipped, ids(&store).len()), (4, 2)); // calls 3 and 6 stored

        assert!(store.save_bytes(b"best", meta(true, 7)).unwrap().entry.is_some());
        assert_eq!(ids(&store).len(), 3);
    }

    #[test]
    fn size_cap_prunes_oldest_during_save() {
        let dir = TempDir::new("cap");
        let store = Store::open(dir.path()).unwrap();
        let mut policy = Policy::default();
        policy.set("max_total_mb", "1").unwrap();
        store.set_policy(&policy).unwrap();

        let chunk = vec![1u8; 600 * 1024];
        for step in 1..=3u64 {
            store.save_bytes(&chunk, meta(false, step)).unwrap();
        }
        // 600 KiB each, cap 1 MiB: only the newest fits.
        assert_eq!(ids(&store), vec![3]);
    }

    // --- store: management -----------------------------------------------------

    #[test]
    fn pinned_checkpoints_survive_pruning_and_need_force_to_remove() {
        let dir = TempDir::new("pin");
        let store = Store::open(dir.path()).unwrap();
        let mut policy = Policy::default();
        policy.keep_last = Some(1);
        store.set_policy(&policy).unwrap();

        store.save_bytes(b"one", meta(false, 1)).unwrap();
        store.set_pinned(1, true).unwrap();
        store.save_bytes(b"two", meta(false, 2)).unwrap();
        store.save_bytes(b"three", meta(false, 3)).unwrap();
        assert_eq!(ids(&store), vec![1, 3]);

        assert!(store.remove(1, false).unwrap_err().contains("pinned"));
        let (removed, warnings) = store.remove(1, true).unwrap();
        assert_eq!(removed.id, 1);
        assert!(warnings.is_empty());
        assert_eq!(ids(&store), vec![3]);
        assert!(!dir.path().join("checkpoint_000001.ckpt").exists());
        assert!(store.remove(1, true).is_err());
        assert!(store.set_pinned(99, true).is_err());
    }

    #[test]
    fn prune_applies_a_new_policy_to_existing_checkpoints_with_dry_run() {
        let dir = TempDir::new("prune");
        let store = Store::open(dir.path()).unwrap();
        for step in 1..=4u64 {
            store.save_bytes(b"data", meta(false, step)).unwrap();
        }
        assert_eq!(ids(&store).len(), 4); // default policy keeps everything

        let mut policy = Policy::default();
        policy.keep_last = Some(1);
        store.set_policy(&policy).unwrap();

        let dry = store.prune(true).unwrap();
        assert_eq!(dry.removed.len(), 3);
        assert_eq!(ids(&store).len(), 4);

        let real = store.prune(false).unwrap();
        assert_eq!(real.removed.len(), 3);
        assert_eq!(ids(&store), vec![4]);
        assert_eq!(files_with_extension(dir.path(), "ckpt"), 1);
    }

    #[test]
    fn latest_best_and_resolve_find_checkpoints() {
        let dir = TempDir::new("resolve");
        let store = Store::open(dir.path()).unwrap();
        assert_eq!(store.latest_path().unwrap(), None);
        assert!(store.find("latest").is_err());

        store.save_bytes(b"a", meta(true, 1)).unwrap();
        store.save_bytes(b"b", meta(false, 2)).unwrap();
        assert!(store.latest_path().unwrap().unwrap().ends_with("checkpoint_000002.ckpt"));
        assert!(store.best_path().unwrap().unwrap().ends_with("checkpoint_000001.ckpt"));
        assert_eq!(store.find("#2").unwrap().id, 2);
        assert_eq!(store.find("BEST").unwrap().id, 1);
        assert!(store.find("9").unwrap_err().contains("No checkpoint"));
        assert!(store.find("abc").unwrap_err().contains("not a checkpoint id"));
    }

    #[test]
    fn verify_reports_tampering_missing_files_and_orphans() {
        let dir = TempDir::new("verify");
        let store = Store::open(dir.path()).unwrap();
        store.save_bytes(b"first!", meta(false, 1)).unwrap();
        store.save_bytes(b"second", meta(false, 2)).unwrap();
        store.save_bytes(b"third!", meta(false, 3)).unwrap();

        fs::write(dir.path().join("checkpoint_000001.ckpt"), b"FIRST!").unwrap(); // same size
        fs::remove_file(dir.path().join("checkpoint_000002.ckpt")).unwrap();
        fs::write(dir.path().join("checkpoint_000003.ckpt"), b"x").unwrap(); // different size
        fs::write(dir.path().join("stray.bin"), b"?").unwrap();

        let report = store.verify().unwrap();
        assert_eq!(report.checked, 3);
        assert_eq!(report.problems.len(), 3);
        assert!(report.problems[0].contains("SHA-256 mismatch"));
        assert!(report.problems[1].contains("missing"));
        assert!(report.problems[2].contains("size"));
        assert_eq!(report.orphans, vec!["stray.bin"]);
    }

    #[test]
    fn export_copies_without_overwriting_and_refuses_damaged_data() {
        let dir = TempDir::new("export");
        let out = TempDir::new("export-out");
        let store = Store::open(dir.path()).unwrap();
        store.save_bytes(b"model weights", meta(true, 1)).unwrap();

        let copied = store.export("best", out.path()).unwrap();
        assert_eq!(copied, out.path().join("checkpoint_000001.ckpt"));
        assert_eq!(fs::read(&copied).unwrap(), b"model weights");
        assert!(store.export("1", out.path()).unwrap_err().contains("already exists"));
        let named = out.path().join("final.pt");
        store.export("latest", &named).unwrap();
        assert!(named.exists());

        fs::write(dir.path().join("checkpoint_000001.ckpt"), b"model WEIGHTS").unwrap();
        let error = store.export("1", &out.path().join("other.pt")).unwrap_err();
        assert!(error.contains("checksum"));
        assert!(!out.path().join("other.pt").exists());
    }

    // --- store: safety ---------------------------------------------------------

    #[test]
    fn a_damaged_index_is_reported_and_never_overwritten() {
        let dir = TempDir::new("damaged");
        let store = Store::open(dir.path()).unwrap();
        fs::write(dir.path().join(INDEX_FILE), b"{ this is not json").unwrap();

        let error = store.save_bytes(b"data", meta(false, 1)).unwrap_err();
        assert!(error.contains("damaged") && error.contains("Nothing was changed"));
        assert_eq!(fs::read(dir.path().join(INDEX_FILE)).unwrap(), b"{ this is not json");
        assert_eq!(files_with_extension(dir.path(), "ckpt"), 0);
        assert_eq!(files_with_extension(dir.path(), "tmp"), 0);
        assert!(store.list().is_err());
    }

    #[test]
    fn index_file_names_that_escape_the_store_are_never_used() {
        let dir = TempDir::new("escape");
        let store = Store::open(dir.path()).unwrap();
        let victim = dir.path().parent().unwrap().join(format!(
            "tc-victim-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&victim, b"do not touch").unwrap();
        let relative = format!("../{}", victim.file_name().unwrap().to_string_lossy());

        let index = serde_json::json!({
            "version": 1, "next_id": 2, "calls": 0,
            "checkpoints": [{"id": 1, "file": relative, "size": 12, "sha256": "00", "created_unix": 0}]
        });
        fs::write(dir.path().join(INDEX_FILE), index.to_string()).unwrap();

        assert!(store.verify().unwrap().problems[0].contains("unsafe file name"));
        assert!(store.export("1", dir.path()).unwrap_err().contains("unsafe file name"));
        let (_, warnings) = store.remove(1, true).unwrap();
        assert!(warnings[0].contains("unsafe file name"));
        assert_eq!(fs::read(&victim).unwrap(), b"do not touch");
        fs::remove_file(victim).unwrap();
    }

    #[test]
    fn a_stale_lock_is_replaced_and_a_fresh_one_blocks() {
        let dir = TempDir::new("lock");
        let store = Store::open(dir.path()).unwrap();

        let lock = dir.path().join(LOCK_FILE);
        fs::write(&lock, b"999999").unwrap();
        let long_ago = SystemTime::now() - Duration::from_secs(600);
        OpenOptions::new().write(true).open(&lock).unwrap().set_modified(long_ago).unwrap();
        assert!(store.save_bytes(b"data", meta(false, 1)).is_ok());
        assert!(!lock.exists(), "the lock is released after saving");

        let held = DirLock::acquire(dir.path()).unwrap();
        assert!(lock.exists());
        drop(held);
        assert!(!lock.exists());
    }

    #[test]
    fn stale_temporary_files_are_cleaned_on_open() {
        let dir = TempDir::new("tmpclean");
        let old = dir.path().join("incoming.1.2.tmp");
        let fresh = dir.path().join("incoming.3.4.tmp");
        fs::write(&old, b"x").unwrap();
        fs::write(&fresh, b"x").unwrap();
        let long_ago = SystemTime::now() - Duration::from_secs(2 * 3600);
        OpenOptions::new().write(true).open(&old).unwrap().set_modified(long_ago).unwrap();

        Store::open(dir.path()).unwrap();
        assert!(!old.exists());
        assert!(fresh.exists(), "a recent temporary file may belong to a running save");
    }

    #[test]
    fn policy_persists_and_resets() {
        let dir = TempDir::new("policyfile");
        let store = Store::open(dir.path()).unwrap();
        assert_eq!(store.policy().unwrap(), Policy::default());

        let mut policy = Policy::default();
        policy.apply_preset("best_only").unwrap();
        store.set_policy(&policy).unwrap();
        assert_eq!(store.policy().unwrap().keep_last, Some(0));

        store.reset_policy().unwrap();
        assert_eq!(store.policy().unwrap(), Policy::default());
        store.reset_policy().unwrap(); // already reset: fine

        fs::write(dir.path().join(POLICY_FILE), br#"{"name_prefix": "../x"}"#).unwrap();
        assert!(store.policy().is_err());
        assert!(store.save_bytes(b"data", meta(false, 1)).is_err());
    }

    // --- small helpers -----------------------------------------------------------

    #[test]
    fn helpers_format_and_sanitize() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(5 * MIB), "5.0 MB");
        assert_eq!(dir_name_for("my script/v2"), "my_script_v2");
        assert_eq!(dir_name_for("..hidden"), "hidden");
        assert_eq!(dir_name_for("..."), "script");
        assert_eq!(dir_name_for(""), "script");
        assert!(is_plain_file_name("a.ckpt") && !is_plain_file_name("../a") && !is_plain_file_name("a\\b"));
    }

    #[test]
    fn resolve_entry_handles_ids_and_keywords() {
        let entries = vec![entry(1, 1, 1, true, 1), entry(5, 1, 2, false, 1), entry(9, 1, 3, true, 1)];
        assert_eq!(resolve_entry(&entries, "latest").unwrap().id, 9);
        assert_eq!(resolve_entry(&entries, "best").unwrap().id, 9);
        assert_eq!(resolve_entry(&entries, "#5").unwrap().id, 5);
        assert!(resolve_entry(&entries, "2").is_err());
        assert!(resolve_entry(&[], "latest").is_err());
    }
}
