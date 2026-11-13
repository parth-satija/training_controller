//! REPL presentation of checkpoint storage: turns `Store` operations into the
//! text the `/checkpoints`, `/set_checkpoint_config`, ... commands print.
//!
//! Every function returns `Ok(text)` or `Err(message)`; the command handlers in
//! `main.rs` print either one. All logic lives in `training_controller::checkpoint`.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use training_controller::checkpoint::{format_size, Entry, Store, PRESETS};

type Reply = Result<String, String>;

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// `95` -> `1m ago`.
pub fn format_age(seconds: u64) -> String {
    match seconds {
        0..=59 => format!("{seconds}s ago"),
        60..=3599 => format!("{}m ago", seconds / 60),
        3600..=86399 => format!("{}h ago", seconds / 3600),
        _ => format!("{}d ago", seconds / 86400),
    }
}

// -- settings ------------------------------------------------------------------

pub fn show_config(script: &str, store: &Store) -> Reply {
    let policy = store.policy()?;
    let rows = policy.describe();
    let key_width = rows.iter().map(|(key, _, _)| key.len()).max().unwrap_or(0);
    let value_width = rows
        .iter()
        .map(|(_, value, _)| value.chars().count())
        .max()
        .unwrap_or(0);

    let mut out = format!(
        "Checkpoint settings for '{script}':\n  folder: {}\n",
        store.dir().display()
    );
    for (key, value, help) in &rows {
        out.push_str(&format!("  {key:<key_width$}  {value:<value_width$}  {help}\n"));
    }
    out.push_str("Retention rules apply per run; a checkpoint is kept if any rule keeps it.\n");
    out.push_str("Change one with /set_checkpoint_config KEY VALUE, or pick a /checkpoint_preset.");
    Ok(out)
}

pub fn set_config(script: &str, store: &Store, key: &str, value: &str) -> Reply {
    let key = key.trim();
    let mut policy = store.policy()?;
    policy.set(key, value)?;
    store.set_policy(&policy)?;

    let shown = policy
        .describe()
        .into_iter()
        .find(|(name, _, _)| *name == key)
        .map(|(_, value, _)| value)
        .unwrap_or_default();
    Ok(format!(
        "Set {key} = {shown} for '{script}'.\n\
         It applies from the next save; /prune_checkpoints applies it to existing checkpoints now."
    ))
}

/// With no name, lists the presets.
pub fn apply_preset(script: &str, store: &Store, name: Option<&str>) -> Reply {
    let Some(name) = name else {
        let width = PRESETS.iter().map(|(preset, _)| preset.len()).max().unwrap_or(0);
        let mut out = String::from("Checkpoint presets:\n");
        for (preset, description) in PRESETS {
            out.push_str(&format!("  {preset:<width$}  {description}\n"));
        }
        out.push_str("Apply one with /checkpoint_preset NAME, then fine-tune with /set_checkpoint_config.");
        return Ok(out);
    };

    let mut policy = store.policy()?;
    policy.apply_preset(name)?;
    store.set_policy(&policy)?;
    let description = PRESETS
        .iter()
        .find(|(preset, _)| *preset == name)
        .map_or("", |(_, description)| *description);
    Ok(format!(
        "Applied preset '{name}' to '{script}': {description}.\n\
         It applies from the next save; /prune_checkpoints applies it to existing checkpoints now."
    ))
}

pub fn reset_config(script: &str, store: &Store) -> Reply {
    store.reset_policy()?;
    Ok(format!(
        "Checkpoint settings of '{script}' reset to the defaults (keep everything). Existing checkpoints were not touched."
    ))
}

// -- listing -------------------------------------------------------------------

fn flags(entry: &Entry) -> String {
    match (entry.best, entry.pinned) {
        (true, true) => "best,pin".to_string(),
        (true, false) => "best".to_string(),
        (false, true) => "pin".to_string(),
        (false, false) => "-".to_string(),
    }
}

fn metrics_text(entry: &Entry) -> String {
    entry
        .metrics
        .iter()
        .take(3)
        .map(|(name, value)| format!("{name}={value:.4}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_row(entry: &Entry, now: u64) -> String {
    let experiment = entry.experiment.map_or("-".to_string(), |id| id.to_string());
    let step = entry.step.map_or("-".to_string(), |step| step.to_string());
    format!(
        "{:>4}  {:>4}  {:>8}  {:<8}  {:>9}  {:<8}  {}",
        entry.id,
        experiment,
        step,
        flags(entry),
        format_size(entry.size),
        format_age(now.saturating_sub(entry.created_unix)),
        metrics_text(entry)
    )
}

/// `/checkpoints [EXPERIMENT_ID]`
pub fn list(script: &str, store: &Store, experiment: Option<&str>) -> Reply {
    let mut entries = store.list()?;
    if let Some(text) = experiment {
        let id: i64 = text
            .trim()
            .trim_start_matches('#')
            .parse()
            .map_err(|_| format!("'{text}' is not an experiment id."))?;
        entries.retain(|entry| entry.experiment == Some(id));
    }
    if entries.is_empty() {
        return Ok(format!(
            "No checkpoints stored for '{script}'. A script stores them with tc.save_checkpoint(data, is_best)."
        ));
    }

    let now = now_unix();
    let mut out = format!(
        "Checkpoints for '{script}' ({}):\n",
        store.dir().display()
    );
    out.push_str(&format!(
        "{:>4}  {:>4}  {:>8}  {:<8}  {:>9}  {:<8}  {}\n",
        "ID", "EXP", "STEP", "FLAGS", "SIZE", "AGE", "METRICS"
    ));
    for entry in &entries {
        out.push_str(&format_row(entry, now));
        out.push('\n');
    }
    let total: u64 = entries.iter().map(|entry| entry.size).sum();
    out.push_str(&format!(
        "{} checkpoint{}, {} in total. Use an ID, 'latest' or 'best' with /pin_checkpoint, /remove_checkpoint and /export_checkpoint.",
        entries.len(),
        if entries.len() == 1 { "" } else { "s" },
        format_size(total)
    ));
    Ok(out)
}

// -- management ------------------------------------------------------------------

pub fn pin(store: &Store, spec: &str, pinned: bool) -> Reply {
    let entry = store.find(spec)?;
    store.set_pinned(entry.id, pinned)?;
    Ok(if pinned {
        format!("Pinned checkpoint #{}: pruning will never delete it.", entry.id)
    } else {
        format!("Unpinned checkpoint #{}: the retention rules apply to it again.", entry.id)
    })
}

pub fn confirm_prompt(entry: &Entry) -> String {
    format!(
        "Delete checkpoint #{} ({}, {})? This cannot be undone. (y/N): ",
        entry.id,
        entry.file,
        format_size(entry.size)
    )
}

pub fn removed_message(entry: &Entry, warnings: &[String]) -> String {
    let mut out = format!("Deleted checkpoint #{} ({}).", entry.id, entry.file);
    for warning in warnings {
        out.push_str(&format!("\nWarning: {warning}"));
    }
    out
}

/// `/prune_checkpoints [dry]`
pub fn prune(script: &str, store: &Store, mode: Option<&str>) -> Reply {
    let dry_run = match mode {
        None => false,
        Some("dry") | Some("dry-run") => true,
        Some(other) => {
            return Err(format!(
                "Unknown option '{other}'. Use /prune_checkpoints, or /prune_checkpoints dry to preview."
            ));
        }
    };
    let report = store.prune(dry_run)?;

    let mut out = if report.removed.is_empty() {
        format!("Nothing to prune for '{script}' with the current settings.")
    } else {
        let verb = if dry_run { "Would remove" } else { "Removed" };
        let mut text = format!(
            "{verb} {} checkpoint{}{}:\n",
            report.removed.len(),
            if report.removed.len() == 1 { "" } else { "s" },
            if dry_run { " (dry run, nothing was deleted)" } else { "" }
        );
        for entry in &report.removed {
            let step = entry.step.map_or(String::new(), |step| format!(" step {step}"));
            text.push_str(&format!("  #{}{step} ({})\n", entry.id, format_size(entry.size)));
        }
        text.trim_end().to_string()
    };
    if report.still_over_cap {
        out.push_str("\nWarning: the size cap (max_total_mb) is still exceeded; the remaining checkpoints are protected (pinned, best or newest).");
    }
    for warning in &report.warnings {
        out.push_str(&format!("\nWarning: {warning}"));
    }
    Ok(out)
}

/// `/verify_checkpoints`
pub fn verify(script: &str, store: &Store) -> Reply {
    let report = store.verify()?;
    let mut out = if report.problems.is_empty() {
        format!(
            "All {} checkpoint{} of '{script}' match their recorded size and SHA-256.",
            report.checked,
            if report.checked == 1 { "" } else { "s" }
        )
    } else {
        let mut text = format!(
            "Checked {} checkpoints of '{script}': {} problem{}:\n",
            report.checked,
            report.problems.len(),
            if report.problems.len() == 1 { "" } else { "s" }
        );
        for problem in &report.problems {
            text.push_str(&format!("  {problem}\n"));
        }
        text.trim_end().to_string()
    };
    if !report.orphans.is_empty() {
        out.push_str(&format!(
            "\nFiles in {} that are not in the index (not deleted): {}",
            store.dir().display(),
            report.orphans.join(", ")
        ));
    }
    Ok(out)
}

/// `/export_checkpoint ID DEST`
pub fn export(store: &Store, spec: &str, destination: &str) -> Reply {
    let entry = store.find(spec)?;
    let target = store.export(spec, Path::new(destination))?;
    Ok(format!("Exported checkpoint #{} to {}", entry.id, target.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use training_controller::checkpoint::NewCheckpoint;

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "tc-checkpoint-cli-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn save(store: &Store, step: u64, best: bool) {
        let mut metrics = std::collections::BTreeMap::new();
        metrics.insert("loss".to_string(), 1.0 / step as f64);
        store
            .save_bytes(
                format!("checkpoint data {step}").as_bytes(),
                NewCheckpoint { is_best: best, step: Some(step), metrics, experiment: Some(4) },
            )
            .unwrap();
    }

    #[test]
    fn age_is_formatted_in_the_largest_unit() {
        assert_eq!(format_age(5), "5s ago");
        assert_eq!(format_age(125), "2m ago");
        assert_eq!(format_age(7300), "2h ago");
        assert_eq!(format_age(200_000), "2d ago");
    }

    #[test]
    fn config_can_be_shown_changed_preset_and_reset() {
        let dir = TempDir::new();
        let store = Store::open(&dir.0).unwrap();

        let shown = show_config("demo", &store).unwrap();
        for key in ["keep_last", "keep_best", "max_total_mb", "save_every", "verify"] {
            assert!(shown.contains(key), "missing {key}");
        }
        assert!(shown.contains("all"));

        let changed = set_config("demo", &store, "keep_last", "2").unwrap();
        assert!(changed.contains("keep_last = 2"));
        assert!(show_config("demo", &store).unwrap().contains("keep_last     2"));
        assert!(set_config("demo", &store, "keep_last", "many").is_err());
        assert!(set_config("demo", &store, "bogus", "1").unwrap_err().contains("Valid settings"));

        let presets = apply_preset("demo", &store, None).unwrap();
        assert!(presets.contains("best_only") && presets.contains("milestones"));
        assert!(apply_preset("demo", &store, Some("best_only")).unwrap().contains("Applied preset"));
        assert_eq!(store.policy().unwrap().keep_last, Some(0));
        assert!(apply_preset("demo", &store, Some("nope")).is_err());

        assert!(reset_config("demo", &store).unwrap().contains("reset"));
        assert_eq!(store.policy().unwrap().keep_last, None);
    }

    #[test]
    fn list_shows_columns_flags_metrics_and_filters_by_run() {
        let dir = TempDir::new();
        let store = Store::open(&dir.0).unwrap();
        assert!(list("demo", &store, None).unwrap().contains("No checkpoints stored"));

        save(&store, 100, false);
        save(&store, 200, true);
        pin(&store, "latest", true).unwrap();

        let text = list("demo", &store, None).unwrap();
        assert!(text.contains("ID") && text.contains("METRICS"));
        assert!(text.contains("best,pin"));
        assert!(text.contains("loss=0.0100") && text.contains("loss=0.0050"));
        assert!(text.contains("2 checkpoints"));

        assert!(list("demo", &store, Some("4")).unwrap().contains("2 checkpoints"));
        assert!(list("demo", &store, Some("#9")).unwrap().contains("No checkpoints"));
        assert!(list("demo", &store, Some("x")).is_err());
    }

    #[test]
    fn pin_prune_verify_and_export_report_clearly() {
        let dir = TempDir::new();
        let out = TempDir::new();
        let store = Store::open(&dir.0).unwrap();
        for step in 1..=3 {
            save(&store, step, false);
        }

        assert!(pin(&store, "1", true).unwrap().contains("Pinned checkpoint #1"));
        assert!(pin(&store, "1", false).unwrap().contains("Unpinned"));
        assert!(pin(&store, "99", true).is_err());

        assert!(prune("demo", &store, None).unwrap().contains("Nothing to prune"));
        set_config("demo", &store, "keep_last", "1").unwrap();
        let preview = prune("demo", &store, Some("dry")).unwrap();
        assert!(preview.contains("Would remove 2 checkpoints") && preview.contains("nothing was deleted"));
        assert_eq!(store.list().unwrap().len(), 3);
        assert!(prune("demo", &store, Some("later")).is_err());
        let done = prune("demo", &store, None).unwrap();
        assert!(done.contains("Removed 2 checkpoints") && done.contains("#1 step 1"));
        assert_eq!(store.list().unwrap().len(), 1);

        assert!(verify("demo", &store).unwrap().contains("All 1 checkpoint of 'demo' match"));
        fs::write(dir.0.join("extra.bin"), b"?").unwrap();
        fs::write(dir.0.join("checkpoint_000003.ckpt"), b"tampered!!").unwrap();
        let report = verify("demo", &store).unwrap();
        assert!(report.contains("1 problem") && report.contains("extra.bin"));

        fs::write(dir.0.join("checkpoint_000003.ckpt"), b"checkpoint data 3").unwrap();
        let exported = export(&store, "latest", out.0.to_str().unwrap()).unwrap();
        assert!(exported.contains("Exported checkpoint #3"));
        assert!(export(&store, "latest", out.0.to_str().unwrap()).unwrap_err().contains("already exists"));
    }

    #[test]
    fn removal_messages_include_warnings() {
        let entry = Entry {
            id: 3,
            file: "checkpoint_000003.ckpt".into(),
            step: Some(30),
            best: false,
            pinned: false,
            size: 2048,
            sha256: String::new(),
            created_unix: 0,
            metrics: Default::default(),
            experiment: None,
        };
        assert!(confirm_prompt(&entry).contains("#3") && confirm_prompt(&entry).contains("2.0 KB"));
        assert_eq!(removed_message(&entry, &[]), "Deleted checkpoint #3 (checkpoint_000003.ckpt).");
        assert!(removed_message(&entry, &["disk busy".to_string()]).contains("Warning: disk busy"));
    }
}
