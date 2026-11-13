//! Unit tests for the training dashboard view.

use super::*;
use crate::dashboard::DashboardFile;

const GLYPHS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

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

fn settings_from(value: serde_json::Value) -> Settings {
    let file: DashboardFile = serde_json::from_value(value).unwrap();
    Settings::compile(&file).unwrap()
}

fn model_with(widgets: &[(&str, WidgetKind)]) -> TrainingModel {
    let mut model =
        TrainingModel::new("test".into(), "events.jsonl".into(), Settings::default(), None);
    model.apply_event(TrainingEvent::Init {
        t: 1_000,
        widgets: widgets
            .iter()
            .map(|(name, kind)| WidgetSpec {
                name: name.to_string(),
                kind: *kind,
            })
            .collect(),
    });
    model
}

fn demo_model() -> TrainingModel {
    let mut model =
        TrainingModel::new("demo".into(), "demo data".into(), Settings::default(), None);
    let mut demo = TrainingDemo::new();
    for _ in 0..30 {
        model.apply(demo.poll());
    }
    model
}

fn text_of(model: &TrainingModel, width: usize, height: usize, now: u64) -> String {
    render_training_frame(model, width, height, now)
        .iter()
        .map(|line| strip_ansi(line))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn braille_curve_has_exact_size_and_traces_a_diagonal() {
    let rows = braille_curve(&[0.0, 1.0], 4, 2);
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|row| row.chars().count() == 4));
    // Rising line: the bottom-left cell and the top-right cell are used.
    assert_ne!(rows[1].chars().next().unwrap(), ' ');
    assert_ne!(rows[0].chars().last().unwrap(), ' ');
    assert_eq!(rows[0].chars().next().unwrap(), ' ');

    assert!(braille_curve(&[], 4, 2).iter().all(|row| row.trim().is_empty()));
    assert!(braille_curve(&[1.0], 0, 2).is_empty());
    // Flat and single-point input must not panic or divide by zero.
    assert!(braille_curve(&[2.0, 2.0, 2.0], 5, 3).iter().any(|row| !row.trim().is_empty()));
    assert!(braille_curve(&[2.0], 5, 3).iter().any(|row| !row.trim().is_empty()));
    // More points than dots.
    let many: Vec<f64> = (0..1000).map(|i| (i as f64 / 50.0).sin()).collect();
    assert!(braille_curve(&many, 10, 3).iter().all(|row| row.chars().count() == 10));
}

#[test]
fn bar_chart_scales_heights_and_keeps_the_latest_values() {
    let rows = bar_chart(&[0.0, 0.5, 1.0], &GLYPHS, 3, 2);
    assert_eq!(rows.len(), 2);
    // The tallest bar fills both rows; the lowest is a single small step.
    assert_eq!(rows[0].chars().nth(2).unwrap(), '█');
    assert_eq!(rows[1].chars().nth(2).unwrap(), '█');
    assert_eq!(rows[1].chars().next().unwrap(), '▁');
    assert_eq!(rows[0].chars().next().unwrap(), ' ');

    // Only the last `cols` values are drawn; unused columns are blank.
    let tail = bar_chart(&[9.0, 9.0, 0.0, 1.0], &GLYPHS, 2, 1);
    assert_eq!(tail[0], "▁█");
    assert_eq!(bar_chart(&[1.0], &GLYPHS, 3, 1)[0].chars().count(), 3);
    assert!(bar_chart(&[1.0], &[], 3, 1).is_empty());
}

#[test]
fn grid_planning_balances_rows_and_splits_widths() {
    assert_eq!(plan_columns(1, 100), 1);
    assert_eq!(plan_columns(4, 80), 2); // 3 fit, but 2 + 2 beats 3 + 1
    assert_eq!(plan_columns(6, 120), 3);
    assert_eq!(plan_columns(5, 30), 1); // too narrow for two panels
    assert_eq!(plan_columns(3, 200), 3);

    let widths = split_widths(80, 3);
    assert_eq!(widths.iter().sum::<usize>() + 2 * GAP, 80);
    assert!(widths.iter().max().unwrap() - widths.iter().min().unwrap() <= 1);
    assert_eq!(split_widths(30, 1), vec![30]);
}

#[test]
fn centering_pads_and_clips() {
    assert_eq!(center("ab", 6), "  ab  ");
    assert_eq!(center("abc", 6), " abc  ");
    assert_eq!(center("abcdef", 4), "abc…");
}

#[test]
fn events_build_the_model_and_a_new_init_resets_it() {
    let mut model = model_with(&[("loss", WidgetKind::Curve), ("lr", WidgetKind::Block)]);
    assert_eq!(model.phase(1_000), Phase::Running);
    model.apply_event(TrainingEvent::Update { t: 2_000, values: vec![Some(0.5), Some(0.1)] });
    model.apply_event(TrainingEvent::Update { t: 3_000, values: vec![None, Some(0.2)] });
    model.apply_event(TrainingEvent::Hero { t: 3_500, text: "hello".into() });

    assert_eq!(model.updates(), 2);
    assert_eq!(model.hero, "hello");
    assert_eq!(model.widgets[0].series.to_vec(), vec![0.5]);
    assert!(model.widgets[0].invalid);
    assert_eq!(model.widgets[1].series.latest(), Some(0.2));

    // Wrong number of values is reported and ignored.
    model.apply_event(TrainingEvent::Update { t: 4_000, values: vec![Some(1.0)] });
    assert_eq!(model.updates(), 2);
    assert!(model.data_warning.as_deref().unwrap().contains("1 values"));

    model.apply_event(TrainingEvent::Init { t: 9_000, widgets: vec![] });
    assert!(model.hero.is_empty());
    assert_eq!(model.updates(), 0);
}

#[test]
fn updates_before_init_and_garbage_lines_are_ignored() {
    let mut model =
        TrainingModel::new("t".into(), "f".into(), Settings::default(), None);
    model.apply(SourceUpdate {
        records: vec![
            serde_json::json!({"event": "update", "t": 1, "values": [1.0]}),
            serde_json::json!({"loss": 0.5}),
            serde_json::json!("nope"),
        ],
        restarted: false,
    });
    assert_eq!(model.phase(10), Phase::Waiting);
    assert_eq!(model.updates(), 0);
    assert!(model.widgets.is_empty());
}

#[test]
fn restart_clears_everything_before_new_records() {
    let mut model = model_with(&[("loss", WidgetKind::Curve)]);
    model.apply_event(TrainingEvent::Update { t: 2_000, values: vec![Some(1.0)] });
    model.apply(SourceUpdate {
        records: vec![serde_json::json!({
            "event": "init", "t": 50_000,
            "widgets": [{"name": "acc", "kind": "graph"}]
        })],
        restarted: true,
    });
    assert_eq!(model.widgets.len(), 1);
    assert_eq!(model.widgets[0].name, "acc");
    assert_eq!(model.updates(), 0);
}

#[test]
fn training_time_runs_while_active_and_freezes_when_idle() {
    let mut model = model_with(&[("loss", WidgetKind::Curve)]);
    model.apply_event(TrainingEvent::Update { t: 61_000, values: vec![Some(1.0)] });
    assert_eq!(model.phase(62_000), Phase::Running);
    assert_eq!(model.elapsed(62_000), Some(Duration::from_millis(61_000)));

    let later = 61_000 + STALE_AFTER_MS + 1;
    assert_eq!(model.phase(later), Phase::Idle);
    assert_eq!(model.elapsed(later), Some(Duration::from_millis(60_000)));
    assert_eq!(model.elapsed(later + 100_000), Some(Duration::from_millis(60_000)));

    // Heartbeats keep a script alive even when it updates only once per epoch...
    model.set_heartbeat(Some(later + 99_000));
    assert_eq!(model.phase(later + 100_000), Phase::Running);
    assert_eq!(model.elapsed(later + 100_000), Some(Duration::from_millis(164_001)));
    // ...and the clock freezes at the last heartbeat once they stop.
    let dead = later + 99_000 + STALE_AFTER_MS + 1;
    assert_eq!(model.phase(dead), Phase::Idle);
    assert_eq!(model.elapsed(dead), Some(Duration::from_millis(later + 99_000 - 1_000)));
    assert_eq!(model.elapsed(dead + 50_000), model.elapsed(dead));

    let fresh = TrainingModel::new("t".into(), "f".into(), Settings::default(), None);
    assert_eq!(fresh.elapsed(5), None);
}

#[test]
fn frame_has_the_exact_size_for_many_terminal_sizes() {
    let model = demo_model();
    let now = api::now_ms();
    for (width, height) in [(30, 8), (40, 12), (80, 24), (120, 40), (200, 60), (60, 9)] {
        let lines = render_training_frame(&model, width, height, now);
        assert_eq!(lines.len(), height, "{width}x{height}");
        for line in &lines {
            assert!(
                strip_ansi(line).chars().count() < width.max(2),
                "{width}x{height}: {:?}",
                strip_ansi(line)
            );
        }
    }
    // Too small: still exactly `height` lines, and the message is shown when it fits.
    for (width, height) in [(10, 4), (29, 30), (80, 7), (60, 5)] {
        assert_eq!(render_training_frame(&model, width, height, now).len(), height);
    }
    assert!(text_of(&model, 80, 7, now).contains("Terminal too small"));
}

#[test]
fn dashboard_shows_widgets_time_and_the_bold_hero_between_the_rows() {
    let mut model = model_with(&[
        ("loss", WidgetKind::Curve),
        ("val_loss", WidgetKind::Curve),
        ("acc", WidgetKind::Graph),
        ("lr", WidgetKind::Block),
    ]);
    model.apply_event(TrainingEvent::Update {
        t: 2_000,
        values: vec![Some(0.5), Some(0.6), Some(0.7), Some(0.001)],
    });
    model.apply_event(TrainingEvent::Update {
        t: 3_661_000,
        values: vec![Some(0.0023), Some(0.0031), Some(0.91), Some(0.0005)],
    });
    model.apply_event(TrainingEvent::Hero {
        t: 3_661_000,
        text: "Training at Epoch 47, step 4700, Total Loss at 0.0023".into(),
    });

    let now = 3_662_000;
    let text = text_of(&model, 100, 30, now);
    for expected in [
        "loss", "val_loss", "acc", "lr", "0.0023", "0.91", "01:01:01",
        "running", "Training at Epoch 47, step 4700, Total Loss at 0.0023",
        "min 0.0023", "max 0.5", "▼ -5.00e-4",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }

    // 4 widgets -> 2 rows of 2; the hero is between them.
    let lines: Vec<&str> = text.lines().collect();
    let hero = lines.iter().position(|l| l.contains("Training at Epoch 47")).unwrap();
    let first_row = lines.iter().position(|l| l.contains("┌─ loss")).unwrap();
    let second_row = lines.iter().position(|l| l.contains("┌─ acc")).unwrap();
    assert!(first_row < hero && hero < second_row);

    // The hero is bold (SGR 1) and horizontally centered.
    let raw = render_training_frame(&model, 100, 30, now);
    assert!(raw[hero].contains("\x1b[1m"));
    let visible = strip_ansi(&raw[hero]);
    let left = visible.chars().take_while(|c| *c == ' ').count();
    let right = 99 - visible.trim_end().chars().count();
    assert!(left.abs_diff(right) <= 2, "left {left} right {right}");
}

#[test]
fn a_single_row_puts_the_hero_above_the_widgets() {
    let mut model = model_with(&[("loss", WidgetKind::Curve)]);
    model.apply_event(TrainingEvent::Hero { t: 1_100, text: "HERO".into() });
    let text = text_of(&model, 80, 20, 2_000);
    let lines: Vec<&str> = text.lines().collect();
    let hero = lines.iter().position(|l| l.contains("HERO")).unwrap();
    let panel = lines.iter().position(|l| l.contains("┌─ loss")).unwrap();
    assert!(hero < panel);
}

#[test]
fn short_terminals_hide_rows_and_say_so() {
    let widgets: Vec<(String, WidgetKind)> =
        (0..8).map(|i| (format!("m{i}"), WidgetKind::Curve)).collect();
    let refs: Vec<(&str, WidgetKind)> =
        widgets.iter().map(|(n, k)| (n.as_str(), *k)).collect();
    let model = model_with(&refs);
    let text = text_of(&model, 60, 14, 2_000);
    assert!(text.contains("more (enlarge the terminal)"), "{text}");
    let tiny = text_of(&model, 60, 8, 2_000);
    assert!(tiny.contains("too short") || tiny.contains("more"));
}

#[test]
fn waiting_and_idle_states_explain_themselves() {
    let fresh = TrainingModel::new("t".into(), "events.jsonl".into(), Settings::default(), None);
    let text = text_of(&fresh, 100, 20, 5);
    assert!(text.contains("Waiting for training data"));
    assert!(text.contains("init_training_dashboard"));
    assert!(text.contains("--:--:--"));

    let mut model = model_with(&[("loss", WidgetKind::Curve)]);
    model.apply_event(TrainingEvent::Update { t: 2_000, values: vec![Some(1.0)] });
    let idle = text_of(&model, 100, 20, 2_000 + STALE_AFTER_MS + 5);
    assert!(idle.contains("idle"));
    assert!(idle.contains("script not running"));
    assert!(idle.contains("waiting for update_hero_text"));
}

#[test]
fn nan_and_missing_data_are_shown_not_crashed_on() {
    let mut model = model_with(&[("loss", WidgetKind::Curve), ("lr", WidgetKind::Block)]);
    let before = text_of(&model, 80, 20, 2_000);
    assert!(before.contains("no data yet"));
    model.apply_event(TrainingEvent::Update { t: 1_500, values: vec![None, None] });
    let after = text_of(&model, 80, 20, 2_000);
    assert!(after.contains("NaN"));
}

#[test]
fn config_overrides_label_unit_color_and_hero_style() {
    let settings = settings_from(serde_json::json!({
        "theme": {"hero": {"fg": "red", "bold": true, "reverse": true}},
        "metrics": {"per_metric": {"loss": {"label": "Total loss", "unit": "u", "decimals": 2, "color": "#ff0000"}}},
        "layout": {"show_footer": false}
    }));
    let mut model = TrainingModel::new("t".into(), "f".into(), settings, None);
    model.apply_event(TrainingEvent::Init {
        t: 1_000,
        widgets: vec![WidgetSpec { name: "loss".into(), kind: WidgetKind::Block }],
    });
    model.apply_event(TrainingEvent::Update { t: 1_100, values: vec![Some(1.23456)] });
    model.apply_event(TrainingEvent::Hero { t: 1_100, text: "hi".into() });

    let raw = render_training_frame(&model, 80, 20, 2_000).join("\n");
    let text = strip_ansi(&raw);
    assert!(text.contains("Total loss") && text.contains("1.23u"));
    assert!(!text.contains("q / Esc"));
    assert!(raw.contains("1;7;31"), "hero style missing");
    assert!(raw.contains("38;2;255;0;0"), "metric color missing");
}

#[test]
fn demo_runs_the_real_protocol_and_fills_every_widget_type() {
    let model = demo_model();
    assert_eq!(model.widgets.len(), 6);
    assert!(model.updates() >= 100);
    assert!(model.hero.starts_with("Training at Epoch"));
    let text = text_of(&model, 120, 40, api::now_ms());
    for name in ["loss", "val_loss", "accuracy", "grad_norm", "lr", "samples/s"] {
        assert!(text.contains(name), "{name} missing");
    }
    assert!(text.chars().any(|c| ('\u{2800}'..='\u{28ff}').contains(&c)), "no braille curve");
    assert!(text.contains('█') || text.contains('▇'), "no bars");
}

#[test]
fn events_file_to_dashboard_round_trip() {
    // The same bytes the Python API writes are what the tail reads.
    let path = std::env::temp_dir().join(format!(
        "training_controller-view-{}-{}.jsonl",
        std::process::id(),
        api::now_ms()
    ));
    let specs = api::parse_widget_specs(&[
        "loss".to_string(),
        "curve".to_string(),
        "lr".to_string(),
        "box".to_string(),
    ])
    .unwrap();
    let mut writer = api::TrainingWriter::create(&path, specs.clone()).unwrap();
    writer.update(&[0.5, 0.01]).unwrap();
    writer.update(&[0.25, 0.02]).unwrap();
    writer.hero("Epoch 2").unwrap();

    let mut source = TrainingSource::file(path.clone());
    let mut model = TrainingModel::new("t".into(), source.label(), Settings::default(), None);
    model.apply(source.poll().unwrap());
    assert_eq!(model.updates(), 2);
    assert_eq!(model.hero, "Epoch 2");

    // A new run truncates the file; the tail reports a restart.
    let mut second = api::TrainingWriter::create(&path, specs).unwrap();
    second.update(&[9.0, 9.0]).unwrap();
    model.apply(source.poll().unwrap());
    assert_eq!(model.updates(), 1);
    assert!(model.hero.is_empty());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn training_path_resolution_follows_precedence() {
    let cwd = std::env::current_dir().unwrap();
    let explicit = cwd.join("events.jsonl").to_string_lossy().into_owned();
    assert_eq!(
        resolve_training_path(Some(&explicit), None).unwrap(),
        PathBuf::from(&explicit)
    );
    assert!(resolve_training_path(Some("relative.jsonl"), None).is_err());

    if std::env::var_os(api::TRAINING_FILE_ENV).is_none() {
        assert!(resolve_training_path(None, None).is_err());
        let script = cwd.join("train.py").to_string_lossy().into_owned();
        assert_eq!(
            resolve_training_path(None, Some(&script)).unwrap(),
            cwd.join(api::TRAINING_FILE_NAME)
        );
    }
}
