//! SQLite storage for training logs and experiment IDs.
//!
//! The Training Controller owns one database (`logs.db`, next to `tc.json`).
//! Every `/run_script` creates an **experiment** row and gets a numeric
//! **experiment ID**. The controller passes the database path and the ID to the
//! script through environment variables ([`DB_FILE_ENV`], [`EXPERIMENT_ID_ENV`]).
//! The Python script then declares its columns with `tc.init_log(...)` and
//! fills the log with `tc.log(...)`; both end up in [`LogWriter`].
//!
//! ```text
//! controller (main.rs)                       python script (api.rs / python.rs)
//! --------------------                       ----------------------------------
//! create_experiment() -> id  ── env vars ──► init_log("loss", "acc")   LogWriter::init
//! finish_experiment() on exit                log(0.5, 0.9)  (per step)  LogWriter::log
//! list_experiments / read_log  ◄── same SQLite file (WAL mode) ──►
//! ```
//!
//! # Schema
//!
//! * `experiments(id, script_name, script_path, config_name, args, status,
//!   exit_code, started_ms, finished_ms, script_seq)`: `id` is the database-wide
//!   key (it is what `TC_EXPERIMENT_ID` and the tables below refer to);
//!   `script_seq` is the experiment ID that users see: 1, 2, 3, ... counted
//!   separately for every script (name and path together), see
//!   [`resolve_experiment`] and [`experiment_seq`]
//! * `log_columns(experiment_id, idx, title)`: the titles given to `init_log`
//! * `log_steps(experiment_id, step, t_ms, time_s)`: one row per `log(...)`
//!   call. `t_ms` is the wall-clock time; `time_s` is the time in seconds since
//!   the *first* `log(...)` call of the experiment (0 for the first row)
//! * `log_values(experiment_id, step, col, value)`: one row per value; NaN and
//!   infinite values are stored as `NULL`
//! * `log_view`: the three log tables joined, handy for ad-hoc SQL:
//!   `SELECT step, title, value FROM log_view WHERE experiment_id = 3`
//!
//! Both the controller and the script open the file at the same time, so it
//! uses WAL mode and a busy timeout.
//!
//! This module uses no PyO3 types, so it is tested with plain `cargo test`.

use crate::api::now_ms;
use rusqlite::{params, Connection, OptionalExtension};
use std::fs;
use std::path::Path;
use std::time::Duration;

/// Environment variable naming the SQLite file. Set by `/run_script`; when it
/// is missing `init_log` uses `logs.db` in the working directory.
pub const DB_FILE_ENV: &str = "TC_LOG_DB";

/// Environment variable holding the experiment ID chosen by the controller.
pub const EXPERIMENT_ID_ENV: &str = "TC_EXPERIMENT_ID";

/// Default database file name.
pub const DB_FILE_NAME: &str = "logs.db";

/// Status of an experiment whose script was started by the controller.
pub const STATUS_RUNNING: &str = "running";
/// Status of an experiment created by a script that was not started by the
/// controller (nobody records when such a run ends).
pub const STATUS_STANDALONE: &str = "standalone";

const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

const SCHEMA: &str = "
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    script_name TEXT,
    script_path TEXT,
    config_name TEXT,
    args        TEXT NOT NULL DEFAULT '[]',
    status      TEXT NOT NULL DEFAULT 'running',
    exit_code   INTEGER,
    started_ms  INTEGER NOT NULL,
    finished_ms INTEGER,
    script_seq  INTEGER
);

CREATE TABLE IF NOT EXISTS log_columns (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    title         TEXT NOT NULL,
    PRIMARY KEY (experiment_id, idx),
    UNIQUE (experiment_id, title)
);

CREATE TABLE IF NOT EXISTS log_steps (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    step          INTEGER NOT NULL,
    t_ms          INTEGER NOT NULL,
    time_s        REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (experiment_id, step)
);

CREATE TABLE IF NOT EXISTS log_values (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    step          INTEGER NOT NULL,
    col           INTEGER NOT NULL,
    value         REAL,
    PRIMARY KEY (experiment_id, step, col)
);
";

/// Bumped whenever [`migrate`] has something new to do.
const SCHEMA_VERSION: i64 = 3;

/// Recreated by every migration so it always matches the current tables.
const VIEW_SQL: &str = "
DROP VIEW IF EXISTS log_view;
CREATE VIEW log_view AS
SELECT v.experiment_id AS experiment_id,
       v.step          AS step,
       s.t_ms          AS t_ms,
       s.time_s        AS time_s,
       c.title         AS title,
       v.value         AS value
FROM log_values v
JOIN log_steps   s ON s.experiment_id = v.experiment_id AND s.step = v.step
JOIN log_columns c ON c.experiment_id = v.experiment_id AND c.idx  = v.col;
";

fn err(what: &'static str) -> impl Fn(rusqlite::Error) -> String {
    move |error| format!("{what}: {error}")
}

/// Creates the tables if they do not exist yet and upgrades older databases.
pub fn init_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(SCHEMA)
        .map_err(err("Failed to create the log database schema"))?;
    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(err("Failed to read the log database version"))?;
    if version < SCHEMA_VERSION {
        migrate(conn)?;
    }
    Ok(())
}

/// Runs in one immediate transaction, so two processes opening an old
/// database at the same time upgrade it one after the other.
fn migrate(conn: &Connection) -> Result<(), String> {
    conn.execute_batch("BEGIN IMMEDIATE")
        .map_err(err("Failed to upgrade the log database"))?;
    match migrate_in_transaction(conn) {
        Ok(()) => conn
            .execute_batch("COMMIT")
            .map_err(err("Failed to upgrade the log database")),
        Err(error) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(error)
        }
    }
}

fn migrate_in_transaction(conn: &Connection) -> Result<(), String> {
    let has_time: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('log_steps') WHERE name = 'time_s'",
            [],
            |row| row.get(0),
        )
        .map_err(err("Failed to upgrade the log database"))?;
    if has_time == 0 {
        // Version 2: time since the first log() call. Old rows are backfilled.
        conn.execute_batch(
            "ALTER TABLE log_steps ADD COLUMN time_s REAL NOT NULL DEFAULT 0;
             UPDATE log_steps SET time_s = (t_ms - (
                 SELECT MIN(s.t_ms) FROM log_steps s
                 WHERE s.experiment_id = log_steps.experiment_id)) / 1000.0;",
        )
        .map_err(err("Failed to upgrade the log database"))?;
    }
    let has_seq: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('experiments') WHERE name = 'script_seq'",
            [],
            |row| row.get(0),
        )
        .map_err(err("Failed to upgrade the log database"))?;
    if has_seq == 0 {
        // Version 3: experiment IDs count per script. Old experiments are
        // numbered in the order they were created, separately for each script.
        conn.execute_batch(
            "ALTER TABLE experiments ADD COLUMN script_seq INTEGER;
             UPDATE experiments SET script_seq = (
                 SELECT COUNT(*) FROM experiments e
                 WHERE e.script_name IS experiments.script_name
                   AND e.script_path IS experiments.script_path
                   AND e.id <= experiments.id);",
        )
        .map_err(err("Failed to upgrade the log database"))?;
    }
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS experiments_by_script
         ON experiments (script_name, script_path, script_seq);",
    )
    .map_err(err("Failed to upgrade the log database"))?;
    conn.execute_batch(VIEW_SQL)
        .map_err(err("Failed to upgrade the log database"))?;
    conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))
        .map_err(err("Failed to upgrade the log database"))
}

/// Opens (creating it and its folder if needed) the database at `path`.
pub fn open(path: &Path) -> Result<Connection, String> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create {}: {error}", parent.display()))?;
    }
    let conn = Connection::open(path)
        .map_err(|error| format!("Failed to open log database {}: {error}", path.display()))?;
    conn.busy_timeout(BUSY_TIMEOUT)
        .map_err(err("Failed to configure the log database"))?;
    conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;")
        .map_err(err("Failed to configure the log database"))?;
    init_schema(&conn)?;
    Ok(conn)
}

// ---------------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------------

/// What the controller knows about a run when it creates the experiment.
pub struct NewExperiment<'a> {
    pub script_name: Option<&'a str>,
    pub script_path: Option<&'a str>,
    pub config_name: Option<&'a str>,
    pub args: &'a [String],
    pub status: &'a str,
}

/// Inserts an experiment and returns its database ID (1, 2, 3, ... never
/// reused). The experiment also gets its number within its script (see
/// [`experiment_seq`]): one more than the newest experiment of the same script.
/// Doing both in one statement keeps the numbering right when two runs start
/// at the same time.
pub fn create_experiment(conn: &Connection, new: &NewExperiment) -> Result<i64, String> {
    let args = serde_json::to_string(new.args)
        .map_err(|error| format!("Failed to encode the script arguments: {error}"))?;
    conn.execute(
        "INSERT INTO experiments
             (script_name, script_path, config_name, args, status, started_ms, script_seq)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6,
             (SELECT COALESCE(MAX(script_seq), 0) + 1 FROM experiments
              WHERE script_name IS ?1 AND script_path IS ?2))",
        params![
            new.script_name,
            new.script_path,
            new.config_name,
            args,
            new.status,
            now_ms() as i64
        ],
    )
    .map_err(err("Failed to create the experiment"))?;
    Ok(conn.last_insert_rowid())
}

/// Records how an experiment ended.
pub fn finish_experiment(
    conn: &Connection,
    experiment_id: i64,
    status: &str,
    exit_code: Option<i32>,
) -> Result<(), String> {
    conn.execute(
        "UPDATE experiments SET status = ?1, exit_code = ?2, finished_ms = ?3 WHERE id = ?4",
        params![status, exit_code, now_ms() as i64, experiment_id],
    )
    .map_err(err("Failed to update the experiment"))?;
    Ok(())
}

/// Whether an experiment with this ID exists.
pub fn experiment_exists(conn: &Connection, experiment_id: i64) -> Result<bool, String> {
    conn.query_row(
        "SELECT 1 FROM experiments WHERE id = ?1",
        [experiment_id],
        |_| Ok(()),
    )
    .optional()
    .map(|found| found.is_some())
    .map_err(err("Failed to look up the experiment"))
}

/// The column titles of an experiment's log, in `init_log` order.
pub fn log_columns(conn: &Connection, experiment_id: i64) -> Result<Vec<String>, String> {
    let mut statement = conn
        .prepare("SELECT title FROM log_columns WHERE experiment_id = ?1 ORDER BY idx")
        .map_err(err("Failed to read the log columns"))?;
    let rows = statement
        .query_map([experiment_id], |row| row.get::<_, String>(0))
        .map_err(err("Failed to read the log columns"))?;
    let titles = rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(err("Failed to read the log columns"))?;
    Ok(titles)
}

/// One line of `/show_experiments`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSummary {
    /// Database ID.
    pub id: i64,
    /// The experiment ID users see: the number within its script.
    pub seq: i64,
    pub script_name: Option<String>,
    pub config_name: Option<String>,
    pub status: String,
    pub exit_code: Option<i64>,
    /// Local time, `YYYY-MM-DD HH:MM:SS`.
    pub started: String,
    /// `None` while the run has not finished.
    pub duration_s: Option<f64>,
    pub steps: i64,
    pub columns: Vec<String>,
}

/// The newest `limit` experiments of all scripts, newest first.
pub fn list_experiments(conn: &Connection, limit: usize) -> Result<Vec<ExperimentSummary>, String> {
    list_experiments_for(conn, None, limit)
}

/// The newest `limit` experiments, newest first; with `script` (name, path)
/// only those of that script.
pub fn list_experiments_for(
    conn: &Connection,
    script: Option<(&str, &str)>,
    limit: usize,
) -> Result<Vec<ExperimentSummary>, String> {
    let mut statement = conn
        .prepare(
            "SELECT e.id, e.script_name, e.config_name, e.status, e.exit_code,
                    datetime(e.started_ms / 1000, 'unixepoch', 'localtime'),
                    CASE WHEN e.finished_ms IS NULL THEN NULL
                         ELSE (e.finished_ms - e.started_ms) / 1000.0 END,
                    (SELECT COUNT(*) FROM log_steps s WHERE s.experiment_id = e.id),
                    e.script_seq
             FROM experiments e
             WHERE (?1 = 0 OR (e.script_name IS ?2 AND e.script_path IS ?3))
             ORDER BY e.id DESC
             LIMIT ?4",
        )
        .map_err(err("Failed to list the experiments"))?;
    let rows = statement
        .query_map(
            params![
                script.is_some() as i64,
                script.map(|(name, _)| name),
                script.map(|(_, path)| path),
                limit as i64
            ],
            |row| {
            let id: i64 = row.get(0)?;
            Ok(ExperimentSummary {
                id,
                seq: row.get::<_, Option<i64>>(8)?.unwrap_or(id),
                script_name: row.get(1)?,
                config_name: row.get(2)?,
                status: row.get(3)?,
                exit_code: row.get(4)?,
                started: row.get(5)?,
                duration_s: row.get(6)?,
                steps: row.get(7)?,
                columns: Vec::new(),
            })
        })
        .map_err(err("Failed to list the experiments"))?;

    let mut summaries = Vec::new();
    for row in rows {
        let mut summary = row.map_err(err("Failed to list the experiments"))?;
        summary.columns = log_columns(conn, summary.id)?;
        summaries.push(summary);
    }
    Ok(summaries)
}

// ---------------------------------------------------------------------------
// Writing the log (the Python side)
// ---------------------------------------------------------------------------

/// Checks and trims the titles given to `init_log`.
///
/// Fails on an empty list, an empty title or a duplicate title.
pub fn validate_titles(titles: &[String]) -> Result<Vec<String>, String> {
    if titles.is_empty() {
        return Err(
            "init_log needs at least one column title, e.g. init_log(\"loss\", \"accuracy\")."
                .into(),
        );
    }
    let mut clean: Vec<String> = Vec::with_capacity(titles.len());
    for title in titles {
        let title = title.trim();
        if title.is_empty() {
            return Err("Log column titles must not be empty.".into());
        }
        if clean.iter().any(|existing| existing == title) {
            return Err(format!("Log column title '{title}' is listed twice."));
        }
        clean.push(title.to_string());
    }
    Ok(clean)
}

/// Appends log rows for one experiment. Owns its database connection; the
/// functions in `api.rs` keep one writer per process.
pub struct LogWriter {
    conn: Connection,
    experiment_id: i64,
    titles: Vec<String>,
    next_step: i64,
    /// Wall-clock time of the first `log` call; `time_s` counts from here.
    first_log_ms: Option<i64>,
}

impl LogWriter {
    /// Declares the columns of `experiment_id` and starts an empty log.
    ///
    /// Any log the experiment already had is discarded, so calling `init_log`
    /// again restarts the log. The experiment must exist.
    pub fn init(
        mut conn: Connection,
        experiment_id: i64,
        titles: Vec<String>,
    ) -> Result<Self, String> {
        {
            let tx = conn
                .transaction()
                .map_err(err("Failed to start a log transaction"))?;
            for table in ["log_values", "log_steps", "log_columns"] {
                tx.execute(
                    &format!("DELETE FROM {table} WHERE experiment_id = ?1"),
                    [experiment_id],
                )
                .map_err(err("Failed to reset the log"))?;
            }
            for (index, title) in titles.iter().enumerate() {
                tx.execute(
                    "INSERT INTO log_columns (experiment_id, idx, title) VALUES (?1, ?2, ?3)",
                    params![experiment_id, index as i64, title],
                )
                .map_err(err("Failed to store the log column titles"))?;
            }
            tx.commit().map_err(err("Failed to store the log column titles"))?;
        }
        Ok(Self {
            conn,
            experiment_id,
            titles,
            next_step: 0,
            first_log_ms: None,
        })
    }

    pub fn experiment_id(&self) -> i64 {
        self.experiment_id
    }

    pub fn titles(&self) -> &[String] {
        &self.titles
    }

    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Gives the connection back, e.g. to start a new log with it.
    pub fn into_connection(self) -> Connection {
        self.conn
    }

    /// Records one value per column as the next step (0, 1, 2, ...).
    ///
    /// The row also stores `time_s`: seconds since the first `log` call of
    /// this run, so the first row has time 0.
    ///
    /// NaN and infinite values are stored as `NULL`. The row is committed
    /// immediately, so the controller sees it while the script keeps running.
    pub fn log(&mut self, values: &[f64]) -> Result<(), String> {
        if values.len() != self.titles.len() {
            return Err(format!(
                "log expected {} values ({}) but got {}.",
                self.titles.len(),
                self.titles.join(", "),
                values.len()
            ));
        }
        let step = self.next_step;
        let t_ms = now_ms() as i64;
        let first_ms = self.first_log_ms.unwrap_or(t_ms);
        let time_s = (t_ms - first_ms).max(0) as f64 / 1000.0;

        let tx = self
            .conn
            .transaction()
            .map_err(err("Failed to start a log transaction"))?;
        tx.prepare_cached(
            "INSERT INTO log_steps (experiment_id, step, t_ms, time_s) VALUES (?1, ?2, ?3, ?4)",
        )
        .map_err(err("Failed to write the log"))?
        .execute(params![self.experiment_id, step, t_ms, time_s])
        .map_err(err("Failed to write the log"))?;
        {
            let mut insert = tx
                .prepare_cached(
                    "INSERT INTO log_values (experiment_id, step, col, value) VALUES (?1, ?2, ?3, ?4)",
                )
                .map_err(err("Failed to write the log"))?;
            for (col, value) in values.iter().enumerate() {
                let value = value.is_finite().then_some(*value);
                insert
                    .execute(params![self.experiment_id, step, col as i64, value])
                    .map_err(err("Failed to write the log"))?;
            }
        }
        tx.commit().map_err(err("Failed to write the log"))?;
        self.first_log_ms.get_or_insert(t_ms);
        self.next_step += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Reading the log (the controller side)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct LogRow {
    pub step: i64,
    /// Seconds since the first `log` call of the experiment (0 for step 0).
    pub time_s: f64,
    /// One entry per column; `None` is a NaN/infinite value.
    pub values: Vec<Option<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogTable {
    /// Database ID.
    pub experiment_id: i64,
    /// The experiment ID users see: the number within its script.
    pub seq: i64,
    pub columns: Vec<String>,
    /// Number of steps in the whole log (`rows` may hold only the last few).
    pub total_steps: i64,
    pub rows: Vec<LogRow>,
}

/// The last `last` steps of an experiment's log, oldest first.
pub fn read_log(conn: &Connection, experiment_id: i64, last: usize) -> Result<LogTable, String> {
    if !experiment_exists(conn, experiment_id)? {
        return Err(format!("Experiment #{experiment_id} does not exist."));
    }

    let columns = log_columns(conn, experiment_id)?;
    let total_steps: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM log_steps WHERE experiment_id = ?1",
            [experiment_id],
            |row| row.get(0),
        )
        .map_err(err("Failed to read the log"))?;

    let mut steps: Vec<(i64, f64)> = {
        let mut statement = conn
            .prepare(
                "SELECT step, time_s FROM log_steps WHERE experiment_id = ?1
                 ORDER BY step DESC LIMIT ?2",
            )
            .map_err(err("Failed to read the log"))?;
        let rows = statement
            .query_map(params![experiment_id, last as i64], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
            })
            .map_err(err("Failed to read the log"))?;
        let collected = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(err("Failed to read the log"))?;
        collected
    };
    steps.reverse();

    let mut value_query = conn
        .prepare("SELECT col, value FROM log_values WHERE experiment_id = ?1 AND step = ?2")
        .map_err(err("Failed to read the log"))?;
    let mut rows = Vec::with_capacity(steps.len());
    for (step, time_s) in steps {
        let mut values: Vec<Option<f64>> = vec![None; columns.len()];
        let cells = value_query
            .query_map(params![experiment_id, step], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<f64>>(1)?))
            })
            .map_err(err("Failed to read the log"))?;
        for cell in cells {
            let (col, value) = cell.map_err(err("Failed to read the log"))?;
            if let Some(slot) = usize::try_from(col).ok().and_then(|col| values.get_mut(col)) {
                *slot = value;
            }
        }
        rows.push(LogRow {
            step,
            time_s,
            values,
        });
    }

    Ok(LogTable {
        experiment_id,
        seq: experiment_seq(conn, experiment_id)?.unwrap_or(experiment_id),
        columns,
        total_steps,
        rows,
    })
}

// ---------------------------------------------------------------------------
// Quantities for graphs (the controller side)
// ---------------------------------------------------------------------------
//
// A *quantity* is something that can be put on a graph axis: any log column,
// `time` (seconds since the first log call) or `step`. A column whose title is
// `time` or `step` takes precedence over the built-in quantity of that name.

/// Built-in quantity: seconds since the first `log` call.
pub const QUANTITY_TIME: &str = "time";
/// Built-in quantity: the step number (0, 1, 2, ...).
pub const QUANTITY_STEP: &str = "step";

/// Every quantity of an experiment: its columns, then `time` and `step`.
pub fn quantity_names(conn: &Connection, experiment_id: i64) -> Result<Vec<String>, String> {
    let mut names = log_columns(conn, experiment_id)?;
    for builtin in [QUANTITY_TIME, QUANTITY_STEP] {
        if !names.iter().any(|name| name == builtin) {
            names.push(builtin.to_string());
        }
    }
    Ok(names)
}

fn query_pairs(
    conn: &Connection,
    sql: &str,
    parameters: impl rusqlite::Params,
) -> Result<Vec<(i64, f64)>, String> {
    let mut statement = conn
        .prepare_cached(sql)
        .map_err(err("Failed to read the log"))?;
    let rows = statement
        .query_map(parameters, |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
        })
        .map_err(err("Failed to read the log"))?;
    let pairs = rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(err("Failed to read the log"))?;
    Ok(pairs)
}

/// `(step, value)` of one quantity for every step after `after_step`, oldest
/// first (use -1 for all steps). Steps whose value is NaN/infinite are left out.
///
/// Fails if the experiment has no such quantity, listing the ones it has.
pub fn read_quantity(
    conn: &Connection,
    experiment_id: i64,
    name: &str,
    after_step: i64,
) -> Result<Vec<(i64, f64)>, String> {
    let columns = log_columns(conn, experiment_id)?;
    if let Some(col) = columns.iter().position(|title| title == name) {
        return query_pairs(
            conn,
            "SELECT step, value FROM log_values
             WHERE experiment_id = ?1 AND col = ?2 AND step > ?3 AND value IS NOT NULL
             ORDER BY step",
            params![experiment_id, col as i64, after_step],
        );
    }
    match name {
        QUANTITY_TIME => query_pairs(
            conn,
            "SELECT step, time_s FROM log_steps
             WHERE experiment_id = ?1 AND step > ?2 ORDER BY step",
            params![experiment_id, after_step],
        ),
        QUANTITY_STEP => query_pairs(
            conn,
            "SELECT step, CAST(step AS REAL) FROM log_steps
             WHERE experiment_id = ?1 AND step > ?2 ORDER BY step",
            params![experiment_id, after_step],
        ),
        _ => {
            let hint = if columns.is_empty() {
                " (the script has not called init_log yet)"
            } else {
                ""
            };
            Err(format!(
                "Experiment #{experiment_id} has no quantity '{name}'. Available: {}{hint}.",
                quantity_names(conn, experiment_id)?.join(", ")
            ))
        }
    }
}

/// The highest logged step of an experiment, `None` while its log is empty.
pub fn max_step(conn: &Connection, experiment_id: i64) -> Result<Option<i64>, String> {
    conn.query_row(
        "SELECT MAX(step) FROM log_steps WHERE experiment_id = ?1",
        [experiment_id],
        |row| row.get::<_, Option<i64>>(0),
    )
    .map_err(err("Failed to read the log"))
}

/// The status text of an experiment (`running`, `finished`, ...), if it exists.
pub fn experiment_status(conn: &Connection, experiment_id: i64) -> Result<Option<String>, String> {
    conn.query_row(
        "SELECT status FROM experiments WHERE id = ?1",
        [experiment_id],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(err("Failed to look up the experiment"))
}

/// The ID of the newest experiment, `None` while there are none.
pub fn latest_experiment_id(conn: &Connection) -> Result<Option<i64>, String> {
    conn.query_row("SELECT MAX(id) FROM experiments", [], |row| {
        row.get::<_, Option<i64>>(0)
    })
    .map_err(err("Failed to look up the experiments"))
}

/// The newest experiment (database ID) of one script, `None` if it has none.
pub fn latest_script_experiment_id(
    conn: &Connection,
    script_name: &str,
    script_path: &str,
) -> Result<Option<i64>, String> {
    conn.query_row(
        "SELECT MAX(id) FROM experiments WHERE script_name IS ?1 AND script_path IS ?2",
        params![script_name, script_path],
        |row| row.get::<_, Option<i64>>(0),
    )
    .map_err(err("Failed to look up the experiments"))
}

/// The experiment ID users see for a database ID: its number within its
/// script (1 for the first run of each script). `None` if there is no such
/// experiment.
pub fn experiment_seq(conn: &Connection, experiment_id: i64) -> Result<Option<i64>, String> {
    conn.query_row(
        "SELECT script_seq FROM experiments WHERE id = ?1",
        [experiment_id],
        |row| row.get::<_, Option<i64>>(0),
    )
    .optional()
    .map(Option::flatten)
    .map_err(err("Failed to look up the experiment"))
}

/// The database ID of experiment number `seq` of a script (the number users
/// type, e.g. `/graph 3`). `None` if the script has no such experiment.
pub fn resolve_experiment(
    conn: &Connection,
    script_name: &str,
    script_path: &str,
    seq: i64,
) -> Result<Option<i64>, String> {
    conn.query_row(
        "SELECT id FROM experiments
         WHERE script_name IS ?1 AND script_path IS ?2 AND script_seq = ?3",
        params![script_name, script_path, seq],
        |row| row.get::<_, i64>(0),
    )
    .optional()
    .map_err(err("Failed to look up the experiment"))
}

// ---------------------------------------------------------------------------
// Text output for the REPL (pure)
// ---------------------------------------------------------------------------

/// Left-aligned columns separated by two spaces.
pub fn render_table(headers: &[String], rows: &[Vec<String>]) -> String {
    let mut widths: Vec<usize> = headers.iter().map(|header| header.chars().count()).collect();
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            if let Some(width) = widths.get_mut(index) {
                *width = (*width).max(cell.chars().count());
            }
        }
    }
    let line = |cells: &[String]| -> String {
        cells
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let width = widths.get(index).copied().unwrap_or(0);
                format!("{cell:<width$}")
            })
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_string()
    };

    let mut out = line(headers);
    for row in rows {
        out.push('\n');
        out.push_str(&line(row));
    }
    out
}

/// Compact number: plain for ordinary magnitudes, scientific for tiny/huge ones.
pub fn format_value(value: Option<f64>) -> String {
    match value {
        None => "-".to_string(),
        Some(value) if value != 0.0 && (value.abs() < 1e-4 || value.abs() >= 1e7) => {
            format!("{value:.4e}")
        }
        Some(value) => {
            let text = format!("{value:.6}");
            let trimmed = text.trim_end_matches('0').trim_end_matches('.');
            if trimmed.is_empty() || trimmed == "-" {
                "0".to_string()
            } else {
                trimmed.to_string()
            }
        }
    }
}

fn format_duration(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let (hours, minutes, secs) = (total / 3600, (total % 3600) / 60, total % 60);
    if hours > 0 {
        format!("{hours}h{minutes:02}m{secs:02}s")
    } else if minutes > 0 {
        format!("{minutes}m{secs:02}s")
    } else {
        format!("{secs}s")
    }
}

/// Text for `/show_experiments`.
pub fn format_experiments(experiments: &[ExperimentSummary]) -> String {
    let headers: Vec<String> = ["id", "script", "config", "status", "started", "took", "steps", "columns"]
        .iter()
        .map(|header| header.to_string())
        .collect();
    let rows: Vec<Vec<String>> = experiments
        .iter()
        .map(|experiment| {
            let status = match experiment.exit_code {
                Some(code) if code != 0 => format!("{} (exit {code})", experiment.status),
                _ => experiment.status.clone(),
            };
            vec![
                format!("#{}", experiment.seq),
                experiment.script_name.clone().unwrap_or_else(|| "-".into()),
                experiment.config_name.clone().unwrap_or_else(|| "-".into()),
                status,
                experiment.started.clone(),
                experiment.duration_s.map_or_else(|| "-".into(), format_duration),
                experiment.steps.to_string(),
                if experiment.columns.is_empty() {
                    "-".to_string()
                } else {
                    experiment.columns.join(", ")
                },
            ]
        })
        .collect();
    format!(
        "Experiments (newest first):\n{}\nShow a log with /show_log ID.",
        render_table(&headers, &rows)
    )
}

/// Text for `/show_log`.
pub fn format_log(table: &LogTable) -> String {
    if table.columns.is_empty() {
        return format!(
            "Experiment #{} has no log yet (the script has not called init_log).",
            table.seq
        );
    }
    let mut headers = vec!["step".to_string(), "time_s".to_string()];
    headers.extend(table.columns.iter().cloned());
    let rows: Vec<Vec<String>> = table
        .rows
        .iter()
        .map(|row| {
            let mut cells = vec![row.step.to_string(), format!("{:.2}", row.time_s)];
            cells.extend(row.values.iter().map(|value| format_value(*value)));
            cells
        })
        .collect();
    format!(
        "Experiment #{}: {} step(s), showing the last {}\n{}",
        table.seq,
        table.total_steps,
        table.rows.len(),
        render_table(&headers, &rows)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn
    }

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| item.to_string()).collect()
    }

    fn new_experiment(conn: &Connection, name: &str) -> i64 {
        create_experiment(
            conn,
            &NewExperiment {
                script_name: Some(name),
                script_path: Some("/train.py"),
                config_name: Some("fast"),
                args: &strings(&["--epochs", "3"]),
                status: STATUS_RUNNING,
            },
        )
        .unwrap()
    }

    #[test]
    fn titles_are_validated_and_trimmed() {
        assert_eq!(
            validate_titles(&strings(&[" loss ", "acc"])).unwrap(),
            strings(&["loss", "acc"])
        );
        assert!(validate_titles(&[]).unwrap_err().contains("at least one"));
        assert!(validate_titles(&strings(&["  "])).unwrap_err().contains("empty"));
        assert!(validate_titles(&strings(&["a", "b", "a"])).unwrap_err().contains("twice"));
    }

    #[test]
    fn experiment_ids_increase_and_are_never_reused() {
        let conn = memory_db();
        let first = new_experiment(&conn, "a");
        let second = new_experiment(&conn, "b");
        assert_eq!((first, second), (1, 2));
        assert!(experiment_exists(&conn, 2).unwrap());
        assert!(!experiment_exists(&conn, 3).unwrap());

        conn.execute("DELETE FROM experiments WHERE id = 2", []).unwrap();
        assert_eq!(new_experiment(&conn, "c"), 3);
    }

    #[test]
    fn log_rows_round_trip_with_nan_as_null() {
        let conn = memory_db();
        let id = new_experiment(&conn, "train");
        let mut writer = LogWriter::init(conn, id, strings(&["loss", "acc"])).unwrap();
        writer.log(&[0.5, 0.25]).unwrap();
        writer.log(&[f64::NAN, f64::INFINITY]).unwrap();
        writer.log(&[0.125, 1.0]).unwrap();

        let table = read_log(writer.connection(), id, 10).unwrap();
        assert_eq!(table.columns, strings(&["loss", "acc"]));
        assert_eq!(table.total_steps, 3);
        let values: Vec<_> = table.rows.iter().map(|row| row.values.clone()).collect();
        assert_eq!(
            values,
            vec![
                vec![Some(0.5), Some(0.25)],
                vec![None, None],
                vec![Some(0.125), Some(1.0)],
            ]
        );
        assert_eq!(
            table.rows.iter().map(|row| row.step).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );

        // Only the last rows are returned, still oldest first.
        let tail = read_log(writer.connection(), id, 2).unwrap();
        assert_eq!(tail.total_steps, 3);
        assert_eq!(tail.rows.iter().map(|row| row.step).collect::<Vec<_>>(), vec![1, 2]);

        // The view joins titles, steps and values.
        let count: i64 = writer
            .connection()
            .query_row(
                "SELECT COUNT(*) FROM log_view WHERE experiment_id = ?1 AND title = 'loss'",
                [id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn wrong_value_count_is_rejected_and_writes_nothing() {
        let conn = memory_db();
        let id = new_experiment(&conn, "train");
        let mut writer = LogWriter::init(conn, id, strings(&["a", "b"])).unwrap();
        let error = writer.log(&[1.0]).unwrap_err();
        assert!(error.contains("expected 2 values (a, b) but got 1"));
        assert_eq!(read_log(writer.connection(), id, 10).unwrap().total_steps, 0);

        writer.log(&[1.0, 2.0]).unwrap();
        assert_eq!(read_log(writer.connection(), id, 10).unwrap().total_steps, 1);
    }

    #[test]
    fn restarting_a_log_discards_old_rows_but_keeps_other_experiments() {
        let conn = memory_db();
        let one = new_experiment(&conn, "one");
        let two = new_experiment(&conn, "two");
        conn.execute(
            "INSERT INTO log_columns (experiment_id, idx, title) VALUES (?1, 0, 'keep')",
            [two],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO log_steps (experiment_id, step, t_ms) VALUES (?1, 0, 1)",
            [two],
        )
        .unwrap();

        let mut writer = LogWriter::init(conn, one, strings(&["x"])).unwrap();
        writer.log(&[1.0]).unwrap();
        writer.log(&[2.0]).unwrap();
        assert_eq!(read_log(writer.connection(), one, 10).unwrap().total_steps, 2);

        // Same connection, new writer for the same experiment: a fresh log.
        let LogWriter { conn, .. } = writer;
        let mut writer = LogWriter::init(conn, one, strings(&["y", "z"])).unwrap();
        assert_eq!(writer.titles(), strings(&["y", "z"]).as_slice());
        let table = read_log(writer.connection(), one, 10).unwrap();
        assert_eq!(table.total_steps, 0);
        assert_eq!(table.columns, strings(&["y", "z"]));
        writer.log(&[1.0, 2.0]).unwrap();
        assert_eq!(read_log(writer.connection(), one, 10).unwrap().rows[0].step, 0);

        // Experiment two was not touched.
        let other = read_log(writer.connection(), two, 10).unwrap();
        assert_eq!(other.columns, strings(&["keep"]));
        assert_eq!(other.total_steps, 1);
    }

    #[test]
    fn unknown_experiments_are_reported() {
        let conn = memory_db();
        assert!(read_log(&conn, 42, 5).unwrap_err().contains("#42 does not exist"));
        // The foreign key stops a log for an experiment that was never created.
        assert!(LogWriter::init(conn, 42, strings(&["a"])).is_err());
    }

    #[test]
    fn experiments_are_listed_and_finished() {
        let conn = memory_db();
        let first = new_experiment(&conn, "alpha");
        let second = new_experiment(&conn, "beta");
        finish_experiment(&conn, first, "finished", Some(0)).unwrap();
        finish_experiment(&conn, second, "failed", Some(3)).unwrap();

        let mut writer = LogWriter::init(conn, second, strings(&["loss"])).unwrap();
        writer.log(&[0.1]).unwrap();

        let list = list_experiments(writer.connection(), 10).unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, second); // newest first
        assert_eq!(list[0].status, "failed");
        assert_eq!(list[0].exit_code, Some(3));
        assert_eq!(list[0].steps, 1);
        assert_eq!(list[0].columns, strings(&["loss"]));
        assert!(list[0].duration_s.is_some());
        assert_eq!(list[1].script_name.as_deref(), Some("alpha"));
        assert_eq!(list[1].config_name.as_deref(), Some("fast"));
        assert!(list[1].columns.is_empty());

        assert_eq!(list_experiments(writer.connection(), 1).unwrap().len(), 1);

        let text = format_experiments(&list);
        assert!(text.contains("#1") && text.contains("failed (exit 3)"));
        assert!(text.contains("beta") && text.contains("loss"));
    }

    #[test]
    fn running_experiments_have_no_duration() {
        let conn = memory_db();
        new_experiment(&conn, "live");
        let list = list_experiments(&conn, 5).unwrap();
        assert_eq!(list[0].status, STATUS_RUNNING);
        assert_eq!(list[0].duration_s, None);
    }

    #[test]
    fn values_and_tables_are_formatted() {
        assert_eq!(format_value(None), "-");
        assert_eq!(format_value(Some(0.0)), "0");
        assert_eq!(format_value(Some(0.5)), "0.5");
        assert_eq!(format_value(Some(2.0)), "2");
        assert_eq!(format_value(Some(-0.25)), "-0.25");
        assert_eq!(format_value(Some(0.0000123)), "1.2300e-5");
        assert_eq!(format_value(Some(123456789.0)), "1.2346e8");

        let table = render_table(
            &strings(&["a", "bb"]),
            &[strings(&["longer", "x"]), strings(&["y", "z"])],
        );
        assert_eq!(table, "a       bb\nlonger  x\ny       z");

        assert_eq!(format_duration(5.0), "5s");
        assert_eq!(format_duration(125.0), "2m05s");
        assert_eq!(format_duration(3725.0), "1h02m05s");
    }

    #[test]
    fn log_text_shows_columns_and_handles_an_empty_log() {
        let conn = memory_db();
        let id = new_experiment(&conn, "train");
        let empty = read_log(&conn, id, 5).unwrap();
        assert!(format_log(&empty).contains("has no log yet"));

        let mut writer = LogWriter::init(conn, id, strings(&["loss"])).unwrap();
        writer.log(&[0.5]).unwrap();
        let text = format_log(&read_log(writer.connection(), id, 5).unwrap());
        assert!(text.contains("Experiment #1: 1 step(s)"));
        assert!(text.contains("step") && text.contains("time_s") && text.contains("loss"));
        assert!(text.contains("0.5"));
    }

    #[test]
    fn time_starts_at_zero_with_the_first_log_call() {
        let conn = memory_db();
        let id = new_experiment(&conn, "train");
        // The experiment exists for a while before the first log() call.
        std::thread::sleep(std::time::Duration::from_millis(30));
        let mut writer = LogWriter::init(conn, id, strings(&["loss"])).unwrap();
        writer.log(&[1.0]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(40));
        writer.log(&[0.5]).unwrap();

        let table = read_log(writer.connection(), id, 10).unwrap();
        assert_eq!(table.rows[0].time_s, 0.0);
        assert!(table.rows[1].time_s >= 0.03, "time_s = {}", table.rows[1].time_s);
        assert!(table.rows[1].time_s < 5.0);

        // Restarting the log starts the clock again.
        let LogWriter { conn, .. } = writer;
        let mut writer = LogWriter::init(conn, id, strings(&["loss"])).unwrap();
        writer.log(&[2.0]).unwrap();
        assert_eq!(read_log(writer.connection(), id, 10).unwrap().rows[0].time_s, 0.0);
    }

    #[test]
    fn old_databases_are_upgraded_and_old_rows_get_a_time() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE experiments (
                 id INTEGER PRIMARY KEY AUTOINCREMENT, script_name TEXT, script_path TEXT,
                 config_name TEXT, args TEXT NOT NULL DEFAULT '[]',
                 status TEXT NOT NULL DEFAULT 'running', exit_code INTEGER,
                 started_ms INTEGER NOT NULL, finished_ms INTEGER);
             CREATE TABLE log_steps (
                 experiment_id INTEGER NOT NULL, step INTEGER NOT NULL, t_ms INTEGER NOT NULL,
                 PRIMARY KEY (experiment_id, step));
             INSERT INTO experiments (started_ms) VALUES (500);
             INSERT INTO log_steps VALUES (1, 0, 1000), (1, 1, 3500);
             CREATE VIEW log_view AS SELECT experiment_id, step FROM log_steps;",
        )
        .unwrap();

        init_schema(&conn).unwrap();
        let times: Vec<f64> = {
            let mut statement = conn
                .prepare("SELECT time_s FROM log_steps ORDER BY step")
                .unwrap();
            let rows = statement.query_map([], |row| row.get::<_, f64>(0)).unwrap();
            rows.map(|time| time.unwrap()).collect()
        };
        assert_eq!(times, vec![0.0, 2.5]);

        // The view was recreated with the new column, and a second call is a no-op.
        conn.prepare("SELECT time_s FROM log_view").unwrap();
        // Experiments that existed before get their number within their script.
        let seq: i64 = conn
            .query_row("SELECT script_seq FROM experiments", [], |row| row.get(0))
            .unwrap();
        assert_eq!(seq, 1);
        init_schema(&conn).unwrap();
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn quantities_cover_columns_time_and_step() {
        let conn = memory_db();
        let id = new_experiment(&conn, "train");
        assert_eq!(
            quantity_names(&conn, id).unwrap(),
            strings(&["time", "step"])
        );

        let mut writer = LogWriter::init(conn, id, strings(&["loss", "acc"])).unwrap();
        writer.log(&[0.5, 0.1]).unwrap();
        writer.log(&[f64::NAN, 0.2]).unwrap();
        writer.log(&[0.25, 0.3]).unwrap();
        let conn = writer.connection();

        assert_eq!(
            quantity_names(conn, id).unwrap(),
            strings(&["loss", "acc", "time", "step"])
        );
        // NaN steps are skipped.
        assert_eq!(read_quantity(conn, id, "loss", -1).unwrap(), vec![(0, 0.5), (2, 0.25)]);
        assert_eq!(read_quantity(conn, id, "acc", 0).unwrap(), vec![(1, 0.2), (2, 0.3)]);
        assert_eq!(
            read_quantity(conn, id, "step", -1).unwrap(),
            vec![(0, 0.0), (1, 1.0), (2, 2.0)]
        );
        let time = read_quantity(conn, id, "time", -1).unwrap();
        assert_eq!(time.len(), 3);
        assert_eq!(time[0], (0, 0.0));
        assert!(read_quantity(conn, id, "loss", i64::MAX).unwrap().is_empty());

        let error = read_quantity(conn, id, "nope", -1).unwrap_err();
        assert!(error.contains("no quantity 'nope'") && error.contains("loss, acc, time, step"));

        assert_eq!(max_step(conn, id).unwrap(), Some(2));
        assert_eq!(experiment_status(conn, id).unwrap().as_deref(), Some(STATUS_RUNNING));
        assert_eq!(experiment_status(conn, 99).unwrap(), None);
        assert_eq!(latest_experiment_id(conn).unwrap(), Some(id));
    }

    #[test]
    fn a_column_named_time_hides_the_builtin_time() {
        let conn = memory_db();
        let id = new_experiment(&conn, "train");
        let mut writer = LogWriter::init(conn, id, strings(&["time"])).unwrap();
        writer.log(&[42.0]).unwrap();
        assert_eq!(
            read_quantity(writer.connection(), id, "time", -1).unwrap(),
            vec![(0, 42.0)]
        );
        assert_eq!(quantity_names(writer.connection(), id).unwrap(), strings(&["time", "step"]));
    }

    #[test]
    fn empty_log_has_no_max_step_and_no_latest_experiment_without_experiments() {
        let conn = memory_db();
        assert_eq!(latest_experiment_id(&conn).unwrap(), None);
        let id = new_experiment(&conn, "train");
        assert_eq!(max_step(&conn, id).unwrap(), None);
        let error = read_quantity(&conn, id, "loss", -1).unwrap_err();
        assert!(error.contains("has not called init_log"));
    }

    #[test]
    fn experiment_ids_count_per_script() {
        let conn = memory_db();
        let make = |name: Option<&str>, path: Option<&str>| {
            create_experiment(
                &conn,
                &NewExperiment {
                    script_name: name,
                    script_path: path,
                    config_name: None,
                    args: &[],
                    status: STATUS_RUNNING,
                },
            )
            .unwrap()
        };
        let a1 = make(Some("a"), Some("/a.py"));
        let b1 = make(Some("b"), Some("/b.py"));
        let a2 = make(Some("a"), Some("/a.py"));
        // The same name at another path is another script.
        let a_elsewhere = make(Some("a"), Some("/other/a.py"));
        // Runs that were not started by the controller are numbered together.
        let hand1 = make(None, None);
        let hand2 = make(None, None);
        // Database IDs stay global...
        assert_eq!([a1, b1, a2, a_elsewhere, hand1, hand2], [1, 2, 3, 4, 5, 6]);

        // ...but the experiment IDs users see count per script.
        let seq = |id| experiment_seq(&conn, id).unwrap();
        assert_eq!(
            [seq(a1), seq(b1), seq(a2), seq(a_elsewhere), seq(hand1), seq(hand2)],
            [Some(1), Some(1), Some(2), Some(1), Some(1), Some(2)]
        );
        assert_eq!(seq(99), None);

        assert_eq!(resolve_experiment(&conn, "a", "/a.py", 2).unwrap(), Some(a2));
        assert_eq!(resolve_experiment(&conn, "b", "/b.py", 1).unwrap(), Some(b1));
        assert_eq!(resolve_experiment(&conn, "a", "/a.py", 3).unwrap(), None);
        assert_eq!(resolve_experiment(&conn, "b", "/a.py", 1).unwrap(), None);

        assert_eq!(latest_script_experiment_id(&conn, "a", "/a.py").unwrap(), Some(a2));
        assert_eq!(latest_script_experiment_id(&conn, "zzz", "/z.py").unwrap(), None);

        let list = list_experiments_for(&conn, Some(("a", "/a.py")), 10).unwrap();
        assert_eq!(list.iter().map(|e| e.id).collect::<Vec<_>>(), vec![a2, a1]);
        assert_eq!(list.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![2, 1]);
        assert_eq!(list_experiments(&conn, 10).unwrap().len(), 6);
        assert!(format_experiments(&list).contains("#2"));

        // The log of an experiment shows its per-script number.
        let table = read_log(&conn, a2, 5).unwrap();
        assert_eq!((table.experiment_id, table.seq), (a2, 2));
        assert!(format_log(&table).contains("Experiment #2"));
    }

    #[test]
    fn a_file_database_is_shared_between_connections() {
        let path = std::env::temp_dir().join(format!(
            "training_controller-logdb-{}-{}.db",
            std::process::id(),
            now_ms()
        ));
        let controller = open(&path).unwrap();
        let id = new_experiment(&controller, "shared");

        // The "script" opens the same file and logs.
        let script = open(&path).unwrap();
        let mut writer = LogWriter::init(script, id, strings(&["loss"])).unwrap();
        writer.log(&[0.75]).unwrap();

        // The controller sees the committed row and can finish the experiment.
        let table = read_log(&controller, id, 5).unwrap();
        assert_eq!(table.rows[0].values, vec![Some(0.75)]);
        finish_experiment(&controller, id, "finished", Some(0)).unwrap();

        drop(writer);
        drop(controller);
        for suffix in ["", "-wal", "-shm"] {
            let mut file = path.clone().into_os_string();
            file.push(suffix);
            let _ = fs::remove_file(file);
        }
    }
}
