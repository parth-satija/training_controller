mod checkpoint_cli;
mod dashboard;
mod graphs;
mod panels;
mod wsl;

use dashboard::{DashboardConfig, TrainingConfig, TrainingSource};
use graphs::{GraphSpec, Selection};
use panels::{PanelHub, RunStatus};
use wsl::{EnvKind, ScriptEnv};
use reedline_repl_rs::clap::{Arg, ArgAction, ArgMatches, Command};
use reedline_repl_rs::{Repl, Result};

use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::process::Stdio;
use training_controller::checkpoint::{self, Store};
use training_controller::{api, control, logdb};

// Shared state across commands
struct AppState {
    count: usize,
    panels: PanelHub,
    tc_json: TcJson,
    active_script_name: Option<String>,
    active_script_path: Option<String>,
    // The associate script (and its config) that `/test` runs. Associate
    // scripts belong to the active training script, so these are cleared
    // whenever another training script is selected.
    active_associate_name: Option<String>,
    active_associate_path: Option<String>,
    active_associate_config: Option<String>,
    // Runs started by /run_script and /resume that /pause can still reach.
    runs: Vec<TrackedRun>,
}

/// A run started by the controller, kept so `/pause` can find its control
/// directory (see control.rs).
#[derive(Debug, Clone)]
struct TrackedRun {
    run_id: u32,
    script_name: String,
    script_path: String,
    control_dir: PathBuf,
}

/// The flags a script was last started with by `/run_script`; `/resume` reuses them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LastRun {
    /// The saved config the run expanded, if any (shown in messages only).
    pub config: Option<String>,
    /// The complete argument list the script got.
    pub args: Vec<String>,
}

pub struct TcJson {
    path: PathBuf,
    document: Value,
}

impl TcJson {
    pub fn load() -> std::result::Result<Self, String> {
        let path = tc_json_path()?;
        let (document, initialize_file) = match fs::read_to_string(&path) {
            Ok(contents) => (
                serde_json::from_str(&contents)
                    .map_err(|error| format!("Failed to parse {}: {error}", path.display()))?,
                false,
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                (Value::Object(serde_json::Map::new()), true)
            }
            Err(error) => {
                return Err(format!("Failed to read {}: {error}", path.display()));
            }
        };

        let store = Self { path, document };
        if initialize_file {
            store.save()?;
        }

        Ok(store)
    }

    pub fn document(&self) -> &Value {
        &self.document
    }

    pub fn document_mut(&mut self) -> &mut Value {
        &mut self.document
    }

    pub fn get(&self, pointer: &str) -> Option<&Value> {
        self.document.pointer(pointer)
    }

    pub fn save(&self) -> std::result::Result<(), String> {
        let parent = self
            .path
            .parent()
            .ok_or_else(|| format!("Invalid tc.json path: {}", self.path.display()))?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create {}: {error}", parent.display()))?;

        let contents = serde_json::to_vec_pretty(&self.document)
            .map_err(|error| format!("Failed to serialize tc.json: {error}"))?;
        fs::write(&self.path, contents)
            .map_err(|error| format!("Failed to write {}: {error}", self.path.display()))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn scripts(&self) -> Option<&Vec<Value>> {
        self.document.get("scripts")?.as_array()
    }

    pub fn add_script(
        &mut self,
        name: &str,
        absolute_path: &str,
    ) -> std::result::Result<(), String> {
        if !Path::new(absolute_path).is_absolute() {
            return Err("Path must be absolute".into());
        }

        let scripts = self
            .document
            .as_object_mut()
            .ok_or_else(|| "tc.json must contain a JSON object".to_string())?
            .entry("scripts")
            .or_insert_with(|| Value::Array(Vec::new()));
        let scripts = scripts
            .as_array_mut()
            .ok_or_else(|| "The scripts entry in tc.json must be an array".to_string())?;

        scripts.push(serde_json::json!({
            "name": name,
            "path": absolute_path,
        }));

        self.save()
    }

    // -- Per-script flag configs ------------------------------------------
    //
    // Stored in tc.json as:
    //   { "scripts": [ { "name": ..., "path": ..., "configs": { "fast": ["--epochs", "5"] } } ] }
    // A script is identified by its name and path together.

    fn script_index(&self, script_name: &str, script_path: &str) -> Option<usize> {
        self.scripts()?.iter().position(|script| {
            script.get("name").and_then(|v| v.as_str()) == Some(script_name)
                && script.get("path").and_then(|v| v.as_str()) == Some(script_path)
        })
    }

    /// All configs of a script as `(config name, flags)`, sorted by name.
    pub fn configs(&self, script_name: &str, script_path: &str) -> Vec<(String, Vec<String>)> {
        let Some(index) = self.script_index(script_name, script_path) else {
            return Vec::new();
        };
        let Some(map) = self
            .scripts()
            .and_then(|scripts| scripts[index].get("configs"))
            .and_then(|configs| configs.as_object())
        else {
            return Vec::new();
        };
        map.iter()
            .map(|(config_name, flags)| {
                let flags = flags
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|item| item.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                (config_name.clone(), flags)
            })
            .collect()
    }

    pub fn get_config(
        &self,
        script_name: &str,
        script_path: &str,
        config_name: &str,
    ) -> Option<Vec<String>> {
        self.configs(script_name, script_path)
            .into_iter()
            .find(|(candidate, _)| candidate == config_name)
            .map(|(_, flags)| flags)
    }

    fn configs_mut(
        &mut self,
        script_name: &str,
        script_path: &str,
    ) -> std::result::Result<&mut serde_json::Map<String, Value>, String> {
        let index = self
            .script_index(script_name, script_path)
            .ok_or_else(|| format!("Script '{script_name}' is not registered in tc.json"))?;
        let script = self
            .document
            .get_mut("scripts")
            .and_then(|scripts| scripts.as_array_mut())
            .and_then(|scripts| scripts.get_mut(index))
            .and_then(|script| script.as_object_mut())
            .ok_or_else(|| "Invalid script entry in tc.json".to_string())?;
        script
            .entry("configs")
            .or_insert_with(|| Value::Object(serde_json::Map::new()))
            .as_object_mut()
            .ok_or_else(|| "The configs entry in tc.json must be an object".to_string())
    }

    fn flags_to_json(flags: &[String]) -> Value {
        Value::Array(flags.iter().cloned().map(Value::String).collect())
    }

    /// Creates a new config; fails if the name is already used.
    pub fn add_config(
        &mut self,
        script_name: &str,
        script_path: &str,
        config_name: &str,
        flags: &[String],
    ) -> std::result::Result<(), String> {
        validate_config_name(config_name)?;
        let configs = self.configs_mut(script_name, script_path)?;
        if configs.contains_key(config_name) {
            return Err(format!(
                "Config '{config_name}' already exists. Use /edit_config to change it."
            ));
        }
        configs.insert(config_name.to_string(), Self::flags_to_json(flags));
        self.save()
    }

    /// Replaces the flags of an existing config.
    pub fn edit_config(
        &mut self,
        script_name: &str,
        script_path: &str,
        config_name: &str,
        flags: &[String],
    ) -> std::result::Result<(), String> {
        let configs = self.configs_mut(script_name, script_path)?;
        if !configs.contains_key(config_name) {
            return Err(format!("Config '{config_name}' not found."));
        }
        configs.insert(config_name.to_string(), Self::flags_to_json(flags));
        self.save()
    }

    pub fn remove_config(
        &mut self,
        script_name: &str,
        script_path: &str,
        config_name: &str,
    ) -> std::result::Result<(), String> {
        let configs = self.configs_mut(script_name, script_path)?;
        if configs.remove(config_name).is_none() {
            return Err(format!("Config '{config_name}' not found."));
        }
        self.save()
    }

    pub fn rename_config(
        &mut self,
        script_name: &str,
        script_path: &str,
        old_name: &str,
        new_name: &str,
    ) -> std::result::Result<(), String> {
        validate_config_name(new_name)?;
        let configs = self.configs_mut(script_name, script_path)?;
        if !configs.contains_key(old_name) {
            return Err(format!("Config '{old_name}' not found."));
        }
        if configs.contains_key(new_name) {
            return Err(format!("Config '{new_name}' already exists."));
        }
        let flags = configs.remove(old_name).unwrap_or_else(|| Value::Array(Vec::new()));
        configs.insert(new_name.to_string(), flags);
        self.save()
    }

    // -- Last run ------------------------------------------------------------
    //
    // Stored in the script's entry, so `/resume` can start the script with the
    // same flags even after the controller was restarted:
    //   { "scripts": [ { "name": ..., "path": ...,
    //       "last_run": { "config": "fast", "args": ["--epochs", "5"] } } ] }
    // `args` is the complete list the script was started with (a config name
    // is already expanded), so editing the config later does not change what
    // `/resume` runs.

    /// The flags of the script's most recent `/run_script`, if it ever ran.
    pub fn last_run(&self, script_name: &str, script_path: &str) -> Option<LastRun> {
        let index = self.script_index(script_name, script_path)?;
        let last = self.scripts()?.get(index)?.get("last_run")?;
        let args = last
            .get("args")?
            .as_array()?
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect();
        let config = last
            .get("config")
            .and_then(|value| value.as_str())
            .map(str::to_string);
        Some(LastRun { config, args })
    }

    /// Remembers the flags a script was just started with.
    pub fn set_last_run(
        &mut self,
        script_name: &str,
        script_path: &str,
        config: Option<&str>,
        args: &[String],
    ) -> std::result::Result<(), String> {
        let index = self
            .script_index(script_name, script_path)
            .ok_or_else(|| format!("Script '{script_name}' is not registered in tc.json"))?;
        let script = self
            .document
            .get_mut("scripts")
            .and_then(|scripts| scripts.as_array_mut())
            .and_then(|scripts| scripts.get_mut(index))
            .and_then(|script| script.as_object_mut())
            .ok_or_else(|| "Invalid script entry in tc.json".to_string())?;
        script.insert(
            "last_run".to_string(),
            serde_json::json!({ "config": config, "args": args }),
        );
        self.save()
    }

    // -- Graph configs ------------------------------------------------------
    //
    // Stored in tc.json as:
    //   { "graph_configs": { "train": [ {"y": "loss", "x": "time"}, ... ] } }
    // They are global and name quantities, not experiments, so one config can
    // be shown for any experiment: `/graph train 3`.

    /// All graph configs as `(name, graphs)`, sorted by name.
    pub fn graph_configs(&self) -> Vec<(String, Vec<GraphSpec>)> {
        let Some(map) = self
            .document
            .get("graph_configs")
            .and_then(|configs| configs.as_object())
        else {
            return Vec::new();
        };
        map.iter()
            .filter_map(|(name, value)| {
                serde_json::from_value::<Vec<GraphSpec>>(value.clone())
                    .ok()
                    .map(|specs| (name.clone(), specs))
            })
            .collect()
    }

    pub fn get_graph_config(&self, name: &str) -> Option<Vec<GraphSpec>> {
        self.graph_configs()
            .into_iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, specs)| specs)
    }

    fn graph_configs_mut(
        &mut self,
    ) -> std::result::Result<&mut serde_json::Map<String, Value>, String> {
        self.document
            .as_object_mut()
            .ok_or_else(|| "tc.json must contain a JSON object".to_string())?
            .entry("graph_configs")
            .or_insert_with(|| Value::Object(serde_json::Map::new()))
            .as_object_mut()
            .ok_or_else(|| "The graph_configs entry in tc.json must be an object".to_string())
    }

    fn graph_specs_to_json(specs: &[GraphSpec]) -> std::result::Result<Value, String> {
        if specs.is_empty() {
            return Err("A graph config needs at least one graph.".to_string());
        }
        serde_json::to_value(specs)
            .map_err(|error| format!("Failed to encode the graph config: {error}"))
    }

    /// Creates a new graph config; fails if the name is already used.
    pub fn add_graph_config(
        &mut self,
        name: &str,
        specs: &[GraphSpec],
    ) -> std::result::Result<(), String> {
        validate_graph_config_name(name)?;
        let value = Self::graph_specs_to_json(specs)?;
        let configs = self.graph_configs_mut()?;
        if configs.contains_key(name) {
            return Err(format!(
                "Graph config '{name}' already exists. Use /edit_graph_config to change it."
            ));
        }
        configs.insert(name.to_string(), value);
        self.save()
    }

    /// Replaces the graphs of an existing graph config.
    pub fn edit_graph_config(
        &mut self,
        name: &str,
        specs: &[GraphSpec],
    ) -> std::result::Result<(), String> {
        let value = Self::graph_specs_to_json(specs)?;
        let configs = self.graph_configs_mut()?;
        if !configs.contains_key(name) {
            return Err(format!("Graph config '{name}' not found."));
        }
        configs.insert(name.to_string(), value);
        self.save()
    }

    pub fn remove_graph_config(&mut self, name: &str) -> std::result::Result<(), String> {
        let configs = self.graph_configs_mut()?;
        if configs.remove(name).is_none() {
            return Err(format!("Graph config '{name}' not found."));
        }
        self.save()
    }

    // -- Associate scripts ----------------------------------------------------
    //
    // Scripts that go with one training script (evaluation, inference, data
    // checks, ...). Each belongs to the training script it was added to and is
    // stored in that script's entry in tc.json:
    //   { "scripts": [ { "name": ..., "path": ..., "associates": [
    //       { "name": ..., "path": ..., "configs": { "quick": ["--n", "5"] } } ] } ] }
    // A training script is identified by name and path together, an associate
    // script by its name within that training script. Removing a training
    // script removes its associate scripts and their configs.

    pub fn associate_scripts(&self, main_name: &str, main_path: &str) -> Option<&Vec<Value>> {
        let index = self.script_index(main_name, main_path)?;
        self.scripts()?.get(index)?.get("associates")?.as_array()
    }

    fn associate_index(&self, main_name: &str, main_path: &str, name: &str) -> Option<usize> {
        self.associate_scripts(main_name, main_path)?
            .iter()
            .position(|script| script.get("name").and_then(|v| v.as_str()) == Some(name))
    }

    fn associates_mut(
        &mut self,
        main_name: &str,
        main_path: &str,
    ) -> std::result::Result<&mut Vec<Value>, String> {
        let index = self
            .script_index(main_name, main_path)
            .ok_or_else(|| format!("Script '{main_name}' is not registered in tc.json"))?;
        let script = self
            .document
            .get_mut("scripts")
            .and_then(|scripts| scripts.as_array_mut())
            .and_then(|scripts| scripts.get_mut(index))
            .and_then(|script| script.as_object_mut())
            .ok_or_else(|| "Invalid script entry in tc.json".to_string())?;
        script
            .entry("associates")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| "The associates entry in tc.json must be an array".to_string())
    }

    pub fn add_associate_script(
        &mut self,
        main_name: &str,
        main_path: &str,
        name: &str,
        absolute_path: &str,
    ) -> std::result::Result<(), String> {
        if name.trim().is_empty() {
            return Err("Name must not be empty".into());
        }
        if !Path::new(absolute_path).is_absolute() {
            return Err("Path must be absolute".into());
        }
        if self.associate_index(main_name, main_path, name).is_some() {
            return Err(format!(
                "'{main_name}' already has an associate script named '{name}'."
            ));
        }

        let associates = self.associates_mut(main_name, main_path)?;
        associates.push(serde_json::json!({
            "name": name,
            "path": absolute_path,
        }));

        self.save()
    }

    pub fn remove_associate_script(
        &mut self,
        main_name: &str,
        main_path: &str,
        index: usize,
    ) -> std::result::Result<(String, String), String> {
        let associates = self.associates_mut(main_name, main_path)?;

        if index >= associates.len() {
            return Err(format!("Index {index} out of bounds (0..{}).", associates.len()));
        }

        let removed = associates.remove(index);
        let name = removed.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
        let path = removed.get("path").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();

        self.save()?;
        Ok((name, path))
    }

    /// All configs of an associate script as `(config name, flags)`, sorted by name.
    pub fn associate_configs(
        &self,
        main_name: &str,
        main_path: &str,
        name: &str,
    ) -> Vec<(String, Vec<String>)> {
        let Some(index) = self.associate_index(main_name, main_path, name) else {
            return Vec::new();
        };
        flag_configs(
            self.associate_scripts(main_name, main_path)
                .and_then(|scripts| scripts.get(index))
                .and_then(|script| script.get("configs")),
        )
    }

    pub fn get_associate_config(
        &self,
        main_name: &str,
        main_path: &str,
        name: &str,
        config_name: &str,
    ) -> Option<Vec<String>> {
        self.associate_configs(main_name, main_path, name)
            .into_iter()
            .find(|(candidate, _)| candidate == config_name)
            .map(|(_, flags)| flags)
    }

    fn associate_configs_mut(
        &mut self,
        main_name: &str,
        main_path: &str,
        name: &str,
    ) -> std::result::Result<&mut serde_json::Map<String, Value>, String> {
        let index = self
            .associate_index(main_name, main_path, name)
            .ok_or_else(|| {
                format!("Associate script '{name}' is not registered for '{main_name}'")
            })?;
        let script = self
            .associates_mut(main_name, main_path)?
            .get_mut(index)
            .and_then(|script| script.as_object_mut())
            .ok_or_else(|| "Invalid associate script entry in tc.json".to_string())?;
        script
            .entry("configs")
            .or_insert_with(|| Value::Object(serde_json::Map::new()))
            .as_object_mut()
            .ok_or_else(|| "The configs entry in tc.json must be an object".to_string())
    }

    /// Creates a new associate config; fails if the name is already used.
    pub fn add_associate_config(
        &mut self,
        main_name: &str,
        main_path: &str,
        name: &str,
        config_name: &str,
        flags: &[String],
    ) -> std::result::Result<(), String> {
        validate_config_name(config_name)?;
        let configs = self.associate_configs_mut(main_name, main_path, name)?;
        if configs.contains_key(config_name) {
            return Err(format!(
                "Config '{config_name}' already exists. Use /edit_associate_config to change it."
            ));
        }
        configs.insert(config_name.to_string(), Self::flags_to_json(flags));
        self.save()
    }

    /// Replaces the flags of an existing associate config.
    pub fn edit_associate_config(
        &mut self,
        main_name: &str,
        main_path: &str,
        name: &str,
        config_name: &str,
        flags: &[String],
    ) -> std::result::Result<(), String> {
        let configs = self.associate_configs_mut(main_name, main_path, name)?;
        if !configs.contains_key(config_name) {
            return Err(format!("Config '{config_name}' not found."));
        }
        configs.insert(config_name.to_string(), Self::flags_to_json(flags));
        self.save()
    }

    pub fn remove_associate_config(
        &mut self,
        main_name: &str,
        main_path: &str,
        name: &str,
        config_name: &str,
    ) -> std::result::Result<(), String> {
        let configs = self.associate_configs_mut(main_name, main_path, name)?;
        if configs.remove(config_name).is_none() {
            return Err(format!("Config '{config_name}' not found."));
        }
        self.save()
    }

    pub fn rename_associate_config(
        &mut self,
        main_name: &str,
        main_path: &str,
        name: &str,
        old_name: &str,
        new_name: &str,
    ) -> std::result::Result<(), String> {
        validate_config_name(new_name)?;
        let configs = self.associate_configs_mut(main_name, main_path, name)?;
        if !configs.contains_key(old_name) {
            return Err(format!("Config '{old_name}' not found."));
        }
        if configs.contains_key(new_name) {
            return Err(format!("Config '{new_name}' already exists."));
        }
        let flags = configs.remove(old_name).unwrap_or_else(|| Value::Array(Vec::new()));
        configs.insert(new_name.to_string(), flags);
        self.save()
    }

    pub fn remove_script(&mut self, index: usize) -> std::result::Result<(String, String), String> {
        let scripts = self
            .document
            .as_object_mut()
            .ok_or_else(|| "tc.json must contain a JSON object".to_string())?
            .entry("scripts")
            .or_insert_with(|| Value::Array(Vec::new()));
        let scripts = scripts
            .as_array_mut()
            .ok_or_else(|| "The scripts entry in tc.json must be an array".to_string())?;

        if index >= scripts.len() {
            return Err(format!("Index {index} out of bounds (0..{}).", scripts.len()));
        }

        let removed = scripts.remove(index);
        let name = removed.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
        let path = removed.get("path").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();

        self.save()?;
        Ok((name, path))
    }
}


/// Reads a `{ "name": ["--flag", "value"], ... }` object as `(name, flags)`
/// pairs, sorted by name. Anything that is not an object gives no configs.
fn flag_configs(value: Option<&Value>) -> Vec<(String, Vec<String>)> {
    let Some(map) = value.and_then(|value| value.as_object()) else {
        return Vec::new();
    };
    map.iter()
        .map(|(config_name, flags)| {
            let flags = flags
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            (config_name.clone(), flags)
        })
        .collect()
}

fn tc_json_path() -> std::result::Result<PathBuf, String> {
    #[cfg(windows)]
    let app_data_dir = std::env::var_os("APPDATA").map(PathBuf::from).or_else(|| {
        std::env::var_os("USERPROFILE")
            .map(PathBuf::from)
            .map(|home| home.join("AppData").join("Roaming"))
    });

    #[cfg(not(windows))]
    let app_data_dir = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".config"))
        });

    app_data_dir
        .map(|directory| directory.join("training_controller").join("tc.json"))
        .ok_or_else(|| "Could not determine the application-data directory".to_string())
}

#[derive(Debug, Deserialize)]
pub struct TerminalArgument {
    pub flags: Vec<String>,
    pub takes_value: bool,
    pub required: bool,
    pub nargs: Option<String>,
    pub action: Option<String>,
    pub value_type: Option<String>,
}

pub fn get_terminal_arguments(
    absolute_path: &str,
) -> std::result::Result<Vec<TerminalArgument>, String> {
    let path = Path::new(absolute_path);

    if !path.is_absolute() {
        return Err("Path must be absolute".into());
    }

    if !path.is_file() {
        return Err("File does not exist".into());
    }

    if path.extension().and_then(|e| e.to_str()) != Some("py") {
        return Err("File must have a .py extension".into());
    }

    let python_code = r#"
import ast
import json
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    tree = ast.parse(f.read())

results = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue

    func = node.func

    # Match parser.add_argument(...)
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "add_argument"
    ):
        continue

    flags = []

    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            flags.append(arg.value)

    if not flags:
        continue

    kwargs = {}

    for kw in node.keywords:
        if isinstance(kw.value, ast.Constant):
            kwargs[kw.arg] = kw.value.value

    action = kwargs.get("action")
    nargs = kwargs.get("nargs")

    takes_value = (
        action not in ("store_true", "store_false", "count", "help", "version")
        and nargs != 0
    )

    results.append({
        "flags": flags,
        "takes_value": takes_value,
        "required": kwargs.get("required", False),
        "nargs": str(nargs) if nargs is not None else None,
        "action": action,
        "value_type": (
            kwargs.get("type")
            if isinstance(kwargs.get("type"), str)
            else None
        )
    })

print(json.dumps(results))
"#;

    let output = ProcessCommand::new("python")
        .arg("-c")
        .arg(python_code)
        .arg(path)
        .output()
        .map_err(|e| format!("Failed to execute Python: {}", e))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }

    let stdout = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;

    serde_json::from_str(&stdout).map_err(|e| format!("JSON parsing failed: {}", e))
}



// ---------------------------------------------------------------------------
// Running scripts with the `training_controller` Python module available
// ---------------------------------------------------------------------------

const PY_MODULE_NAME: &str = "training_controller";

/// Name Python expects for an extension module on this platform.
fn python_module_file_name() -> String {
    if cfg!(windows) {
        format!("{PY_MODULE_NAME}.pyd")
    } else {
        format!("{PY_MODULE_NAME}.so")
    }
}

/// Name Cargo gives the compiled cdylib on this platform.
fn built_library_file_name() -> String {
    if cfg!(windows) {
        format!("{PY_MODULE_NAME}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{PY_MODULE_NAME}.dylib")
    } else {
        format!("lib{PY_MODULE_NAME}.so")
    }
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|window| window == needle)
}

/// A cdylib built without `--features python` has no `PyInit_...` symbol and
/// cannot be imported, so check for it before copying.
fn exports_python_module(path: &Path) -> bool {
    let needle = format!("PyInit_{PY_MODULE_NAME}");
    fs::read(path)
        .map(|bytes| contains_bytes(&bytes, needle.as_bytes()))
        .unwrap_or(false)
}

fn modified_time(path: &Path) -> Option<std::time::SystemTime> {
    fs::metadata(path).ok()?.modified().ok()
}

/// Compiled libraries near this executable or the working directory, newest first.
fn built_library_candidates() -> Vec<PathBuf> {
    let file_name = built_library_file_name();
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            dirs.push(exe_dir.to_path_buf());
            if let Some(parent) = exe_dir.parent() {
                dirs.push(parent.to_path_buf());
                dirs.push(parent.join("debug"));
                dirs.push(parent.join("release"));
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("target").join("debug"));
        dirs.push(cwd.join("target").join("release"));
    }

    let mut candidates: Vec<PathBuf> = dirs
        .into_iter()
        .map(|dir| dir.join(&file_name))
        .filter(|path| path.is_file())
        .collect();
    candidates.sort();
    candidates.dedup();
    candidates.sort_by_key(|path| std::cmp::Reverse(modified_time(path)));
    candidates
}

/// Makes `import training_controller` resolvable by copying the built extension
/// into `module_dir` under the file name Python expects. That directory is then
/// put on `PYTHONPATH`, so no virtualenv or `pip install` is needed.
fn prepare_python_module(module_dir: &Path) -> std::result::Result<(), String> {
    install_python_module(
        module_dir,
        &python_module_file_name(),
        built_library_candidates(),
        "Run `cargo build --features python` and try again.",
    )
}

/// Copies the newest usable build among `candidates` to `module_dir/file_name`
/// (unless that copy is already current). `build_hint` tells how to build the
/// extension when there is none.
fn install_python_module(
    module_dir: &Path,
    file_name: &str,
    candidates: Vec<PathBuf>,
    build_hint: &str,
) -> std::result::Result<(), String> {
    let destination = module_dir.join(file_name);

    let up_to_date = match (candidates.first(), modified_time(&destination)) {
        (Some(newest), Some(installed)) => modified_time(newest).is_some_and(|time| time <= installed),
        (None, Some(_)) => true,
        _ => false,
    };
    if up_to_date {
        return Ok(());
    }

    let Some(source) = candidates.into_iter().find(|path| exports_python_module(path)) else {
        // Keep using a previously installed copy if there is one.
        return if destination.exists() {
            Ok(())
        } else {
            Err(format!(
                "The `{PY_MODULE_NAME}` Python module has not been built. {build_hint}"
            ))
        };
    };

    fs::create_dir_all(module_dir)
        .map_err(|error| format!("Failed to create {}: {error}", module_dir.display()))?;
    match fs::copy(&source, &destination) {
        Ok(_) => Ok(()),
        // Typically the old copy is loaded by a running Python process; keep it.
        Err(_) if destination.exists() => Ok(()),
        Err(error) => Err(format!(
            "Failed to copy {} to {}: {error}",
            source.display(),
            destination.display()
        )),
    }
}

/// `module_dir` followed by whatever `PYTHONPATH` already contained.
fn python_path_with(
    module_dir: &Path,
    existing: Option<std::ffi::OsString>,
) -> std::result::Result<std::ffi::OsString, String> {
    let existing = existing.unwrap_or_default();
    let entries = std::iter::once(module_dir.to_path_buf()).chain(
        std::env::split_paths(&existing).filter(|path| !path.as_os_str().is_empty()),
    );
    std::env::join_paths(entries).map_err(|error| format!("Failed to build PYTHONPATH: {error}"))
}

/// Linux builds of the extension for WSL2, newest first. They are built inside
/// WSL into `target-wsl` (see `wsl::BUILD_COMMAND`), so they never mix with the
/// Windows build in `target`.
fn wsl_library_candidates() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        // <project>/target/<profile>/training_controller.exe
        if let Some(project) = exe.parent().and_then(Path::parent).and_then(Path::parent) {
            roots.push(project.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    for root in &roots {
        for profile in ["debug", "release"] {
            let path = root
                .join(wsl::TARGET_DIR_NAME)
                .join(profile)
                .join(wsl::BUILT_LIB_FILE_NAME);
            if path.is_file() {
                candidates.push(path);
            }
        }
    }
    candidates.sort();
    candidates.dedup();
    candidates.sort_by_key(|path| std::cmp::Reverse(modified_time(path)));
    candidates
}

/// Like `prepare_python_module`, for the Linux extension that scripts import in WSL2.
fn prepare_wsl_module(module_dir: &Path) -> std::result::Result<(), String> {
    install_python_module(
        module_dir,
        wsl::MODULE_FILE_NAME,
        wsl_library_candidates(),
        &format!(
            "Build the Linux extension inside WSL (in the project folder, from a WSL shell): `{}`, then try again.",
            wsl::BUILD_COMMAND
        ),
    )
}

/// Where a script runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunTarget {
    /// The Python on this system's PATH.
    Native,
    /// The Python inside WSL2, started through `wsl.exe` (see wsl.rs).
    Wsl,
}

impl RunTarget {
    /// Added to "started" / "resumed" messages.
    fn place_note(self) -> &'static str {
        match self {
            RunTarget::Native => "",
            RunTarget::Wsl => " in WSL2",
        }
    }
}

/// The programs to try, in order, to start a script.
fn interpreters(target: RunTarget) -> &'static [&'static str] {
    match target {
        RunTarget::Native => python_interpreters(),
        RunTarget::Wsl => &[wsl::WSL_EXE],
    }
}

/// The process for one start attempt: the interpreter (or `wsl.exe`), the
/// script and its flags, the environment, and captured output.
fn script_command(
    target: RunTarget,
    interpreter: &str,
    script: &Path,
    script_args: &[String],
    env: &ScriptEnv,
) -> ProcessCommand {
    let mut command = match target {
        RunTarget::Native => {
            let mut command = ProcessCommand::new(interpreter);
            command.arg("-u").arg(script).args(script_args);
            env.apply_native(&mut command);
            if let Some(directory) = script.parent() {
                command.current_dir(directory);
            }
            command
        }
        RunTarget::Wsl => {
            let mut command = wsl::script_command(script, script_args);
            // Activate the venv next to the script: its Python is already the
            // interpreter, VIRTUAL_ENV makes tools the script starts see it too.
            let mut env = env.clone();
            if let Some(venv) = wsl::venv_for(script) {
                env.set("VIRTUAL_ENV", &venv, EnvKind::Path);
            }
            env.apply_wsl(&mut command);
            command
        }
    };
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn missing_runtime_message(target: RunTarget, not_found: &[&str]) -> String {
    match target {
        RunTarget::Native => format!(
            "Error: No Python interpreter found (tried {}). Install Python and add it to PATH.",
            not_found.join(", ")
        ),
        RunTarget::Wsl => format!(
            "Error: {} was not found. Install WSL2 first (`wsl --install` in an administrator terminal).",
            wsl::WSL_EXE
        ),
    }
}

fn python_interpreters() -> &'static [&'static str] {
    if cfg!(windows) {
        &["python", "py"]
    } else {
        &["python3", "python"]
    }
}

fn cmd_run_script(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    run_script_on(args, state, RunTarget::Native)
}

/// `/run_wsl`: `/run_script`, but the script runs with the Python inside WSL2.
fn cmd_run_wsl(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    run_script_on(args, state, RunTarget::Wsl)
}

fn run_script_on(
    args: ArgMatches,
    state: &mut AppState,
    target: RunTarget,
) -> Result<Option<String>> {
    let (Some(name), Some(script_path)) = (
        state.active_script_name.clone(),
        state.active_script_path.clone(),
    ) else {
        return Ok(Some(
            "Error: No active script. Use /select_script or /open_script first.".to_string(),
        ));
    };

    if !Path::new(&script_path).is_file() {
        return Ok(Some(format!("Error: Script file not found: {script_path}")));
    }

    let typed_args = collect_args(&args);
    let configs = state.tc_json.configs(&name, &script_path);
    let (script_args, used_config) = resolve_run_args(&configs, typed_args);
    if let Some(config) = &used_config {
        println!("Using config '{config}': {}", format_args_display(&script_args));
    }

    launch_script(
        state,
        &name,
        &script_path,
        script_args,
        used_config,
        LaunchKind::Fresh,
        target,
    )
}

/// Why a script is being started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LaunchKind {
    /// `/run_script`: the flags are remembered for `/resume`.
    Fresh,
    /// `/resume`: the flags come from the last run and are not overwritten.
    Resume,
}

/// Starts a script in the background with the given flags: sets up the
/// `training_controller` module, an experiment, the checkpoint folder and the
/// control directory for `/pause`, and captures the output into panels.
fn launch_script(
    state: &mut AppState,
    name: &str,
    script_path: &str,
    script_args: Vec<String>,
    used_config: Option<String>,
    kind: LaunchKind,
    target: RunTarget,
) -> Result<Option<String>> {
    let name = name.to_string();
    let script_path = script_path.to_string();
    let script = Path::new(&script_path);
    if !script.is_file() {
        return Ok(Some(format!("Error: Script file not found: {script_path}")));
    }

    let python_path = match python_path_for(state, target) {
        Ok(path) => path,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };

    // Every run is an experiment with its own ID; the script finds the
    // database and the ID in its environment (see `logdb.rs`).
    let db_path = log_db_path(state);
    let experiment_id = db_path.as_deref().and_then(|db| {
        start_experiment(db, &name, &script_path, used_config.as_deref(), &script_args)
    });
    let experiment_note = match experiment_id {
        Some(id) => format!(", experiment #{id}"),
        None => String::new(),
    };

    // A private directory for the pause protocol of this run (see control.rs).
    let control_dir = match control_dir_for(state, &name).and_then(|dir| {
        control::prepare(&dir)?;
        Ok(dir)
    }) {
        Ok(dir) => Some(dir),
        Err(error) => {
            println!("Warning: {error}");
            println!("/pause will not work for this run.");
            None
        }
    };

    // One environment for every start attempt. Paths are marked as paths so
    // that WSL can translate them (see wsl.rs); a native run sets them as they are.
    let mut env = ScriptEnv::new();
    env.set("PYTHONPATH", &python_path, EnvKind::PathList);
    env.set("PYTHONUNBUFFERED", "1", EnvKind::Plain);
    env.set("PYTHONIOENCODING", "utf-8", EnvKind::Plain);
    // Where tc.save_checkpoint() stores this script's checkpoints (see checkpoint.rs).
    if let Ok(directory) = checkpoint_dir_for(state, &name) {
        env.set(checkpoint::DIR_ENV, directory, EnvKind::Path);
    }
    // Where /pause leaves its request for tc.register_pause() (see control.rs).
    if let Some(directory) = &control_dir {
        env.set(control::DIR_ENV, directory, EnvKind::Path);
    }
    // Where init_training_dashboard() writes; `/dashboard training` reads the same file.
    if let Some(events_file) = api::training_file_for_script(script) {
        env.set(api::TRAINING_FILE_ENV, events_file, EnvKind::Path);
    }
    if let (Some(db), Some(id)) = (&db_path, experiment_id) {
        env.set(logdb::DB_FILE_ENV, db, EnvKind::Path);
        env.set(logdb::EXPERIMENT_ID_ENV, id.to_string(), EnvKind::Plain);
    }

    let mut not_found: Vec<&str> = Vec::new();
    for interpreter in interpreters(target) {
        let mut command = script_command(target, interpreter, script, &script_args, &env);

        // The script runs in the background; its output is captured into
        // panels (see /panel) so the REPL stays free.
        match command.spawn() {
            Ok(child) => {
                let mut label = match &used_config {
                    Some(config) => format!("{name} ({config})"),
                    None => name.clone(),
                };
                if kind == LaunchKind::Resume {
                    label.push_str(" [resume]");
                }
                if target == RunTarget::Wsl {
                    label.push_str(" [wsl]");
                }
                let exit_db = db_path.clone();
                let exit_dir = control_dir.clone();
                let run_id =
                    panels::attach_child_with(&state.panels, &label, child, move |status| {
                        if let (Some(db), Some(id)) = (&exit_db, experiment_id) {
                            end_experiment(db, id, status);
                        }
                        if let Some(dir) = &exit_dir {
                            control::cleanup(dir);
                        }
                    });

                if let Some(dir) = control_dir.clone() {
                    track_run(
                        state,
                        TrackedRun {
                            run_id,
                            script_name: name.clone(),
                            script_path: script_path.clone(),
                            control_dir: dir,
                        },
                    );
                }
                if kind == LaunchKind::Fresh {
                    if let Err(error) = state.tc_json.set_last_run(
                        &name,
                        &script_path,
                        used_config.as_deref(),
                        &script_args,
                    ) {
                        println!("Warning: could not remember the flags for /resume: {error}");
                    }
                }

                let verb = match kind {
                    LaunchKind::Fresh => "started",
                    LaunchKind::Resume => "resumed",
                };
                let place = target.place_note();
                return Ok(Some(format!(
                    "Script '{name}' {verb}{place} (run #{run_id}{experiment_note}). \
                     View its output with /panel output, /panel error, /panel log or /panel other; \
                     list all panels with /show_active_panels."
                )));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                not_found.push(interpreter);
            }
            Err(error) => {
                abort_experiment(db_path.as_deref(), experiment_id);
                if let Some(dir) = &control_dir {
                    control::cleanup(dir);
                }
                return Ok(Some(format!("Error: Failed to run {interpreter}: {error}")));
            }
        }
    }

    abort_experiment(db_path.as_deref(), experiment_id);
    if let Some(dir) = &control_dir {
        control::cleanup(dir);
    }
    Ok(Some(missing_runtime_message(target, &not_found)))
}

// ---------------------------------------------------------------------------
// /pause and /resume
// ---------------------------------------------------------------------------
//
// `/pause` asks the running script to call the pause method it registered with
// `tc.register_pause(fn)` (the request travels through a control directory, see
// control.rs). Stopping is up to that method; when the script then exits with
// code 0 the run is recorded as paused. `/resume` starts the script again with
// the flags of its last `/run_script` (kept in tc.json, so they survive
// restarting the controller) plus `--resume`; the script is expected to pick
// up from its latest checkpoint (`tc.latest_checkpoint()`).

/// The flag `/resume` adds to the script's arguments.
const RESUME_FLAG: &str = "--resume";

/// The arguments `/resume` starts a script with: those of the last run plus
/// `--resume` (never twice).
fn resume_args(last: &[String]) -> Vec<String> {
    let mut args = last.to_vec();
    if !args.iter().any(|arg| arg == RESUME_FLAG) {
        args.push(RESUME_FLAG.to_string());
    }
    args
}

/// Every run gets its own control directory next to tc.json.
fn control_dir_for(state: &AppState, script_name: &str) -> std::result::Result<PathBuf, String> {
    let parent = state
        .tc_json
        .path()
        .parent()
        .ok_or_else(|| "Invalid tc.json path.".to_string())?;
    Ok(parent
        .join("control")
        .join(checkpoint::dir_name_for(script_name))
        .join(format!("run-{}", api::now_ms())))
}

/// Remembers a run for `/pause`, forgetting the ones that have ended.
fn track_run(state: &mut AppState, run: TrackedRun) {
    let panels = state.panels.clone();
    state.runs.retain(|tracked| {
        panels.with(|store| store.run_status(tracked.run_id)) == Some(RunStatus::Running)
    });
    state.runs.push(run);
}

/// The newest run of this script that is still running.
fn running_run(state: &AppState, script_name: &str, script_path: &str) -> Option<TrackedRun> {
    state
        .runs
        .iter()
        .rev()
        .find(|run| {
            run.script_name == script_name
                && run.script_path == script_path
                && state.panels.with(|store| store.run_status(run.run_id))
                    == Some(RunStatus::Running)
        })
        .cloned()
}

fn pause_requested_message(script_name: &str, run_id: u32) -> String {
    format!(
        "Pause requested for '{script_name}' (run #{run_id}). The script's pause method is running \
         (follow it with /panel pause). When the script exits, the run is marked paused; \
         continue it with /resume."
    )
}

/// `/pause`: asks the newest running run of the active script to call its pause method.
fn cmd_pause(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (name, script_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let Some(run) = running_run(state, &name, &script_path) else {
        return Ok(Some(format!(
            "Error: '{name}' has no running run to pause. Start one with /run_script or /resume."
        )));
    };
    if state.panels.with(|store| store.pause_requested(run.run_id)) {
        return Ok(Some(format!(
            "Pause was already requested for run #{}. Once the script has exited, continue with /resume.",
            run.run_id
        )));
    }

    if let Err(error) = control::request_pause(&run.control_dir) {
        return Ok(Some(format!("Error: {error}")));
    }
    let answered = control::wait_for_ack(&run.control_dir, control::ACK_TIMEOUT)
        // Not answered in time, but taken just now: the script has it.
        || !control::cancel_request(&run.control_dir);
    if !answered {
        return Ok(Some(format!(
            "Error: Run #{} did not answer within {} seconds, so the request was withdrawn. \
             The script must call tc.register_pause(fn) with its pause method, and it must have \
             been started by the controller. Try /pause again if it was only busy.",
            run.run_id,
            control::ACK_TIMEOUT.as_secs()
        )));
    }

    state
        .panels
        .with(|store| store.mark_pause_requested(run.run_id));
    Ok(Some(pause_requested_message(&name, run.run_id)))
}

/// `/resume`: starts the active script again with the flags of its last run
/// plus `--resume`.
fn cmd_resume(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    resume_on(state, RunTarget::Native)
}

/// `/resume_wsl`: `/resume`, but the script runs with the Python inside WSL2.
fn cmd_resume_wsl(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    resume_on(state, RunTarget::Wsl)
}

fn resume_on(state: &mut AppState, target: RunTarget) -> Result<Option<String>> {
    let (name, script_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    if let Some(run) = running_run(state, &name, &script_path) {
        return Ok(Some(format!(
            "Error: '{name}' is still running (run #{}). Use /pause first and /resume once it has exited.",
            run.run_id
        )));
    }
    let Some(last) = state.tc_json.last_run(&name, &script_path) else {
        return Ok(Some(format!(
            "Error: There is no earlier run of '{name}' to resume. Start it with /run_script first."
        )));
    };

    let script_args = resume_args(&last.args);
    println!(
        "Resuming '{name}' with the flags of its last run: {}",
        format_args_display(&script_args)
    );
    launch_script(
        state,
        &name,
        &script_path,
        script_args,
        last.config,
        LaunchKind::Resume,
        target,
    )
}

// ---------------------------------------------------------------------------
// Per-script flag configs
// ---------------------------------------------------------------------------

/// Graph config names share one namespace with experiment IDs on the
/// `/graph` command line, so a name must not look like an ID list.
fn validate_graph_config_name(name: &str) -> std::result::Result<(), String> {
    validate_config_name(name)?;
    if graphs::parse_id_list(name).is_some() {
        return Err(format!(
            "Invalid graph config name '{name}': it looks like an experiment ID."
        ));
    }
    Ok(())
}

fn validate_config_name(name: &str) -> std::result::Result<(), String> {
    let valid = !name.is_empty()
        && !name.starts_with('-')
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'));
    if valid {
        Ok(())
    } else {
        Err(format!(
            "Invalid config name '{name}'. Use letters, digits, '_', '-' or '.', and do not start with '-'."
        ))
    }
}

/// Flags as one display line; arguments with spaces are wrapped in quotes.
fn format_args_display(args: &[String]) -> String {
    if args.is_empty() {
        return "(no flags)".to_string();
    }
    args.iter()
        .map(|arg| {
            if arg.is_empty() || arg.chars().any(char::is_whitespace) {
                format!("\"{arg}\"")
            } else {
                arg.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn collect_args(args: &ArgMatches) -> Vec<String> {
    args.get_many::<String>("args")
        .map(|values| values.cloned().collect())
        .unwrap_or_default()
}

/// If the first typed argument names a saved config, expands it to the saved
/// flags followed by any remaining typed arguments. Otherwise the typed
/// arguments are used as they are.
fn resolve_run_args(
    configs: &[(String, Vec<String>)],
    typed: Vec<String>,
) -> (Vec<String>, Option<String>) {
    if let Some(first) = typed.first() {
        if let Some((config_name, saved)) = configs.iter().find(|(name, _)| name == first) {
            let mut combined = saved.clone();
            combined.extend(typed[1..].iter().cloned());
            return (combined, Some(config_name.clone()));
        }
    }
    (typed, None)
}

fn active_script(state: &AppState) -> std::result::Result<(String, String), String> {
    match (&state.active_script_name, &state.active_script_path) {
        (Some(name), Some(path)) => Ok((name.clone(), path.clone())),
        _ => Err("No active script. Use /select_script or /open_script first.".to_string()),
    }
}

fn cmd_add_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (script_name, script_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let config = args.get_one::<String>("name").unwrap();
    let flags = collect_args(&args);

    match state.tc_json.add_config(&script_name, &script_path, config, &flags) {
        Ok(()) => Ok(Some(format!(
            "Saved config '{config}' for '{script_name}': {}\nRun it with: /run_script {config}",
            format_args_display(&flags)
        ))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_edit_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (script_name, script_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let config = args.get_one::<String>("name").unwrap();
    let flags = collect_args(&args);

    match state.tc_json.edit_config(&script_name, &script_path, config, &flags) {
        Ok(()) => Ok(Some(format!(
            "Updated config '{config}' for '{script_name}': {}",
            format_args_display(&flags)
        ))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_show_configs(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (script_name, script_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let configs = state.tc_json.configs(&script_name, &script_path);

    if let Some(wanted) = args.get_one::<String>("name") {
        return Ok(Some(match configs.iter().find(|(name, _)| name == wanted) {
            Some((name, flags)) => format!(
                "Config '{name}' for '{script_name}': {}",
                format_args_display(flags)
            ),
            None => format!("Error: Config '{wanted}' not found for '{script_name}'."),
        }));
    }

    if configs.is_empty() {
        return Ok(Some(format!(
            "No configs saved for '{script_name}'. Create one with /add_config NAME [FLAGS...]"
        )));
    }

    let mut output = format!("Configs for '{script_name}':\n");
    for (name, flags) in &configs {
        output.push_str(&format!("  {name} -> {}\n", format_args_display(flags)));
    }
    Ok(Some(output.trim_end().to_string()))
}

fn cmd_remove_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (script_name, script_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let config = args.get_one::<String>("name").unwrap();

    let Some(flags) = state.tc_json.get_config(&script_name, &script_path, config) else {
        return Ok(Some(format!("Error: Config '{config}' not found for '{script_name}'.")));
    };

    println!(
        "Are you sure you want to delete config '{config}' ({}) of '{script_name}'? (y/N): ",
        format_args_display(&flags)
    );
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return Ok(Some("Error: Failed to read confirmation input.".to_string()));
    }
    let answer = input.trim().to_lowercase();
    if answer != "y" && answer != "yes" {
        return Ok(Some("Deletion cancelled.".to_string()));
    }

    match state.tc_json.remove_config(&script_name, &script_path, config) {
        Ok(()) => Ok(Some(format!("Deleted config '{config}' of '{script_name}'."))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_rename_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (script_name, script_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let old_name = args.get_one::<String>("old_name").unwrap();
    let new_name = args.get_one::<String>("new_name").unwrap();

    match state
        .tc_json
        .rename_config(&script_name, &script_path, old_name, new_name)
    {
        Ok(()) => Ok(Some(format!("Renamed config '{old_name}' to '{new_name}'."))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

// ---------------------------------------------------------------------------
// Experiments and logs (SQLite, see logdb.rs)
// ---------------------------------------------------------------------------

/// The log database lives next to tc.json.
fn log_db_path(state: &AppState) -> Option<PathBuf> {
    state
        .tc_json
        .path()
        .parent()
        .map(|directory| directory.join(logdb::DB_FILE_NAME))
}

/// Creates the experiment row for a run and returns its ID. On failure the
/// script still runs, just without logging.
fn start_experiment(
    db_path: &Path,
    script_name: &str,
    script_path: &str,
    config_name: Option<&str>,
    args: &[String],
) -> Option<i64> {
    let created = logdb::open(db_path).and_then(|conn| {
        logdb::create_experiment(
            &conn,
            &logdb::NewExperiment {
                script_name: Some(script_name),
                script_path: Some(script_path),
                config_name,
                args,
                status: logdb::STATUS_RUNNING,
            },
        )
    });
    match created {
        Ok(id) => Some(id),
        Err(error) => {
            println!("Warning: {error}");
            println!("Running the script without log storage.");
            None
        }
    }
}

/// Records how a run ended. Runs on a background thread, so it stays silent.
fn end_experiment(db_path: &Path, experiment_id: i64, status: &RunStatus) {
    let (text, code) = match status {
        RunStatus::Running => ("running", None),
        RunStatus::Exited(0) => ("finished", Some(0)),
        RunStatus::Exited(code) => ("failed", Some(*code)),
        RunStatus::Paused => ("paused", Some(0)),
        RunStatus::Killed => ("killed", None),
        RunStatus::Failed(_) => ("failed", None),
    };
    if let Ok(conn) = logdb::open(db_path) {
        let _ = logdb::finish_experiment(&conn, experiment_id, text, code);
    }
}

/// Marks an experiment whose script could not be started as failed.
fn abort_experiment(db_path: Option<&Path>, experiment_id: Option<i64>) {
    if let (Some(db), Some(id)) = (db_path, experiment_id) {
        end_experiment(db, id, &RunStatus::Failed("could not start".to_string()));
    }
}

fn cmd_show_experiments(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let limit = match args.get_one::<String>("limit") {
        Some(text) => match text.parse::<usize>() {
            Ok(limit) if limit > 0 => limit,
            _ => return Ok(Some("Error: LIMIT must be a positive integer.".to_string())),
        },
        None => 20,
    };
    let Some(db_path) = log_db_path(state) else {
        return Ok(Some("Error: Invalid tc.json path.".to_string()));
    };
    if !db_path.is_file() {
        return Ok(Some(
            "No experiments yet. Every /run_script creates one.".to_string(),
        ));
    }

    let result = logdb::open(&db_path).and_then(|conn| logdb::list_experiments(&conn, limit));
    match result {
        Ok(list) if list.is_empty() => Ok(Some(
            "No experiments yet. Every /run_script creates one.".to_string(),
        )),
        Ok(list) => Ok(Some(logdb::format_experiments(&list))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_show_log(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let id_text = args.get_one::<String>("id").unwrap();
    let Ok(experiment_id) = id_text.trim_start_matches('#').parse::<i64>() else {
        return Ok(Some("Error: EXPERIMENT_ID must be an integer.".to_string()));
    };
    let last = match args.get_one::<String>("last") {
        Some(text) => match text.parse::<usize>() {
            Ok(last) if last > 0 => last,
            _ => return Ok(Some("Error: N must be a positive integer.".to_string())),
        },
        None => 20,
    };
    let Some(db_path) = log_db_path(state) else {
        return Ok(Some("Error: Invalid tc.json path.".to_string()));
    };
    if !db_path.is_file() {
        return Ok(Some(
            "No experiments yet. Every /run_script creates one.".to_string(),
        ));
    }

    let result = logdb::open(&db_path).and_then(|conn| logdb::read_log(&conn, experiment_id, last));
    match result {
        Ok(table) => Ok(Some(logdb::format_log(&table))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

// ---------------------------------------------------------------------------
// Checkpoints (storage and retention live in checkpoint.rs)
// ---------------------------------------------------------------------------

/// Each script has its own checkpoint folder next to tc.json.
fn checkpoint_dir_for(state: &AppState, script_name: &str) -> std::result::Result<PathBuf, String> {
    let parent = state
        .tc_json
        .path()
        .parent()
        .ok_or_else(|| "Invalid tc.json path.".to_string())?;
    Ok(parent
        .join("checkpoints")
        .join(checkpoint::dir_name_for(script_name)))
}

/// The active script's name and its checkpoint store.
fn checkpoint_store(state: &AppState) -> std::result::Result<(String, Store), String> {
    let (name, _) = active_script(state)?;
    let directory = checkpoint_dir_for(state, &name)?;
    Ok((name, Store::open(directory)?))
}

fn reply(result: std::result::Result<String, String>) -> Result<Option<String>> {
    Ok(Some(result.unwrap_or_else(|error| format!("Error: {error}"))))
}

fn cmd_show_checkpoint_config(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    reply(checkpoint_store(state).and_then(|(name, store)| checkpoint_cli::show_config(&name, &store)))
}

fn cmd_set_checkpoint_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let key = args.get_one::<String>("key").unwrap();
    let value = args.get_one::<String>("value").unwrap();
    reply(checkpoint_store(state).and_then(|(name, store)| {
        checkpoint_cli::set_config(&name, &store, key, value)
    }))
}

fn cmd_checkpoint_preset(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let preset = args.get_one::<String>("name").map(String::as_str);
    reply(checkpoint_store(state).and_then(|(name, store)| {
        checkpoint_cli::apply_preset(&name, &store, preset)
    }))
}

fn cmd_reset_checkpoint_config(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    reply(checkpoint_store(state).and_then(|(name, store)| checkpoint_cli::reset_config(&name, &store)))
}

fn cmd_checkpoints(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let experiment = args.get_one::<String>("experiment").map(String::as_str);
    reply(checkpoint_store(state).and_then(|(name, store)| {
        checkpoint_cli::list(&name, &store, experiment)
    }))
}

fn cmd_pin_checkpoint(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let id = args.get_one::<String>("id").unwrap();
    reply(checkpoint_store(state).and_then(|(_, store)| checkpoint_cli::pin(&store, id, true)))
}

fn cmd_unpin_checkpoint(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let id = args.get_one::<String>("id").unwrap();
    reply(checkpoint_store(state).and_then(|(_, store)| checkpoint_cli::pin(&store, id, false)))
}

fn cmd_remove_checkpoint(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let spec = args.get_one::<String>("id").unwrap();
    reply((|| -> std::result::Result<String, String> {
        let (_, store) = checkpoint_store(state)?;
        let entry = store.find(spec)?;
        if entry.pinned {
            return Err(format!(
                "Checkpoint #{id} is pinned. Unpin it first with /unpin_checkpoint {id}.",
                id = entry.id
            ));
        }

        println!("{}", checkpoint_cli::confirm_prompt(&entry));
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|_| "Failed to read confirmation input.".to_string())?;
        let answer = input.trim().to_lowercase();
        if answer != "y" && answer != "yes" {
            return Ok("Deletion cancelled.".to_string());
        }

        let (removed, warnings) = store.remove(entry.id, false)?;
        Ok(checkpoint_cli::removed_message(&removed, &warnings))
    })())
}

fn cmd_prune_checkpoints(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let mode = args.get_one::<String>("mode").map(String::as_str);
    reply(checkpoint_store(state).and_then(|(name, store)| checkpoint_cli::prune(&name, &store, mode)))
}

fn cmd_verify_checkpoints(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    reply(checkpoint_store(state).and_then(|(name, store)| checkpoint_cli::verify(&name, &store)))
}

fn cmd_export_checkpoint(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let id = args.get_one::<String>("id").unwrap();
    let destination = args.get_one::<String>("destination").unwrap();
    reply(checkpoint_store(state).and_then(|(_, store)| checkpoint_cli::export(&store, id, destination)))
}

// ---------------------------------------------------------------------------
// Graphs (see graphs.rs)
// ---------------------------------------------------------------------------

const NO_EXPERIMENTS: &str = "No experiments yet. Every /run_script creates one.";

/// The `/graph` command. The arguments are parsed by hand (see
/// `graphs::parse_graph_command`), so `--save` and `+` need no clap support.
fn graph_command() -> Command {
    Command::new("/graph")
        .arg(
            Arg::new("args")
                .required(true)
                .num_args(1..)
                .allow_hyphen_values(true)
                .value_name("ARGS")
                .help("IDS [Y [X]] [+ Y [X] ...] [--save NAME]   or   CONFIG [IDS]"),
        )
        .about(
            "Graph logged quantities live. IDS = experiment ID(s), e.g. 3 or 3,4 (overlaid). \
             Y and X are log columns, `time` (default x) or `step`; Y may be a comma list; \
             `+` shows several graphs at once; a saved graph config name replaces the graphs",
        )
}

fn cmd_graph(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let tokens = collect_args(&args);
    let config_names: Vec<String> = state
        .tc_json
        .graph_configs()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    let command = match graphs::parse_graph_command(&tokens, |name| {
        config_names.iter().any(|config| config == name)
    }) {
        Ok(command) => command,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };

    let Some(db_path) = log_db_path(state) else {
        return Ok(Some("Error: Invalid tc.json path.".to_string()));
    };
    if !db_path.is_file() {
        return Ok(Some(NO_EXPERIMENTS.to_string()));
    }
    let conn = match logdb::open(&db_path) {
        Ok(conn) => conn,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };

    let ids: Vec<i64> = match &command.ids {
        Some(ids) => ids.clone(),
        None => match logdb::latest_experiment_id(&conn) {
            Ok(Some(id)) => vec![id],
            Ok(None) => return Ok(Some(NO_EXPERIMENTS.to_string())),
            Err(error) => return Ok(Some(format!("Error: {error}"))),
        },
    };
    for &id in &ids {
        match logdb::experiment_exists(&conn, id) {
            Ok(true) => {}
            Ok(false) => return Ok(Some(format!("Error: Experiment #{id} does not exist."))),
            Err(error) => return Ok(Some(format!("Error: {error}"))),
        }
    }

    let specs: Vec<GraphSpec> = match &command.selection {
        Selection::Explicit(specs) => specs.clone(),
        Selection::AllColumns => match logdb::log_columns(&conn, ids[0]) {
            Ok(columns) if columns.is_empty() => {
                return Ok(Some(format!(
                    "Error: Experiment #{} has no log columns yet (the script has not called init_log).",
                    ids[0]
                )));
            }
            Ok(columns) => columns
                .iter()
                .map(|column| GraphSpec::new(column, graphs::DEFAULT_X))
                .collect(),
            Err(error) => return Ok(Some(format!("Error: {error}"))),
        },
        Selection::Config(name) => match state.tc_json.get_graph_config(name) {
            Some(specs) => specs,
            None => return Ok(Some(format!("Error: Graph config '{name}' not found."))),
        },
    };

    let ids_text = ids
        .iter()
        .map(|id| format!("#{id}"))
        .collect::<Vec<_>>()
        .join(", ");
    let noun = if ids.len() == 1 { "experiment" } else { "experiments" };
    let heading = match &command.selection {
        Selection::Config(name) => format!("config '{name}'  ·  {noun} {ids_text}"),
        _ => format!("{noun} {ids_text}"),
    };
    // Checks that every experiment has every quantity before the screen opens.
    let view = match graphs::GraphView::new(&conn, heading, &ids, specs.clone()) {
        Ok(view) => view,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    drop(conn);

    let mut saved_note = String::new();
    if let Some(name) = &command.save {
        match state.tc_json.add_graph_config(name, &specs) {
            Ok(()) => {
                saved_note = format!(
                    "\nSaved graph config '{name}': {}\nShow it for any experiment with: /graph {name} [IDS]",
                    graphs::format_specs(&specs)
                );
            }
            Err(error) => return Ok(Some(format!("Error: {error}"))),
        }
    }

    match graphs::run_graphs(&db_path, view) {
        Ok(summary) => Ok(Some(format!("{summary}{saved_note}"))),
        Err(error) => Ok(Some(format!("Error: {error}{saved_note}"))),
    }
}

fn cmd_add_graph_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let name = args.get_one::<String>("name").unwrap();
    let specs = match graphs::parse_specs(&collect_args(&args)) {
        Ok(specs) if specs.is_empty() => {
            return Ok(Some("Error: Give at least one graph: Y [X] [+ Y [X] ...]".to_string()));
        }
        Ok(specs) => specs,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    match state.tc_json.add_graph_config(name, &specs) {
        Ok(()) => Ok(Some(format!(
            "Saved graph config '{name}': {}\nShow it with: /graph {name} [IDS]",
            graphs::format_specs(&specs)
        ))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_edit_graph_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let name = args.get_one::<String>("name").unwrap();
    let specs = match graphs::parse_specs(&collect_args(&args)) {
        Ok(specs) if specs.is_empty() => {
            return Ok(Some("Error: Give at least one graph: Y [X] [+ Y [X] ...]".to_string()));
        }
        Ok(specs) => specs,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    match state.tc_json.edit_graph_config(name, &specs) {
        Ok(()) => Ok(Some(format!(
            "Updated graph config '{name}': {}",
            graphs::format_specs(&specs)
        ))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_show_graph_configs(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let configs = state.tc_json.graph_configs();

    if let Some(wanted) = args.get_one::<String>("name") {
        return Ok(Some(match configs.iter().find(|(name, _)| name == wanted) {
            Some((name, specs)) => {
                format!("Graph config '{name}': {}", graphs::format_specs(specs))
            }
            None => format!("Error: Graph config '{wanted}' not found."),
        }));
    }

    if configs.is_empty() {
        return Ok(Some(
            "No graph configs saved. Create one with /add_graph_config NAME Y [X] [+ Y [X] ...] \
             or /graph IDS Y [X] --save NAME"
                .to_string(),
        ));
    }
    let mut output = String::from("Graph configs:\n");
    for (name, specs) in &configs {
        output.push_str(&format!("  {name} -> {}\n", graphs::format_specs(specs)));
    }
    output.push_str("Show one with /graph NAME [IDS].");
    Ok(Some(output))
}

fn cmd_remove_graph_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let name = args.get_one::<String>("name").unwrap();
    let Some(specs) = state.tc_json.get_graph_config(name) else {
        return Ok(Some(format!("Error: Graph config '{name}' not found.")));
    };

    println!(
        "Are you sure you want to delete graph config '{name}' ({})? (y/N): ",
        graphs::format_specs(&specs)
    );
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return Ok(Some("Error: Failed to read confirmation input.".to_string()));
    }
    let answer = input.trim().to_lowercase();
    if answer != "y" && answer != "yes" {
        return Ok(Some("Deletion cancelled.".to_string()));
    }

    match state.tc_json.remove_graph_config(name) {
        Ok(()) => Ok(Some(format!("Deleted graph config '{name}'."))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

// ---------------------------------------------------------------------------
// Associate scripts, their configs, and /test
// ---------------------------------------------------------------------------
//
// An associate script is a script that goes with one training script (an
// evaluation, an inference run, a data check, ...). It belongs to the training
// script that was active when it was added, and every associate command works
// on the active training script's associates. Like training scripts they have
// named flag configs. One associate script and one of its configs can be
// selected; `/test` runs that pair. Selecting another training script clears
// the associate selection.

/// The active training script and associate script:
/// `(training name, training path, associate name, associate path)`.
fn active_associate(
    state: &AppState,
) -> std::result::Result<(String, String, String, String), String> {
    let (main_name, main_path) = active_script(state)?;
    match (&state.active_associate_name, &state.active_associate_path) {
        (Some(name), Some(path)) => Ok((main_name, main_path, name.clone(), path.clone())),
        _ => Err(
            "No active associate script. Use /select_associate_script or /open_associate_script first."
                .to_string(),
        ),
    }
}

/// Makes a training script the active one. The associate selection belongs to
/// the previous training script, so it is cleared when the script changes.
fn activate_script(state: &mut AppState, name: &str, path: &str) {
    let unchanged = state.active_script_name.as_deref() == Some(name)
        && state.active_script_path.as_deref() == Some(path);
    if !unchanged {
        clear_associate(state);
    }
    state.active_script_name = Some(name.to_string());
    state.active_script_path = Some(path.to_string());
}

/// Makes an associate script the active one. Its selected config is kept when
/// the same script is selected again and dropped when another one is chosen,
/// because a config belongs to one associate script.
fn activate_associate(state: &mut AppState, name: &str, path: &str) {
    if state.active_associate_name.as_deref() != Some(name) {
        state.active_associate_config = None;
    }
    state.active_associate_name = Some(name.to_string());
    state.active_associate_path = Some(path.to_string());
}

fn clear_associate(state: &mut AppState) {
    state.active_associate_name = None;
    state.active_associate_path = None;
    state.active_associate_config = None;
}

/// `(name, path)` of a script entry in tc.json.
fn script_entry(script: &Value) -> (String, String) {
    let text = |key: &str| {
        script
            .get(key)
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string()
    };
    (text("name"), text("path"))
}

fn confirm(prompt: &str) -> std::result::Result<bool, String> {
    println!("{prompt} (y/N): ");
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|_| "Failed to read confirmation input.".to_string())?;
    let answer = input.trim().to_lowercase();
    Ok(answer == "y" || answer == "yes")
}

fn cmd_add_associate_script(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (main_name, main_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let name = args.get_one::<String>("name").unwrap();
    let absolute_path = args.get_one::<String>("absolute_path").unwrap();

    match state
        .tc_json
        .add_associate_script(&main_name, &main_path, name, absolute_path)
    {
        Ok(()) => Ok(Some(format!(
            "Registered associate script '{name}' for '{main_name}' in {}\nSelect it with: /open_associate_script {name}",
            state.tc_json.path().display()
        ))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_show_associate_scripts(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (main_name, main_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let scripts = match state.tc_json.associate_scripts(&main_name, &main_path) {
        Some(scripts) if !scripts.is_empty() => scripts,
        _ => {
            return Ok(Some(format!(
                "No associate scripts registered for '{main_name}' yet. Add one with /add_associate_script NAME ABSOLUTE_PATH"
            )));
        }
    };

    let mut output = format!("Associate scripts of '{main_name}' (* = selected):\n");
    for (index, script) in scripts.iter().enumerate() {
        let (name, path) = script_entry(script);
        let marker = if state.active_associate_name.as_deref() == Some(name.as_str()) {
            "*"
        } else {
            " "
        };
        output.push_str(&format!(" {marker}[{index}] {name} -> {path}\n"));
    }
    Ok(Some(output.trim_end().to_string()))
}

fn cmd_select_associate_script(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let index_str = args.get_one::<String>("index").unwrap();
    let Ok(index) = index_str.parse::<usize>() else {
        return Ok(Some("Error: Index must be a valid integer.".to_string()));
    };

    let (main_name, main_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let found = state
        .tc_json
        .associate_scripts(&main_name, &main_path)
        .and_then(|scripts| scripts.get(index))
        .map(script_entry);
    let Some((name, path)) = found else {
        let count = state
            .tc_json
            .associate_scripts(&main_name, &main_path)
            .map_or(0, Vec::len);
        return Ok(Some(format!("Error: Index {index} out of bounds (0..{count}).")));
    };

    activate_associate(state, &name, &path);
    Ok(Some(format!("Selected associate script [{index}] '{name}' -> {path}")))
}

fn cmd_open_associate_script(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let name = args.get_one::<String>("name").unwrap();

    let (main_name, main_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let found = state
        .tc_json
        .associate_scripts(&main_name, &main_path)
        .and_then(|scripts| {
            scripts
                .iter()
                .find(|script| script.get("name").and_then(|v| v.as_str()) == Some(name.as_str()))
        })
        .map(script_entry);
    let Some((name, path)) = found else {
        return Ok(Some(format!("Error: Associate script '{name}' not found.")));
    };

    activate_associate(state, &name, &path);
    Ok(Some(format!("Opened associate script '{name}' -> {path}")))
}

fn cmd_remove_associate_script(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let index_str = args.get_one::<String>("index").unwrap();
    let Ok(index) = index_str.parse::<usize>() else {
        return Ok(Some("Error: Index must be a valid integer.".to_string()));
    };

    let (main_name, main_path) = match active_script(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let found = state
        .tc_json
        .associate_scripts(&main_name, &main_path)
        .and_then(|scripts| scripts.get(index))
        .map(script_entry);
    let Some((name, path)) = found else {
        let count = state
            .tc_json
            .associate_scripts(&main_name, &main_path)
            .map_or(0, Vec::len);
        return Ok(Some(format!("Error: Index {index} out of bounds (0..{count}).")));
    };

    match confirm(&format!(
        "Are you sure you want to delete associate script [{index}] '{name}' -> {path} and all of its configs?"
    )) {
        Ok(true) => {}
        Ok(false) => return Ok(Some("Deletion cancelled.".to_string())),
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    }

    match state
        .tc_json
        .remove_associate_script(&main_name, &main_path, index)
    {
        Ok((removed_name, removed_path)) => {
            if state.active_associate_name.as_deref() == Some(removed_name.as_str()) {
                clear_associate(state);
            }
            Ok(Some(format!(
                "Successfully deleted associate script [{index}] '{removed_name}' -> {removed_path}"
            )))
        }
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_add_associate_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (main_name, main_path, script_name, _) = match active_associate(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let config = args.get_one::<String>("name").unwrap();
    let flags = collect_args(&args);

    match state
        .tc_json
        .add_associate_config(&main_name, &main_path, &script_name, config, &flags)
    {
        Ok(()) => Ok(Some(format!(
            "Saved config '{config}' for associate script '{script_name}': {}\nSelect it with: /select_associate_config {config}",
            format_args_display(&flags)
        ))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_edit_associate_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (main_name, main_path, script_name, _) = match active_associate(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let config = args.get_one::<String>("name").unwrap();
    let flags = collect_args(&args);

    match state
        .tc_json
        .edit_associate_config(&main_name, &main_path, &script_name, config, &flags)
    {
        Ok(()) => Ok(Some(format!(
            "Updated config '{config}' for associate script '{script_name}': {}",
            format_args_display(&flags)
        ))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_show_associate_configs(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (main_name, main_path, script_name, _) = match active_associate(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let configs = state
        .tc_json
        .associate_configs(&main_name, &main_path, &script_name);

    if let Some(wanted) = args.get_one::<String>("name") {
        return Ok(Some(match configs.iter().find(|(name, _)| name == wanted) {
            Some((name, flags)) => format!(
                "Config '{name}' for associate script '{script_name}': {}",
                format_args_display(flags)
            ),
            None => format!(
                "Error: Config '{wanted}' not found for associate script '{script_name}'."
            ),
        }));
    }

    if configs.is_empty() {
        return Ok(Some(format!(
            "No configs saved for associate script '{script_name}'. Create one with /add_associate_config NAME [FLAGS...]"
        )));
    }

    let mut output = format!("Configs for associate script '{script_name}' (* = selected):\n");
    for (name, flags) in &configs {
        let marker = if state.active_associate_config.as_deref() == Some(name.as_str()) {
            "*"
        } else {
            " "
        };
        output.push_str(&format!(" {marker}{name} -> {}\n", format_args_display(flags)));
    }
    Ok(Some(output.trim_end().to_string()))
}

fn cmd_remove_associate_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (main_name, main_path, script_name, _) = match active_associate(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let config = args.get_one::<String>("name").unwrap();

    let Some(flags) = state
        .tc_json
        .get_associate_config(&main_name, &main_path, &script_name, config)
    else {
        return Ok(Some(format!(
            "Error: Config '{config}' not found for associate script '{script_name}'."
        )));
    };

    match confirm(&format!(
        "Are you sure you want to delete config '{config}' ({}) of associate script '{script_name}'?",
        format_args_display(&flags)
    )) {
        Ok(true) => {}
        Ok(false) => return Ok(Some("Deletion cancelled.".to_string())),
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    }

    match state
        .tc_json
        .remove_associate_config(&main_name, &main_path, &script_name, config)
    {
        Ok(()) => {
            if state.active_associate_config.as_deref() == Some(config.as_str()) {
                state.active_associate_config = None;
            }
            Ok(Some(format!(
                "Deleted config '{config}' of associate script '{script_name}'."
            )))
        }
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_rename_associate_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (main_name, main_path, script_name, _) = match active_associate(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let old_name = args.get_one::<String>("old_name").unwrap();
    let new_name = args.get_one::<String>("new_name").unwrap();

    match state
        .tc_json
        .rename_associate_config(&main_name, &main_path, &script_name, old_name, new_name)
    {
        Ok(()) => {
            // Keep the selection pointing at the same config.
            if state.active_associate_config.as_deref() == Some(old_name.as_str()) {
                state.active_associate_config = Some(new_name.clone());
            }
            Ok(Some(format!("Renamed config '{old_name}' to '{new_name}'.")))
        }
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_select_associate_config(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let (main_name, main_path, script_name, _) = match active_associate(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };

    // No name: go back to running the associate script without a config.
    let Some(config) = args.get_one::<String>("name") else {
        state.active_associate_config = None;
        return Ok(Some(format!(
            "Deselected the config of associate script '{script_name}'; /test will run it without saved flags."
        )));
    };

    match state
        .tc_json
        .get_associate_config(&main_name, &main_path, &script_name, config)
    {
        Some(flags) => {
            state.active_associate_config = Some(config.clone());
            Ok(Some(format!(
                "Selected config '{config}' for associate script '{script_name}': {}",
                format_args_display(&flags)
            )))
        }
        None => Ok(Some(format!(
            "Error: Config '{config}' not found for associate script '{script_name}'."
        ))),
    }
}

/// `PYTHONPATH` that makes `import training_controller` work for a script.
///
/// Native runs: the module directory next to tc.json (refreshed from the built
/// extension), followed by whatever `PYTHONPATH` already contained.
///
/// WSL2 runs: only the WSL module directory (it holds the Linux build). The
/// Windows `PYTHONPATH` is left out because it belongs to the Windows Python.
fn python_path_for(
    state: &AppState,
    target: RunTarget,
) -> std::result::Result<std::ffi::OsString, String> {
    let parent = state
        .tc_json
        .path()
        .parent()
        .ok_or_else(|| "Invalid tc.json path.".to_string())?;
    match target {
        RunTarget::Native => {
            let module_dir = parent.join("pymodule");
            if let Err(warning) = prepare_python_module(&module_dir) {
                println!("Warning: {warning}");
                println!("Running the script without the `{PY_MODULE_NAME}` module.");
            }
            python_path_with(&module_dir, std::env::var_os("PYTHONPATH"))
        }
        RunTarget::Wsl => {
            let module_dir = parent.join(wsl::MODULE_DIR_NAME);
            if let Err(warning) = prepare_wsl_module(&module_dir) {
                println!("Warning: {warning}");
                println!("Running the script without the Linux `{PY_MODULE_NAME}` module.");
            }
            Ok(module_dir.into_os_string())
        }
    }
}

/// `/test`: runs the selected associate script with the selected associate
/// config (plus any extra arguments typed after `/test`) in the background.
///
/// The script gets the same environment as a `/run_script` run (the
/// `training_controller` module, captured into panels) with two differences:
/// it does not create an experiment, and the checkpoint folder of the training
/// script it belongs to is passed along, so `tc.latest_checkpoint()` finds the
/// checkpoints that script stored. Anything it logs with `tc.init_log`
/// goes to the controller's database as a standalone experiment.
fn cmd_test(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    test_on(args, state, RunTarget::Native)
}

/// `/test_wsl`: `/test`, but the associate script runs with the Python inside WSL2.
fn cmd_test_wsl(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    test_on(args, state, RunTarget::Wsl)
}

fn test_on(
    args: ArgMatches,
    state: &mut AppState,
    target: RunTarget,
) -> Result<Option<String>> {
    let (main_name, main_path, name, script_path) = match active_associate(state) {
        Ok(active) => active,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };

    let script = Path::new(&script_path);
    if !script.is_file() {
        return Ok(Some(format!(
            "Error: Associate script file not found: {script_path}"
        )));
    }

    let mut script_args: Vec<String> = Vec::new();
    let used_config = match state.active_associate_config.clone() {
        Some(config) => match state
            .tc_json
            .get_associate_config(&main_name, &main_path, &name, &config)
        {
            Some(flags) => {
                script_args = flags;
                Some(config)
            }
            None => {
                return Ok(Some(format!(
                    "Error: The selected config '{config}' no longer exists for associate script '{name}'. \
                     Pick one with /select_associate_config."
                )));
            }
        },
        None => None,
    };
    script_args.extend(collect_args(&args));
    match &used_config {
        Some(config) => println!(
            "Using associate config '{config}': {}",
            format_args_display(&script_args)
        ),
        None => println!(
            "No associate config selected. Arguments: {}",
            format_args_display(&script_args)
        ),
    }

    let python_path = match python_path_for(state, target) {
        Ok(path) => path,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let training_checkpoints = checkpoint_dir_for(state, &main_name).ok();
    let db_path = log_db_path(state);
    let label = match &used_config {
        Some(config) => format!("test: {name} ({config})"),
        None => format!("test: {name}"),
    };
    let label = match target {
        RunTarget::Native => label,
        RunTarget::Wsl => format!("{label} [wsl]"),
    };

    let mut env = ScriptEnv::new();
    env.set("PYTHONPATH", &python_path, EnvKind::PathList);
    env.set("PYTHONUNBUFFERED", "1", EnvKind::Plain);
    env.set("PYTHONIOENCODING", "utf-8", EnvKind::Plain);
    if let Some(directory) = &training_checkpoints {
        env.set(checkpoint::DIR_ENV, directory, EnvKind::Path);
    }
    if let Some(db) = &db_path {
        env.set(logdb::DB_FILE_ENV, db, EnvKind::Path);
    }

    let mut not_found: Vec<&str> = Vec::new();
    for interpreter in interpreters(target) {
        let mut command = script_command(target, interpreter, script, &script_args, &env);

        match command.spawn() {
            Ok(child) => {
                let run_id = panels::attach_child(&state.panels, &label, child);
                let place = target.place_note();
                return Ok(Some(format!(
                    "Associate script '{name}' started{place} (run #{run_id}). \
                     View its output with /panel output, /panel error, /panel log or /panel other; \
                     list all panels with /show_active_panels."
                )));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                not_found.push(interpreter);
            }
            Err(error) => {
                return Ok(Some(format!("Error: Failed to run {interpreter}: {error}")));
            }
        }
    }

    Ok(Some(missing_runtime_message(target, &not_found)))
}

/// The `/test` command. Hyphenated arguments are passed through untouched.
fn test_command() -> Command {
    Command::new("/test")
        .arg(
            Arg::new("args")
                .num_args(0..)
                .allow_hyphen_values(true)
                .value_name("ARGS")
                .help("Extra arguments added after the selected config's flags"),
        )
        .about("Run the selected associate script with the selected associate config in the background (output goes to /panel)")
}

/// The `/test_wsl` command: `/test` with the Python inside WSL2.
fn test_wsl_command() -> Command {
    Command::new("/test_wsl")
        .arg(
            Arg::new("args")
                .num_args(0..)
                .allow_hyphen_values(true)
                .value_name("ARGS")
                .help("Extra arguments added after the selected config's flags"),
        )
        .about("Like /test, but runs the associate script with the Python inside WSL2 (output goes to /panel)")
}

/// `/wsl_status`: checks that scripts can run in WSL2 and what is still missing.
fn cmd_wsl_status(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let Some(parent) = state.tc_json.path().parent() else {
        return Ok(Some("Error: Invalid tc.json path.".to_string()));
    };
    let module_dir = parent.join(wsl::MODULE_DIR_NAME);
    let build_note = prepare_wsl_module(&module_dir).err();

    let probe = match wsl::probe(&module_dir) {
        Ok(probe) => probe,
        Err(error) => return Ok(Some(format!("Error: WSL is not ready: {error}"))),
    };

    let mut lines = vec![format!(
        "WSL: Python {} (kernel {})",
        probe.python_version, probe.kernel
    )];
    if !probe.kernel.to_ascii_uppercase().contains("WSL2") {
        lines.push(
            "Warning: this does not look like WSL2. Convert the distribution with `wsl --set-version <distro> 2`."
                .to_string(),
        );
    }
    match &probe.module {
        Ok(()) => lines.push(format!(
            "The `{PY_MODULE_NAME}` module can be imported in WSL."
        )),
        Err(reason) => {
            lines.push(format!(
                "The `{PY_MODULE_NAME}` module cannot be imported in WSL ({reason})."
            ));
            if let Some(note) = build_note {
                lines.push(note);
            }
            lines.push(format!(
                "To use another interpreter (e.g. a virtualenv with the module installed), set {} \
                 to its absolute path before starting the controller.",
                wsl::PYTHON_ENV
            ));
        }
    }
    lines.push("Run scripts there with /run_wsl, /resume_wsl and /test_wsl.".to_string());
    Ok(Some(lines.join("\n")))
}

fn cmd_help_ping(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    state.count += 1;
    Ok(Some(format!("Pong! (Command run {} times)", state.count)))
}

fn cmd_echo(args: ArgMatches, _state: &mut AppState) -> Result<Option<String>> {
    let msg = args.get_one::<String>("text").unwrap();
    Ok(Some(format!("You said: {msg}")))
}

fn cmd_test_get_terminal_arguments(
    args: ArgMatches,
    _state: &mut AppState,
) -> Result<Option<String>> {
    let path = args.get_one::<String>("path").unwrap();

    match get_terminal_arguments(path) {
        Ok(arguments) => Ok(Some(format!("{arguments:#?}"))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_show_scripts(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let scripts = match state.tc_json.scripts() {
        Some(s) => s,
        None => return Ok(Some("No scripts registered yet.".to_string())),
    };

    if scripts.is_empty() {
        return Ok(Some("No scripts registered yet.".to_string()));
    }

    let mut output = String::from("Registered scripts:\n");
    for (index, script) in scripts.iter().enumerate() {
        let name = script.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
        let path = script.get("path").and_then(|v| v.as_str()).unwrap_or("unknown");
        output.push_str(&format!("  [{index}] {name} -> {path}\n"));
    }

    Ok(Some(output.trim_end().to_string()))
}
fn cmd_select_script(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let index_str = args.get_one::<String>("index").unwrap();
    let index = match index_str.parse::<usize>() {
        Ok(i) => i,
        Err(_) => return Ok(Some("Error: Index must be a valid integer.".to_string())),
    };

    let scripts = match state.tc_json.scripts() {
        Some(s) => s,
        None => return Ok(Some("Error: No scripts available.".to_string())),
    };

    if index >= scripts.len() {
        return Ok(Some(format!("Error: Index {index} out of bounds (0..{}).", scripts.len())));
    }

    let script = &scripts[index];
    let name = script.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
    let path = script.get("path").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();

    activate_script(state, &name, &path);

    Ok(Some(format!("Selected script [{index}] '{name}' -> {path}")))
}
fn cmd_remove_script(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let index_str = args.get_one::<String>("index").unwrap();
    let index = match index_str.parse::<usize>() {
        Ok(i) => i,
        Err(_) => return Ok(Some("Error: Index must be a valid integer.".to_string())),
    };

    let scripts = match state.tc_json.scripts() {
        Some(s) => s,
        None => return Ok(Some("Error: No scripts available.".to_string())),
    };

    if index >= scripts.len() {
        return Ok(Some(format!("Error: Index {index} out of bounds (0..{}).", scripts.len())));
    }

    let script = &scripts[index];
    let name = script.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
    let path = script.get("path").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();

    println!("Are you sure you want to delete script [{index}] '{name}' -> {path}? (y/N): ");
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return Ok(Some("Error: Failed to read confirmation input.".to_string()));
    }

    let trimmed = input.trim().to_lowercase();
    if trimmed != "y" && trimmed != "yes" {
        return Ok(Some("Deletion cancelled.".to_string()));
    }

    match state.tc_json.remove_script(index) {
        Ok((del_name, del_path)) => {
            if state.active_script_name.as_deref() == Some(&del_name) {
                state.active_script_name = None;
                state.active_script_path = None;
                clear_associate(state);
            }
            Ok(Some(format!("Successfully deleted script [{index}] '{del_name}' -> {del_path}")))
        }
        Err(err) => Ok(Some(format!("Error: {err}"))),
    }
}

fn cmd_open_script(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let name = args.get_one::<String>("name").unwrap();

    let scripts = match state.tc_json.scripts() {
        Some(s) => s,
        None => return Ok(Some("Error: No scripts available.".to_string())),
    };

    let script = scripts.iter().find(|s| {
        s.get("name").and_then(|v| v.as_str()) == Some(name)
    });

    let script = match script {
        Some(s) => s,
        None => return Ok(Some(format!("Error: Script '{name}' not found."))),
    };

    let path = script.get("path").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();

    activate_script(state, &name, &path);

    Ok(Some(format!("Opened script '{name}' -> {path}")))
}

fn cmd_add_scripts(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let name = args.get_one::<String>("name").unwrap();
    let absolute_path = args.get_one::<String>("absolute_path").unwrap();

    match state.tc_json.add_script(name, absolute_path) {
        Ok(()) => Ok(Some(format!(
            "Registered script '{name}' in {}",
            state.tc_json.path().display()
        ))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_panel(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let name = args
        .get_one::<String>("name")
        .map(String::as_str)
        .unwrap_or(panels::OUTPUT_PANEL);
    match panels::open_panel(&state.panels, name) {
        Ok(summary) => Ok(Some(summary)),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_show_active_panels(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    Ok(Some(state.panels.with(|store| panels::describe_active(store))))
}

fn cmd_clear_panels(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let name = args.get_one::<String>("name").map(String::as_str);
    match state.panels.with(|store| store.clear(name)) {
        Ok(message) => Ok(Some(message)),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn dashboard_config_path(state: &AppState) -> std::result::Result<PathBuf, String> {
    let app_dir = state
        .tc_json
        .path()
        .parent()
        .ok_or_else(|| format!("Invalid tc.json path: {}", state.tc_json.path().display()))?;
    Ok(dashboard::config_path(app_dir))
}

fn cmd_dashboard_config(_args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let path = match dashboard_config_path(state) {
        Ok(path) => path,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };

    match dashboard::ensure_config_file(&path) {
        Ok(()) => Ok(Some(format!(
            "Dashboard config: {}\nEdit it in any editor; changes apply live while a dashboard is open.",
            path.display()
        ))),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

/// `/dashboard training`: the live view fed by the Python training API.
fn cmd_dashboard_training(args: &ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    let config_path = match dashboard_config_path(state) {
        Ok(path) => path,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let (settings, watcher) = match dashboard::open_settings(&config_path) {
        Ok(opened) => opened,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };

    let interval_override = match args.get_one::<String>("interval_ms") {
        Some(text) => match dashboard::parse_interval(text) {
            Ok(interval) => Some(interval),
            Err(error) => return Ok(Some(format!("Error: {error}"))),
        },
        None => None,
    };

    let source = if args.get_flag("demo") {
        TrainingSource::demo()
    } else {
        match dashboard::resolve_training_path(
            args.get_one::<String>("path").map(String::as_str),
            state.active_script_path.as_deref(),
        ) {
            Ok(path) => TrainingSource::file(path),
            Err(error) => return Ok(Some(format!("Error: {error}"))),
        }
    };

    let title = match state.active_script_name.as_deref() {
        Some(name) => format!("Training dashboard - {name}"),
        None => "Training dashboard".to_string(),
    };

    match dashboard::run_training_dashboard(TrainingConfig {
        title,
        source,
        settings,
        watcher,
        interval_override,
    }) {
        Ok(summary) => Ok(Some(summary)),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

fn cmd_dashboard(args: ArgMatches, state: &mut AppState) -> Result<Option<String>> {
    if let Some(("training", training_args)) = args.subcommand() {
        return cmd_dashboard_training(training_args, state);
    }

    let config_path = match dashboard_config_path(state) {
        Ok(path) => path,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };
    let (settings, watcher) = match dashboard::open_settings(&config_path) {
        Ok(opened) => opened,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };

    let interval_override = match args.get_one::<String>("interval_ms") {
        Some(text) => match dashboard::parse_interval(text) {
            Ok(interval) => Some(interval),
            Err(error) => return Ok(Some(format!("Error: {error}"))),
        },
        None => None,
    };

    let source = match dashboard::resolve_source(
        args.get_flag("demo"),
        args.get_one::<String>("metrics_path").map(String::as_str),
        &settings,
        state.active_script_path.as_deref(),
    ) {
        Ok(source) => source,
        Err(error) => return Ok(Some(format!("Error: {error}"))),
    };

    let title = match state.active_script_name.as_deref() {
        Some(name) => format!("Training dashboard - {name}"),
        None => "Training dashboard".to_string(),
    };

    match dashboard::run_dashboard(DashboardConfig::new(
        title,
        source,
        settings,
        watcher,
        interval_override,
    )) {
        Ok(summary) => Ok(Some(summary)),
        Err(error) => Ok(Some(format!("Error: {error}"))),
    }
}

/// The `/dashboard` command, including its `training` subcommand.
fn dashboard_command() -> Command {
    Command::new("/dashboard")
        .arg(
            Arg::new("metrics_path")
                .required(false)
                .value_name("ABSOLUTE_PATH")
                .help("JSON-lines metrics file (default: metrics.jsonl next to the active script)"),
        )
        .arg(
            Arg::new("interval_ms")
                .long("interval")
                .value_name("MS")
                .help("Refresh interval in ms (50-60000); overrides the config file"),
        )
        .arg(
            Arg::new("demo")
                .long("demo")
                .action(ArgAction::SetTrue)
                .help("Show synthetic training data instead of reading a file"),
        )
        .subcommand(
            Command::new("training")
                .arg(
                    Arg::new("path")
                        .required(false)
                        .value_name("ABSOLUTE_PATH")
                        .help("Events file (default: training_dashboard.jsonl next to the active script)"),
                )
                .arg(
                    Arg::new("interval_ms")
                        .long("interval")
                        .value_name("MS")
                        .help("Refresh interval in ms (50-60000); overrides the config file"),
                )
                .arg(
                    Arg::new("demo")
                        .long("demo")
                        .action(ArgAction::SetTrue)
                        .help("Show a synthetic training run instead of reading a file"),
                )
                .about("Live dashboard for a script using init_training_dashboard / update_training_dashboard / update_hero_text"),
        )
        .about("Open a live dashboard of training metrics, or `/dashboard training` for the Python API view (q / Esc / Ctrl+C to exit)")
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let tc_json = TcJson::load().map_err(std::io::Error::other)?;
    let initial_state = AppState {
        count: 0,
        panels: PanelHub::new(),
        tc_json,
        active_script_name: None,
        active_script_path: None,
        active_associate_name: None,
        active_associate_path: None,
        active_associate_config: None,
        runs: Vec::new(),
    };

    let mut repl = Repl::new(initial_state)
        .with_name("TrainingController")
        .with_version("v0.1.0")
        .with_banner("Type / and press Tab to see available commands.")
        .with_command(
            Command::new("/ping").about("Check responsiveness and increment counter"),
            cmd_help_ping,
        )
        .with_command(
            Command::new("/echo")
                .arg(Arg::new("text").required(true))
                .about("Echo back your input"),
            cmd_echo,
        )
        .with_command(
            Command::new("/test_get_terminal_arguments")
                .arg(
                    Arg::new("path")
                        .required(true)
                        .value_name("ABSOLUTE_PATH")
                        .help("Absolute path to a Python file"),
                )
                .about("Test get_terminal_arguments on a Python file"),
            cmd_test_get_terminal_arguments,
        )
        .with_command(
            Command::new("/run_script")
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .allow_hyphen_values(true)
                        .value_name("ARGS")
                        .help("A saved config name and/or arguments passed to the script"),
                )
                .about("Start the active script in the background (output goes to /panel); first argument may be a saved config name"),
            cmd_run_script,
        )
        .with_command(
            Command::new("/run_python")
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .allow_hyphen_values(true)
                        .value_name("ARGS")
                        .help("A saved config name and/or arguments passed to the script"),
                )
                .about("Alias of /run_script"),
            cmd_run_script,
        )
        .with_command(
            Command::new("/pause").about(
                "Ask the running active script to pause: calls the pause method it registered \
                 with tc.register_pause(fn)",
            ),
            cmd_pause,
        )
        .with_command(
            Command::new("/resume").about(
                "Start the active script again with the flags of its last run plus --resume",
            ),
            cmd_resume,
        )
        .with_command(
            Command::new("/run_wsl")
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .allow_hyphen_values(true)
                        .value_name("ARGS")
                        .help("A saved config name and/or arguments passed to the script"),
                )
                .about("Like /run_script, but runs the active script with the Python inside WSL2"),
            cmd_run_wsl,
        )
        .with_command(
            Command::new("/resume_wsl").about(
                "Like /resume, but runs the active script with the Python inside WSL2",
            ),
            cmd_resume_wsl,
        )
        .with_command(
            Command::new("/wsl_status").about(
                "Check that WSL2 can run scripts: its Python, and the Linux training_controller module",
            ),
            cmd_wsl_status,
        )
        .with_command(
            Command::new("/add_config")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .allow_hyphen_values(true)
                        .value_name("FLAGS")
                        .help("Flags saved under NAME, e.g. --epochs 5 --lr 0.01"),
                )
                .about("Save a named set of flags for the active script"),
            cmd_add_config,
        )
        .with_command(
            Command::new("/edit_config")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .allow_hyphen_values(true)
                        .value_name("FLAGS")
                        .help("New flags that replace the saved ones"),
                )
                .about("Replace the flags of an existing config"),
            cmd_edit_config,
        )
        .with_command(
            Command::new("/show_configs")
                .arg(Arg::new("name").required(false).value_name("NAME"))
                .about("List the active script's configs (or show one by name)"),
            cmd_show_configs,
        )
        .with_command(
            Command::new("/remove_config")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .about("Delete a config of the active script, with confirmation"),
            cmd_remove_config,
        )
        .with_command(
            Command::new("/rename_config")
                .arg(Arg::new("old_name").required(true).value_name("OLD"))
                .arg(Arg::new("new_name").required(true).value_name("NEW"))
                .about("Rename a config of the active script"),
            cmd_rename_config,
        )
        .with_command(
            Command::new("/open_script")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .about("Select a script by name to be the active script"),
            cmd_open_script,
        )
        .with_command(
            Command::new("/add_scripts")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .arg(
                    Arg::new("absolute_path")
                        .required(true)
                        .value_name("ABSOLUTE_PATH")
                        .help("Absolute path to the script"),
                )
                .about("Register a script in tc.json"),
            cmd_add_scripts,
        )
        .with_command(
            Command::new("/show_scripts")
                .about("Show all saved scripts and their paths with an index"),
            cmd_show_scripts,
        )
        .with_command(
            Command::new("/select_script")
                .arg(Arg::new("index").required(true).value_name("INDEX"))
                .about("Select a script by index to be the active script"),
            cmd_select_script,
        )
        .with_command(
            Command::new("/remove_script")
                .arg(Arg::new("index").required(true).value_name("INDEX"))
                .about("Delete a script by index from tc.json with confirmation"),
            cmd_remove_script,
        )
        .with_command(
            Command::new("/show_experiments")
                .arg(
                    Arg::new("limit")
                        .required(false)
                        .value_name("LIMIT")
                        .help("How many experiments to list (default: 20)"),
                )
                .about("List recorded experiments (newest first) with status and log columns"),
            cmd_show_experiments,
        )
        .with_command(
            Command::new("/show_log")
                .arg(Arg::new("id").required(true).value_name("EXPERIMENT_ID"))
                .arg(
                    Arg::new("last")
                        .long("last")
                        .value_name("N")
                        .help("Show only the last N steps (default: 20)"),
                )
                .about("Show the logged values of an experiment"),
            cmd_show_log,
        )
        .with_command(graph_command(), cmd_graph)
        .with_command(
            Command::new("/add_graph_config")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .value_name("GRAPHS")
                        .help("Y [X] [+ Y [X] ...]; Y may be a comma list (one graph each)"),
                )
                .about("Save a named set of graphs (applies to any experiment: /graph NAME IDS)"),
            cmd_add_graph_config,
        )
        .with_command(
            Command::new("/edit_graph_config")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .value_name("GRAPHS")
                        .help("New graphs that replace the saved ones"),
                )
                .about("Replace the graphs of an existing graph config"),
            cmd_edit_graph_config,
        )
        .with_command(
            Command::new("/show_graph_configs")
                .arg(Arg::new("name").required(false).value_name("NAME"))
                .about("List the saved graph configs (or show one by name)"),
            cmd_show_graph_configs,
        )
        .with_command(
            Command::new("/remove_graph_config")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .about("Delete a graph config, with confirmation"),
            cmd_remove_graph_config,
        )
        .with_command(dashboard_command(), cmd_dashboard)
        .with_command(
            Command::new("/dashboard_config")
                .about("Show the dashboard config file path (created with defaults if missing)"),
            cmd_dashboard_config,
        )
        .with_command(
            Command::new("/panel")
                .arg(
                    Arg::new("name")
                        .required(false)
                        .value_name("NAME")
                        .help("output, error, log, other, or any [tag] used by the script (default: output)"),
                )
                .about("Open a live view of a script output panel (q / Esc / Ctrl+C to exit)"),
            cmd_panel,
        )
        .with_command(
            Command::new("/show_active_panels")
                .about("List the panels with their line counts and the recent script runs"),
            cmd_show_active_panels,
        )
        .with_command(
            Command::new("/clear_panels")
                .arg(
                    Arg::new("name")
                        .required(false)
                        .value_name("NAME")
                        .help("Panel to clear (default: all panels)"),
                )
                .about("Clear the captured output of one panel or of all panels"),
            cmd_clear_panels,
        )
        .with_command(
            Command::new("/show_checkpoint_config")
                .about("Show the checkpoint storage settings of the active script"),
            cmd_show_checkpoint_config,
        )
        .with_command(
            Command::new("/set_checkpoint_config")
                .arg(
                    Arg::new("key")
                        .required(true)
                        .value_name("KEY")
                        .help("keep_last, keep_best, keep_first, keep_every, max_total_mb, save_every, verify, name_prefix or extension"),
                )
                .arg(
                    Arg::new("value")
                        .required(true)
                        .value_name("VALUE")
                        .help("A number, true/false, text, or all/off depending on the setting"),
                )
                .about("Change one checkpoint storage setting of the active script"),
            cmd_set_checkpoint_config,
        )
        .with_command(
            Command::new("/checkpoint_preset")
                .arg(
                    Arg::new("name")
                        .required(false)
                        .value_name("NAME")
                        .help("all, best_only, last_and_best or milestones (omit to list them)"),
                )
                .about("Apply a ready-made retention preset (or list the presets)"),
            cmd_checkpoint_preset,
        )
        .with_command(
            Command::new("/reset_checkpoint_config")
                .about("Reset the checkpoint settings of the active script to the defaults (keep everything)"),
            cmd_reset_checkpoint_config,
        )
        .with_command(
            Command::new("/checkpoints")
                .arg(
                    Arg::new("experiment")
                        .required(false)
                        .value_name("EXPERIMENT_ID")
                        .help("Only show checkpoints saved by this experiment"),
                )
                .about("List the stored checkpoints of the active script"),
            cmd_checkpoints,
        )
        .with_command(
            Command::new("/pin_checkpoint")
                .arg(Arg::new("id").required(true).value_name("ID").help("A checkpoint id, 'latest' or 'best'"))
                .about("Protect a checkpoint from pruning"),
            cmd_pin_checkpoint,
        )
        .with_command(
            Command::new("/unpin_checkpoint")
                .arg(Arg::new("id").required(true).value_name("ID").help("A checkpoint id, 'latest' or 'best'"))
                .about("Let the retention rules apply to a checkpoint again"),
            cmd_unpin_checkpoint,
        )
        .with_command(
            Command::new("/remove_checkpoint")
                .arg(Arg::new("id").required(true).value_name("ID").help("A checkpoint id, 'latest' or 'best'"))
                .about("Delete one checkpoint, with confirmation (pinned ones are refused)"),
            cmd_remove_checkpoint,
        )
        .with_command(
            Command::new("/prune_checkpoints")
                .arg(
                    Arg::new("mode")
                        .required(false)
                        .value_name("dry")
                        .help("Write 'dry' to preview without deleting"),
                )
                .about("Apply the retention settings to the existing checkpoints now"),
            cmd_prune_checkpoints,
        )
        .with_command(
            Command::new("/verify_checkpoints")
                .about("Check every checkpoint against its recorded size and SHA-256"),
            cmd_verify_checkpoints,
        )
        .with_command(
            Command::new("/export_checkpoint")
                .arg(Arg::new("id").required(true).value_name("ID").help("A checkpoint id, 'latest' or 'best'"))
                .arg(
                    Arg::new("destination")
                        .required(true)
                        .value_name("DESTINATION")
                        .help("A folder, or a new file path (existing files are never overwritten)"),
                )
                .about("Copy a checkpoint out of the store"),
            cmd_export_checkpoint,
        )
        .with_command(
            Command::new("/add_associate_script")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .arg(
                    Arg::new("absolute_path")
                        .required(true)
                        .value_name("ABSOLUTE_PATH")
                        .help("Absolute path to the associate script"),
                )
                .about("Register an associate script (evaluation, inference, ...) in tc.json"),
            cmd_add_associate_script,
        )
        .with_command(
            Command::new("/show_associate_scripts")
                .about("Show all associate scripts and their paths with an index"),
            cmd_show_associate_scripts,
        )
        .with_command(
            Command::new("/select_associate_script")
                .arg(Arg::new("index").required(true).value_name("INDEX"))
                .about("Select an associate script by index for /test"),
            cmd_select_associate_script,
        )
        .with_command(
            Command::new("/open_associate_script")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .about("Select an associate script by name for /test"),
            cmd_open_associate_script,
        )
        .with_command(
            Command::new("/remove_associate_script")
                .arg(Arg::new("index").required(true).value_name("INDEX"))
                .about("Delete an associate script and its configs by index, with confirmation"),
            cmd_remove_associate_script,
        )
        .with_command(
            Command::new("/add_associate_config")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .allow_hyphen_values(true)
                        .value_name("FLAGS")
                        .help("Flags saved under NAME, e.g. --split val --samples 100"),
                )
                .about("Save a named set of flags for the selected associate script"),
            cmd_add_associate_config,
        )
        .with_command(
            Command::new("/edit_associate_config")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .allow_hyphen_values(true)
                        .value_name("FLAGS")
                        .help("New flags that replace the saved ones"),
                )
                .about("Replace the flags of an associate config"),
            cmd_edit_associate_config,
        )
        .with_command(
            Command::new("/show_associate_configs")
                .arg(Arg::new("name").required(false).value_name("NAME"))
                .about("List the selected associate script's configs (or show one by name)"),
            cmd_show_associate_configs,
        )
        .with_command(
            Command::new("/remove_associate_config")
                .arg(Arg::new("name").required(true).value_name("NAME"))
                .about("Delete a config of the selected associate script, with confirmation"),
            cmd_remove_associate_config,
        )
        .with_command(
            Command::new("/rename_associate_config")
                .arg(Arg::new("old_name").required(true).value_name("OLD"))
                .arg(Arg::new("new_name").required(true).value_name("NEW"))
                .about("Rename a config of the selected associate script"),
            cmd_rename_associate_config,
        )
        .with_command(
            Command::new("/select_associate_config")
                .arg(
                    Arg::new("name")
                        .required(false)
                        .value_name("NAME")
                        .help("Config to use for /test (omit to run without a config)"),
                )
                .about("Select the associate config that /test uses"),
            cmd_select_associate_config,
        )
        .with_command(test_command(), cmd_test)
        .with_command(test_wsl_command(), cmd_test_wsl);

    repl.run()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_tc_json(document: Value) -> TcJson {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        TcJson {
            path: std::env::temp_dir().join(format!(
                "training_controller-configs-{}-{unique_id}.json",
                std::process::id()
            )),
            document,
        }
    }

    #[test]
    fn configs_are_added_edited_renamed_and_removed_per_script() {
        let mut tc = temp_tc_json(serde_json::json!({
            "scripts": [
                {"name": "a", "path": "/a.py"},
                {"name": "b", "path": "/b.py"}
            ]
        }));
        let flags = vec!["--epochs".to_string(), "5".to_string()];

        tc.add_config("a", "/a.py", "fast", &flags).unwrap();
        assert_eq!(tc.get_config("a", "/a.py", "fast"), Some(flags.clone()));
        // Configs belong to one script only.
        assert_eq!(tc.get_config("b", "/b.py", "fast"), None);
        assert!(tc.add_config("a", "/a.py", "fast", &flags).is_err());
        assert!(tc.add_config("a", "/a.py", "bad name", &flags).is_err());
        assert!(tc.add_config("zzz", "/zzz.py", "x", &flags).is_err());

        tc.edit_config("a", "/a.py", "fast", &["--x".to_string()]).unwrap();
        assert!(tc.edit_config("a", "/a.py", "missing", &[]).is_err());

        tc.rename_config("a", "/a.py", "fast", "slow").unwrap();
        assert_eq!(tc.get_config("a", "/a.py", "fast"), None);
        assert_eq!(
            tc.get_config("a", "/a.py", "slow"),
            Some(vec!["--x".to_string()])
        );
        assert!(tc.rename_config("a", "/a.py", "missing", "other").is_err());

        tc.remove_config("a", "/a.py", "slow").unwrap();
        assert!(tc.remove_config("a", "/a.py", "slow").is_err());
        assert!(tc.configs("a", "/a.py").is_empty());

        // Everything was persisted to disk.
        let saved: Value = serde_json::from_slice(&fs::read(tc.path()).unwrap()).unwrap();
        assert_eq!(saved["scripts"][0]["configs"], serde_json::json!({}));
        fs::remove_file(tc.path()).unwrap();
    }

    #[test]
    fn graph_configs_are_added_edited_and_removed() {
        let mut tc = temp_tc_json(serde_json::json!({
            "scripts": [{"name": "a", "path": "/a.py"}]
        }));
        assert!(tc.graph_configs().is_empty());

        let specs = vec![GraphSpec::new("loss", "time"), GraphSpec::new("acc", "step")];
        tc.add_graph_config("train", &specs).unwrap();
        assert_eq!(tc.get_graph_config("train"), Some(specs.clone()));
        assert_eq!(tc.get_graph_config("other"), None);

        // Names must be valid, new, and must not look like experiment IDs.
        assert!(tc.add_graph_config("train", &specs).unwrap_err().contains("already exists"));
        assert!(tc.add_graph_config("bad name", &specs).is_err());
        assert!(tc.add_graph_config("12", &specs).unwrap_err().contains("experiment ID"));
        assert!(tc.add_graph_config("3,4", &specs).is_err());
        assert!(tc.add_graph_config("empty", &[]).unwrap_err().contains("at least one"));

        let edited = vec![GraphSpec::new("lr", "time")];
        tc.edit_graph_config("train", &edited).unwrap();
        assert_eq!(tc.get_graph_config("train"), Some(edited.clone()));
        assert!(tc.edit_graph_config("missing", &edited).is_err());

        tc.add_graph_config("another", &specs).unwrap();
        let names: Vec<String> = tc.graph_configs().into_iter().map(|(name, _)| name).collect();
        assert_eq!(names, vec!["another", "train"]);

        // Saved to disk next to the scripts, in the documented shape.
        let saved: Value = serde_json::from_slice(&fs::read(tc.path()).unwrap()).unwrap();
        assert_eq!(
            saved["graph_configs"]["train"],
            serde_json::json!([{"y": "lr", "x": "time"}])
        );
        assert_eq!(saved["scripts"][0]["name"], "a");

        tc.remove_graph_config("train").unwrap();
        assert!(tc.remove_graph_config("train").is_err());
        assert_eq!(tc.get_graph_config("train"), None);
        fs::remove_file(tc.path()).unwrap();
    }

    #[test]
    fn graph_command_keeps_plus_signs_and_save_flags() {
        let matches = graph_command()
            .try_get_matches_from([
                "/graph", "3,4", "loss", "acc", "+", "lr", "--save", "mine",
            ])
            .unwrap();
        let tokens = collect_args(&matches);
        assert_eq!(tokens, vec!["3,4", "loss", "acc", "+", "lr", "--save", "mine"]);

        let parsed = graphs::parse_graph_command(&tokens, |_| false).unwrap();
        assert_eq!(parsed.ids, Some(vec![3, 4]));
        assert_eq!(parsed.save.as_deref(), Some("mine"));
        assert_eq!(
            parsed.selection,
            Selection::Explicit(vec![
                GraphSpec::new("loss", "acc"),
                GraphSpec::new("lr", "time")
            ])
        );

        // At least one argument is required.
        assert!(graph_command().try_get_matches_from(["/graph"]).is_err());
    }

    #[test]
    fn associate_scripts_belong_to_one_script() {
        let directory = std::env::current_dir().unwrap();
        let path_of = |file: &str| directory.join(file).to_string_lossy().into_owned();
        let (train_path, other_path) = (path_of("train.py"), path_of("other.py"));
        let (eval, plot) = (path_of("eval.py"), path_of("plot.py"));
        let mut tc = temp_tc_json(serde_json::json!({
            "scripts": [
                {"name": "train", "path": train_path},
                {"name": "other", "path": other_path}
            ]
        }));
        assert!(tc.associate_scripts("train", &train_path).is_none());

        tc.add_associate_script("train", &train_path, "eval", &eval).unwrap();
        tc.add_associate_script("train", &train_path, "plot", &plot).unwrap();
        // The same name is fine under another training script.
        tc.add_associate_script("other", &other_path, "eval", &plot).unwrap();

        // Names are unique per training script, paths absolute, names not blank,
        // and the training script must be registered.
        assert!(tc
            .add_associate_script("train", &train_path, "eval", &plot)
            .unwrap_err()
            .contains("already has"));
        assert!(tc
            .add_associate_script("train", &train_path, "rel", "relative.py")
            .unwrap_err()
            .contains("absolute"));
        assert!(tc.add_associate_script("train", &train_path, "  ", &plot).is_err());
        assert!(tc
            .add_associate_script("missing", "/missing.py", "x", &eval)
            .unwrap_err()
            .contains("not registered"));
        assert_eq!(tc.associate_scripts("train", &train_path).unwrap().len(), 2);
        assert_eq!(tc.associate_scripts("other", &other_path).unwrap().len(), 1);

        // Saved inside each training script's entry.
        let saved: Value = serde_json::from_slice(&fs::read(tc.path()).unwrap()).unwrap();
        assert_eq!(
            saved["scripts"][0]["associates"],
            serde_json::json!([
                {"name": "eval", "path": eval},
                {"name": "plot", "path": plot}
            ])
        );
        assert_eq!(saved["scripts"][1]["associates"][0]["path"], plot.as_str());
        assert!(saved.get("associate_scripts").is_none());

        assert_eq!(
            tc.remove_associate_script("train", &train_path, 0).unwrap(),
            ("eval".to_string(), eval.clone())
        );
        assert!(tc
            .remove_associate_script("train", &train_path, 5)
            .unwrap_err()
            .contains("out of bounds"));
        assert_eq!(tc.associate_scripts("train", &train_path).unwrap().len(), 1);
        assert_eq!(tc.associate_scripts("other", &other_path).unwrap().len(), 1);

        // Removing a training script takes its associate scripts with it.
        tc.remove_script(0).unwrap();
        assert!(tc.associate_scripts("train", &train_path).is_none());
        assert_eq!(tc.associate_scripts("other", &other_path).unwrap().len(), 1);
        fs::remove_file(tc.path()).unwrap();
    }

    #[test]
    fn associate_configs_belong_to_one_associate_script() {
        let directory = std::env::current_dir().unwrap();
        let path_of = |file: &str| directory.join(file).to_string_lossy().into_owned();
        let (train_path, other_path) = (path_of("train.py"), path_of("other.py"));
        let (eval, plot) = (path_of("eval.py"), path_of("plot.py"));
        let mut tc = temp_tc_json(serde_json::json!({
            "scripts": [
                {"name": "train", "path": train_path},
                {"name": "other", "path": other_path}
            ]
        }));
        tc.add_associate_script("train", &train_path, "eval", &eval).unwrap();
        tc.add_associate_script("train", &train_path, "plot", &plot).unwrap();
        tc.add_associate_script("other", &other_path, "eval", &eval).unwrap();
        let flags = vec!["--split".to_string(), "val".to_string()];
        let (t, tp, o, op) = ("train", train_path.as_str(), "other", other_path.as_str());

        tc.add_associate_config(t, tp, "eval", "quick", &flags).unwrap();
        assert_eq!(tc.get_associate_config(t, tp, "eval", "quick"), Some(flags.clone()));
        // Configs belong to one associate script of one training script only.
        assert_eq!(tc.get_associate_config(t, tp, "plot", "quick"), None);
        assert_eq!(tc.get_associate_config(o, op, "eval", "quick"), None);
        assert!(tc
            .add_associate_config(t, tp, "eval", "quick", &flags)
            .unwrap_err()
            .contains("already exists"));
        assert!(tc.add_associate_config(t, tp, "eval", "bad name", &flags).is_err());
        assert!(tc.add_associate_config(t, tp, "missing", "x", &flags).is_err());
        assert!(tc.add_associate_config("nope", "/nope.py", "eval", "x", &flags).is_err());

        tc.edit_associate_config(t, tp, "eval", "quick", &["--x".to_string()]).unwrap();
        assert!(tc.edit_associate_config(t, tp, "eval", "nope", &[]).is_err());

        tc.rename_associate_config(t, tp, "eval", "quick", "full").unwrap();
        assert_eq!(tc.get_associate_config(t, tp, "eval", "quick"), None);
        assert_eq!(
            tc.get_associate_config(t, tp, "eval", "full"),
            Some(vec!["--x".to_string()])
        );
        assert!(tc.rename_associate_config(t, tp, "eval", "nope", "other").is_err());

        tc.remove_associate_config(t, tp, "eval", "full").unwrap();
        assert!(tc.remove_associate_config(t, tp, "eval", "full").is_err());
        assert!(tc.associate_configs(t, tp, "eval").is_empty());

        // Removing an associate script takes its configs with it.
        tc.add_associate_config(t, tp, "plot", "a", &flags).unwrap();
        tc.remove_associate_script(t, tp, 1).unwrap();
        assert!(tc.associate_configs(t, tp, "plot").is_empty());
        fs::remove_file(tc.path()).unwrap();
    }

    #[test]
    fn selecting_another_script_clears_the_associate_selection() {
        let mut state = AppState {
            count: 0,
            panels: PanelHub::new(),
            tc_json: temp_tc_json(serde_json::json!({})),
            active_script_name: None,
            active_script_path: None,
            active_associate_name: None,
            active_associate_path: None,
            active_associate_config: None,
            runs: Vec::new(),
        };
        activate_script(&mut state, "train", "/train.py");
        activate_associate(&mut state, "eval", "/eval.py");
        state.active_associate_config = Some("quick".to_string());

        // Selecting the same training script again keeps the selection.
        activate_script(&mut state, "train", "/train.py");
        assert_eq!(state.active_associate_name.as_deref(), Some("eval"));
        assert_eq!(state.active_associate_config.as_deref(), Some("quick"));

        // Another training script has its own associate scripts.
        activate_script(&mut state, "other", "/other.py");
        assert_eq!(state.active_script_name.as_deref(), Some("other"));
        assert!(state.active_associate_name.is_none());
        assert!(state.active_associate_path.is_none());
        assert!(state.active_associate_config.is_none());
        assert!(active_associate(&state).is_err());
    }

    #[test]
    fn test_command_keeps_hyphenated_arguments() {
        let matches = test_command()
            .try_get_matches_from(["/test", "--split", "val", "--n", "3"])
            .unwrap();
        assert_eq!(collect_args(&matches), vec!["--split", "val", "--n", "3"]);

        let bare = test_command().try_get_matches_from(["/test"]).unwrap();
        assert!(collect_args(&bare).is_empty());
    }

    #[test]
    fn last_run_is_remembered_per_script_and_saved_to_disk() {
        let mut tc = temp_tc_json(serde_json::json!({
            "scripts": [
                {"name": "a", "path": "/a.py"},
                {"name": "b", "path": "/b.py"}
            ]
        }));
        assert_eq!(tc.last_run("a", "/a.py"), None);

        let flags = vec!["--epochs".to_string(), "5".to_string()];
        tc.set_last_run("a", "/a.py", Some("fast"), &flags).unwrap();
        assert_eq!(
            tc.last_run("a", "/a.py"),
            Some(LastRun { config: Some("fast".into()), args: flags.clone() })
        );
        // It belongs to one script only, and the script must be registered.
        assert_eq!(tc.last_run("b", "/b.py"), None);
        assert!(tc.set_last_run("zzz", "/zzz.py", None, &flags).is_err());

        // The next run replaces it (a run without a config clears the name).
        tc.set_last_run("a", "/a.py", None, &[]).unwrap();
        assert_eq!(
            tc.last_run("a", "/a.py"),
            Some(LastRun { config: None, args: Vec::new() })
        );

        // Saved in the script's entry, next to its configs, and readable after a reload.
        tc.add_config("a", "/a.py", "fast", &flags).unwrap();
        tc.set_last_run("a", "/a.py", Some("fast"), &flags).unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(tc.path()).unwrap()).unwrap();
        assert_eq!(
            saved["scripts"][0]["last_run"],
            serde_json::json!({"config": "fast", "args": ["--epochs", "5"]})
        );
        assert_eq!(saved["scripts"][0]["configs"]["fast"], serde_json::json!(["--epochs", "5"]));
        let reloaded = temp_tc_json(saved);
        assert_eq!(
            reloaded.last_run("a", "/a.py"),
            Some(LastRun { config: Some("fast".into()), args: flags.clone() })
        );

        // Removing the script removes its last run.
        tc.remove_script(0).unwrap();
        assert_eq!(tc.last_run("a", "/a.py"), None);
        fs::remove_file(tc.path()).unwrap();
    }

    #[test]
    fn resume_adds_the_resume_flag_once() {
        let last = vec!["--epochs".to_string(), "5".to_string()];
        assert_eq!(resume_args(&last), vec!["--epochs", "5", "--resume"]);
        assert_eq!(resume_args(&[]), vec!["--resume"]);

        let already = vec!["--resume".to_string(), "--lr".to_string(), "0.1".to_string()];
        assert_eq!(resume_args(&already), already);
    }

    #[test]
    fn pause_finds_the_newest_running_run_of_the_active_script() {
        let mut state = AppState {
            count: 0,
            panels: PanelHub::new(),
            tc_json: temp_tc_json(serde_json::json!({})),
            active_script_name: None,
            active_script_path: None,
            active_associate_name: None,
            active_associate_path: None,
            active_associate_config: None,
            runs: Vec::new(),
        };
        let tracked = |run_id: u32, name: &str| TrackedRun {
            run_id,
            script_name: name.to_string(),
            script_path: format!("/{name}.py"),
            control_dir: PathBuf::from("control"),
        };
        let (first, second) = state
            .panels
            .with(|store| (store.start_run("a"), store.start_run("b")));
        track_run(&mut state, tracked(first, "a"));
        track_run(&mut state, tracked(second, "b"));

        let found = |state: &AppState, name: &str| {
            running_run(state, name, &format!("/{name}.py")).map(|run| run.run_id)
        };
        assert_eq!(found(&state, "a"), Some(first));
        assert_eq!(found(&state, "b"), Some(second));
        assert_eq!(found(&state, "c"), None);
        // A script is identified by name and path together.
        assert!(running_run(&state, "a", "/elsewhere/a.py").is_none());

        // Two runs of one script: the newest one is paused.
        let third = state.panels.with(|store| store.start_run("a"));
        track_run(&mut state, tracked(third, "a"));
        assert_eq!(found(&state, "a"), Some(third));

        // Ended runs are not found, and are dropped from the list on the next launch.
        state.panels.with(|store| {
            store.finish_run(third, RunStatus::Exited(0));
            store.finish_run(first, RunStatus::Exited(0));
        });
        assert_eq!(found(&state, "a"), None);
        let fourth = state.panels.with(|store| store.start_run("c"));
        track_run(&mut state, tracked(fourth, "c"));
        let kept: Vec<u32> = state.runs.iter().map(|run| run.run_id).collect();
        assert_eq!(kept, vec![second, fourth]);
    }

    #[test]
    fn config_names_are_validated() {
        assert!(validate_config_name("fast_run-1.v2").is_ok());
        assert!(validate_config_name("").is_err());
        assert!(validate_config_name("has space").is_err());
        assert!(validate_config_name("--flag").is_err());
    }

    #[test]
    fn run_args_expand_a_leading_config_name() {
        let configs = vec![("fast".to_string(), vec!["--epochs".to_string(), "1".to_string()])];

        let (args, used) = resolve_run_args(&configs, vec!["fast".into(), "--seed".into(), "7".into()]);
        assert_eq!(args, vec!["--epochs", "1", "--seed", "7"]);
        assert_eq!(used.as_deref(), Some("fast"));

        let (args, used) = resolve_run_args(&configs, vec!["--epochs".into(), "9".into()]);
        assert_eq!(args, vec!["--epochs", "9"]);
        assert_eq!(used, None);

        let (args, used) = resolve_run_args(&configs, Vec::new());
        assert!(args.is_empty() && used.is_none());
    }

    #[test]
    fn format_args_display_quotes_arguments_with_spaces() {
        assert_eq!(format_args_display(&[]), "(no flags)");
        assert_eq!(
            format_args_display(&["--name".into(), "my run".into()]),
            "--name \"my run\""
        );
    }

    #[test]
    fn contains_bytes_finds_symbol_names() {
        assert!(contains_bytes(b"xx PyInit_training_controller yy", b"PyInit_training_controller"));
        assert!(!contains_bytes(b"nothing here", b"PyInit_training_controller"));
        assert!(!contains_bytes(b"abc", b""));
    }

    #[test]
    fn python_path_puts_module_dir_first_and_keeps_existing_entries() {
        let module_dir = PathBuf::from("module_dir");
        let existing = std::env::join_paths(["first", "second"]).unwrap();

        let joined = python_path_with(&module_dir, Some(existing)).unwrap();
        let entries: Vec<PathBuf> = std::env::split_paths(&joined).collect();
        assert_eq!(
            entries,
            vec![
                PathBuf::from("module_dir"),
                PathBuf::from("first"),
                PathBuf::from("second")
            ]
        );

        let alone = python_path_with(&module_dir, None).unwrap();
        assert_eq!(
            std::env::split_paths(&alone).collect::<Vec<_>>(),
            vec![PathBuf::from("module_dir")]
        );
    }

    #[test]
    fn add_script_appends_and_saves_script_entry() {
        let unique_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "training_controller-{}-{unique_id}.json",
            std::process::id()
        ));
        let mut tc_json = TcJson {
            path: path.clone(),
            document: serde_json::json!({
                "scripts": [{"name": "existing", "path": "C:\\scripts\\existing.py"}]
            }),
        };
        let absolute_path = std::env::current_dir()
            .unwrap()
            .join("new_script.py")
            .to_string_lossy()
            .into_owned();

        tc_json.add_script("new", &absolute_path).unwrap();

        let saved: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            saved["scripts"],
            serde_json::json!([
                {"name": "existing", "path": "C:\\scripts\\existing.py"},
                {"name": "new", "path": absolute_path}
            ])
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn show_scripts_handles_empty_and_populated_scripts() {
        let tc_json = TcJson {
            path: PathBuf::from("unused.json"),
            document: serde_json::json!({
                "scripts": [
                    {"name": "first", "path": "C:\\scripts\\first.py"},
                    {"name": "second", "path": "C:\\scripts\\second.py"}
                ]
            }),
        };

        let scripts = tc_json.scripts().unwrap();
        assert_eq!(scripts.len(), 2);
        assert_eq!(scripts[0]["name"], "first");
        assert_eq!(scripts[1]["path"], "C:\\scripts\\second.py");
    }

    #[test]
    fn dashboard_command_parses_every_form() {
        let parse = |args: &[&str]| dashboard_command().try_get_matches_from(args.to_vec());

        // The original forms keep working, including a path that is not "training".
        let plain = parse(&["/dashboard"]).unwrap();
        assert!(plain.subcommand().is_none());
        assert!(parse(&["/dashboard", "--demo"]).unwrap().get_flag("demo"));
        let with_path = parse(&["/dashboard", "C:\\runs\\metrics.jsonl", "--interval", "250"]).unwrap();
        assert!(with_path.subcommand().is_none());
        assert_eq!(
            with_path.get_one::<String>("metrics_path").map(String::as_str),
            Some("C:\\runs\\metrics.jsonl")
        );

        // The new view.
        let training = parse(&["/dashboard", "training"]).unwrap();
        let (name, sub) = training.subcommand().unwrap();
        assert_eq!(name, "training");
        assert!(!sub.get_flag("demo"));
        assert!(sub.get_one::<String>("path").is_none());

        let demo = parse(&["/dashboard", "training", "--demo", "--interval", "100"]).unwrap();
        let (_, sub) = demo.subcommand().unwrap();
        assert!(sub.get_flag("demo"));
        assert_eq!(sub.get_one::<String>("interval_ms").map(String::as_str), Some("100"));

        let path = parse(&["/dashboard", "training", "D:\\runs\\events.jsonl"]).unwrap();
        let (_, sub) = path.subcommand().unwrap();
        assert_eq!(
            sub.get_one::<String>("path").map(String::as_str),
            Some("D:\\runs\\events.jsonl")
        );
    }
}
