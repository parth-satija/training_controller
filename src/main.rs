use reedline_repl_rs::clap::{Arg, ArgMatches, Command};
use reedline_repl_rs::{Repl, Result};

use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

// Shared state across commands
struct AppState {
    count: usize,
    tc_json: TcJson,
    active_script_name: Option<String>,
    active_script_path: Option<String>,
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

    state.active_script_name = Some(name.clone());
    state.active_script_path = Some(path.clone());

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

    state.active_script_name = Some(name.clone());
    state.active_script_path = Some(path.clone());

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

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let tc_json = TcJson::load().map_err(std::io::Error::other)?;
    let initial_state = AppState {
        count: 0,
        tc_json,
        active_script_name: None,
        active_script_path: None,
    };

    let mut repl = Repl::new(initial_state)
        .with_name("MyInteractiveCLI")
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
        );

    repl.run()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

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
}
