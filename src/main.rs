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

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let _tc_json = TcJson::load().map_err(std::io::Error::other)?;
    let initial_state = AppState { count: 0 };

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
        );

    repl.run()?;
    Ok(())
}
