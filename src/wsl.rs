//! Running scripts inside WSL2.
//!
//! The controller is a Windows program, so a script that should run in WSL2 is
//! started through `wsl.exe`:
//!
//! ```text
//! wsl.exe [--distribution D] --cd <script folder> --exec python3 -u <script> ARGS...
//! ```
//!
//! Two things have to be translated, because the script sees a Linux system:
//!
//! * **Paths** on the command line (the script and its folder) are converted
//!   with [`to_wsl_path`]: `D:\x\train.py` becomes `/mnt/d/x/train.py`.
//! * **Environment variables.** `wsl.exe` only passes the variables listed in
//!   `WSLENV`, and a variable flagged `/p` (path) or `/l` (list of paths) is
//!   translated on the way (`C:\a;C:\b` becomes `/mnt/c/a:/mnt/c/b`). [`ScriptEnv`]
//!   collects the variables a run needs together with their kind and fills in
//!   both the variables and `WSLENV`, so the script finds its checkpoint
//!   folder, control folder, log database and so on exactly as a native run does.
//!
//! The Python module (`import training_controller`) must be the **Linux**
//! build of the extension; the Windows `.pyd` cannot be loaded there. It is
//! built inside WSL ([`BUILD_COMMAND`]) into its own target folder, so it never
//! mixes with the Windows build, and the controller copies it into
//! `pymodule_wsl` next to `tc.json` (see `prepare_wsl_module` in `main.rs`).
//!
//! Two optional environment variables of the controller pick the interpreter:
//! [`DISTRO_ENV`] (a WSL distribution, default: your default one) and
//! [`PYTHON_ENV`] (the Python to run inside WSL, default `python3`; use an
//! absolute path such as `/home/me/venv/bin/python` for a virtualenv, because
//! no shell is involved and `~` is not expanded).

use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::{Command, Stdio};

/// The program that starts something inside WSL.
pub const WSL_EXE: &str = "wsl.exe";

/// Optional: the WSL distribution to run scripts in (default: the default one).
pub const DISTRO_ENV: &str = "TC_WSL_DISTRO";

/// Optional: the Python to run inside WSL (default: [`DEFAULT_PYTHON`]).
pub const PYTHON_ENV: &str = "TC_WSL_PYTHON";

pub const DEFAULT_PYTHON: &str = "python3";

/// Folder next to tc.json that holds the Linux extension module for WSL.
pub const MODULE_DIR_NAME: &str = "pymodule_wsl";

/// Name Python expects for the extension module on Linux.
pub const MODULE_FILE_NAME: &str = "training_controller.so";

/// Name Cargo gives the compiled library on Linux.
pub const BUILT_LIB_FILE_NAME: &str = "libtraining_controller.so";

/// Cargo target folder (inside the project) for the Linux build.
pub const TARGET_DIR_NAME: &str = "target-wsl";

/// Run in the project folder from a WSL shell to build the Linux extension.
/// Only the library is built: the REPL binary must not be linked with
/// `extension-module`.
pub const BUILD_COMMAND: &str =
    "CARGO_TARGET_DIR=target-wsl cargo build --lib --features python,pyo3/extension-module";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Converts a Windows path to the path WSL uses for the same file.
///
/// * `D:\x\y.py` and `D:/x/y.py` become `/mnt/d/x/y.py`
/// * `\\?\D:\x` (verbatim prefix) is handled like `D:\x`
/// * `\\wsl$\Ubuntu\home\me` and `\\wsl.localhost\Ubuntu\home\me` become `/home/me`
/// * a path that already is a Linux path is returned unchanged
///
/// This assumes WSL's default mount point `/mnt`.
pub fn to_wsl_path(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let normalized = normalized.strip_prefix("//?/").unwrap_or(&normalized);

    let bytes = normalized.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
        let drive = bytes[0].to_ascii_lowercase() as char;
        let rest = normalized[2..].trim_start_matches('/');
        return if rest.is_empty() {
            format!("/mnt/{drive}")
        } else {
            format!("/mnt/{drive}/{rest}")
        };
    }

    for prefix in ["//wsl$/", "//wsl.localhost/"] {
        if let Some(rest) = normalized.strip_prefix(prefix) {
            // The first component is the distribution name.
            return match rest.split_once('/') {
                Some((_, tail)) => format!("/{tail}"),
                None => "/".to_string(),
            };
        }
    }

    normalized.to_string()
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// How WSL should treat an environment variable when it crosses over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvKind {
    /// Passed as it is.
    Plain,
    /// One Windows path, translated to a Linux path (`/p`).
    Path,
    /// A list of Windows paths, translated to a Linux path list (`/l`).
    PathList,
}

impl EnvKind {
    fn wslenv_flag(self) -> &'static str {
        match self {
            EnvKind::Plain => "",
            EnvKind::Path => "/p",
            EnvKind::PathList => "/l",
        }
    }
}

/// The environment variables of one script run, set the same way for a native
/// run ([`ScriptEnv::apply_native`]) and for a run in WSL ([`ScriptEnv::apply_wsl`]).
#[derive(Debug, Clone, Default)]
pub struct ScriptEnv {
    vars: Vec<(String, OsString, EnvKind)>,
}

impl ScriptEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, name: &str, value: impl AsRef<OsStr>, kind: EnvKind) {
        self.vars
            .push((name.to_string(), value.as_ref().to_os_string(), kind));
    }

    /// Sets the variables on a process that runs on this system.
    pub fn apply_native(&self, command: &mut Command) {
        for (name, value, _) in &self.vars {
            command.env(name, value);
        }
    }

    /// Sets the variables on a `wsl.exe` process and lists them in `WSLENV`
    /// (keeping whatever `WSLENV` the controller itself was started with).
    pub fn apply_wsl(&self, command: &mut Command) {
        for (name, value, _) in &self.vars {
            command.env(name, value);
        }
        let existing = std::env::var("WSLENV").ok();
        command.env("WSLENV", self.wslenv(existing.as_deref()));
    }

    /// The `WSLENV` value: `NAME[/flag]` entries separated by `:`.
    pub fn wslenv(&self, existing: Option<&str>) -> String {
        let mut entries: Vec<String> = self
            .vars
            .iter()
            .map(|(name, _, kind)| format!("{name}{}", kind.wslenv_flag()))
            .collect();
        if let Some(existing) = existing.map(str::trim).filter(|value| !value.is_empty()) {
            entries.push(existing.to_string());
        }
        entries.join(":")
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// `wsl.exe [--distribution D] [--cd DIR] --exec PYTHON`, ready for the
/// Python arguments. No shell is involved, so the arguments arrive untouched.
fn python_command(workdir: Option<&Path>, python: &str) -> Command {
    let mut command = Command::new(WSL_EXE);
    if let Some(distro) = env_value(DISTRO_ENV) {
        command.arg("--distribution").arg(distro);
    }
    if let Some(directory) = workdir.filter(|dir| !dir.as_os_str().is_empty()) {
        command
            .arg("--cd")
            .arg(to_wsl_path(&directory.to_string_lossy()));
    }
    command.arg("--exec").arg(python);
    command
}

/// The Python to use when nothing more specific applies: `TC_WSL_PYTHON` or `python3`.
fn default_python() -> String {
    env_value(PYTHON_ENV).unwrap_or_else(|| DEFAULT_PYTHON.to_string())
}

/// Folder names searched for a Linux virtualenv in the script's folder.
pub const VENV_DIR_NAMES: [&str; 3] = ["venv", ".venv", "env"];

/// The Linux virtualenv in the script's folder, if there is one.
///
/// A Linux venv has `pyvenv.cfg` and a `bin` folder; a Windows venv has
/// `Scripts` instead and is skipped. (Only existence is checked from the
/// Windows side: the `bin/python` symlink may not be readable there.)
pub fn venv_for(script: &Path) -> Option<std::path::PathBuf> {
    let folder = script.parent()?;
    VENV_DIR_NAMES
        .iter()
        .map(|name| folder.join(name))
        .find(|venv| venv.join("pyvenv.cfg").is_file() && venv.join("bin").is_dir())
}

/// The Python for a script, as a WSL path or command name. Order: the
/// `TC_WSL_PYTHON` override, the virtualenv next to the script, `python3`.
pub fn python_for(script: &Path) -> String {
    if let Some(python) = env_value(PYTHON_ENV) {
        return python;
    }
    match venv_for(script) {
        Some(venv) => format!("{}/bin/python", to_wsl_path(&venv.to_string_lossy())),
        None => DEFAULT_PYTHON.to_string(),
    }
}

/// The process that runs `script` with `script_args` inside WSL, started in
/// the script's folder. Environment and stdio are up to the caller.
pub fn script_command(script: &Path, script_args: &[String]) -> Command {
    let mut command = python_command(script.parent(), &python_for(script));
    command
        .arg("-u")
        .arg(to_wsl_path(&script.to_string_lossy()))
        .args(script_args);
    command
}

// ---------------------------------------------------------------------------
// Status check
// ---------------------------------------------------------------------------

/// What `/wsl_status` found out.
#[derive(Debug, Clone)]
pub struct Probe {
    pub python_version: String,
    /// Kernel release; WSL2 kernels say `WSL2`.
    pub kernel: String,
    /// `Ok` if `import training_controller` works inside WSL, else the reason.
    pub module: Result<(), String>,
}

/// Runs two tiny Python snippets inside WSL. Fails if WSL or its Python is not
/// usable at all; a missing `training_controller` module is reported in
/// [`Probe::module`] instead.
pub fn probe(module_dir: &Path) -> Result<Probe, String> {
    let mut env = ScriptEnv::new();
    env.set("PYTHONPATH", module_dir, EnvKind::PathList);
    env.set("PYTHONIOENCODING", "utf-8", EnvKind::Plain);

    let info = run_python(
        &env,
        "import sys, platform; print(sys.version.split()[0]); print(platform.release())",
    )?;
    let mut lines = info.lines();
    let python_version = lines.next().unwrap_or("unknown").trim().to_string();
    let kernel = lines.next().unwrap_or("unknown").trim().to_string();

    let module = run_python(&env, "import training_controller")
        .map(|_| ())
        .map_err(|error| last_line(&error));

    Ok(Probe {
        python_version,
        kernel,
        module,
    })
}

fn run_python(env: &ScriptEnv, code: &str) -> Result<String, String> {
    let mut command = python_command(None, &default_python());
    command.arg("-c").arg(code).stdin(Stdio::null());
    env.apply_wsl(&mut command);

    let output = command.output().map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            format!("{WSL_EXE} was not found. Install WSL2 with `wsl --install` in an administrator terminal.")
        } else {
            format!("Failed to run {WSL_EXE}: {error}")
        }
    })?;

    if output.status.success() {
        Ok(clean(&output.stdout))
    } else {
        let stderr = clean(&output.stderr);
        Err(if stderr.is_empty() {
            clean(&output.stdout)
        } else {
            stderr
        })
    }
}

/// Output as text. `wsl.exe` prints its own messages as UTF-16, which shows up
/// as text with NUL bytes between the letters; dropping them makes it readable.
fn clean(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .replace('\0', "")
        .trim()
        .to_string()
}

fn last_line(text: &str) -> String {
    text.lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("unknown error")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drive_paths_become_mnt_paths() {
        assert_eq!(
            to_wsl_path(r"D:\training_controller\a.py"),
            "/mnt/d/training_controller/a.py"
        );
        assert_eq!(to_wsl_path("C:/Users/me"), "/mnt/c/Users/me");
        assert_eq!(to_wsl_path(r"c:\"), "/mnt/c");
        assert_eq!(to_wsl_path("E:"), "/mnt/e");
        assert_eq!(to_wsl_path(r"\\?\D:\x\y.py"), "/mnt/d/x/y.py");
    }

    #[test]
    fn linux_paths_and_wsl_shares_translate_to_linux_paths() {
        assert_eq!(to_wsl_path("/home/me/train.py"), "/home/me/train.py");
        assert_eq!(
            to_wsl_path(r"\\wsl$\Ubuntu\home\me\train.py"),
            "/home/me/train.py"
        );
        assert_eq!(to_wsl_path(r"\\wsl.localhost\Ubuntu\home\me"), "/home/me");
        assert_eq!(to_wsl_path(r"\\wsl$\Ubuntu"), "/");
    }

    #[test]
    fn wslenv_lists_every_variable_with_its_flag() {
        let mut env = ScriptEnv::new();
        env.set("TC_CONTROL_DIR", r"C:\x", EnvKind::Path);
        env.set("PYTHONUNBUFFERED", "1", EnvKind::Plain);
        env.set("PYTHONPATH", r"C:\a;C:\b", EnvKind::PathList);

        let own = "TC_CONTROL_DIR/p:PYTHONUNBUFFERED:PYTHONPATH/l";
        assert_eq!(env.wslenv(None), own);
        assert_eq!(env.wslenv(Some("  ")), own);
        assert_eq!(env.wslenv(Some("FOO/p")), format!("{own}:FOO/p"));
        assert_eq!(ScriptEnv::new().wslenv(None), "");
    }

    #[test]
    fn apply_sets_the_variables_and_only_wsl_gets_wslenv() {
        let mut env = ScriptEnv::new();
        env.set("TC_LOG_DB", r"C:\tc\logs.db", EnvKind::Path);

        let lookup = |command: &Command, name: &str| -> Option<String> {
            command
                .get_envs()
                .find(|(key, _)| *key == OsStr::new(name))
                .and_then(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
        };

        let mut native = Command::new("python");
        env.apply_native(&mut native);
        assert_eq!(lookup(&native, "TC_LOG_DB").as_deref(), Some(r"C:\tc\logs.db"));
        assert_eq!(lookup(&native, "WSLENV"), None);

        let mut wsl = Command::new(WSL_EXE);
        env.apply_wsl(&mut wsl);
        assert_eq!(lookup(&wsl, "TC_LOG_DB").as_deref(), Some(r"C:\tc\logs.db"));
        assert!(lookup(&wsl, "WSLENV").unwrap().starts_with("TC_LOG_DB/p"));
    }

    #[test]
    fn the_script_runs_through_wsl_with_translated_paths() {
        let command = script_command(
            Path::new(r"D:\runs\train.py"),
            &["--epochs".to_string(), "5".to_string()],
        );
        assert_eq!(command.get_program(), OsStr::new(WSL_EXE));

        let args: Vec<String> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        assert!(args.windows(2).any(|pair| pair == ["--cd", "/mnt/d/runs"]));
        assert!(args.contains(&"--exec"));
        assert_eq!(
            &args[args.len() - 4..],
            &["-u", "/mnt/d/runs/train.py", "--epochs", "5"]
        );
    }

    #[test]
    fn wsl_error_output_loses_its_nul_bytes() {
        let utf16_like = b"N\0o\0 \0d\0i\0s\0t\0r\0o\0\n\0";
        assert_eq!(clean(utf16_like), "No distro");
        assert_eq!(last_line("Traceback...\n  File x\nModuleNotFoundError: no module\n\n"), "ModuleNotFoundError: no module");
        assert_eq!(last_line(""), "unknown error");
    }
}
