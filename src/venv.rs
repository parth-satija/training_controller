//! Native virtualenvs: running scripts and shell commands inside the venv next
//! to the script, the way `activate` would.
//!
//! (The WSL counterpart, which looks for a *Linux* venv, is in `wsl.rs`.)
//!
//! A venv is found in the script's folder under the names in
//! [`VENV_DIR_NAMES`], in that order of priority. It counts when it has
//! `pyvenv.cfg` and a Python of this system (`Scripts\python.exe` on Windows,
//! `bin/python` elsewhere), so a Linux venv on a Windows drive, or a hollow
//! folder, is skipped in favour of the next candidate.
//!
//! Activating is done through the environment instead of running
//! `Activate.ps1` / `activate`: that works the same for scripts, PowerShell
//! and bash, and is not blocked by PowerShell's execution policy. What
//! `activate` really does is set `VIRTUAL_ENV` and put the venv's executable
//! folder first on `PATH`, which [`activate`] does.

use crate::wsl::{EnvKind, ScriptEnv};
use std::path::{Path, PathBuf};

/// Folder names searched in the script's folder, in priority order.
pub const VENV_DIR_NAMES: [&str; 3] = [".venv", "venv", "env"];

/// The folder with the venv's executables.
pub fn bin_dir(venv: &Path) -> PathBuf {
    venv.join(if cfg!(windows) { "Scripts" } else { "bin" })
}

/// The venv's Python interpreter.
pub fn python_in(venv: &Path) -> PathBuf {
    bin_dir(venv).join(if cfg!(windows) { "python.exe" } else { "python" })
}

/// The venv directly inside `folder`, if there is a usable one.
pub fn venv_in(folder: &Path) -> Option<PathBuf> {
    VENV_DIR_NAMES
        .iter()
        .map(|name| folder.join(name))
        .find(|venv| venv.join("pyvenv.cfg").is_file() && python_in(venv).is_file())
}

/// The venv next to `script`, if there is a usable one.
pub fn venv_for(script: &Path) -> Option<PathBuf> {
    venv_in(script.parent()?)
}

/// Adds what activating `venv` changes to `env`: `VIRTUAL_ENV`, and the venv's
/// executable folder at the front of `PATH`, so `python`, `pip` and anything
/// installed in the venv are found first.
pub fn activate(env: &mut ScriptEnv, venv: &Path) {
    env.set("VIRTUAL_ENV", venv, EnvKind::Path);
    let mut paths = vec![bin_dir(venv)];
    if let Some(existing) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    if let Ok(joined) = std::env::join_paths(paths) {
        env.set("PATH", joined, EnvKind::PathList);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::process::Command;

    fn fake_venv(folder: &Path, name: &str, with_python: bool) {
        let venv = folder.join(name);
        std::fs::create_dir_all(bin_dir(&venv)).unwrap();
        std::fs::write(venv.join("pyvenv.cfg"), "home = somewhere\n").unwrap();
        if with_python {
            std::fs::write(python_in(&venv), "").unwrap();
        }
    }

    #[test]
    fn dot_venv_wins_and_hollow_venvs_are_skipped() {
        let root = std::env::temp_dir().join(format!("tc_native_venv_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        assert_eq!(venv_in(&root), None);

        fake_venv(&root, "env", true);
        assert_eq!(venv_in(&root), Some(root.join("env")));
        fake_venv(&root, "venv", true);
        assert_eq!(venv_in(&root), Some(root.join("venv")));
        fake_venv(&root, ".venv", true);
        assert_eq!(venv_in(&root), Some(root.join(".venv")));
        assert_eq!(venv_for(&root.join("train.py")), Some(root.join(".venv")));

        // A `.venv` without a Python (or of the other OS) falls back to the next one.
        let _ = std::fs::remove_dir_all(root.join(".venv"));
        fake_venv(&root, ".venv", false);
        assert_eq!(venv_in(&root), Some(root.join("venv")));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn activating_sets_virtual_env_and_puts_the_venv_first_on_path() {
        let venv = Path::new("some_venv");
        let mut env = ScriptEnv::new();
        activate(&mut env, venv);

        let mut command = Command::new("python");
        env.apply_native(&mut command);
        let lookup = |name: &str| {
            command
                .get_envs()
                .find(|(key, _)| *key == OsStr::new(name))
                .and_then(|(_, value)| value.map(|v| v.to_os_string()))
        };
        assert_eq!(lookup("VIRTUAL_ENV"), Some(venv.as_os_str().to_os_string()));
        let path = lookup("PATH").unwrap();
        assert_eq!(std::env::split_paths(&path).next(), Some(bin_dir(venv)));
    }
}
