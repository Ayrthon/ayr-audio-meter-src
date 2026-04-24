//! Persistent JSON settings under the OS config (or local data) directory.
//!
//! File: `{config_dir or data_local_dir}/audio-meter-rs/settings.json`

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// On-disk shape; extend with new optional fields as needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UserSettingsData {
    /// When `None`, clips use the built‑in default under local app data (`audio-meter-rs/clips`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clips_save_dir: Option<PathBuf>,

    /// Remember whether the user had the left column collapsed when the app exited,
    /// so the next launch opens with the same layout. Default (unset) = expanded.
    #[serde(default)]
    pub left_panel_collapsed: bool,

    /// First-run trial welcome modal has been dismissed. Stays `false` until
    /// the user closes the one-shot "Trial started — 14 days" dialog; after
    /// that we never show it again even across reinstalls (the anti-rollback
    /// trial anchor makes showing it again pointless anyway).
    #[serde(default)]
    pub license_welcome_shown: bool,

    /// Output endpoint the headphone monitor routes to. Matches OBS's
    /// "Settings → Audio → Monitoring Device" model:
    ///
    /// * `None`  — monitoring is disabled (listen button inert). This is the
    ///   default so users opt in explicitly and never accidentally create a
    ///   loopback feedback loop on first launch.
    /// * `Some(name)` — friendly name (`IMMDevice` friendly name on Windows,
    ///   `cpal::Device::name()` elsewhere) of the render endpoint to use.
    ///
    /// Lookup is by display name because cpal on Windows doesn't surface
    /// stable device IDs; names stay stable across app launches unless the
    /// user renames the device in Sound Settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monitor_device_name: Option<String>,
}

impl Default for UserSettingsData {
    fn default() -> Self {
        Self {
            clips_save_dir: None,
            left_panel_collapsed: false,
            license_welcome_shown: false,
            monitor_device_name: None,
        }
    }
}

pub type SharedUserSettings = Arc<Mutex<UserSettingsData>>;

pub fn make_shared_user_settings(data: UserSettingsData) -> SharedUserSettings {
    Arc::new(Mutex::new(data))
}

pub fn settings_file_path() -> Option<PathBuf> {
    let base = dirs::config_dir().or_else(dirs::data_local_dir)?;
    Some(base.join("audio-meter-rs").join("settings.json"))
}

pub fn load_settings_file() -> (UserSettingsData, Option<PathBuf>) {
    let path = settings_file_path();
    let Some(ref p) = path else {
        return (UserSettingsData::default(), None);
    };
    if let Ok(bytes) = std::fs::read(p) {
        if let Ok(data) = serde_json::from_slice::<UserSettingsData>(&bytes) {
            return (data, path);
        }
    }
    (UserSettingsData::default(), path)
}

pub fn save_settings_file(path: &Path, data: &UserSettingsData) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec_pretty(data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

/// Default `%LOCALAPPDATA%/audio-meter-rs/clips` (or platform equivalent).
pub fn default_clips_dir() -> Option<PathBuf> {
    let base = dirs::data_local_dir()?;
    let dir = base.join("audio-meter-rs").join("clips");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Effective WAV output directory for the given settings.
pub fn resolve_clips_dir(data: &UserSettingsData) -> Option<PathBuf> {
    if let Some(ref custom) = data.clips_save_dir {
        if custom.as_os_str().is_empty() {
            return default_clips_dir();
        }
        std::fs::create_dir_all(custom).ok()?;
        return Some(custom.clone());
    }
    default_clips_dir()
}

pub fn clips_path_display(data: &UserSettingsData) -> String {
    resolve_clips_dir(data)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(could not resolve clips folder)".into())
}
