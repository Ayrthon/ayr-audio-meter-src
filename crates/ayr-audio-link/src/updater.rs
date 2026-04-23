//! GitHub Releases auto-updater for AYR Audio Link.
//!
//! This is a trimmed-down port of the meter's `src/updater.rs`. The state
//! machine, version parsing, and download-and-launch flow are deliberately
//! kept byte-for-byte compatible so fixes made to one can be ported to the
//! other with `diff` rather than a rewrite. The two differences:
//!
//! * `UPDATE_REPO` points at the Link's own sidecar repo
//!   (`Ayrthon/ayr-audio-link`). Releases there are independent of the
//!   meter's — the Link ships on its own cadence.
//! * We log through `log::info!` / `log::error!` rather than the meter's
//!   `startup_log` module, which is tied to the meter's splash-screen
//!   diagnostic file. The Link doesn't need that, and dragging it across
//!   would add a dependency for no user-visible benefit.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::Deserialize;

/// Sidecar public repo that hosts Link installers. Separate from the
/// meter's (`Ayrthon/ayr-audio-meter`) so the two products can version
/// independently and a "latest release" probe on either side is unambiguous.
pub const UPDATE_REPO: &str = "Ayrthon/ayr-audio-link";

#[derive(Clone, Debug)]
pub struct ReleaseInfo {
    pub version: String,
    pub installer_url: String,
    pub installer_filename: String,
    #[allow(dead_code)]
    pub html_url: String,
    #[allow(dead_code)]
    pub notes: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug)]
pub enum UpdateStatus {
    Idle,
    Checking,
    UpToDate,
    Available(ReleaseInfo),
    Downloading { progress: f32, total_bytes: u64 },
    Downloaded(PathBuf),
    LaunchFailed(String),
    Launching,
    #[allow(dead_code)]
    NetworkError(String),
}

pub type SharedUpdateStatus = Arc<Mutex<UpdateStatus>>;

pub fn make_shared_status() -> SharedUpdateStatus {
    Arc::new(Mutex::new(UpdateStatus::Idle))
}

// ---- Startup check ---------------------------------------------------------

pub fn spawn_check(repo: &'static str, current: &'static str, shared: SharedUpdateStatus) {
    {
        let mut s = shared.lock();
        if !matches!(*s, UpdateStatus::Idle) {
            return;
        }
        *s = UpdateStatus::Checking;
    }
    std::thread::Builder::new()
        .name("link-update-check".into())
        .spawn(move || {
            log::info!("updater: checking GitHub releases for {repo} (have {current})");
            let outcome = match fetch_latest_release(repo) {
                Ok(info) => {
                    if is_newer(&info.version, current) {
                        log::info!(
                            "updater: update available: {} (have {current})",
                            info.version
                        );
                        UpdateStatus::Available(info)
                    } else {
                        log::info!(
                            "updater: up to date ({current}; latest is {})",
                            info.version
                        );
                        UpdateStatus::UpToDate
                    }
                }
                Err(e) => {
                    log::warn!("updater: check failed: {e}");
                    UpdateStatus::NetworkError(e.to_string())
                }
            };
            *shared.lock() = outcome;
        })
        .expect("spawn link-update-check thread");
}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    body: String,
    assets: Vec<GhAsset>,
}

#[derive(Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: u64,
}

fn fetch_latest_release(repo: &str) -> anyhow::Result<ReleaseInfo> {
    let url = format!("https://api.github.com/repos/{repo}/releases/latest");
    let resp = ureq::get(&url)
        .set(
            "User-Agent",
            concat!("AYR-Audio-Link/", env!("CARGO_PKG_VERSION")),
        )
        .set("Accept", "application/vnd.github+json")
        .set("X-GitHub-Api-Version", "2022-11-28")
        .timeout(Duration::from_secs(10))
        .call()
        .map_err(|e| anyhow::anyhow!("github api: {e}"))?;

    let release: GhRelease = resp
        .into_json()
        .map_err(|e| anyhow::anyhow!("parse release json: {e}"))?;

    let installer = release
        .assets
        .iter()
        .find(|a| a.name.to_ascii_lowercase().ends_with(".exe"))
        .ok_or_else(|| anyhow::anyhow!("release has no .exe asset"))?;

    let version = release
        .tag_name
        .trim_start_matches(|c: char| c == 'v' || c == 'V')
        .to_string();

    Ok(ReleaseInfo {
        version,
        installer_url: installer.browser_download_url.clone(),
        installer_filename: installer.name.clone(),
        html_url: release.html_url.clone(),
        notes: release.body.clone(),
        size_bytes: installer.size,
    })
}

// ---- Version compare -------------------------------------------------------

fn parse_triple(v: &str) -> (u32, u32, u32) {
    let v = v.trim().trim_start_matches(|c: char| c == 'v' || c == 'V');
    let core = v
        .split(|c: char| c == '-' || c == '+')
        .next()
        .unwrap_or(v);
    let mut parts = core.split('.');
    let major = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let patch = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    (major, minor, patch)
}

pub fn is_newer(latest: &str, current: &str) -> bool {
    parse_triple(latest) > parse_triple(current)
}

// ---- Download + launch -----------------------------------------------------

pub fn spawn_download_and_run(info: ReleaseInfo, shared: SharedUpdateStatus) {
    {
        let mut s = shared.lock();
        *s = UpdateStatus::Downloading {
            progress: 0.0,
            total_bytes: info.size_bytes,
        };
    }
    let shared_bg = shared.clone();
    std::thread::Builder::new()
        .name("link-update-download".into())
        .spawn(move || {
            log::info!(
                "updater: downloading {} ({} bytes) from {}",
                info.installer_filename,
                info.size_bytes,
                info.installer_url
            );
            let outcome = match download_to_temp(&info, &shared_bg) {
                Ok(path) => {
                    log::info!("updater: downloaded to {}", path.display());
                    UpdateStatus::Downloaded(path)
                }
                Err(e) => {
                    log::error!("updater: download failed: {e}");
                    UpdateStatus::LaunchFailed(format!("Download failed: {e}"))
                }
            };
            *shared_bg.lock() = outcome;
        })
        .expect("spawn link-update-download thread");
}

fn download_to_temp(info: &ReleaseInfo, shared: &SharedUpdateStatus) -> anyhow::Result<PathBuf> {
    let resp = ureq::get(&info.installer_url)
        .set(
            "User-Agent",
            concat!("AYR-Audio-Link/", env!("CARGO_PKG_VERSION")),
        )
        .timeout(Duration::from_secs(60))
        .call()
        .map_err(|e| anyhow::anyhow!("download request: {e}"))?;

    let total: u64 = resp
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(info.size_bytes);

    let dest = std::env::temp_dir().join(&info.installer_filename);
    let mut file = std::fs::File::create(&dest)
        .map_err(|e| anyhow::anyhow!("create {}: {e}", dest.display()))?;

    let mut reader = resp.into_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded: u64 = 0;
    let mut last_notified: u64 = 0;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| anyhow::anyhow!("read body: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| anyhow::anyhow!("write {}: {e}", dest.display()))?;
        downloaded += n as u64;

        if downloaded - last_notified >= 256 * 1024 {
            last_notified = downloaded;
            let progress = if total > 0 {
                (downloaded as f32 / total as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            *shared.lock() = UpdateStatus::Downloading {
                progress,
                total_bytes: total,
            };
        }
    }

    file.sync_all()
        .map_err(|e| anyhow::anyhow!("flush {}: {e}", dest.display()))?;
    Ok(dest)
}

#[allow(dead_code)]
pub fn launch_installer(path: &Path) -> anyhow::Result<()> {
    log::info!("updater: spawning installer {}", path.display());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        std::process::Command::new(path)
            .creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
            .spawn()
            .map_err(|e| anyhow::anyhow!("spawn installer: {e}"))?;
    }
    #[cfg(not(windows))]
    {
        std::process::Command::new(path)
            .spawn()
            .map_err(|e| anyhow::anyhow!("spawn installer: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_compare_triples() {
        assert!(is_newer("1.0.1", "1.0.0"));
        assert!(is_newer("0.2.0", "0.1.99"));
        assert!(!is_newer("0.1.0", "0.1.0"));
        assert!(!is_newer("0.0.9", "0.1.0"));
    }

    #[test]
    fn version_compare_ignores_prefix_and_suffix() {
        assert!(is_newer("v0.2.0", "0.1.0"));
        assert!(is_newer("0.2.0-beta.1", "0.1.0"));
        assert!(!is_newer("0.1.0-rc.1", "0.1.0"));
    }
}
