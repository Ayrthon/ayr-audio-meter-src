//! In-app update checker + installer runner against GitHub Releases.
//!
//! # Flow
//!
//! 1. `spawn_check` hits `https://api.github.com/repos/{repo}/releases/latest` on a
//!    background thread, parses `tag_name` + the first `*.exe` asset, and compares
//!    the semver triple against the current `CARGO_PKG_VERSION`. The UI thread
//!    reads the shared [`UpdateStatus`] each frame and renders a modal when
//!    [`UpdateStatus::Available`] arrives.
//! 2. On "Update now" the UI calls [`spawn_download_and_run`], which streams the
//!    `.exe` asset into `%TEMP%`, transitions through `Downloading { progress }`
//!    so the modal can show a progress bar, and finally flips to `Downloaded`.
//! 3. The UI picks `Downloaded` up, calls [`launch_installer`] (which spawns the
//!    Inno Setup installer detached so it survives our exit), then sends
//!    `ViewportCommand::Close` to let the installer replace the running binary
//!    without the Restart Manager prompt.
//!
//! # Why GitHub Releases
//!
//! No auth anywhere (public repo), no tokens shipped in the binary, purpose-built
//! assets API, and the per-release workflow is one `gh release create …` call.
//! See `docs/release-checklist.md` for the exact commands.
//!
//! Network errors avoid a modal (logged via [`crate::startup_log`]); the title-row
//! chip shows a short "Couldn't reach GitHub…" line so offline isn't mistaken for
//! "skipped checking".
//!
//! When **`releases/latest` equals your build**, there is no update popup — that is
//! expected; [`UpdateStatus::UpToDate`] still proves the GitHub round-trip ran (chip text).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::Deserialize;

use crate::startup_log;

/// GitHub repo slug we poll for releases. This is **not** the source repo
/// (which is kept private) — it's a dedicated public "downloads" repo that
/// holds only compiled installers and release notes. Splitting them means we
/// can keep the code closed while still using the public GitHub Releases API
/// for the updater (the API returns 404 for private repos to unauthenticated
/// clients, and we intentionally don't ship a token).
///
/// Baked in at compile time. If you fork the project, point this at your own
/// public releases repo.
pub const UPDATE_REPO: &str = "Ayrthon/ayr-audio-meter";

/// Parsed view of a GitHub release's relevant fields.
#[derive(Clone, Debug)]
pub struct ReleaseInfo {
    /// Semver string as presented to the user (no leading `v`). Example: `"1.6.6"`.
    pub version: String,
    /// Direct download URL for the installer `.exe` asset (GitHub's
    /// `browser_download_url`, which is a signed redirect to the asset object).
    pub installer_url: String,
    /// Asset filename as it appears on the release (e.g.
    /// `AYR-Audio-Meter-Setup-1.6.6.exe`). Used as the basename when saving into
    /// `%TEMP%` so duplicate downloads don't collide with prior versions.
    pub installer_filename: String,
    /// GitHub release web page (for "View release notes on GitHub" fallbacks).
    pub html_url: String,
    /// Release body (markdown). Rendered as plain text in the modal — we don't
    /// have a markdown widget wired up.
    pub notes: String,
    /// Declared asset size in bytes (for progress-bar totals).
    pub size_bytes: u64,
}

/// State machine consumed by the UI thread every frame. The updater thread is
/// the only writer for `Checking`, `UpToDate`, `Available`, `Downloading`,
/// `Downloaded`, and `Error`. The UI thread flips `Launching` / `Dismissed`
/// after reacting to terminal states.
#[derive(Clone, Debug)]
pub enum UpdateStatus {
    /// Pre-`spawn_check` default. Never shown in UI.
    Idle,
    /// `spawn_check` is awaiting the GitHub API response.
    Checking,
    /// Latest release tag ≤ current version. Modal stays hidden.
    /// `github_latest` is the semver from `releases/latest` (after stripping `v`),
    /// so the UI can show that a network check completed.
    UpToDate {
        github_latest: String,
    },
    /// Latest release tag > current version. Show the "Update available" modal.
    Available(ReleaseInfo),
    /// User clicked "Update now". `progress` ∈ [0, 1]; `total_bytes` is 0 when
    /// the server didn't send `Content-Length` (GitHub always does, but be
    /// defensive).
    Downloading { progress: f32, total_bytes: u64 },
    /// Download finished at `path`. UI picks this up, runs the installer, and
    /// closes the app.
    Downloaded(PathBuf),
    /// We attempted to spawn the installer but `Command::spawn` errored (rare —
    /// usually "file not found" if AV quarantined the `.exe`). Shown to the
    /// user as a modal so they can retry via the browser.
    LaunchFailed(String),
    /// Installer has been spawned and `ViewportCommand::Close` is in flight.
    /// Prevents us from launching the same installer twice if the close cmd
    /// gets coalesced across frames.
    Launching,
    /// Network / parse error from [`spawn_check`]. Silent: logged only. The
    /// inner message is kept so a future "Check for updates…" menu item could
    /// surface it, but the current UI deliberately never reads it (offline
    /// state is not user-actionable at startup).
    #[allow(dead_code)]
    NetworkError(String),
}

pub type SharedUpdateStatus = Arc<Mutex<UpdateStatus>>;

pub fn make_shared_status() -> SharedUpdateStatus {
    Arc::new(Mutex::new(UpdateStatus::Idle))
}

// ---- Startup check ----------------------------------------------------------

/// Kick off a background thread that queries GitHub for the latest release and
/// writes the result into `shared`. Idempotent: a second call while a check is
/// already in flight (or already resolved) is a no-op.
pub fn spawn_check(repo: &'static str, current: &'static str, shared: SharedUpdateStatus) {
    {
        let mut s = shared.lock();
        if !matches!(*s, UpdateStatus::Idle) {
            return;
        }
        *s = UpdateStatus::Checking;
    }
    std::thread::Builder::new()
        .name("update-check".into())
        .spawn(move || {
            startup_log::log(format!(
                "updater: checking GitHub releases for {repo} (have {current})"
            ));
            let outcome = match fetch_latest_release(repo) {
                Ok(info) => {
                    if is_newer(&info.version, current) {
                        startup_log::log(format!(
                            "updater: update available: {} (have {current})",
                            info.version
                        ));
                        UpdateStatus::Available(info)
                    } else {
                        startup_log::log(format!(
                            "updater: up to date ({current}; latest is {})",
                            info.version
                        ));
                        UpdateStatus::UpToDate {
                            github_latest: info.version,
                        }
                    }
                }
                Err(e) => {
                    // Intentionally quiet in the UI: offline, proxy, GH outage,
                    // rate limit — none of these are user-actionable.
                    startup_log::log(format!("updater: check failed: {e}"));
                    UpdateStatus::NetworkError(e.to_string())
                }
            };
            *shared.lock() = outcome;
        })
        .expect("spawn update-check thread");
}

/// Minimal shape of GitHub's `GET /repos/{o}/{r}/releases/latest` we care about.
/// Anything else returned by the API is ignored; serde silently drops unknown
/// fields.
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
    // GitHub requires a User-Agent; refuses the request otherwise.
    let resp = ureq::get(&url)
        .set(
            "User-Agent",
            concat!("AYR-Audio-Meter/", env!("CARGO_PKG_VERSION")),
        )
        .set("Accept", "application/vnd.github+json")
        .set("X-GitHub-Api-Version", "2022-11-28")
        .timeout(Duration::from_secs(10))
        .call()
        .map_err(|e| anyhow::anyhow!("github api: {e}"))?;

    let release: GhRelease = resp
        .into_json()
        .map_err(|e| anyhow::anyhow!("parse release json: {e}"))?;

    // Pick the first asset whose name looks like an installer `.exe`. We accept
    // both `AYR-Audio-Meter-Setup-1.6.6.exe` (current convention) and anything
    // else ending in `.exe` in case the filename scheme ever changes.
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

// ---- Version compare --------------------------------------------------------

/// Parse `"1.6.6"` / `"v1.6.6"` / `"1.6.6-beta.1"` into a `(major, minor, patch)`
/// triple. Any pre-release / build suffix is ignored. Non-numeric segments
/// evaluate to `0`, so malformed tags never panic — they just compare low.
fn parse_triple(v: &str) -> (u32, u32, u32) {
    let v = v.trim().trim_start_matches(|c: char| c == 'v' || c == 'V');
    // Strip any `-...` or `+...` suffix before splitting on `.`.
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

/// `true` iff `latest > current` by semver triple.
pub fn is_newer(latest: &str, current: &str) -> bool {
    parse_triple(latest) > parse_triple(current)
}

// ---- Download + launch ------------------------------------------------------

/// Kick off a download thread that streams the installer into `%TEMP%` and, on
/// success, flips `shared` to [`UpdateStatus::Downloaded`] carrying the on-disk
/// path. Progress is written back to `shared` as `Downloading { progress, .. }`
/// at roughly 4 Hz (every ~256 KB) so the UI can animate a progress bar
/// without ping-ponging the lock.
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
        .name("update-download".into())
        .spawn(move || {
            startup_log::log(format!(
                "updater: downloading {} ({} bytes) from {}",
                info.installer_filename, info.size_bytes, info.installer_url
            ));
            let outcome = match download_to_temp(&info, &shared_bg) {
                Ok(path) => {
                    startup_log::log(format!("updater: downloaded to {}", path.display()));
                    UpdateStatus::Downloaded(path)
                }
                Err(e) => {
                    startup_log::log(format!("updater: download failed: {e}"));
                    UpdateStatus::LaunchFailed(format!("Download failed: {e}"))
                }
            };
            *shared_bg.lock() = outcome;
        })
        .expect("spawn update-download thread");
}

/// Stream the installer body into `%TEMP%/{filename}`. Progress ticks every
/// ~256 KB so modal repaint churn is bounded.
fn download_to_temp(info: &ReleaseInfo, shared: &SharedUpdateStatus) -> anyhow::Result<PathBuf> {
    let resp = ureq::get(&info.installer_url)
        .set(
            "User-Agent",
            concat!("AYR-Audio-Meter/", env!("CARGO_PKG_VERSION")),
        )
        // GH serves asset bodies from objects.githubusercontent.com with long
        // signed URLs; keep the per-request timeout generous because first-byte
        // latency can be a few seconds and the installer is 10+ MB. ureq 2.x
        // collapsed the old split `timeout_connect`/`timeout_read` knobs into a
        // single total-request timeout.
        .timeout(Duration::from_secs(60))
        .call()
        .map_err(|e| anyhow::anyhow!("download request: {e}"))?;

    let total: u64 = resp
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(info.size_bytes);

    let dest = std::env::temp_dir().join(&info.installer_filename);
    // Clobber any prior partial download (e.g. previous aborted attempt).
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

/// Spawn the Inno Setup installer detached from the current process. The
/// installer's `.iss` has `CloseApplications=yes` + `RestartApplications=yes`,
/// so it will silently close + relaunch the running `.exe` after the upgrade;
/// we still send `ViewportCommand::Close` immediately after spawning so the
/// transition feels instant rather than waiting for Restart Manager.
///
/// On non-Windows targets this just `spawn`s the binary as-is for parity with
/// the tests, but in practice we only ever produce `.exe` installers.
pub fn launch_installer(path: &Path) -> anyhow::Result<()> {
    startup_log::log(format!("updater: spawning installer {}", path.display()));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP: installer survives our
        // `ViewportCommand::Close`-driven exit and doesn't inherit our stdio.
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
        assert!(is_newer("1.6.6", "1.6.5"));
        assert!(is_newer("1.7.0", "1.6.99"));
        assert!(is_newer("2.0.0", "1.99.99"));
        assert!(!is_newer("1.6.5", "1.6.5"));
        assert!(!is_newer("1.6.4", "1.6.5"));
    }

    #[test]
    fn version_compare_ignores_prefix_and_suffix() {
        assert!(is_newer("v1.6.6", "1.6.5"));
        assert!(is_newer("1.6.6", "v1.6.5"));
        assert!(is_newer("1.6.6-beta.2", "1.6.5"));
        // Pre-release suffixes on equal triples collapse to equal; neither is newer.
        assert!(!is_newer("1.6.5-rc.1", "1.6.5"));
    }

    #[test]
    fn version_compare_survives_garbage() {
        assert!(!is_newer("potato", "1.6.5"));
        assert!(is_newer("1.6.5", "potato"));
    }
}
