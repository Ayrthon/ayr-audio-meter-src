//! Persistent sender identity + pairing-trust cache.
//!
//! Two small pieces of state live on disk at
//! `{dirs::config_dir}/ayr-audio-link/state.json`:
//!
//! * `instance_name`   — random-per-install mDNS label. Generated on first
//!   run and never changed, so a meter that paired once sees the "same"
//!   sender across reboots / updates.
//! * `trusted_peers`   — fingerprints of meters the user previously
//!   clicked "Allow always" for, so they auto-connect next time.
//!
//! Everything is optional / defaulted, so a missing or corrupted file just
//! downgrades to "new install" behaviour rather than crashing.

use std::path::PathBuf;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use ayr_audio_link_core::fingerprint;

const STATE_FILE_NAME: &str = "state.json";
const APP_DIR_NAME: &str = "ayr-audio-link";

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SenderState {
    /// Monotonic layout version — bump when renaming / moving fields so
    /// older installs don't silently lose data.
    pub schema: u32,
    /// Stable mDNS instance label. Generated on first run; never changes.
    pub instance_name: Option<String>,
    /// Display name the user edited (falls back to hostname if `None`).
    pub custom_display_name: Option<String>,
    /// Peers the user has clicked "Allow always" for. Keyed by
    /// `peer_fingerprint` from [`ayr_audio_link_core::fingerprint`].
    pub trusted_peers: Vec<TrustedPeer>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrustedPeer {
    pub fingerprint: String,
    /// Human-readable label the user saw when they accepted — useful for a
    /// future "Manage trusted devices" screen. Not part of the matching key.
    pub label: String,
    pub first_trusted_at: chrono::DateTime<chrono::Utc>,
}

pub fn state_file_path() -> Option<PathBuf> {
    let base = dirs::config_dir()?;
    Some(base.join(APP_DIR_NAME).join(STATE_FILE_NAME))
}

pub fn load() -> SenderState {
    let Some(path) = state_file_path() else {
        return SenderState::default();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return SenderState::default();
    };
    match serde_json::from_slice::<SenderState>(&bytes) {
        Ok(s) => s,
        Err(e) => {
            log::warn!(
                "ayr-audio-link: {} is unreadable ({e}); starting fresh",
                path.display()
            );
            SenderState::default()
        }
    }
}

pub fn save(state: &SenderState) {
    let Some(path) = state_file_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            log::warn!("ayr-audio-link: create {}: {e}", parent.display());
            return;
        }
    }
    let json = match serde_json::to_vec_pretty(state) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("ayr-audio-link: serialise state: {e}");
            return;
        }
    };
    // Write through a temp file so a power cut mid-write doesn't leave us
    // with zero-byte garbage on disk.
    let tmp = path.with_extension("json.tmp");
    if let Err(e) = std::fs::write(&tmp, &json) {
        log::warn!("ayr-audio-link: write {}: {e}", tmp.display());
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        log::warn!("ayr-audio-link: rename {}: {e}", path.display());
    }
}

/// Thin wrapper owned by the UI so load/save is coordinated through one
/// mutex and every "state changed" path can just call `commit`.
pub struct IdentityStore {
    inner: Mutex<SenderState>,
}

impl IdentityStore {
    pub fn load_or_init() -> Self {
        let mut state = load();
        let mut dirty = false;
        if state.instance_name.is_none() {
            // First-run label: `{hostname}-{first-8-of-fingerprint}`. The
            // hostname part is human-readable for the meter's device picker;
            // the suffix disambiguates two PCs on the same LAN with the
            // same machine name (rare, but VMs / imaged fleets can collide).
            let host = fingerprint::best_effort_hostname();
            let fp = local_peer_fingerprint();
            let short = fp.chars().take(8).collect::<String>();
            state.instance_name = Some(format!("{host}-{short}"));
            dirty = true;
        }
        if state.schema == 0 {
            state.schema = 1;
            dirty = true;
        }
        if dirty {
            save(&state);
        }
        Self {
            inner: Mutex::new(state),
        }
    }

    pub fn snapshot(&self) -> SenderState {
        self.inner.lock().clone()
    }

    pub fn instance_name(&self) -> String {
        self.inner
            .lock()
            .instance_name
            .clone()
            .unwrap_or_else(fingerprint::best_effort_hostname)
    }

    pub fn display_name(&self) -> String {
        let g = self.inner.lock();
        g.custom_display_name
            .clone()
            .unwrap_or_else(fingerprint::best_effort_hostname)
    }

    pub fn set_display_name(&self, name: Option<String>) {
        let mut g = self.inner.lock();
        g.custom_display_name = name.filter(|s| !s.trim().is_empty());
        let snapshot = g.clone();
        drop(g);
        save(&snapshot);
    }

    pub fn is_trusted(&self, fingerprint: &str) -> bool {
        self.inner
            .lock()
            .trusted_peers
            .iter()
            .any(|p| p.fingerprint == fingerprint)
    }

    pub fn trust(&self, fingerprint: String, label: String) {
        let mut g = self.inner.lock();
        if g.trusted_peers.iter().any(|p| p.fingerprint == fingerprint) {
            return;
        }
        g.trusted_peers.push(TrustedPeer {
            fingerprint,
            label,
            first_trusted_at: chrono::Utc::now(),
        });
        let snapshot = g.clone();
        drop(g);
        save(&snapshot);
    }

    pub fn forget(&self, fingerprint: &str) {
        let mut g = self.inner.lock();
        g.trusted_peers.retain(|p| p.fingerprint != fingerprint);
        let snapshot = g.clone();
        drop(g);
        save(&snapshot);
    }
}

/// Our own device fingerprint — what receivers will display when asking to
/// pair. Currently best-effort; on Windows we read MachineGuid + username,
/// elsewhere we fall back to hostname.
pub fn local_peer_fingerprint() -> String {
    #[cfg(windows)]
    {
        if let Some((machine, user)) = windows_machine_user() {
            return fingerprint::peer_fingerprint(&machine, &user);
        }
    }
    fingerprint::peer_fingerprint(
        &fingerprint::best_effort_hostname(),
        &std::env::var("USER").unwrap_or_default(),
    )
}

#[cfg(windows)]
fn windows_machine_user() -> Option<(String, String)> {
    use winreg::RegKey;
    use winreg::enums::HKEY_LOCAL_MACHINE;
    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    let key = hklm
        .open_subkey(r"SOFTWARE\Microsoft\Cryptography")
        .ok()?;
    let guid: String = key.get_value("MachineGuid").ok()?;
    let domain = std::env::var("USERDOMAIN").ok();
    let user = std::env::var("USERNAME").ok()?;
    let ident = match domain {
        Some(d) if !d.is_empty() => format!("{d}\\{user}"),
        _ => user,
    };
    Some((guid, ident))
}
