//! Paid-license + free-trial gate, backed by Lemon Squeezy's License Keys API.
//!
//! # Why Lemon Squeezy
//!
//! LS is a Merchant-of-Record: they handle checkout, global VAT/sales-tax
//! collection, invoices, refunds, and — crucially for us — ship a first-class
//! **License Keys API** with per-key activation-instance limits built in. Our
//! three touch points:
//!
//! * `POST /v1/licenses/activate`   — first-time activation on a new device.
//! * `POST /v1/licenses/validate`   — periodic re-check that the key hasn't
//!   been refunded / deactivated / moved.
//! * `POST /v1/licenses/deactivate` — user-initiated "remove this PC" from the
//!   Manage License panel.
//!
//! These endpoints only require the user-entered license key as the credential
//! — **no API key is embedded in the binary**, so there's nothing for a reverse
//! engineer to steal.
//!
//! # Trial model
//!
//! 14 days of fully-functional use per Windows user. The trial anchor
//! (`trial_started_at`) is the **min** of three independent sources, stamped
//! on first launch, to resist naïve clock-rollback and file-delete resets:
//!
//! 1. `trial_started_at` inside `%APPDATA%/audio-meter-rs/license.json`
//! 2. `HKCU\Software\AYR Audio\AudioMeter\TrialStart` (REG_QWORD unix seconds)
//! 3. File-system `created()` time of `license.json` itself
//!
//! Deleting the JSON, wiping the registry key, or rolling the wall clock back
//! only resets the trial if *all* sources disappear simultaneously.
//!
//! # Per-PC per-user fingerprint
//!
//! `instance_name` sent to LS is
//! `SHA-256("ayr-audio-meter|" + MachineGuid + "|" + domain\username)`.
//! MachineGuid is the Windows crypto-install unique id; combining it with the
//! per-user login means two accounts on the same PC consume two activation
//! slots, which is what "one per PC per user" literally means. The raw values
//! never leave the device — only the hex-encoded hash does.
//!
//! # State machine
//!
//! The UI thread reads a shared [`LicenseStatus`] every frame, the same way
//! [`crate::updater`] drives its modal. All transitions are driven from this
//! module: the UI only calls `spawn_activate` / `spawn_deactivate` / clears
//! its local "modal dismissed" flags.
//!
//! # Dev / QA override
//!
//! `AYR_AUDIO_METER_LICENSE_OVERRIDE=<mode>` short-circuits the real flow for
//! developer iteration and agent-driven UI testing. See
//! [`parse_override`] for the accepted tokens. Debug builds default to
//! `licensed` when no override is set so `launch-dev.ps1` never nags during
//! development.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, TimeZone, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::startup_log;

// ---- Tunables ---------------------------------------------------------------

/// Free-trial length in days. Counted from `trial::anchor()`.
pub const TRIAL_DAYS: i64 = 14;

/// How often we re-hit LS `/validate` while the app is running. If this many
/// days have elapsed since `last_validated_at`, the initial state is still
/// `Licensed` but we kick off a background revalidate that may demote us to
/// `Invalid` (refunded / deactivated) or `OfflineGrace` (network down).
pub const REVALIDATE_EVERY_DAYS: i64 = 7;

/// After a successful online validation, we'll keep trusting the cached grant
/// for this many days without a fresh re-check. Beyond this, offline use is
/// blocked until the app can reach LS again. Deliberately twice the
/// revalidate interval so normal users never see the grace countdown.
pub const OFFLINE_GRACE_DAYS: i64 = 14;

/// Base URL for the LS public API.
pub const LS_API_BASE: &str = "https://api.lemonsqueezy.com";

/// Public checkout URL for the one-time "AYR Audio Meter" license
/// (€24.95, two activations per key, lifetime updates within the 2.x line).
/// Hosted by Lemon Squeezy — safe to embed in the binary; it's a "buy now"
/// deep link with no auth attached.
///
/// NOTE: this is currently the **Test mode** URL. Swap to the Live checkout
/// URL before cutting the public 2.0.0 release (they're different UUIDs on
/// the same path).
pub const CHECKOUT_URL: &str =
    "https://ayraudio.lemonsqueezy.com/checkout/buy/db938c33-16d5-4c1f-93bf-2661d0d654c0?logo=0&discount=0";
/// LS customer portal landing page. Users can manage orders / download receipts
/// / self-deactivate instances here. Magic-link login: no credentials needed.
pub const CUSTOMER_PORTAL_URL: &str = "https://app.lemonsqueezy.com/my-orders";

/// Env var name for the dev/QA override. Kept as a module constant so it
/// appears in one place for grep / documentation.
pub const LICENSE_OVERRIDE_ENV: &str = "AYR_AUDIO_METER_LICENSE_OVERRIDE";

/// Registry path used as one of the three trial-anchor sources on Windows.
#[cfg(windows)]
const TRIAL_REG_PATH: &str = r"Software\AYR Audio\AudioMeter";
#[cfg(windows)]
const TRIAL_REG_VALUE: &str = "TrialStart";

// ---- Shared status ----------------------------------------------------------

/// State machine the UI thread reads every frame. Only this module writes to
/// it; the UI writes are limited to calling one of the `spawn_*` helpers
/// below, which flip through `Activating` / `Deactivating` into a terminal
/// state.
#[derive(Clone, Debug)]
pub enum LicenseStatus {
    /// Pre-`load_or_init` default. Never shown in UI.
    Unknown,
    /// Inside the free-trial window, no key entered.
    TrialActive { days_left: u32 },
    /// 14+ days since first launch, no valid key.
    TrialExpired,
    /// User submitted a key; LS `/activate` call in flight.
    Activating,
    /// Cached + recently-validated grant. `offline_days_left == Some` means
    /// we're past `REVALIDATE_EVERY_DAYS` but still inside the grace window
    /// and haven't been able to reach LS; the UI should nudge the user to
    /// come online soon.
    Licensed {
        email: String,
        tier: String,
        activations_used: u32,
        activations_limit: u32,
        /// Hard expiry from LS (e.g. subscription end). `None` for perpetual.
        expires_at: Option<DateTime<Utc>>,
        /// `Some(n)` while we're inside the offline grace slope; `None` when
        /// we've successfully hit `/validate` within REVALIDATE_EVERY_DAYS.
        offline_days_left: Option<u32>,
    },
    /// LS told us the key is not valid (disabled / refunded / expired /
    /// out of activations). Inner string is a user-visible message.
    Invalid(String),
    /// User clicked "Deactivate this PC"; LS `/deactivate` call in flight.
    Deactivating,
    /// Silent network / parse failure from any LS call. Not shown directly
    /// to the user; if we were Licensed we degrade to `Licensed { offline_days_left }`
    /// via the validate path instead. Reserved for future paths where the
    /// UI wants to express "still trying" without nagging; right now every
    /// activate/validate failure collapses into `Invalid` or the offline
    /// grace slope, so this variant is intentionally unused.
    #[allow(dead_code)]
    NetworkError(String),
}

impl LicenseStatus {
    /// The UI uses this to decide whether to show the full product or draw the
    /// "trial expired / unlicensed" overlay. Activating/deactivating count as
    /// "grant access" to avoid a momentary lockout flicker during an API round
    /// trip.
    pub fn grants_full_access(&self) -> bool {
        match self {
            LicenseStatus::TrialActive { .. }
            | LicenseStatus::Licensed { .. }
            | LicenseStatus::Activating
            | LicenseStatus::Deactivating
            | LicenseStatus::Unknown => true,
            LicenseStatus::TrialExpired
            | LicenseStatus::Invalid(_)
            | LicenseStatus::NetworkError(_) => false,
        }
    }

    /// `true` when the user currently holds a real (non-trial) license.
    pub fn is_licensed(&self) -> bool {
        matches!(self, LicenseStatus::Licensed { .. })
    }
}

pub type SharedLicense = Arc<Mutex<LicenseStatus>>;

pub fn make_shared_license() -> SharedLicense {
    Arc::new(Mutex::new(LicenseStatus::Unknown))
}

// ---- license.json on disk ---------------------------------------------------

/// Persisted license cache. Everything is optional so an empty file
/// round-trips through serde cleanly and unknown future fields are preserved
/// as long as we bump `schema` when changing shape.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LicenseFile {
    /// Layout version. Bump when we break backward-compat.
    pub schema: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub license_key: Option<String>,
    /// Opaque id returned by LS on `/activate`; needed as the primary key for
    /// all subsequent `/validate` + `/deactivate` calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    /// Echoed back from `/activate`; stored so we can confirm no one else on
    /// this machine has re-activated against our slot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    pub activations_used: u32,
    pub activations_limit: u32,

    /// Hard expiry date from LS (e.g. subscription end). `None` for perpetual.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    /// Wall-clock time of the last successful LS `/validate`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_validated_at: Option<DateTime<Utc>>,

    /// Free-trial start anchor. Min of the three redundant sources; see
    /// [`trial::resolve_anchor`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trial_started_at: Option<DateTime<Utc>>,
}

/// `%APPDATA%/audio-meter-rs/license.json` (or the platform equivalent). Kept
/// in the same directory as `user_settings.json` so the app has a single state
/// folder, but in its own file so security-adjacent state is easy to audit.
pub fn license_file_path() -> Option<PathBuf> {
    let base = dirs::config_dir().or_else(dirs::data_local_dir)?;
    Some(base.join("audio-meter-rs").join("license.json"))
}

pub fn load_license_file() -> (LicenseFile, Option<PathBuf>) {
    let path = license_file_path();
    let Some(ref p) = path else {
        return (LicenseFile::default(), None);
    };
    if let Ok(bytes) = std::fs::read(p) {
        if let Ok(data) = serde_json::from_slice::<LicenseFile>(&bytes) {
            return (data, path);
        }
    }
    (LicenseFile::default(), path)
}

pub fn save_license_file(path: &Path, data: &LicenseFile) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec_pretty(data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

// ---- Fingerprint (SHA-256 of machine id + user id) --------------------------

/// Produces the opaque per-(PC, Windows user) string we hand to LS as
/// `instance_name`. Hex SHA-256 so:
///
/// * The raw MachineGuid / username never leaves the device.
/// * It's stable across reboots, app reinstalls, and app updates.
/// * It changes on Windows reinstall (MachineGuid regenerates) and on
///   username change — both of which the user can repair by self-deactivating
///   the old instance from the LS portal, which our Manage License panel links
///   to.
pub fn device_instance_name() -> String {
    let raw = raw_identity();
    let mut hasher = Sha256::new();
    hasher.update(b"ayr-audio-meter|");
    hasher.update(raw.as_bytes());
    let out = hasher.finalize();
    let mut hex = String::with_capacity(out.len() * 2);
    for b in out.iter() {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

fn raw_identity() -> String {
    let machine = machine_guid().unwrap_or_else(|| "unknown-machine".into());
    let user = user_identity();
    format!("{machine}|{user}")
}

#[cfg(windows)]
fn machine_guid() -> Option<String> {
    use winreg::RegKey;
    use winreg::enums::HKEY_LOCAL_MACHINE;
    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    let k = hklm.open_subkey(r"SOFTWARE\Microsoft\Cryptography").ok()?;
    k.get_value::<String, _>("MachineGuid").ok()
}

#[cfg(not(windows))]
fn machine_guid() -> Option<String> {
    // Linux/macOS dev fallback so the fingerprint code compiles + tests pass.
    // Not a supported production target (we ship as a Windows-only app).
    Some(format!("{}-dev", host_name()))
}

fn user_identity() -> String {
    let username = std::env::var("USERNAME")
        .or_else(|_| std::env::var("USER"))
        .unwrap_or_else(|_| "unknown".into());
    let domain = std::env::var("USERDOMAIN")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| host_name());
    format!("{domain}\\{username}")
}

fn host_name() -> String {
    #[cfg(windows)]
    {
        std::env::var("COMPUTERNAME").unwrap_or_else(|_| "dev-host".into())
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOSTNAME").unwrap_or_else(|_| "dev-host".into())
    }
}

// ---- Trial clock ------------------------------------------------------------

pub mod trial {
    //! Anti-rollback trial-start resolver.
    //!
    //! A naïve single-source anchor lets a motivated user uninstall, delete
    //! `license.json`, and come back to a fresh 14-day window. We stamp three
    //! independent sources and always take the *minimum* (earliest) of
    //! whatever we find, so restoring the trial requires wiping all three in
    //! the same session.

    use super::*;

    /// Returns the resolved trial anchor and a `needs_stamp` flag telling the
    /// caller to write the resolved value back into `LicenseFile::trial_started_at`.
    /// Consults all three sources (JSON anchor, HKCU registry, file ctime).
    pub fn resolve_anchor(
        file: &LicenseFile,
        file_path: Option<&Path>,
    ) -> (DateTime<Utc>, bool) {
        let json_src = file.trial_started_at;
        let reg_src = registry_anchor();
        let ctime_src = file_path.and_then(file_ctime_utc);
        let (resolved, needs_stamp) =
            resolve_anchor_pure(json_src, reg_src, ctime_src, Utc::now());
        // Write back to the weaker sources so next launch has three anchors.
        // We ignore errors: if the registry write fails on a locked-down
        // machine, the JSON anchor still does its job.
        if reg_src.is_none() {
            stamp_registry(resolved);
        }
        (resolved, needs_stamp)
    }

    /// Pure core of [`resolve_anchor`] — no I/O, no side effects, fully
    /// testable. Public within the module so `#[cfg(test)]` can call it.
    pub fn resolve_anchor_pure(
        json: Option<DateTime<Utc>>,
        registry: Option<DateTime<Utc>>,
        ctime: Option<DateTime<Utc>>,
        now: DateTime<Utc>,
    ) -> (DateTime<Utc>, bool) {
        let candidates: Vec<DateTime<Utc>> =
            [json, registry, ctime].into_iter().flatten().collect();
        match candidates.iter().min() {
            Some(&earliest) => (earliest, json != Some(earliest)),
            None => (now, true),
        }
    }

    pub fn days_left(anchor: DateTime<Utc>) -> i64 {
        let elapsed_days = (Utc::now() - anchor).num_days();
        TRIAL_DAYS - elapsed_days
    }

    #[cfg(windows)]
    fn registry_anchor() -> Option<DateTime<Utc>> {
        use winreg::RegKey;
        use winreg::enums::HKEY_CURRENT_USER;
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        let k = hkcu.open_subkey(TRIAL_REG_PATH).ok()?;
        let secs: u64 = k.get_value(TRIAL_REG_VALUE).ok()?;
        Utc.timestamp_opt(secs as i64, 0).single()
    }

    #[cfg(not(windows))]
    fn registry_anchor() -> Option<DateTime<Utc>> {
        None
    }

    #[cfg(windows)]
    fn stamp_registry(anchor: DateTime<Utc>) {
        use winreg::RegKey;
        use winreg::enums::{HKEY_CURRENT_USER, KEY_ALL_ACCESS};
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        let Ok((k, _)) =
            hkcu.create_subkey_with_flags(TRIAL_REG_PATH, KEY_ALL_ACCESS)
        else {
            return;
        };
        let secs = anchor.timestamp().max(0) as u64;
        let _ = k.set_value(TRIAL_REG_VALUE, &secs);
    }

    #[cfg(not(windows))]
    fn stamp_registry(_anchor: DateTime<Utc>) {}

    fn file_ctime_utc(p: &Path) -> Option<DateTime<Utc>> {
        let md = std::fs::metadata(p).ok()?;
        let created = md.created().ok()?;
        let dur = created
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?;
        Utc.timestamp_opt(dur.as_secs() as i64, dur.subsec_nanos())
            .single()
    }
}

// ---- Dev / QA override ------------------------------------------------------

/// Parses the `AYR_AUDIO_METER_LICENSE_OVERRIDE` env var. Public for tests.
///
/// Accepted tokens (case-insensitive):
/// * `licensed` or `licensed:<tier>` — Licensed with synthetic data
/// * `trial` or `trial:<days>` — TrialActive; defaults to 7 days
/// * `expired` — TrialExpired
/// * `invalid` — Invalid("debug override")
/// * `offline:<days>` — Licensed inside the offline grace slope
/// * anything else — `None` (no override)
pub fn parse_override(raw: &str) -> Option<LicenseStatus> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Lowercase only the *kind* portion so a `licensed:Studio` override keeps
    // the tier's original casing (it flows into the UI verbatim).
    let (kind, arg): (String, Option<String>) = match trimmed.split_once(':') {
        Some((k, v)) => (k.trim().to_ascii_lowercase(), Some(v.trim().to_string())),
        None => (trimmed.to_ascii_lowercase(), None),
    };
    let kind = kind.as_str();
    let arg = arg.filter(|s| !s.is_empty());
    match kind {
        "licensed" => {
            let tier = arg.unwrap_or_else(|| "Personal".into());
            Some(LicenseStatus::Licensed {
                email: "dev@ayrthon.local".into(),
                tier,
                activations_used: 1,
                activations_limit: 2,
                expires_at: None,
                offline_days_left: None,
            })
        }
        "trial" => {
            let days: u32 = arg
                .as_deref()
                .and_then(|s| s.parse().ok())
                .unwrap_or(7);
            Some(LicenseStatus::TrialActive { days_left: days })
        }
        "expired" | "trial-expired" => Some(LicenseStatus::TrialExpired),
        "invalid" => Some(LicenseStatus::Invalid("debug override".into())),
        "offline" => {
            let days: u32 = arg
                .as_deref()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            Some(LicenseStatus::Licensed {
                email: "dev@ayrthon.local".into(),
                tier: "Personal".into(),
                activations_used: 1,
                activations_limit: 2,
                expires_at: None,
                offline_days_left: Some(days),
            })
        }
        _ => None,
    }
}

fn active_override() -> Option<LicenseStatus> {
    let raw = std::env::var(LICENSE_OVERRIDE_ENV).ok()?;
    parse_override(&raw)
}

fn debug_build_default() -> LicenseStatus {
    // Debug builds default to a synthetic licensed state so `launch-dev.ps1`
    // iteration doesn't force the developer through the activation modal
    // every time. Set `AYR_AUDIO_METER_LICENSE_OVERRIDE=...` to exercise a
    // specific branch, including `unlicensed`-ish states like `trial:3`.
    LicenseStatus::Licensed {
        email: "dev@ayrthon.local".into(),
        tier: "Personal".into(),
        activations_used: 1,
        activations_limit: 2,
        expires_at: None,
        offline_days_left: None,
    }
}

// ---- Load or init -----------------------------------------------------------

/// Reads `license.json`, resolves the trial anchor, decides the initial
/// [`LicenseStatus`], and (for a cached licensed key) optionally spawns a
/// background revalidate. Call this once during `AudioMeterApp::new`.
pub fn load_or_init(shared: SharedLicense) {
    // Dev override wins over everything.
    if let Some(status) = active_override() {
        startup_log::log(format!(
            "license: using AYR_AUDIO_METER_LICENSE_OVERRIDE -> {:?}",
            status
        ));
        *shared.lock() = status;
        return;
    }
    #[cfg(debug_assertions)]
    {
        startup_log::log("license: debug build default -> licensed");
        *shared.lock() = debug_build_default();
        return;
    }

    #[cfg(not(debug_assertions))]
    {
        let (mut file, path) = load_license_file();
        let path_ref = path.as_deref();
        let (anchor, needs_stamp) = trial::resolve_anchor(&file, path_ref);
        if needs_stamp {
            file.trial_started_at = Some(anchor);
            file.schema = file.schema.max(1);
            if let Some(p) = path_ref {
                if let Err(e) = save_license_file(p, &file) {
                    startup_log::log(format!("license: stamp trial anchor failed: {e}"));
                }
            }
        }

        let status = initial_status(&file, anchor);
        startup_log::log(format!(
            "license: initial status = {:?} (anchor {})",
            status, anchor
        ));
        *shared.lock() = status;

        // If we booted into a Licensed-from-cache state and it's been long
        // enough, kick off a background revalidate so a refunded / deactivated
        // key doesn't stay trusted forever.
        if let (Some(_), Some(last)) = (file.license_key.as_deref(), file.last_validated_at)
        {
            let age_days = (Utc::now() - last).num_days();
            if age_days >= REVALIDATE_EVERY_DAYS {
                spawn_validate(shared.clone());
            }
        }
    }
}

fn initial_status(file: &LicenseFile, anchor: DateTime<Utc>) -> LicenseStatus {
    if let (Some(_key), Some(last)) = (file.license_key.as_deref(), file.last_validated_at) {
        let age_days = (Utc::now() - last).num_days();
        let offline_days_left = if age_days <= REVALIDATE_EVERY_DAYS {
            None
        } else if age_days <= OFFLINE_GRACE_DAYS {
            Some((OFFLINE_GRACE_DAYS - age_days).max(0) as u32)
        } else {
            // Grace burned through while the app was offline. Fall through to
            // Invalid rather than silently keep the old grant.
            return LicenseStatus::Invalid(
                "License needs an online re-check. Please reconnect to the internet.".into(),
            );
        };
        return LicenseStatus::Licensed {
            email: file.email.clone().unwrap_or_default(),
            tier: file.tier.clone().unwrap_or_else(|| "Personal".into()),
            activations_used: file.activations_used,
            activations_limit: file.activations_limit,
            expires_at: file.expires_at,
            offline_days_left,
        };
    }
    // No key: fall through to trial vs expired.
    let days = trial::days_left(anchor);
    if days > 0 {
        LicenseStatus::TrialActive {
            days_left: days as u32,
        }
    } else {
        LicenseStatus::TrialExpired
    }
}

// ---- LS API client ----------------------------------------------------------

#[derive(Deserialize, Debug)]
struct LsActivateResp {
    activated: Option<bool>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    instance: Option<LsInstance>,
    #[serde(default)]
    meta: Option<LsMeta>,
    #[serde(default)]
    license_key: Option<LsLicenseKey>,
}

#[derive(Deserialize, Debug)]
struct LsValidateResp {
    valid: Option<bool>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    meta: Option<LsMeta>,
    #[serde(default)]
    license_key: Option<LsLicenseKey>,
    #[serde(default)]
    instance: Option<LsInstance>,
}

#[derive(Deserialize, Debug)]
struct LsDeactivateResp {
    deactivated: Option<bool>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize, Debug)]
struct LsInstance {
    id: String,
    #[serde(default)]
    name: String,
}

#[derive(Deserialize, Debug)]
struct LsLicenseKey {
    #[serde(default)]
    status: String,
    #[serde(default)]
    activation_limit: u32,
    #[serde(default)]
    activation_usage: u32,
    #[serde(default)]
    expires_at: Option<DateTime<Utc>>,
}

#[derive(Deserialize, Debug)]
struct LsMeta {
    #[serde(default)]
    customer_email: Option<String>,
    #[serde(default)]
    variant_name: Option<String>,
}

/// User-initiated activation. Runs on a background thread; flips `shared`
/// to `Activating` immediately, then terminal state when the call returns.
pub fn spawn_activate(key: String, shared: SharedLicense) {
    *shared.lock() = LicenseStatus::Activating;
    let shared_bg = shared.clone();
    std::thread::Builder::new()
        .name("license-activate".into())
        .spawn(move || {
            let instance_name = device_instance_name();
            startup_log::log(format!(
                "license: activating key ending …{} instance={}",
                tail4(&key),
                short_hash(&instance_name)
            ));
            let outcome = match activate_call(&key, &instance_name) {
                Ok(file) => persist_and_status(file),
                Err(e) => {
                    startup_log::log(format!("license: activate failed: {e}"));
                    LicenseStatus::Invalid(e.to_string())
                }
            };
            *shared_bg.lock() = outcome;
        })
        .expect("spawn license-activate thread");
}

/// Background revalidate of the cached grant. Called on startup when the last
/// validate is more than REVALIDATE_EVERY_DAYS old, or manually via the
/// Manage License panel. Demotes `Licensed` → `Licensed { offline_days_left }`
/// on soft failures and `Licensed` → `Invalid` on hard failures.
pub fn spawn_validate(shared: SharedLicense) {
    let shared_bg = shared.clone();
    std::thread::Builder::new()
        .name("license-validate".into())
        .spawn(move || {
            let (file, path) = load_license_file();
            let Some(key) = file.license_key.clone() else {
                return;
            };
            let Some(instance_id) = file.instance_id.clone() else {
                return;
            };
            startup_log::log(format!(
                "license: validating key ending …{}",
                tail4(&key)
            ));
            match validate_call(&key, &instance_id) {
                Ok(fresh) => {
                    // Merge fresh response onto the existing file so we keep
                    // trial_started_at etc.
                    let mut merged = file.clone();
                    merged.license_key = fresh.license_key;
                    merged.instance_id = fresh.instance_id;
                    merged.instance_name = fresh.instance_name;
                    merged.email = fresh.email;
                    merged.tier = fresh.tier;
                    merged.activations_used = fresh.activations_used;
                    merged.activations_limit = fresh.activations_limit;
                    merged.expires_at = fresh.expires_at;
                    merged.last_validated_at = fresh.last_validated_at;
                    if let Some(ref p) = path {
                        let _ = save_license_file(p, &merged);
                    }
                    *shared_bg.lock() = LicenseStatus::Licensed {
                        email: merged.email.unwrap_or_default(),
                        tier: merged.tier.unwrap_or_else(|| "Personal".into()),
                        activations_used: merged.activations_used,
                        activations_limit: merged.activations_limit,
                        expires_at: merged.expires_at,
                        offline_days_left: None,
                    };
                }
                Err(ValidateError::Network(msg)) => {
                    startup_log::log(format!("license: validate network error: {msg}"));
                    // Soft failure: if we're still inside the offline grace
                    // window, keep the user in business. Otherwise flip to
                    // Invalid — but `initial_status` should normally have
                    // caught this case at startup, so we only land here on a
                    // manual retry well past the grace cliff.
                    if let Some(last) = file.last_validated_at {
                        let age_days = (Utc::now() - last).num_days();
                        if age_days <= OFFLINE_GRACE_DAYS {
                            let left = (OFFLINE_GRACE_DAYS - age_days).max(0) as u32;
                            *shared_bg.lock() = LicenseStatus::Licensed {
                                email: file.email.unwrap_or_default(),
                                tier: file.tier.unwrap_or_else(|| "Personal".into()),
                                activations_used: file.activations_used,
                                activations_limit: file.activations_limit,
                                expires_at: file.expires_at,
                                offline_days_left: Some(left),
                            };
                            return;
                        }
                    }
                    *shared_bg.lock() = LicenseStatus::Invalid(
                        "Could not reach the license server. Please reconnect and retry."
                            .into(),
                    );
                }
                Err(ValidateError::Rejected(msg)) => {
                    startup_log::log(format!("license: validate rejected: {msg}"));
                    // Hard failure: refunded, deactivated, key disabled.
                    if let Some(ref p) = path {
                        let _ = save_license_file(p, &LicenseFile {
                            trial_started_at: file.trial_started_at,
                            schema: file.schema.max(1),
                            ..Default::default()
                        });
                    }
                    *shared_bg.lock() = LicenseStatus::Invalid(msg);
                }
            }
        })
        .expect("spawn license-validate thread");
}

/// User-initiated "remove this PC". Tells LS to free the instance slot, then
/// wipes everything-but-trial-anchor locally so the user returns to their
/// existing trial clock (or TrialExpired if it already burned out).
pub fn spawn_deactivate(shared: SharedLicense) {
    *shared.lock() = LicenseStatus::Deactivating;
    let shared_bg = shared.clone();
    std::thread::Builder::new()
        .name("license-deactivate".into())
        .spawn(move || {
            let (file, path) = load_license_file();
            let Some(key) = file.license_key.clone() else {
                // Nothing to deactivate; reset UI.
                let anchor = file.trial_started_at.unwrap_or_else(Utc::now);
                *shared_bg.lock() = initial_status(
                    &LicenseFile {
                        trial_started_at: Some(anchor),
                        ..Default::default()
                    },
                    anchor,
                );
                return;
            };
            let Some(instance_id) = file.instance_id.clone() else {
                return;
            };
            startup_log::log(format!(
                "license: deactivating key ending …{}",
                tail4(&key)
            ));
            let _ = deactivate_call(&key, &instance_id);
            // Regardless of the server result, clear the local grant — the user
            // asked us to remove this PC.
            let cleared = LicenseFile {
                trial_started_at: file.trial_started_at,
                schema: file.schema.max(1),
                ..Default::default()
            };
            if let Some(ref p) = path {
                let _ = save_license_file(p, &cleared);
            }
            let anchor = cleared.trial_started_at.unwrap_or_else(Utc::now);
            let days = trial::days_left(anchor);
            *shared_bg.lock() = if days > 0 {
                LicenseStatus::TrialActive {
                    days_left: days as u32,
                }
            } else {
                LicenseStatus::TrialExpired
            };
        })
        .expect("spawn license-deactivate thread");
}

// ---- Internal HTTP --------------------------------------------------------

enum ValidateError {
    /// Network / parse / unexpected non-200 — safe to fall back to grace.
    Network(String),
    /// LS explicitly said "this key is not valid anymore".
    Rejected(String),
}

fn activate_call(key: &str, instance_name: &str) -> anyhow::Result<LicenseFile> {
    let url = format!("{LS_API_BASE}/v1/licenses/activate");
    let resp = ureq::post(&url)
        .set(
            "User-Agent",
            concat!("AYR-Audio-Meter/", env!("CARGO_PKG_VERSION")),
        )
        .set("Accept", "application/json")
        .timeout(Duration::from_secs(15))
        .send_form(&[
            ("license_key", key),
            ("instance_name", instance_name),
        ])
        .map_err(|e| anyhow::anyhow!("lemon squeezy activate: {e}"))?;
    let parsed: LsActivateResp = resp
        .into_json()
        .map_err(|e| anyhow::anyhow!("parse activate resp: {e}"))?;
    if parsed.activated != Some(true) {
        let msg = parsed
            .error
            .unwrap_or_else(|| "License key could not be activated.".into());
        return Err(anyhow::anyhow!(msg));
    }
    let instance = parsed
        .instance
        .ok_or_else(|| anyhow::anyhow!("activate: missing instance"))?;
    let lk = parsed.license_key.unwrap_or_default_key();
    let meta = parsed.meta.unwrap_or_default_meta();
    Ok(LicenseFile {
        schema: 1,
        license_key: Some(key.to_string()),
        instance_id: Some(instance.id),
        instance_name: Some(instance.name),
        email: meta.customer_email.clone(),
        tier: meta.variant_name.clone(),
        activations_used: lk.activation_usage,
        activations_limit: lk.activation_limit,
        expires_at: lk.expires_at,
        last_validated_at: Some(Utc::now()),
        trial_started_at: None,
    })
}

fn validate_call(key: &str, instance_id: &str) -> Result<LicenseFile, ValidateError> {
    let url = format!("{LS_API_BASE}/v1/licenses/validate");
    let req = ureq::post(&url)
        .set(
            "User-Agent",
            concat!("AYR-Audio-Meter/", env!("CARGO_PKG_VERSION")),
        )
        .set("Accept", "application/json")
        .timeout(Duration::from_secs(15));
    let resp = match req.send_form(&[
        ("license_key", key),
        ("instance_id", instance_id),
    ]) {
        Ok(r) => r,
        Err(ureq::Error::Status(code, _)) if (400..500).contains(&code) => {
            // 4xx from LS means the key is genuinely not OK. Anything else is
            // "network weather" that should fall back to grace.
            return Err(ValidateError::Rejected(format!(
                "License server rejected this key (HTTP {code})."
            )));
        }
        Err(e) => return Err(ValidateError::Network(e.to_string())),
    };
    let parsed: LsValidateResp = resp
        .into_json()
        .map_err(|e| ValidateError::Network(format!("parse validate resp: {e}")))?;
    if parsed.valid != Some(true) {
        return Err(ValidateError::Rejected(
            parsed
                .error
                .unwrap_or_else(|| "License key is no longer valid.".into()),
        ));
    }
    let lk = parsed.license_key.unwrap_or_default_key();
    let meta = parsed.meta.unwrap_or_default_meta();
    let instance = parsed.instance.unwrap_or_default_instance();
    Ok(LicenseFile {
        schema: 1,
        license_key: Some(key.to_string()),
        instance_id: Some(instance.id),
        instance_name: Some(instance.name),
        email: meta.customer_email,
        tier: meta.variant_name,
        activations_used: lk.activation_usage,
        activations_limit: lk.activation_limit,
        expires_at: lk.expires_at,
        last_validated_at: Some(Utc::now()),
        trial_started_at: None,
    })
}

fn deactivate_call(key: &str, instance_id: &str) -> anyhow::Result<()> {
    let url = format!("{LS_API_BASE}/v1/licenses/deactivate");
    let resp = ureq::post(&url)
        .set(
            "User-Agent",
            concat!("AYR-Audio-Meter/", env!("CARGO_PKG_VERSION")),
        )
        .set("Accept", "application/json")
        .timeout(Duration::from_secs(15))
        .send_form(&[
            ("license_key", key),
            ("instance_id", instance_id),
        ])
        .map_err(|e| anyhow::anyhow!("lemon squeezy deactivate: {e}"))?;
    let parsed: LsDeactivateResp = resp
        .into_json()
        .map_err(|e| anyhow::anyhow!("parse deactivate resp: {e}"))?;
    if parsed.deactivated != Some(true) {
        let msg = parsed
            .error
            .unwrap_or_else(|| "License server did not confirm deactivation.".into());
        return Err(anyhow::anyhow!(msg));
    }
    Ok(())
}

fn persist_and_status(file: LicenseFile) -> LicenseStatus {
    let path = license_file_path();
    let mut merged = file;
    // Preserve the existing trial anchor if we had one.
    let (prev, _) = load_license_file();
    if merged.trial_started_at.is_none() {
        merged.trial_started_at = prev.trial_started_at;
    }
    if let Some(ref p) = path {
        if let Err(e) = save_license_file(p, &merged) {
            startup_log::log(format!("license: save license.json failed: {e}"));
        }
    }
    LicenseStatus::Licensed {
        email: merged.email.unwrap_or_default(),
        tier: merged.tier.unwrap_or_else(|| "Personal".into()),
        activations_used: merged.activations_used,
        activations_limit: merged.activations_limit,
        expires_at: merged.expires_at,
        offline_days_left: None,
    }
}

// ---- Small utilities --------------------------------------------------------

fn tail4(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    let n = chars.len();
    if n <= 4 {
        return "****".into();
    }
    chars[n - 4..].iter().collect()
}

fn short_hash(h: &str) -> String {
    let take: usize = 10;
    h.chars().take(take).collect()
}

// Provide `_or_default_*` helpers on the Option<Ls*> types so the call sites
// stay readable even when LS omits a sub-object.
trait LsLicenseKeyExt {
    fn unwrap_or_default_key(self) -> LsLicenseKey;
}
impl LsLicenseKeyExt for Option<LsLicenseKey> {
    fn unwrap_or_default_key(self) -> LsLicenseKey {
        self.unwrap_or(LsLicenseKey {
            status: String::new(),
            activation_limit: 0,
            activation_usage: 0,
            expires_at: None,
        })
    }
}
trait LsMetaExt {
    fn unwrap_or_default_meta(self) -> LsMeta;
}
impl LsMetaExt for Option<LsMeta> {
    fn unwrap_or_default_meta(self) -> LsMeta {
        self.unwrap_or(LsMeta {
            customer_email: None,
            variant_name: None,
        })
    }
}
trait LsInstanceExt {
    fn unwrap_or_default_instance(self) -> LsInstance;
}
impl LsInstanceExt for Option<LsInstance> {
    fn unwrap_or_default_instance(self) -> LsInstance {
        self.unwrap_or(LsInstance {
            id: String::new(),
            name: String::new(),
        })
    }
}

// Avoid "unused `Read`" when no test references it on non-Windows builds.
#[allow(dead_code)]
fn _assert_read_in_scope<R: Read>(_r: R) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_parses_licensed_variants() {
        let status = parse_override("licensed").expect("parse");
        assert!(status.is_licensed());
        let status = parse_override("licensed:Studio").expect("parse");
        match status {
            LicenseStatus::Licensed { tier, .. } => assert_eq!(tier, "Studio"),
            _ => panic!("expected Licensed"),
        }
    }

    #[test]
    fn override_parses_trial_days() {
        let status = parse_override("trial:9").expect("parse");
        match status {
            LicenseStatus::TrialActive { days_left } => assert_eq!(days_left, 9),
            _ => panic!("expected TrialActive"),
        }
        let status = parse_override("trial").expect("parse default");
        assert!(matches!(status, LicenseStatus::TrialActive { .. }));
    }

    #[test]
    fn override_parses_expired_invalid_offline() {
        assert!(matches!(
            parse_override("expired"),
            Some(LicenseStatus::TrialExpired)
        ));
        assert!(matches!(
            parse_override("invalid"),
            Some(LicenseStatus::Invalid(_))
        ));
        match parse_override("offline:5") {
            Some(LicenseStatus::Licensed {
                offline_days_left: Some(d),
                ..
            }) => assert_eq!(d, 5),
            _ => panic!("expected offline Licensed"),
        }
    }

    #[test]
    fn override_ignores_garbage_and_empty() {
        assert!(parse_override("").is_none());
        assert!(parse_override("   ").is_none());
        assert!(parse_override("potato").is_none());
        assert!(parse_override("licensed:").is_some()); // trailing colon is OK
    }

    #[test]
    fn grants_full_access_covers_unlocked_states() {
        assert!(LicenseStatus::Licensed {
            email: "".into(),
            tier: "Personal".into(),
            activations_used: 1,
            activations_limit: 2,
            expires_at: None,
            offline_days_left: None,
        }
        .grants_full_access());
        assert!(LicenseStatus::TrialActive { days_left: 1 }.grants_full_access());
        assert!(!LicenseStatus::TrialExpired.grants_full_access());
        assert!(!LicenseStatus::Invalid("nope".into()).grants_full_access());
    }

    #[test]
    fn trial_days_left_counts_down() {
        let anchor = Utc::now() - chrono::Duration::days(3);
        let left = trial::days_left(anchor);
        assert!(left >= TRIAL_DAYS - 4 && left <= TRIAL_DAYS - 2);
    }

    #[test]
    fn trial_days_left_negative_after_expiry() {
        let anchor = Utc::now() - chrono::Duration::days(30);
        assert!(trial::days_left(anchor) < 0);
    }

    #[test]
    fn resolve_anchor_pure_takes_earliest() {
        let now = Utc::now();
        let two_days_ago = now - chrono::Duration::days(2);
        let ten_days_ago = now - chrono::Duration::days(10);
        let five_days_ago = now - chrono::Duration::days(5);

        // All three sources present: earliest wins, JSON needs a re-stamp.
        let (resolved, needs_stamp) = trial::resolve_anchor_pure(
            Some(two_days_ago),
            Some(ten_days_ago),
            Some(five_days_ago),
            now,
        );
        assert_eq!(resolved, ten_days_ago);
        assert!(needs_stamp);

        // JSON already has the earliest value: no stamp needed.
        let (resolved, needs_stamp) = trial::resolve_anchor_pure(
            Some(ten_days_ago),
            Some(two_days_ago),
            None,
            now,
        );
        assert_eq!(resolved, ten_days_ago);
        assert!(!needs_stamp);

        // Nothing at all: fall back to `now`, caller must stamp.
        let (resolved, needs_stamp) = trial::resolve_anchor_pure(None, None, None, now);
        assert_eq!(resolved, now);
        assert!(needs_stamp);
    }

    #[test]
    fn device_instance_name_is_stable_and_hex() {
        let a = device_instance_name();
        let b = device_instance_name();
        assert_eq!(a, b, "fingerprint should be deterministic for this process");
        assert_eq!(a.len(), 64, "hex SHA-256 is 64 chars");
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn tail4_masks_short_and_long_keys() {
        assert_eq!(tail4("abc"), "****");
        assert_eq!(tail4("12345678"), "5678");
        assert_eq!(tail4("ABCDEF-GHIJKL-MNOPQR-STUV"), "STUV");
    }

    #[test]
    fn license_file_round_trips_empty_and_full() {
        let empty = LicenseFile::default();
        let s = serde_json::to_string(&empty).unwrap();
        let back: LicenseFile = serde_json::from_str(&s).unwrap();
        assert!(back.license_key.is_none());

        let full = LicenseFile {
            schema: 1,
            license_key: Some("AAAA-BBBB-CCCC-DDDD".into()),
            instance_id: Some("inst-1".into()),
            instance_name: Some("fp-abcd".into()),
            email: Some("buyer@example.com".into()),
            tier: Some("Personal".into()),
            activations_used: 1,
            activations_limit: 2,
            expires_at: None,
            last_validated_at: Some(Utc::now()),
            trial_started_at: Some(Utc::now() - chrono::Duration::days(1)),
        };
        let s = serde_json::to_string(&full).unwrap();
        let back: LicenseFile = serde_json::from_str(&s).unwrap();
        assert_eq!(back.license_key.as_deref(), Some("AAAA-BBBB-CCCC-DDDD"));
        assert_eq!(back.activations_limit, 2);
    }
}
