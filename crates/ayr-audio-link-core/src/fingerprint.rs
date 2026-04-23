//! Stable per-device identity used by the pairing trust cache.
//!
//! Same shape as the meter's license-side `device_instance_name` (a SHA-256
//! hash of machine-GUID + user identity) but kept in this shared crate so
//! the sender can identify *which receiver* is connecting without pulling
//! in the meter's license module.
//!
//! We intentionally hash rather than expose raw GUIDs / usernames to keep
//! identifiers opaque when they're displayed in the "Allow PC X to tap
//! your audio?" pairing dialog — the short fingerprint prefix acts as a
//! verification code the user can read over voice if needed, without
//! leaking OS-level identifiers.

use sha2::{Digest, Sha256};

/// Salt/domain-separator so the fingerprint can't collide with hashes the
/// license code produces from similar input. Changing this invalidates all
/// cached trust entries, so bump only on an intentional protocol break.
const SALT: &[u8] = b"ayr-audio-link-peer-v1|";

/// Compute the pairing fingerprint for a given `(machine_id, user_ident)`
/// pair. `machine_id` is the OS's stable device GUID (Windows MachineGuid,
/// Linux /etc/machine-id, macOS IOPlatformUUID); `user_ident` is
/// typically `domain\\username` or just `username`. Both are free-form —
/// we hash them verbatim so missing values downgrade cleanly rather than
/// being rejected.
pub fn peer_fingerprint(machine_id: &str, user_ident: &str) -> String {
    let mut h = Sha256::new();
    h.update(SALT);
    h.update(machine_id.as_bytes());
    h.update(b"|");
    h.update(user_ident.as_bytes());
    let digest = h.finalize();
    // Hex string keeps the fingerprint printable in JSON trust files and in
    // logs. 64 chars is fine — we display a short prefix in UI.
    hex_lower(&digest)
}

/// Human-facing short code, e.g. `"a9f2-c4b1"`. Used in the pairing dialog
/// so the user can eyeball-match against what the peer shows.
pub fn short_code(fingerprint: &str) -> String {
    let trimmed: String = fingerprint.chars().take(8).collect();
    if trimmed.len() < 8 {
        return trimmed;
    }
    format!("{}-{}", &trimmed[..4], &trimmed[4..])
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0F) as usize] as char);
    }
    s
}

/// Best-effort OS hostname. Shown as the default sender display name so
/// receivers see something meaningful even before the user customises it.
pub fn best_effort_hostname() -> String {
    hostname::get()
        .ok()
        .and_then(|os| os.into_string().ok())
        .unwrap_or_else(|| "ayr-audio-link".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_stable_and_hex() {
        let a = peer_fingerprint("GUID-ABC", "ACME\\alice");
        let b = peer_fingerprint("GUID-ABC", "ACME\\alice");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn fingerprint_differs_per_user() {
        let a = peer_fingerprint("GUID-ABC", "ACME\\alice");
        let b = peer_fingerprint("GUID-ABC", "ACME\\bob");
        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_differs_per_machine() {
        let a = peer_fingerprint("GUID-ABC", "ACME\\alice");
        let b = peer_fingerprint("GUID-DEF", "ACME\\alice");
        assert_ne!(a, b);
    }

    #[test]
    fn short_code_is_pronounceable() {
        let fp = peer_fingerprint("GUID-ABC", "ACME\\alice");
        let short = short_code(&fp);
        assert_eq!(short.len(), 9); // 4 + '-' + 4
        assert!(short.contains('-'));
    }
}
