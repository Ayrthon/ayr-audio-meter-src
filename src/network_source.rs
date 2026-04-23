//! Receiver-side AYR Audio Link glue.
//!
//! Owns the mDNS browser and converts discovered senders into
//! [`audio::DeviceInfo`] entries that the existing device dropdown can
//! display alongside inputs / loopbacks.
//!
//! Lifetime model: the browser is started once when the app boots and
//! runs for the lifetime of the process. `list_known_senders` returns a
//! cheap snapshot the UI polls every refresh / every frame.

use std::sync::Arc;

use parking_lot::Mutex;

use ayr_audio_link_core::discovery::{Browser, DiscoveredSender};

use crate::audio::{DeviceInfo, NetworkEndpoint, SourceKind};

/// Browser wrapped so boot-time failures degrade gracefully — a broken
/// mDNS stack (firewall, disabled multicast, missing IPv4) shouldn't
/// prevent the meter from launching; it should just mean the NETWORK
/// section stays empty.
pub struct NetworkSources {
    browser: Mutex<Option<Browser>>,
}

impl NetworkSources {
    pub fn start() -> Arc<Self> {
        let browser = match Browser::new() {
            Ok(b) => Some(b),
            Err(e) => {
                log::warn!("mdns browser unavailable: {e}");
                None
            }
        };
        Arc::new(Self {
            browser: Mutex::new(browser),
        })
    }

    /// Snapshot of currently visible senders as `DeviceInfo` entries
    /// ready to splice into the meter's device list.
    pub fn discovered(&self) -> Vec<DeviceInfo> {
        let Some(browser) = self.browser.lock().as_ref().map(|b| b.current()) else {
            return Vec::new();
        };
        browser.into_iter().map(sender_to_device_info).collect()
    }

    /// Raw snapshot, handy for UI code that needs fingerprint / host info
    /// the `DeviceInfo` shape throws away.
    #[allow(dead_code)]
    pub fn senders(&self) -> Vec<DiscoveredSender> {
        self.browser
            .lock()
            .as_ref()
            .map(|b| b.current())
            .unwrap_or_default()
    }
}

fn sender_to_device_info(sender: DiscoveredSender) -> DeviceInfo {
    // Prefer IPv4 — that's what Windows Firewall is almost always set to
    // allow, and the sender's installer opens an IPv4 rule explicitly.
    let host = sender.best_addr().unwrap_or(std::net::IpAddr::V4(
        std::net::Ipv4Addr::new(0, 0, 0, 0),
    ));
    let service_id = format!(
        "{}#{}",
        sender.instance_name,
        // Short fingerprint prefix so two devices with the same hostname
        // don't collide on the "stable identity" key.
        sender.fingerprint.chars().take(8).collect::<String>()
    );
    DeviceInfo {
        name: sender.display_name.clone(),
        kind: SourceKind::Network,
        max_channels: 2,
        network: Some(NetworkEndpoint {
            service_id,
            host,
            port: sender.port,
            protocol_version: sender.protocol_version,
        }),
    }
}
