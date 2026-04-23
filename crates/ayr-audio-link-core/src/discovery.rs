//! mDNS-SD advertise / browse for AYR Audio Link.
//!
//! The sender calls [`Advertiser::new`] on startup to publish itself on
//! `_ayraudiolink._tcp.local.`; the meter calls [`Browser::new`] to watch
//! for senders appearing / disappearing on the LAN. All the ugly
//! cross-thread bookkeeping is hidden behind these two handles.
//!
//! Key design points:
//!
//! * A single `mdns_sd::ServiceDaemon` is wrapped once inside each struct.
//!   Shutdown is done in `Drop` so callers don't leak the background
//!   thread.
//! * The Browser returns a *snapshot* `Vec<DiscoveredSender>` via
//!   [`Browser::current`], instead of a stream of events. This matches the
//!   receiver UI which polls at ~60Hz and wants an easy answer to "what's
//!   on the LAN *right now?*".
//! * TXT records carry protocol version + the sender's pairing fingerprint
//!   so the receiver can render the pairing short code without a separate
//!   round trip.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use parking_lot_fallback::Mutex;
use serde::{Deserialize, Serialize};

use crate::{LinkError, LinkResult, MDNS_SERVICE_TYPE, PROTOCOL_VERSION};

/// A sender seen on the local network. The meter keeps a `Vec<Self>` and
/// uses it to populate the `NETWORK` section of the device dropdown.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveredSender {
    /// mDNS service instance name (the "Alice-PC (Input 1/2)" piece before
    /// the service type). Stable across restarts of the same sender because
    /// the sender derives it from hostname + device.
    pub instance_name: String,
    /// Human-readable display name the sender puts in its TXT record.
    pub display_name: String,
    /// Pairing fingerprint (SHA-256 hex) the sender publishes so the UI
    /// can show a short matching code without an extra round trip.
    pub fingerprint: String,
    /// Protocol major as advertised in the TXT record.
    pub protocol_version: u16,
    /// First reachable IPv4/v6 address resolved for the sender's hostname.
    /// May be empty if mDNS resolved only link-local IPv6 we can't dial.
    pub addresses: Vec<IpAddr>,
    /// TCP port the sender is listening on.
    pub port: u16,
}

impl DiscoveredSender {
    /// Best address to dial. Prefers IPv4 (firewalls on Windows almost
    /// always allow v4 but block unsolicited v6 traffic); falls back to v6.
    pub fn best_addr(&self) -> Option<IpAddr> {
        self.addresses
            .iter()
            .find(|a| a.is_ipv4())
            .copied()
            .or_else(|| self.addresses.first().copied())
    }
}

/// TXT-record keys. Keep these short — DNS TXT records have per-entry
/// length limits and mDNS implementations deal poorly with verbose keys.
const TXT_KEY_VERSION: &str = "v";
const TXT_KEY_FP: &str = "fp";
const TXT_KEY_DISPLAY: &str = "n";
const TXT_KEY_CHANNELS: &str = "ch";
const TXT_KEY_SAMPLE_RATE: &str = "sr";

/// Sender-side: publish our service so meters on the LAN see us.
///
/// Holds the running `ServiceDaemon` for its lifetime and unregisters
/// + shuts it down on `Drop`, so a sender that crashes mid-stream doesn't
/// leave a stale advert on the network for the next 2 minutes.
pub struct Advertiser {
    daemon: ServiceDaemon,
    service_fullname: String,
}

impl Advertiser {
    /// `instance_name` is the unique label that identifies *this* sender
    /// (not the service *type*, which is always `MDNS_SERVICE_TYPE`).
    /// Pick something like `"{hostname}-ayraudiolink"` so multiple senders
    /// on the same LAN don't collide.
    pub fn new(
        instance_name: &str,
        port: u16,
        display_name: &str,
        fingerprint: &str,
        sample_rate: u32,
        channels: u8,
    ) -> LinkResult<Self> {
        let daemon = ServiceDaemon::new().map_err(|e| LinkError::Mdns(e.to_string()))?;

        // Let the daemon auto-detect interface addresses. mdns-sd handles
        // this when you pass an empty host IP list.
        let host_ipv4: &[IpAddr] = &[];
        let host_name = format!("{}.local.", instance_name);

        // TXT record with a small set of sanity-check fields. The meter is
        // also told all this over the handshake once it connects, but
        // having it in mDNS lets us filter in the UI *before* dialling.
        let txt: HashMap<String, String> = [
            (TXT_KEY_VERSION.into(), PROTOCOL_VERSION.to_string()),
            (TXT_KEY_FP.into(), fingerprint.to_string()),
            (TXT_KEY_DISPLAY.into(), display_name.to_string()),
            (TXT_KEY_CHANNELS.into(), channels.to_string()),
            (TXT_KEY_SAMPLE_RATE.into(), sample_rate.to_string()),
        ]
        .into_iter()
        .collect();

        let info = ServiceInfo::new(
            MDNS_SERVICE_TYPE,
            instance_name,
            &host_name,
            host_ipv4,
            port,
            txt,
        )
        .map_err(|e| LinkError::Mdns(e.to_string()))?
        .enable_addr_auto();

        let service_fullname = info.get_fullname().to_string();
        daemon
            .register(info)
            .map_err(|e| LinkError::Mdns(e.to_string()))?;
        Ok(Self {
            daemon,
            service_fullname,
        })
    }

    pub fn service_fullname(&self) -> &str {
        &self.service_fullname
    }
}

impl Drop for Advertiser {
    fn drop(&mut self) {
        // Tell the network we're gone. `unregister` is best-effort — if the
        // daemon has already died (e.g. the whole process is mid-shutdown
        // and the socket is gone), we just log and move on.
        if let Err(e) = self.daemon.unregister(&self.service_fullname) {
            log::debug!("mdns unregister: {e}");
        }
        if let Err(e) = self.daemon.shutdown() {
            log::debug!("mdns shutdown: {e}");
        }
    }
}

/// Receiver-side: browse the LAN for senders. Spins up a background thread
/// that drains the `ServiceDaemon` event receiver and keeps a shared
/// `Vec<DiscoveredSender>` up to date.
pub struct Browser {
    daemon: ServiceDaemon,
    shared: Arc<Mutex<Vec<DiscoveredSender>>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Browser {
    pub fn new() -> LinkResult<Self> {
        let daemon = ServiceDaemon::new().map_err(|e| LinkError::Mdns(e.to_string()))?;
        let shared: Arc<Mutex<Vec<DiscoveredSender>>> = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let receiver = daemon
            .browse(MDNS_SERVICE_TYPE)
            .map_err(|e| LinkError::Mdns(e.to_string()))?;

        let shared_bg = shared.clone();
        let stop_bg = stop.clone();
        let thread = thread::Builder::new()
            .name("ayr-link-mdns-browse".into())
            .spawn(move || {
                // Periodically wake up so we can honour `stop` promptly even
                // when the LAN is idle (no mDNS traffic).
                let idle_tick = Duration::from_millis(250);
                let mut last_gc = Instant::now();
                while !stop_bg.load(std::sync::atomic::Ordering::Relaxed) {
                    match receiver.recv_timeout(idle_tick) {
                        Ok(event) => handle_event(event, &shared_bg),
                        Err(flume::RecvTimeoutError::Timeout) => {}
                        Err(flume::RecvTimeoutError::Disconnected) => break,
                    }
                    // Drop stale adverts that haven't refreshed in 60s —
                    // `mdns-sd` emits ServiceRemoved on clean exits but an
                    // abrupt power-off leaves the record in our list until
                    // the LAN expires it.
                    if last_gc.elapsed() > Duration::from_secs(5) {
                        last_gc = Instant::now();
                        // No-op for now; ServiceRemoved handling keeps the
                        // list fresh. Kept as an explicit hook so a future
                        // TTL-based GC can slot in without touching the
                        // event loop.
                    }
                }
            })
            .map_err(|e| LinkError::Mdns(format!("spawn browse thread: {e}")))?;

        Ok(Self {
            daemon,
            shared,
            stop,
            thread: Some(thread),
        })
    }

    /// Returns a cloned snapshot of the currently visible senders, sorted
    /// by instance name for stable UI ordering.
    pub fn current(&self) -> Vec<DiscoveredSender> {
        let mut v = self.shared.lock().clone();
        v.sort_by(|a, b| a.instance_name.cmp(&b.instance_name));
        v
    }
}

impl Drop for Browser {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Err(e) = self.daemon.stop_browse(MDNS_SERVICE_TYPE) {
            log::debug!("stop_browse: {e}");
        }
        if let Err(e) = self.daemon.shutdown() {
            log::debug!("shutdown: {e}");
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn handle_event(event: ServiceEvent, shared: &Mutex<Vec<DiscoveredSender>>) {
    match event {
        ServiceEvent::ServiceResolved(info) => {
            let sender = sender_from_info(&info);
            let mut guard = shared.lock();
            match guard.iter().position(|s| s.instance_name == sender.instance_name) {
                Some(i) => guard[i] = sender,
                None => guard.push(sender),
            }
        }
        ServiceEvent::ServiceRemoved(_type, fullname) => {
            // Fullname is "<instance>.<service-type>", we only care about
            // the instance label.
            let instance = fullname
                .strip_suffix(&format!(".{MDNS_SERVICE_TYPE}"))
                .unwrap_or(&fullname)
                .to_string();
            let mut guard = shared.lock();
            guard.retain(|s| s.instance_name != instance);
        }
        _ => {}
    }
}

fn sender_from_info(info: &mdns_sd::ServiceInfo) -> DiscoveredSender {
    let txt_map: HashMap<String, String> = info
        .get_properties()
        .iter()
        .map(|p| (p.key().to_string(), p.val_str().to_string()))
        .collect();
    let display_name = txt_map
        .get(TXT_KEY_DISPLAY)
        .cloned()
        .unwrap_or_else(|| info.get_hostname().to_string());
    let fingerprint = txt_map.get(TXT_KEY_FP).cloned().unwrap_or_default();
    let protocol_version = txt_map
        .get(TXT_KEY_VERSION)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    let addresses: Vec<IpAddr> = info.get_addresses().iter().copied().collect();
    DiscoveredSender {
        instance_name: info
            .get_fullname()
            .strip_suffix(&format!(".{MDNS_SERVICE_TYPE}"))
            .unwrap_or(info.get_fullname())
            .to_string(),
        display_name,
        fingerprint,
        protocol_version,
        addresses,
        port: info.get_port(),
    }
}

// -- `parking_lot` is a convenient alias so the callers in this crate don't
//    need to pull in the dep themselves. It's the same crate the rest of the
//    workspace uses, so no duplicate monomorphisation.
mod parking_lot_fallback {
    // Re-export under the name the discovery module imports, keeping the
    // compile-time dependency surface to this tiny indirection.
    pub use parking_lot::Mutex;
}
