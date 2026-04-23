//! Shared protocol + discovery helpers for AYR Audio Link.
//!
//! This crate is the only place where the wire format between the AYR Audio
//! Link **sender** and the AYR Audio Meter **receiver** is defined, so both
//! binaries stay byte-compatible as the protocol evolves. There is **no**
//! audio DSP in here and **no** GUI — the sender's capture loop calls
//! [`wire::write_handshake`] + [`wire::write_frame`], the meter's network
//! source calls [`wire::read_handshake`] + [`wire::read_frame`], and
//! [`discovery`] glues them together on a LAN.
//!
//! Design notes:
//!
//! * The payload is **uncompressed `f32` interleaved PCM**. A lossy codec
//!   would alter peak / RMS readings and break dBFS / PR accuracy, which
//!   defeats the point of shipping a meter companion — so we trade a few
//!   Mbps of LAN bandwidth for integrity.
//! * All integers are little-endian. Handshake and frame payloads are
//!   length-prefixed so we can evolve the protocol without forcing a hard
//!   version break.
//! * Service type for Zeroconf is `_ayraudiolink._tcp.local.` — registered
//!   nowhere central; it's a vendor-specific subtype in the `_tcp` tree,
//!   which is the standard way to publish private protocols.

pub mod discovery;
pub mod error;
pub mod fingerprint;
pub mod wire;

pub use error::{LinkError, LinkResult};

/// Wire-protocol version emitted in every handshake. Bump this when the
/// layout of either the handshake or the frame header changes in a way the
/// old reader cannot parse. Keep additive changes at `PROTOCOL_VERSION` —
/// anything that changes the handshake layout warrants a new major.
pub const PROTOCOL_VERSION: u16 = 1;

/// Default TCP port the sender listens on. Not assigned by IANA — picked
/// from the user-registered range, far from common audio ports. End users
/// can override this on the sender's settings if they hit a clash.
pub const DEFAULT_PORT: u16 = 45451;

/// mDNS service type. The trailing dot is required by DNS-SD conventions.
pub const MDNS_SERVICE_TYPE: &str = "_ayraudiolink._tcp.local.";

/// Upper bound on channels the receiver is willing to accept. Matches
/// `MAX_CHANNELS` in the meter's `audio.rs` so a non-conforming sender is
/// rejected at handshake time rather than triggering an assertion deep in
/// the DSP path.
pub const MAX_CHANNELS: u8 = 8;

/// Maximum PCM payload bytes per frame. Guards the reader against a hostile
/// or buggy sender pushing unbounded allocations. 64 KiB comfortably fits
/// ~340 ms of 48 kHz stereo `f32` — far larger than any sane frame the
/// sender would emit (we target 5–10 ms).
pub const MAX_FRAME_PAYLOAD_BYTES: u32 = 64 * 1024;
