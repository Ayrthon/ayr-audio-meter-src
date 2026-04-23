//! Shared error type for the wire protocol and discovery layer.

use thiserror::Error;

pub type LinkResult<T> = Result<T, LinkError>;

#[derive(Debug, Error)]
pub enum LinkError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// The handshake's 4-byte magic wasn't `"AYRL"`. Almost always means the
    /// receiver dialled the wrong host or some other TCP service is listening
    /// on the same port.
    #[error("not an AYR Audio Link stream (bad magic)")]
    BadMagic,

    /// The sender advertised a protocol version we don't understand. Usually
    /// fixed by updating the older of the two binaries.
    #[error("unsupported protocol version {got}; this build understands versions <= {max}")]
    UnsupportedVersion { got: u16, max: u16 },

    /// Handshake declared a channel count the receiver will not accept (0 or
    /// above `MAX_CHANNELS`).
    #[error("invalid channel count {got}; must be 1..={max}")]
    InvalidChannels { got: u8, max: u8 },

    /// Handshake declared an unknown `sample_fmt` byte. Only `1` (interleaved
    /// `f32` little-endian) is supported today.
    #[error("unsupported sample format code {0}")]
    UnsupportedSampleFormat(u8),

    /// Frame payload length exceeded `MAX_FRAME_PAYLOAD_BYTES`. Treated as a
    /// framing / corruption error rather than a recoverable condition — we
    /// close the connection.
    #[error("frame payload too large: {got} bytes (limit {limit})")]
    FramePayloadTooLarge { got: u32, limit: u32 },

    /// Declared name length exceeds what we're willing to allocate for a
    /// single handshake. Paranoia guard against malicious senders; real
    /// display names are a few dozen bytes.
    #[error("sender display name too long: {0} bytes")]
    NameTooLong(u16),

    /// String in the wire format wasn't valid UTF-8. Display names are the
    /// only string field today.
    #[error("display name is not valid UTF-8: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    /// The frame payload size didn't match the channels/sample-format
    /// expectations agreed at handshake time (e.g. odd byte count for a
    /// 2-channel `f32` stream).
    #[error("frame payload size {got} not a multiple of frame stride {stride}")]
    UnalignedFramePayload { got: u32, stride: u32 },

    /// mDNS advertise / browse operations failed.
    #[error("mDNS error: {0}")]
    Mdns(String),
}
