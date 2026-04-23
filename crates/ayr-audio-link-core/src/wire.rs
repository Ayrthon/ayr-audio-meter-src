//! Binary wire format for an AYR Audio Link TCP stream.
//!
//! Layout on the wire (all integers little-endian):
//!
//! ```text
//! ┌───────────── Handshake (once, sender -> receiver after accept) ─────────────┐
//! │  0..4    magic          "AYRL"                                              │
//! │  4..6    version        u16    (currently 1, see PROTOCOL_VERSION)          │
//! │  6..10   sample_rate    u32    Hz                                           │
//! │ 10..11   channels       u8     1..=MAX_CHANNELS                             │
//! │ 11..12   sample_fmt     u8     1 = f32le interleaved (only variant today)   │
//! │ 12..14   name_len       u16    UTF-8 byte length of the display name        │
//! │ 14..+N   name           bytes  sender's human-readable name                 │
//! ├─────────────────────── Frames (N, loop, sender -> receiver) ────────────────┤
//! │   0..4   payload_len    u32    payload byte count                           │
//! │   4..12  sender_pts_ns  u64    sender monotonic clock (drift / jitter hint) │
//! │  12..+N  payload        bytes  interleaved f32le PCM                        │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Payload length is the **only** source of truth for frame size — we don't
//! encode the sample count in the header because `channels * bytes_per_sample`
//! trivially derives it from `payload_len`, and duplicating it opens a
//! validation footgun. The reader enforces
//! `payload_len % (channels * 4) == 0`.

use std::io::{Read, Write};

use crate::{
    LinkError, LinkResult, MAX_CHANNELS, MAX_FRAME_PAYLOAD_BYTES, PROTOCOL_VERSION,
};

pub const MAGIC: [u8; 4] = *b"AYRL";

/// Only wire variant supported today: interleaved 32-bit float, little-endian.
pub const SAMPLE_FMT_F32LE: u8 = 1;

/// Paranoia cap on handshake name length. 256 bytes is way more than any real
/// hostname / display string and still fits in a single `Vec<u8>` without
/// worry. Above this we assume the stream is malformed and abort.
pub const MAX_NAME_LEN: u16 = 256;

/// Data exchanged at connection setup. Immutable for the lifetime of one TCP
/// stream — the sender's device / format is not allowed to change mid-stream,
/// and the meter reacts to a format change by reconnecting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Handshake {
    pub version: u16,
    pub sample_rate: u32,
    pub channels: u8,
    /// Human-readable display name the receiver shows in its device picker.
    /// Typically something like `"MikeStudio-PC (Focusrite Input 1/2)"`.
    pub display_name: String,
}

impl Handshake {
    /// Convenience constructor used by the sender. Always stamps the current
    /// `PROTOCOL_VERSION`.
    pub fn new(sample_rate: u32, channels: u8, display_name: impl Into<String>) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            sample_rate,
            channels,
            display_name: display_name.into(),
        }
    }

    /// Bytes per interleaved audio *frame* (all channels of one sample).
    pub fn frame_stride_bytes(&self) -> u32 {
        // f32 = 4 bytes. Only sample format we support today.
        u32::from(self.channels) * 4
    }
}

pub fn write_handshake<W: Write>(w: &mut W, hs: &Handshake) -> LinkResult<()> {
    if hs.channels == 0 || hs.channels > MAX_CHANNELS {
        return Err(LinkError::InvalidChannels {
            got: hs.channels,
            max: MAX_CHANNELS,
        });
    }
    let name_bytes = hs.display_name.as_bytes();
    if name_bytes.len() > MAX_NAME_LEN as usize {
        return Err(LinkError::NameTooLong(name_bytes.len() as u16));
    }
    w.write_all(&MAGIC)?;
    w.write_all(&hs.version.to_le_bytes())?;
    w.write_all(&hs.sample_rate.to_le_bytes())?;
    w.write_all(&[hs.channels, SAMPLE_FMT_F32LE])?;
    w.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
    w.write_all(name_bytes)?;
    Ok(())
}

pub fn read_handshake<R: Read>(r: &mut R) -> LinkResult<Handshake> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(LinkError::BadMagic);
    }
    let version = read_u16(r)?;
    if version > PROTOCOL_VERSION {
        return Err(LinkError::UnsupportedVersion {
            got: version,
            max: PROTOCOL_VERSION,
        });
    }
    let sample_rate = read_u32(r)?;
    let mut chfmt = [0u8; 2];
    r.read_exact(&mut chfmt)?;
    let channels = chfmt[0];
    let sample_fmt = chfmt[1];
    if channels == 0 || channels > MAX_CHANNELS {
        return Err(LinkError::InvalidChannels {
            got: channels,
            max: MAX_CHANNELS,
        });
    }
    if sample_fmt != SAMPLE_FMT_F32LE {
        return Err(LinkError::UnsupportedSampleFormat(sample_fmt));
    }
    let name_len = read_u16(r)?;
    if name_len > MAX_NAME_LEN {
        return Err(LinkError::NameTooLong(name_len));
    }
    let mut name_buf = vec![0u8; name_len as usize];
    r.read_exact(&mut name_buf)?;
    let display_name = String::from_utf8(name_buf)?;
    Ok(Handshake {
        version,
        sample_rate,
        channels,
        display_name,
    })
}

/// Reuse this buffer across calls so a streaming reader doesn't allocate per
/// frame. `samples` is refilled on each [`read_frame`] and is only valid
/// until the next call.
#[derive(Default)]
pub struct FrameBuf {
    pub pts_ns: u64,
    /// Raw little-endian bytes straight off the wire. Kept around so
    /// `read_frame` can reuse the allocation across frames; external
    /// consumers should read from [`samples`](Self::samples) instead.
    raw: Vec<u8>,
    /// Decoded interleaved `f32` samples. Populated by [`read_frame`].
    /// Decoded into a dedicated `Vec<f32>` rather than transmuting from
    /// `raw` so we avoid the alignment footgun of `Vec<u8>::align_to` on
    /// platforms where the allocator doesn't guarantee 4-byte alignment.
    pub samples: Vec<f32>,
}

impl FrameBuf {
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn write_frame<W: Write>(
    w: &mut W,
    pts_ns: u64,
    payload: &[u8],
) -> LinkResult<()> {
    let len = payload.len();
    if len > MAX_FRAME_PAYLOAD_BYTES as usize {
        return Err(LinkError::FramePayloadTooLarge {
            got: len as u32,
            limit: MAX_FRAME_PAYLOAD_BYTES,
        });
    }
    w.write_all(&(len as u32).to_le_bytes())?;
    w.write_all(&pts_ns.to_le_bytes())?;
    w.write_all(payload)?;
    Ok(())
}

/// Reads one frame into `buf`. Returns `Ok(())` on success, propagates
/// `UnexpectedEof` via `LinkError::Io` on a clean remote close.
pub fn read_frame<R: Read>(
    r: &mut R,
    stride: u32,
    buf: &mut FrameBuf,
) -> LinkResult<()> {
    let payload_len = read_u32(r)?;
    if payload_len > MAX_FRAME_PAYLOAD_BYTES {
        return Err(LinkError::FramePayloadTooLarge {
            got: payload_len,
            limit: MAX_FRAME_PAYLOAD_BYTES,
        });
    }
    if stride > 0 && payload_len % stride != 0 {
        return Err(LinkError::UnalignedFramePayload {
            got: payload_len,
            stride,
        });
    }
    let mut pts = [0u8; 8];
    r.read_exact(&mut pts)?;
    buf.pts_ns = u64::from_le_bytes(pts);
    buf.raw.resize(payload_len as usize, 0);
    r.read_exact(&mut buf.raw)?;
    // Decode into the typed sample buffer. Only `f32le` exists today; when
    // we add `i16le` or `i24le` this is the only spot that needs a match.
    let sample_count = payload_len as usize / 4;
    buf.samples.clear();
    buf.samples.reserve(sample_count);
    for chunk in buf.raw.chunks_exact(4) {
        buf.samples
            .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(())
}

fn read_u16<R: Read>(r: &mut R) -> LinkResult<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> LinkResult<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn handshake_round_trips_basic() {
        let original = Handshake::new(48_000, 2, "ALICE-PC (Line In 1/2)");
        let mut buf = Vec::new();
        write_handshake(&mut buf, &original).unwrap();
        let got = read_handshake(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(original, got);
    }

    #[test]
    fn handshake_round_trips_unicode_name() {
        let original = Handshake::new(44_100, 1, "Björk's Studio 🎙");
        let mut buf = Vec::new();
        write_handshake(&mut buf, &original).unwrap();
        let got = read_handshake(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(original, got);
    }

    #[test]
    fn handshake_rejects_zero_channels() {
        let hs = Handshake {
            version: PROTOCOL_VERSION,
            sample_rate: 48_000,
            channels: 0,
            display_name: "bogus".into(),
        };
        let mut buf = Vec::new();
        assert!(matches!(
            write_handshake(&mut buf, &hs),
            Err(LinkError::InvalidChannels { got: 0, .. })
        ));
    }

    #[test]
    fn handshake_rejects_too_many_channels() {
        let hs = Handshake {
            version: PROTOCOL_VERSION,
            sample_rate: 48_000,
            channels: MAX_CHANNELS + 1,
            display_name: "bogus".into(),
        };
        let mut buf = Vec::new();
        assert!(matches!(
            write_handshake(&mut buf, &hs),
            Err(LinkError::InvalidChannels { .. })
        ));
    }

    #[test]
    fn handshake_rejects_bad_magic() {
        let mut buf = b"AYRX\x01\x00".to_vec(); // wrong magic
        buf.extend_from_slice(&[0u8; 12]);
        let err = read_handshake(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, LinkError::BadMagic));
    }

    #[test]
    fn handshake_rejects_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&999u16.to_le_bytes());
        buf.extend_from_slice(&48_000u32.to_le_bytes());
        buf.extend_from_slice(&[2u8, SAMPLE_FMT_F32LE]);
        buf.extend_from_slice(&0u16.to_le_bytes());
        assert!(matches!(
            read_handshake(&mut Cursor::new(&buf)),
            Err(LinkError::UnsupportedVersion { got: 999, .. })
        ));
    }

    #[test]
    fn handshake_rejects_unknown_sample_fmt() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        buf.extend_from_slice(&48_000u32.to_le_bytes());
        buf.extend_from_slice(&[2u8, 0xFFu8]); // channels 2, bogus fmt
        buf.extend_from_slice(&0u16.to_le_bytes());
        assert!(matches!(
            read_handshake(&mut Cursor::new(&buf)),
            Err(LinkError::UnsupportedSampleFormat(0xFF))
        ));
    }

    #[test]
    fn frame_round_trips_with_stride_check() {
        let stride = 2 * 4; // 2ch * 4B f32
        let samples = vec![0.25f32, -0.25, 0.5, -0.5, 1.0, -1.0];
        let mut payload = Vec::with_capacity(samples.len() * 4);
        for s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }
        let mut wire = Vec::new();
        write_frame(&mut wire, 1234, &payload).unwrap();
        let mut fb = FrameBuf::new();
        read_frame(&mut Cursor::new(&wire), stride, &mut fb).unwrap();
        assert_eq!(fb.pts_ns, 1234);
        assert_eq!(fb.samples.as_slice(), samples.as_slice());
    }

    #[test]
    fn frame_rejects_unaligned_payload_for_stride() {
        let stride = 8; // 2ch * 4B
        let mut wire = Vec::new();
        write_frame(&mut wire, 7, &[0u8; 3]).unwrap();
        let mut fb = FrameBuf::new();
        assert!(matches!(
            read_frame(&mut Cursor::new(&wire), stride, &mut fb),
            Err(LinkError::UnalignedFramePayload { got: 3, .. })
        ));
    }

    #[test]
    fn frame_rejects_oversize_payload() {
        let oversize = MAX_FRAME_PAYLOAD_BYTES + 1;
        // We can't actually allocate that much reliably in a unit test, so
        // hand-craft the header and let the reader reject it immediately.
        let mut wire = Vec::new();
        wire.extend_from_slice(&oversize.to_le_bytes());
        wire.extend_from_slice(&0u64.to_le_bytes());
        let mut fb = FrameBuf::new();
        assert!(matches!(
            read_frame(&mut Cursor::new(&wire), 0, &mut fb),
            Err(LinkError::FramePayloadTooLarge { .. })
        ));
    }
}
