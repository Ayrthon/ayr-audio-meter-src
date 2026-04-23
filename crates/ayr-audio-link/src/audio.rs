//! Audio capture for the sender.
//!
//! Structured as a tiny, self-contained analogue of the meter's
//! `src/audio.rs`:
//!
//! * [`list_devices`] enumerates cpal input devices (cross-platform) and,
//!   on Windows, appends WASAPI render endpoints for loopback capture.
//! * [`start_capture`] spins up a background thread that repeatedly
//!   decodes audio from the chosen device and calls a user-supplied sink
//!   closure with interleaved `f32` blocks + timing metadata.
//!
//! The sender doesn't need the meter's DSP / LUFS / spectrum plumbing —
//! its only job is to hand raw PCM to the TCP writer in `net.rs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, anyhow};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Cap matches `ayr_audio_link_core::MAX_CHANNELS` so a sender can never
/// advertise more channels than the meter will accept.
pub const MAX_CHANNELS: u8 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Input,
    Loopback,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub kind: SourceKind,
    pub max_channels: u8,
}

pub fn list_devices() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let mut out: Vec<DeviceInfo> = Vec::new();

    for d in host.input_devices().context("enumerate cpal inputs")? {
        let name = d
            .name()
            .unwrap_or_else(|_| "(unnamed input)".to_string());
        let max_channels = d
            .default_input_config()
            .map(|c| c.channels() as u8)
            .unwrap_or(2)
            .min(MAX_CHANNELS);
        out.push(DeviceInfo {
            name,
            kind: SourceKind::Input,
            max_channels,
        });
    }

    #[cfg(windows)]
    out.extend(windows_loopback::list_output_devices()?);

    // Deterministic order — inputs alphabetical, then loopbacks alphabetical.
    out.sort_by(|a, b| (a.kind as u8, a.name.clone()).cmp(&(b.kind as u8, b.name.clone())));
    Ok(out)
}

pub fn default_input_name() -> Option<String> {
    cpal::default_host()
        .default_input_device()
        .and_then(|d| d.name().ok())
}

#[cfg(windows)]
pub fn default_output_name() -> Option<String> {
    windows_loopback::default_output_name().ok().flatten()
}

#[cfg(not(windows))]
pub fn default_output_name() -> Option<String> {
    None
}

/// Called from the capture thread for every block of samples produced by
/// the chosen device. Must not block the thread for long — typical
/// implementations push the samples onto a lock-free queue for the TCP
/// writer.
///
/// `samples` is interleaved `f32`; `channels * sample_count` total.
pub trait SampleSink: Send + 'static {
    fn on_block(&mut self, samples: &[f32], channels: u8, sample_rate: u32);
}

impl<F> SampleSink for F
where
    F: FnMut(&[f32], u8, u32) + Send + 'static,
{
    fn on_block(&mut self, samples: &[f32], channels: u8, sample_rate: u32) {
        (self)(samples, channels, sample_rate)
    }
}

/// Running capture. Drop (or call `stop`) to shut the thread down cleanly.
pub struct Capture {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    /// Reported by the capture thread after its first successful block so
    /// the UI / TCP handshake can adopt the correct values without
    /// guessing from cpal's declared config.
    pub format: Arc<parking_lot::Mutex<Option<StreamFormat>>>,
}

impl Capture {
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StreamFormat {
    pub sample_rate: u32,
    pub channels: u8,
}

pub fn start_capture<S: SampleSink>(
    device: DeviceInfo,
    sink: S,
) -> Result<Capture> {
    let stop = Arc::new(AtomicBool::new(false));
    let format = Arc::new(parking_lot::Mutex::new(None::<StreamFormat>));

    // Wrap the sink once so both the cpal callback closure and the WASAPI
    // loopback loop can share ownership through a single `Arc`. The mutex is
    // `parking_lot`'s — no poisoning concerns if a single callback panics.
    let sink_shared: Arc<parking_lot::Mutex<S>> = Arc::new(parking_lot::Mutex::new(sink));

    let stop_bg = stop.clone();
    let format_bg = format.clone();
    let sink_bg = sink_shared.clone();
    let handle = thread::Builder::new()
        .name(format!("ayr-link-capture-{}", device.kind_as_str()))
        .spawn(move || {
            let outcome = match device.kind {
                SourceKind::Input => run_cpal_input(&device.name, &stop_bg, &format_bg, sink_bg),
                SourceKind::Loopback => {
                    #[cfg(windows)]
                    {
                        windows_loopback::run(&device.name, &stop_bg, &format_bg, sink_bg)
                    }
                    #[cfg(not(windows))]
                    {
                        let _ = sink_bg;
                        Err(anyhow!(
                            "loopback capture is only supported on Windows in this build"
                        ))
                    }
                }
            };
            if let Err(e) = outcome {
                log::error!("capture stopped: {e:#}");
            }
        })
        .context("spawn capture thread")?;

    Ok(Capture {
        stop,
        handle: Some(handle),
        format,
    })
}

impl DeviceInfo {
    fn kind_as_str(&self) -> &'static str {
        match self.kind {
            SourceKind::Input => "input",
            SourceKind::Loopback => "loopback",
        }
    }
}

fn run_cpal_input<S: SampleSink>(
    device_name: &str,
    stop: &AtomicBool,
    format_out: &parking_lot::Mutex<Option<StreamFormat>>,
    sink: Arc<parking_lot::Mutex<S>>,
) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .input_devices()
        .context("enumerate cpal inputs")?
        .find(|d| d.name().map(|n| n == device_name).unwrap_or(false))
        .ok_or_else(|| anyhow!("input device not found: {device_name}"))?;

    let cfg = device
        .default_input_config()
        .context("query cpal default_input_config")?;
    let sample_rate: u32 = cfg.sample_rate().into();
    let channels = (cfg.channels() as u8).min(MAX_CHANNELS);

    *format_out.lock() = Some(StreamFormat {
        sample_rate,
        channels,
    });

    let stream_cfg: cpal::StreamConfig = cfg.clone().into();
    let err_fn = |e| log::error!("cpal stream error: {e}");

    // Build one stream per sample format with format-specific conversion
    // inlined into the callback. `scratch` is owned per-closure so we don't
    // re-allocate per block; the closure keeps it alive for its lifetime.
    let stream = match cfg.sample_format() {
        SampleFormat::F32 => {
            let sink_cb = sink.clone();
            let mut scratch: Vec<f32> = Vec::with_capacity(8192);
            device.build_input_stream(
                &stream_cfg,
                move |data: &[f32], _| {
                    scratch.clear();
                    scratch.extend_from_slice(data);
                    sink_cb.lock().on_block(&scratch, channels, sample_rate);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let sink_cb = sink.clone();
            let mut scratch: Vec<f32> = Vec::with_capacity(8192);
            device.build_input_stream(
                &stream_cfg,
                move |data: &[i16], _| {
                    scratch.clear();
                    scratch.extend(data.iter().map(|&s| s as f32 / i16::MAX as f32));
                    sink_cb.lock().on_block(&scratch, channels, sample_rate);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let sink_cb = sink.clone();
            let mut scratch: Vec<f32> = Vec::with_capacity(8192);
            device.build_input_stream(
                &stream_cfg,
                move |data: &[u16], _| {
                    scratch.clear();
                    scratch.extend(
                        data.iter().map(|&s| (s as f32 / u16::MAX as f32) * 2.0 - 1.0),
                    );
                    sink_cb.lock().on_block(&scratch, channels, sample_rate);
                },
                err_fn,
                None,
            )?
        }
        other => return Err(anyhow!("unsupported cpal sample format: {other:?}")),
    };

    stream.play().context("cpal stream.play")?;
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(std::time::Duration::from_millis(50));
    }
    drop(stream);
    Ok(())
}

// ---- Windows loopback -------------------------------------------------------

#[cfg(windows)]
mod windows_loopback {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};

    use anyhow::{Context, Result, anyhow};
    use wasapi::{
        Direction, DeviceEnumerator, SampleType, StreamMode, WaveFormat,
        initialize_mta,
    };

    use super::{DeviceInfo, MAX_CHANNELS, SampleSink, SourceKind, StreamFormat};

    pub fn list_output_devices() -> Result<Vec<DeviceInfo>> {
        initialize_mta().ok();
        let enumerator = DeviceEnumerator::new().context("DeviceEnumerator::new")?;
        let collection = enumerator
            .get_device_collection(&Direction::Render)
            .context("get_device_collection(Render)")?;
        let count = collection.get_nbr_devices().context("get_nbr_devices")?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let Ok(dev) = collection.get_device_at_index(i) else {
                continue;
            };
            let name = dev
                .get_friendlyname()
                .unwrap_or_else(|_| format!("Render {i}"));
            let max_channels = dev
                .get_iaudioclient()
                .and_then(|client| client.get_mixformat())
                .map(|fmt| fmt.get_nchannels() as u8)
                .unwrap_or(2)
                .min(MAX_CHANNELS);
            out.push(DeviceInfo {
                name,
                kind: SourceKind::Loopback,
                max_channels,
            });
        }
        Ok(out)
    }

    pub fn default_output_name() -> Result<Option<String>> {
        initialize_mta().ok();
        let enumerator = DeviceEnumerator::new().context("DeviceEnumerator::new")?;
        let dev = enumerator
            .get_default_device(&Direction::Render)
            .context("default render device")?;
        Ok(Some(dev.get_friendlyname().unwrap_or_default()))
    }

    pub fn run<S: SampleSink>(
        device_name: &str,
        stop: &AtomicBool,
        format_out: &parking_lot::Mutex<Option<StreamFormat>>,
        sink: std::sync::Arc<parking_lot::Mutex<S>>,
    ) -> Result<()> {
        initialize_mta().ok();
        let enumerator = DeviceEnumerator::new().context("DeviceEnumerator::new")?;
        let collection = enumerator
            .get_device_collection(&Direction::Render)
            .context("get_device_collection")?;
        let count = collection.get_nbr_devices().context("get_nbr_devices")?;
        let mut chosen = None;
        for i in 0..count {
            let Ok(dev) = collection.get_device_at_index(i) else {
                continue;
            };
            if dev
                .get_friendlyname()
                .map(|n| n == device_name)
                .unwrap_or(false)
            {
                chosen = Some(dev);
                break;
            }
        }
        let device = chosen.ok_or_else(|| anyhow!("render device `{device_name}` not found"))?;

        let mut audio_client = device.get_iaudioclient().context("get_iaudioclient")?;
        let mix_format = audio_client.get_mixformat().context("get_mixformat")?;
        let sample_rate = mix_format.get_samplespersec();
        let channels_u = mix_format.get_nchannels() as usize;
        let channels = (channels_u as u8).min(MAX_CHANNELS);
        let bits = mix_format.get_bitspersample() as usize;
        let valid_bits = mix_format.get_validbitspersample() as usize;
        let block_align = mix_format.get_blockalign() as usize;

        *format_out.lock() = Some(StreamFormat {
            sample_rate,
            channels,
        });

        // Ask WASAPI for its native mix format in shared mode — guaranteed
        // supported, and `autoconvert: true` lets the driver do any
        // resampling we'd otherwise have to implement ourselves.
        let desired_format = WaveFormat::new(
            bits,
            valid_bits,
            &if bits == 32 && valid_bits == 32 {
                SampleType::Float
            } else {
                SampleType::Int
            },
            sample_rate as usize,
            channels_u,
            None,
        );

        let (_def_period, min_period) = audio_client
            .get_device_period()
            .context("get_device_period")?;
        let buffer_duration_hns = min_period.max(100_000);

        let mode = StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns,
        };
        audio_client
            .initialize_client(&desired_format, &Direction::Capture, &mode)
            .context("initialize_client loopback")?;
        let h_event = audio_client
            .set_get_eventhandle()
            .context("set_get_eventhandle")?;
        let capture_client = audio_client
            .get_audiocaptureclient()
            .context("get_audiocaptureclient")?;

        audio_client.start_stream().context("start_stream")?;

        let bytes_per_sample = bits / 8;
        let is_float = bits == 32 && valid_bits == 32;

        // Ring buffer that holds partial frames between WASAPI reads, so we
        // always emit whole frames to the sink.
        let mut sample_queue: VecDeque<u8> =
            VecDeque::with_capacity(block_align * sample_rate as usize / 10);
        let mut f32_scratch: Vec<f32> = Vec::with_capacity(channels_u * sample_rate as usize / 50);
        let mut contig: Vec<u8> = Vec::with_capacity(block_align * sample_rate as usize / 50);

        while !stop.load(Ordering::Relaxed) {
            // Drain everything WASAPI has queued before blocking again.
            while sample_queue.len() >= block_align {
                let frames = sample_queue.len() / block_align;
                contig.clear();
                contig.reserve(frames * block_align);
                for _ in 0..frames * block_align {
                    if let Some(b) = sample_queue.pop_front() {
                        contig.push(b);
                    }
                }
                f32_scratch.clear();
                decode_samples(
                    &contig,
                    bytes_per_sample,
                    is_float,
                    channels_u,
                    &mut f32_scratch,
                );
                sink.lock().on_block(&f32_scratch, channels, sample_rate);
            }

            match capture_client.get_next_packet_size() {
                Ok(Some(n)) if n > 0 => {
                    let _ = capture_client.read_from_device_to_deque(&mut sample_queue);
                }
                Ok(_) => {
                    if h_event.wait_for_event(200).is_err() {
                        continue;
                    }
                }
                Err(err) => {
                    log::warn!("get_next_packet_size: {err:?}");
                    let _ = h_event.wait_for_event(200);
                }
            }
        }
        audio_client.stop_stream().ok();
        Ok(())
    }

    fn decode_samples(
        bytes: &[u8],
        bytes_per_sample: usize,
        is_float: bool,
        _channels: usize,
        out: &mut Vec<f32>,
    ) {
        match (bytes_per_sample, is_float) {
            (4, true) => {
                for chunk in bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            (4, false) => {
                for chunk in bytes.chunks_exact(4) {
                    let s = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    out.push(s as f32 / i32::MAX as f32);
                }
            }
            (3, _) => {
                for chunk in bytes.chunks_exact(3) {
                    let raw = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0]);
                    let signed = (raw << 8) >> 8;
                    out.push(signed as f32 / 8_388_608.0);
                }
            }
            (2, _) => {
                for chunk in bytes.chunks_exact(2) {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(s as f32 / i16::MAX as f32);
                }
            }
            _ => {}
        }
    }
}
