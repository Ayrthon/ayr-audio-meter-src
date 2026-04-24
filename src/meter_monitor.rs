//! Low-latency **listen** path: tap the same interleaved float samples the meters
//! use and play them through the default output device (cpal). Windows-only
//! shipping target, but the implementation is cross-platform for `cargo check`.
//!
//! Design notes:
//! * The capture callback (`push_interleaved_f32`) writes into a bounded FIFO
//!   under a `parking_lot::Mutex`. Writes are tiny copies so the critical
//!   section is short.
//! * The cpal output callback (`fill_f32`) drains the FIFO in a single lock.
//!   To avoid the classic "both callbacks fire on uncorrelated cadences ⇒
//!   frequent underruns ⇒ clicks" pitfall, we hold off draining until the
//!   queue contains at least `PREFILL_MS` of input audio. If the queue then
//!   drains to under `REFILL_MS` we re-arm the prefill flag and output
//!   silence until the buffer fills back up. That converts micro-underruns
//!   into a single short silence instead of a stream of clicks.
//! * When the input and output sample rates match we skip the resampler
//!   entirely (a linear resampler is fine but pointless when ratio == 1).
//!   Mismatched rates fall back to cursor-driven linear interpolation.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use parking_lot::Mutex;

use crate::audio::MAX_CHANNELS;

/// Cap queued float samples (~0.5 s at 48 kHz × 8 ch) so memory stays bounded
/// if the output device stalls.
const MAX_QUEUE_SAMPLES: usize = 48_000 * MAX_CHANNELS / 2;

/// How much input audio must accumulate before the output thread is allowed
/// to start draining. Tuned for low monitor latency: ~2× the typical WASAPI
/// shared-mode input callback period (10 ms), which is enough headroom to
/// absorb callback jitter without adding noticeable delay. Bump to 40–60
/// if you hear clicks on a slower system.
const PREFILL_MS: u32 = 20;

/// If the queue drains below this watermark (in ms of *input* audio) we
/// re-arm the prefill latch and output silence until it fills again. Kept
/// small so an underrun becomes one brief dropout rather than a running
/// stutter.
const REFILL_MS: u32 = 3;

#[derive(Default)]
struct DrainState {
    /// Fractional position into the next input frame pair at the deque head,
    /// advanced by `in_rate / out_rate` per output frame. Only used when the
    /// output rate differs from the input rate.
    cursor: f64,
    /// `false` until the queue is primed with at least `PREFILL_MS` of audio.
    /// Reset by `set_enabled(false)` and by dry spells during playback.
    primed: bool,
}

impl DrainState {
    fn reset(&mut self) {
        self.cursor = 0.0;
        self.primed = false;
    }
}

/// Shared between the UI (toggle), the capture callbacks (push), and the output
/// thread (drain + resample).
pub struct MeterMonitorBus {
    enabled: AtomicBool,
    in_rate: AtomicU32,
    in_ch: AtomicU32,
    queue: Mutex<VecDeque<f32>>,
    drain: Mutex<DrainState>,
    output_thread: Mutex<Option<JoinHandle<()>>>,
}

impl MeterMonitorBus {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            enabled: AtomicBool::new(false),
            in_rate: AtomicU32::new(48_000),
            in_ch: AtomicU32::new(2),
            queue: Mutex::new(VecDeque::with_capacity(MAX_QUEUE_SAMPLES)),
            drain: Mutex::new(DrainState::default()),
            output_thread: Mutex::new(None),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
        // Always clear stale audio + drain state on a toggle so a fresh
        // enable starts cleanly primed (or silent).
        self.queue.lock().clear();
        self.drain.lock().reset();
    }

    /// Called from the capture thread once the stream format is known. Spawns the
    /// cpal output thread the first time; later calls (e.g. network reconnect) only
    /// refresh rate/channel metadata and clear queued audio.
    pub fn begin_capture(self: &Arc<Self>, in_rate: u32, in_ch: usize, stop: Arc<AtomicBool>) {
        self.in_rate.store(in_rate.max(1), Ordering::Relaxed);
        self.in_ch.store(
            (in_ch.max(1).min(MAX_CHANNELS)) as u32,
            Ordering::Relaxed,
        );
        self.queue.lock().clear();
        self.drain.lock().reset();

        let mut slot = self.output_thread.lock();
        if slot.is_some() {
            return;
        }
        let this = Arc::clone(self);
        *slot = Some(thread::spawn(move || this.output_main(stop)));
    }

    /// Join the output thread (no-op if it never started).
    pub fn end_capture(&self) {
        self.enabled.store(false, Ordering::Relaxed);
        self.queue.lock().clear();
        self.drain.lock().reset();
        if let Some(h) = self.output_thread.lock().take() {
            let _ = h.join();
        }
    }

    /// Push one interleaved block of float samples (`channels` wide). Safe to call
    /// with `enabled == false` (returns immediately).
    pub fn push_interleaved_f32(&self, data: &[f32], channels: usize) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let ch = channels.max(1).min(MAX_CHANNELS);
        if ch == 0 || data.len() < ch || data.len() % ch != 0 {
            return;
        }
        let mut q = self.queue.lock();
        q.extend(data.iter().copied());
        // Bound the queue by dropping the oldest whole frames. Losing a
        // handful of frames after a long stall is better than runaway RAM.
        while q.len() > MAX_QUEUE_SAMPLES {
            for _ in 0..ch {
                q.pop_front();
            }
        }
    }

    fn output_main(self: Arc<Self>, stop_engine: Arc<AtomicBool>) {
        let host = cpal::default_host();
        let Some(device) = host.default_output_device() else {
            log::warn!("meter monitor: no default output device");
            return;
        };
        let Ok(default_conf) = device.default_output_config() else {
            log::warn!("meter monitor: no default output config");
            return;
        };
        let sample_format = default_conf.sample_format();
        let config: StreamConfig = default_conf.config().into();
        let out_ch = config.channels.max(1) as usize;
        let out_rate: u32 = config.sample_rate;

        // Preallocate the i16/u16 conversion scratch so we never allocate in
        // the realtime callback. cpal gives us `nf * out_ch` samples per
        // callback; 2 s at 192 kHz × 8 ch is a comfortable upper bound.
        let scratch_cap = 192_000 * out_ch.max(1) * 2;
        let tmp_i16 = Arc::new(Mutex::new(Vec::<f32>::with_capacity(scratch_cap)));
        let tmp_u16 = Arc::clone(&tmp_i16);

        let bus_f32 = Arc::clone(&self);
        let bus_i16 = Arc::clone(&self);
        let bus_u16 = Arc::clone(&self);
        let stop_f32 = stop_engine.clone();
        let stop_i16 = stop_engine.clone();
        let stop_u16 = stop_engine.clone();

        let stream_result = match sample_format {
            SampleFormat::F32 => device.build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    if stop_f32.load(Ordering::Relaxed) {
                        data.fill(0.0);
                        return;
                    }
                    bus_f32.fill_f32(data, out_ch, out_rate);
                },
                |e| log::error!("meter monitor output stream error: {e}"),
                None,
            ),
            SampleFormat::I16 => device.build_output_stream(
                &config,
                move |data: &mut [i16], _| {
                    if stop_i16.load(Ordering::Relaxed) {
                        data.fill(0);
                        return;
                    }
                    let mut tmp = tmp_i16.lock();
                    tmp.clear();
                    tmp.resize(data.len(), 0.0);
                    bus_i16.fill_f32(&mut tmp, out_ch, out_rate);
                    for (d, &s) in data.iter_mut().zip(tmp.iter()) {
                        *d = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    }
                },
                |e| log::error!("meter monitor output stream error: {e}"),
                None,
            ),
            SampleFormat::U16 => device.build_output_stream(
                &config,
                move |data: &mut [u16], _| {
                    if stop_u16.load(Ordering::Relaxed) {
                        data.fill(32768);
                        return;
                    }
                    let mut tmp = tmp_u16.lock();
                    tmp.clear();
                    tmp.resize(data.len(), 0.0);
                    bus_u16.fill_f32(&mut tmp, out_ch, out_rate);
                    for (d, &s) in data.iter_mut().zip(tmp.iter()) {
                        let v = ((s.clamp(-1.0, 1.0) * 32767.0) as i32 + 32768).clamp(0, 65535);
                        *d = v as u16;
                    }
                },
                |e| log::error!("meter monitor output stream error: {e}"),
                None,
            ),
            other => {
                log::warn!("meter monitor: unsupported output sample format {other:?}");
                return;
            }
        };

        let stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                log::warn!("meter monitor: build_output_stream failed: {e}");
                return;
            }
        };

        if let Err(e) = stream.play() {
            log::warn!("meter monitor: stream.play failed: {e}");
            return;
        }

        while !stop_engine.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(40));
        }
        drop(stream);
    }

    fn fill_f32(&self, data: &mut [f32], out_ch: usize, out_rate: u32) {
        data.fill(0.0);
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let in_rate_u = self.in_rate.load(Ordering::Relaxed).max(1);
        let in_rate = in_rate_u as f64;
        let out_rate_u = out_rate.max(1);
        let out_rate_f = out_rate_u as f64;
        let in_ch = (self.in_ch.load(Ordering::Relaxed).max(1) as usize).min(MAX_CHANNELS);
        let out_ch = out_ch.max(1);

        let n_out_frames = data.len() / out_ch;
        if n_out_frames == 0 {
            return;
        }

        let prefill_samples = (in_rate_u as usize) * in_ch * PREFILL_MS as usize / 1000;
        let refill_samples = (in_rate_u as usize) * in_ch * REFILL_MS as usize / 1000;

        let mut q = self.queue.lock();
        let mut st = self.drain.lock();

        // Prefill latch: don't start draining until the queue holds
        // `PREFILL_MS` of audio. This gives the capture thread enough lead
        // time to absorb the natural jitter between the two callbacks.
        if !st.primed {
            if q.len() >= prefill_samples.max(in_ch * 2) {
                st.primed = true;
                st.cursor = 0.0;
            } else {
                return;
            }
        }

        if in_rate_u == out_rate_u {
            // Matched rates: straight drain, one input frame per output frame,
            // no interpolation at all. This is the common case on Windows
            // because input and output typically share the system mix rate.
            for of in 0..n_out_frames {
                if q.len() < in_ch {
                    // Underrun: re-arm prefill and leave the rest silent.
                    st.primed = false;
                    st.cursor = 0.0;
                    return;
                }
                let mut frame = [0.0_f32; MAX_CHANNELS];
                for c in 0..in_ch {
                    frame[c] = q.pop_front().unwrap_or(0.0);
                }
                let (l, r) = stereo_downmix(&frame[..in_ch], in_ch);
                write_output_frame(data, of, out_ch, l, r);
            }
        } else {
            // Linear-interpolating resampler between `in_rate` and `out_rate`.
            // Only active when the host device is running at a different rate
            // than the capture source (e.g. 44.1 k input → 48 k output).
            let ratio = in_rate / out_rate_f;
            for of in 0..n_out_frames {
                // Advance cursor; pop whole input frames when it rolls over 1.
                st.cursor += ratio;
                while st.cursor >= 1.0 {
                    if q.len() >= in_ch {
                        for _ in 0..in_ch {
                            q.pop_front();
                        }
                        st.cursor -= 1.0;
                    } else {
                        // Underrun mid-resample: re-arm + silent tail.
                        st.primed = false;
                        st.cursor = 0.0;
                        return;
                    }
                }

                if q.len() < in_ch * 2 {
                    st.primed = false;
                    st.cursor = 0.0;
                    return;
                }

                let t = st.cursor.clamp(0.0, 1.0) as f32;
                let mut frame = [0.0_f32; MAX_CHANNELS];
                for c in 0..in_ch {
                    let a = *q.get(c).unwrap_or(&0.0);
                    let b = *q.get(c + in_ch).unwrap_or(&a);
                    frame[c] = a * (1.0 - t) + b * t;
                }
                let (l, r) = stereo_downmix(&frame[..in_ch], in_ch);
                write_output_frame(data, of, out_ch, l, r);
            }
        }

        // Drain-watermark: if we've consumed nearly everything the capture
        // thread had buffered, re-arm the prefill latch so the *next*
        // callback starts silent instead of stuttering on a near-empty
        // queue. This turns micro-underruns into one clean gap instead of
        // an ongoing series of clicks.
        if q.len() < refill_samples {
            st.primed = false;
            st.cursor = 0.0;
        }
    }
}

fn write_output_frame(data: &mut [f32], frame_idx: usize, out_ch: usize, l: f32, r: f32) {
    let base = frame_idx * out_ch;
    if out_ch == 1 {
        data[base] = 0.5 * (l + r);
    } else {
        data[base] = l;
        data[base + 1] = r;
        for c in 2..out_ch {
            data[base + c] = 0.0;
        }
    }
}

fn stereo_downmix(frame: &[f32], in_ch: usize) -> (f32, f32) {
    if in_ch == 0 || frame.is_empty() {
        return (0.0, 0.0);
    }
    if in_ch == 1 {
        let m = frame[0];
        return (m, m);
    }
    let mid = in_ch / 2;
    let left_n = mid.max(1);
    let right_n = (in_ch - mid).max(1);
    let mut sl = 0.0_f32;
    for i in 0..mid {
        sl += frame.get(i).copied().unwrap_or(0.0);
    }
    let mut sr = 0.0_f32;
    for i in mid..in_ch {
        sr += frame.get(i).copied().unwrap_or(0.0);
    }
    (sl / left_n as f32, sr / right_n as f32)
}
