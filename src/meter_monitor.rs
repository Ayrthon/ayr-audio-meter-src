//! Low-latency **listen** path: tap the same interleaved float samples the meters
//! use and play them through the default output device (cpal). Windows-only
//! shipping target, but the implementation is cross-platform for `cargo check`.

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

/// Cap queued float samples (~0.35 s at 48 kHz × 8 ch) so memory stays bounded if
/// the output device stalls.
const MAX_QUEUE_SAMPLES: usize = 48_000 * MAX_CHANNELS * 35 / 100;

#[derive(Default)]
struct DrainState {
    /// Fractional position into the next input frame pair at the deque head,
    /// advanced by `in_rate / out_rate` per output frame.
    cursor: f64,
}

impl DrainState {
    fn reset(&mut self) {
        self.cursor = 0.0;
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
            queue: Mutex::new(VecDeque::new()),
            drain: Mutex::new(DrainState::default()),
            output_thread: Mutex::new(None),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
        if !on {
            self.queue.lock().clear();
            self.drain.lock().reset();
        }
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
                    let nf = data.len() / out_ch;
                    let mut tmp = vec![0.0_f32; nf * out_ch];
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
                    let nf = data.len() / out_ch;
                    let mut tmp = vec![0.0_f32; nf * out_ch];
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
        let in_rate = self.in_rate.load(Ordering::Relaxed).max(1) as f64;
        let out_rate = out_rate.max(1) as f64;
        let ratio = in_rate / out_rate;
        let in_ch = self.in_ch.load(Ordering::Relaxed).max(1) as usize;
        let in_ch = in_ch.min(MAX_CHANNELS);
        let out_ch = out_ch.max(1);

        let n_out_frames = data.len() / out_ch;
        if n_out_frames == 0 {
            return;
        }

        let mut q = self.queue.lock();
        let mut st = self.drain.lock();

        let mut tmp = [0.0_f32; MAX_CHANNELS];
        let mut tmp2 = [0.0_f32; MAX_CHANNELS];

        for of in 0..n_out_frames {
            st.cursor += ratio;
            while st.cursor >= 1.0 {
                if q.len() >= in_ch {
                    for _ in 0..in_ch {
                        q.pop_front();
                    }
                    st.cursor -= 1.0;
                } else {
                    st.cursor = st.cursor.min(1.0);
                    break;
                }
            }

            let (l, r) = if q.len() >= in_ch * 2 {
                let t = st.cursor.clamp(0.0, 1.0) as f32;
                for c in 0..in_ch {
                    let a = *q.get(c).unwrap_or(&0.0);
                    let b = *q.get(c + in_ch).unwrap_or(&a);
                    tmp[c] = a * (1.0 - t) + b * t;
                }
                stereo_downmix(&tmp[..in_ch], in_ch)
            } else if q.len() >= in_ch {
                for c in 0..in_ch {
                    tmp2[c] = *q.get(c).unwrap_or(&0.0);
                }
                stereo_downmix(&tmp2[..in_ch], in_ch)
            } else {
                (0.0, 0.0)
            };

            let base = of * out_ch;
            if out_ch == 1 {
                data[base] = 0.5 * (l + r);
            } else {
                data[base] = l;
                if out_ch > 1 {
                    data[base + 1] = r;
                }
                for c in 2..out_ch {
                    data[base + c] = 0.0;
                }
            }
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
