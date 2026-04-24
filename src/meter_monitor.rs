//! Headphone-monitor routing — matches OBS's audio-monitor model.
//!
//! ## Windows (direct WASAPI, matches `audio-monitoring-win32.c`)
//!
//! On Windows the output path talks to WASAPI directly via the `wasapi`
//! crate — no cpal in the middle. We open an `IAudioClient` in shared
//! event-driven mode with:
//!
//! * `AUDCLNT_STREAMFLAGS_EVENTCALLBACK` — a kernel event fires when the
//!   engine needs more samples. We block on that event and fill the render
//!   buffer from our queue; no polling, no cpal callback scheduling layer
//!   adding its own jitter.
//! * `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY`
//!   — WASAPI itself resamples our stream to the device mix rate. This is
//!   exactly how OBS handles sample-rate conversion on Windows (it doesn't
//!   use libswresample for monitoring), and it lets us submit audio at the
//!   capture-side rate without any Rust-side resampler latency.
//!
//! End-to-end latency is one WASAPI shared-mode buffer period (~10 ms) plus
//! the ~20 ms prefill latch we keep for jitter absorption — ballpark the
//! same ~25–30 ms OBS's monitor shows.
//!
//! ## Non-Windows (cpal + rubato)
//!
//! Elsewhere we keep the cpal output stream + `rubato::SincFixedOut`
//! polyphase resampler. Same API surface, different backend.
//!
//! ## Shared UX
//!
//! * Target device is user-chosen via `set_target_device` (default = `None`
//!   i.e. disabled), matching OBS's "Settings → Audio → Monitoring Device".
//! * Self-monitor protection (`self_monitor_conflict`): we refuse to open a
//!   stream on a device that's also the capture source — same guard OBS's
//!   `OBS_SOURCE_DO_NOT_SELF_MONITOR` provides.
//! * Prefill latch (~20 ms of input) keeps the output muted until the queue
//!   is primed, so uncorrelated jitter between capture and render threads
//!   can't underrun on the first frames.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use parking_lot::{Mutex, RwLock};

use crate::audio::MAX_CHANNELS;

/// Cap queued float samples (~0.5 s at 48 kHz × 8 ch) so memory stays bounded
/// if the render side stalls or the target endpoint disappears.
const MAX_QUEUE_SAMPLES: usize = 48_000 * MAX_CHANNELS / 2;

/// How much input audio must accumulate before the output is allowed to
/// start draining. 20 ms is long enough to absorb jitter between the
/// capture and render audio threads without adding audible delay.
const PREFILL_MS: u32 = 20;

/// Drain-below watermark (ms of *input* audio). When the queue dips below
/// this we re-arm the prefill latch so a dropout becomes one brief gap
/// instead of a sustained stutter.
const REFILL_MS: u32 = 3;

/// Maximum playback-rate correction the drift compensator will apply, as a
/// ratio. 200 ppm covers typical crystal oscillator drift between two
/// consumer-grade audio interfaces (each ±50 ppm) plus headroom for NIC
/// hardware jitter on Ethernet/Wi-Fi. Capping is essential: if an
/// underrun-triggered re-prime tried to "catch up" at large ratios the
/// audio would pitch-shift audibly.
const DRIFT_MAX_RATIO_DELTA: f64 = 200e-6;

/// Seconds of per-call slew applied by the PI controller. Kept small so
/// transient dropouts (a single late RTP packet, a Wi-Fi burst) don't
/// perturb the rate — only sustained depth bias does.
const DRIFT_KP: f64 = 1.5e-6; // per-frame-of-error
const DRIFT_KI: f64 = 8.0e-9; // per-frame-of-error · per-ms-elapsed

struct DrainState {
    /// `false` until the queue is primed with at least `PREFILL_MS` of audio.
    primed: bool,
    /// Most recently consumed interleaved frame (channel-major, up to
    /// MAX_CHANNELS). Held across drain calls so the fractional read
    /// position can interpolate between `cur_frame` and `next_frame`.
    cur_frame: [f32; MAX_CHANNELS],
    next_frame: [f32; MAX_CHANNELS],
    /// Number of input channels currently held in `cur_frame`/`next_frame`.
    /// Zero means neither slot is valid yet (fresh session or post-reset).
    frame_width: usize,
    /// Fractional read position between `cur_frame` (0.0) and
    /// `next_frame` (1.0). Advances by `play_rate` per output frame.
    frac: f64,
    /// Current long-term playback rate multiplier. `1.0` = nominal;
    /// `1.0 + δ` where |δ| ≤ `DRIFT_MAX_RATIO_DELTA` = slow drift
    /// correction. Converges toward whatever ratio keeps the queue depth
    /// centred on `target_depth_frames`.
    play_rate: f64,
    /// Integrator state for the PI controller.
    integral: f64,
    /// Wall-clock timestamp of the previous drain call — used to convert
    /// the integral term from "per call" to "per millisecond" so the
    /// controller behaves identically under different audio callback
    /// periods.
    last_tick: Option<std::time::Instant>,
    /// Periodic drift log throttle.
    last_log: Option<std::time::Instant>,
}

impl Default for DrainState {
    fn default() -> Self {
        Self {
            primed: false,
            cur_frame: [0.0; MAX_CHANNELS],
            next_frame: [0.0; MAX_CHANNELS],
            frame_width: 0,
            frac: 0.0,
            play_rate: 1.0,
            integral: 0.0,
            last_tick: None,
            last_log: None,
        }
    }
}

impl DrainState {
    fn reset(&mut self) {
        *self = DrainState::default();
    }
}

/// Shared between the UI (toggle + device selector), the capture callbacks
/// (push), and the output thread (drain + render).
pub struct MeterMonitorBus {
    /// User's on/off toggle (headphone button).
    enabled: AtomicBool,

    /// Most recent source sample-rate / channel count, set by the capture
    /// thread once the stream format is known.
    in_rate: AtomicU32,
    in_ch: AtomicU32,

    /// Friendly name of the render endpoint the user wants to monitor on.
    /// `None` = monitoring is disabled (matches OBS's default).
    target_device: RwLock<Option<String>>,

    /// Friendly name of the device currently being captured (set by the
    /// audio engine just before it starts a capture stream). Used by
    /// [`Self::self_monitor_conflict`] to detect loopback-feedback setups.
    source_device: RwLock<Option<String>>,

    /// Latched after an output-stream open fails or self-monitor protection
    /// engages. Surfaced to the UI so the listen button can explain why it
    /// did nothing instead of silently ignoring the click.
    last_error: RwLock<Option<String>>,

    /// Interleaved f32 source samples, `[ch0, ch1, …, chN-1]` per frame.
    queue: Mutex<VecDeque<f32>>,
    drain: Mutex<DrainState>,

    /// Monotonically-incrementing generation. Bumped whenever the target
    /// device changes so the output thread can tear down its current stream
    /// and reopen against the new endpoint on its next poll.
    device_generation: AtomicU32,

    output_thread: Mutex<Option<JoinHandle<()>>>,
}

impl MeterMonitorBus {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            enabled: AtomicBool::new(false),
            in_rate: AtomicU32::new(48_000),
            in_ch: AtomicU32::new(2),
            target_device: RwLock::new(None),
            source_device: RwLock::new(None),
            last_error: RwLock::new(None),
            queue: Mutex::new(VecDeque::with_capacity(MAX_QUEUE_SAMPLES)),
            drain: Mutex::new(DrainState::default()),
            device_generation: AtomicU32::new(0),
            output_thread: Mutex::new(None),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
        self.queue.lock().clear();
        self.drain.lock().reset();
        if on {
            *self.last_error.write() = None;
        }
    }

    /// Update the chosen monitor render endpoint (friendly name) or `None`
    /// to disable monitoring entirely. Bumps a generation counter so the
    /// output thread rebuilds its render stream on the new device.
    pub fn set_target_device(&self, name: Option<String>) {
        let same = {
            let cur = self.target_device.read();
            cur.as_deref() == name.as_deref()
        };
        if !same {
            *self.target_device.write() = name;
            self.device_generation.fetch_add(1, Ordering::Relaxed);
            self.queue.lock().clear();
            self.drain.lock().reset();
            *self.last_error.write() = None;
        }
    }

    pub fn target_device(&self) -> Option<String> {
        self.target_device.read().clone()
    }

    pub fn source_device(&self) -> Option<String> {
        self.source_device.read().clone()
    }

    /// `true` when the user's monitor device equals the currently-captured
    /// device (typical loopback feedback scenario). Mirrors OBS's
    /// `OBS_SOURCE_DO_NOT_SELF_MONITOR` check.
    pub fn self_monitor_conflict(&self) -> bool {
        let t = self.target_device.read();
        let s = self.source_device.read();
        match (t.as_deref(), s.as_deref()) {
            (Some(a), Some(b)) => a.eq_ignore_ascii_case(b),
            _ => false,
        }
    }

    pub fn last_error(&self) -> Option<String> {
        self.last_error.read().clone()
    }

    /// Called from the audio engine just before it starts pumping a capture
    /// stream. Records the source format + device name, and spawns the
    /// (single, long-lived) output-manager thread if needed.
    pub fn begin_capture(
        self: &Arc<Self>,
        in_rate: u32,
        in_ch: usize,
        source_device_name: Option<String>,
        stop: Arc<AtomicBool>,
    ) {
        self.in_rate.store(in_rate.max(1), Ordering::Relaxed);
        self.in_ch.store(
            (in_ch.max(1).min(MAX_CHANNELS)) as u32,
            Ordering::Relaxed,
        );
        *self.source_device.write() = source_device_name;
        self.queue.lock().clear();
        self.drain.lock().reset();
        // Force a re-check of conflict state on the next output-manager
        // iteration so newly-captured devices immediately invalidate any
        // in-flight stream that now conflicts.
        self.device_generation.fetch_add(1, Ordering::Relaxed);

        let mut slot = self.output_thread.lock();
        if slot.is_some() {
            return;
        }
        let this = Arc::clone(self);
        *slot = Some(thread::spawn(move || this.output_main(stop)));
    }

    /// Join the output-manager thread (no-op if it never started). Called by
    /// the audio engine on shutdown.
    pub fn end_capture(&self) {
        self.enabled.store(false, Ordering::Relaxed);
        self.queue.lock().clear();
        self.drain.lock().reset();
        *self.source_device.write() = None;
        self.device_generation.fetch_add(1, Ordering::Relaxed);
        if let Some(h) = self.output_thread.lock().take() {
            let _ = h.join();
        }
    }

    /// Push one interleaved block of float samples (`channels` wide). Safe
    /// to call with `enabled == false` — returns immediately so capture-only
    /// users pay no cost.
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
        // Bound memory use by dropping whole frames if the render side
        // stalls. A few lost frames beats unbounded growth.
        while q.len() > MAX_QUEUE_SAMPLES {
            for _ in 0..ch {
                q.pop_front();
            }
        }
    }

    /// Drain up to `n_frames` stereo-downmixed frames into `(out_l, out_r)`.
    /// Returns number of frames actually filled; the caller pads the tail
    /// with silence. Handles the prefill latch so the output unmutes only
    /// once the queue is primed.
    ///
    /// Both WASAPI (Windows) and cpal (non-Windows) output backends funnel
    /// through this helper so the jitter-absorb / self-monitor / disabled
    /// logic stays in one place.
    fn drain_stereo(&self, out_l: &mut [f32], out_r: &mut [f32]) -> usize {
        debug_assert_eq!(out_l.len(), out_r.len());
        let n_frames = out_l.len();
        out_l.fill(0.0);
        out_r.fill(0.0);
        if !self.enabled.load(Ordering::Relaxed) || n_frames == 0 {
            return 0;
        }

        let in_rate_u = self.in_rate.load(Ordering::Relaxed).max(1);
        let in_ch = (self.in_ch.load(Ordering::Relaxed).max(1) as usize).min(MAX_CHANNELS);
        // Prefill / refill thresholds in *samples* (not frames) — the
        // existing conventions elsewhere in this file are sample-indexed.
        let prefill_samples = (in_rate_u as usize) * in_ch * PREFILL_MS as usize / 1000;
        let refill_samples = (in_rate_u as usize) * in_ch * REFILL_MS as usize / 1000;
        // Target queue depth the PI controller tries to keep us at: the
        // midpoint between refill and prefill in *frames*, which gives the
        // drift compensator equal headroom to slow down or speed up.
        let target_depth_frames =
            ((prefill_samples + refill_samples) / (2 * in_ch.max(1))).max(1) as f64;

        let mut q = self.queue.lock();
        let mut st = self.drain.lock();

        if !st.primed {
            // Need at least two input frames queued so we have both
            // `cur_frame` and `next_frame` before interpolation starts.
            if q.len() >= prefill_samples.max(in_ch * 2) {
                // Pop the first two frames into the interpolation slots.
                pop_frame_into(&mut q, &mut st.cur_frame, in_ch);
                pop_frame_into(&mut q, &mut st.next_frame, in_ch);
                st.frame_width = in_ch;
                st.frac = 0.0;
                st.primed = true;
                // Keep the converged rate across prime/reprime cycles:
                // if we've been tracking the sender's actual rate at
                // +50 ppm for 10 minutes, a single 3 ms underrun
                // shouldn't throw that away. Only zero the integral so
                // the controller doesn't ramp from a stale windup state.
                st.integral = 0.0;
            } else {
                return 0;
            }
        }

        // Update the PI controller exactly once per drain call. `error`
        // is in *input frames*: positive = queue deeper than target
        // (producer fast, we must speed up playback), negative =
        // shallower than target (producer slow, slow down playback).
        let now = std::time::Instant::now();
        let dt_ms = match st.last_tick {
            Some(prev) => now.saturating_duration_since(prev).as_secs_f64() * 1000.0,
            None => 0.0,
        };
        st.last_tick = Some(now);
        let depth_frames = (q.len() / in_ch.max(1)) as f64;
        let error = depth_frames - target_depth_frames;
        st.integral += error * dt_ms;
        // Mild integral windup clamp (20 seconds of worst-case accumulation).
        let int_clamp = target_depth_frames * 20_000.0;
        if st.integral > int_clamp {
            st.integral = int_clamp;
        } else if st.integral < -int_clamp {
            st.integral = -int_clamp;
        }
        let raw_delta = DRIFT_KP * error + DRIFT_KI * st.integral;
        let delta = raw_delta.clamp(-DRIFT_MAX_RATIO_DELTA, DRIFT_MAX_RATIO_DELTA);
        st.play_rate = 1.0 + delta;

        // Linear-interpolated drain. We advance `frac` by `play_rate`
        // per output frame; whenever it crosses an integer boundary we
        // slide next_frame → cur_frame and pop a fresh next_frame from
        // the queue. This gives a continuous, fractional resampling
        // across the whole drain call without any block boundaries.
        let mut filled = 0usize;
        while filled < n_frames {
            let cur = &st.cur_frame;
            let nxt = &st.next_frame;
            let f = st.frac as f32;
            let mut mixed = [0.0_f32; MAX_CHANNELS];
            for c in 0..in_ch {
                mixed[c] = cur[c] + (nxt[c] - cur[c]) * f;
            }
            let (l, r) = stereo_downmix(&mixed[..in_ch], in_ch);
            out_l[filled] = l;
            out_r[filled] = r;
            filled += 1;

            st.frac += st.play_rate;
            // Consume as many whole input frames as the ratio demands.
            // Ratios stay within ±200 ppm so this loop typically runs 0
            // or 1 times per output frame — 2 only under extreme
            // correction.
            while st.frac >= 1.0 {
                if q.len() < in_ch {
                    // Queue dry mid-frame — fall back to holding
                    // `next_frame` so we output a flat hold instead of
                    // silence until the next push. Re-arm the prefill
                    // latch so the next dropout gets a proper refill.
                    st.primed = false;
                    break;
                }
                st.cur_frame = st.next_frame;
                pop_frame_into(&mut q, &mut st.next_frame, in_ch);
                st.frac -= 1.0;
            }
            if !st.primed {
                break;
            }
        }

        if q.len() < refill_samples {
            // Dropping into re-prime territory. Don't clear the state
            // mid-stream; just flip the latch so the next call waits for
            // enough samples before it starts draining again.
            st.primed = false;
        }

        // Periodic drift log so long sessions have a breadcrumb trail.
        let should_log = match st.last_log {
            Some(prev) => now.saturating_duration_since(prev) >= std::time::Duration::from_secs(30),
            None => true,
        };
        if should_log {
            st.last_log = Some(now);
            let ppm = delta * 1.0e6;
            log::debug!(
                "monitor drift: depth={depth_frames:.0}f target={target_depth_frames:.0}f rate={:+.1}ppm",
                ppm
            );
        }

        filled
    }

    fn output_main(self: Arc<Self>, stop_engine: Arc<AtomicBool>) {
        let mut current: Option<PlatformStream> = None;
        let mut last_generation = self.device_generation.load(Ordering::Relaxed);
        let mut last_enabled = self.enabled.load(Ordering::Relaxed);

        while !stop_engine.load(Ordering::Relaxed) {
            let enabled = self.enabled.load(Ordering::Relaxed);
            let generation = self.device_generation.load(Ordering::Relaxed);
            let target = self.target_device.read().clone();
            let source = self.source_device.read().clone();
            let in_rate_now = self.in_rate.load(Ordering::Relaxed);
            let in_ch_now = (self.in_ch.load(Ordering::Relaxed) as usize)
                .max(1)
                .min(MAX_CHANNELS);

            let conflict = match (target.as_deref(), source.as_deref()) {
                (Some(a), Some(b)) => a.eq_ignore_ascii_case(b),
                _ => false,
            };
            let want_stream = enabled && target.is_some() && !conflict;

            let need_rebuild = match &current {
                Some(c) => {
                    generation != last_generation
                        || enabled != last_enabled
                        || target.as_deref() != Some(c.device_name.as_str())
                        || in_rate_now != c.in_rate_when_opened
                        || in_ch_now != c.in_ch_when_opened
                        || c.is_dead()
                }
                None => want_stream,
            };

            if need_rebuild {
                current = None; // drop existing stream before opening a new one
                self.queue.lock().clear();
                self.drain.lock().reset();

                if !want_stream {
                    if enabled {
                        if conflict {
                            *self.last_error.write() = Some(format!(
                                "Monitor device matches the capture source ({}). Pick a different output to avoid a feedback loop.",
                                target.unwrap_or_default()
                            ));
                        } else if target.is_none() {
                            *self.last_error.write() =
                                Some("Pick a monitor output device first.".into());
                        }
                    } else {
                        *self.last_error.write() = None;
                    }
                } else if let Some(target_name) = target.as_deref() {
                    match PlatformStream::open(target_name, Arc::clone(&self)) {
                        Ok(os) => {
                            *self.last_error.write() = None;
                            current = Some(os);
                        }
                        Err(e) => {
                            *self.last_error.write() = Some(e);
                        }
                    }
                }

                last_generation = generation;
                last_enabled = enabled;
            }

            thread::sleep(Duration::from_millis(40));
        }

        drop(current);
    }
}

/// Pop one interleaved frame (`in_ch` samples) off the queue into `dst`.
/// Any missing samples are filled with 0 so `dst` is always valid after
/// the call. Used by `drain_stereo`'s fractional interpolator to slide
/// new input frames into `cur_frame` / `next_frame`.
fn pop_frame_into(q: &mut VecDeque<f32>, dst: &mut [f32; MAX_CHANNELS], in_ch: usize) {
    for c in 0..in_ch.min(MAX_CHANNELS) {
        dst[c] = q.pop_front().unwrap_or(0.0);
    }
    for c in in_ch..MAX_CHANNELS {
        dst[c] = 0.0;
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

// ------------------------------------------------------------------
// Platform-specific output backends
// ------------------------------------------------------------------

/// Common metadata tracked by `output_main` regardless of which backend owns
/// the stream. `inner` holds the OS-specific handle.
struct PlatformStream {
    inner: PlatformInner,
    device_name: String,
    in_rate_when_opened: u32,
    in_ch_when_opened: usize,
}

impl PlatformStream {
    fn open(target_name: &str, bus: Arc<MeterMonitorBus>) -> Result<Self, String> {
        let in_rate = bus.in_rate.load(Ordering::Relaxed).max(1);
        let in_ch = (bus.in_ch.load(Ordering::Relaxed) as usize)
            .max(1)
            .min(MAX_CHANNELS);
        let inner = PlatformInner::open(target_name, Arc::clone(&bus))?;
        Ok(Self {
            inner,
            device_name: target_name.to_string(),
            in_rate_when_opened: in_rate,
            in_ch_when_opened: in_ch,
        })
    }

    fn is_dead(&self) -> bool {
        self.inner.is_dead()
    }
}

// ------------- Windows: direct WASAPI ---------------------------------

#[cfg(windows)]
mod wasapi_out {
    use super::MeterMonitorBus;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::thread::{self, JoinHandle};
    use std::time::Duration;
    use wasapi::{
        Direction, DeviceEnumerator, SampleType, StreamMode, WaveFormat, initialize_mta,
    };

    /// Owns the render thread + its stop flag. Dropping joins the thread.
    pub struct RenderStream {
        thread: Option<JoinHandle<()>>,
        stop: Arc<AtomicBool>,
        died: Arc<AtomicBool>,
    }

    impl RenderStream {
        pub fn is_dead(&self) -> bool {
            self.died.load(Ordering::Relaxed)
        }
    }

    impl Drop for RenderStream {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(h) = self.thread.take() {
                let _ = h.join();
            }
        }
    }

    /// Open the render stream. Blocks until WASAPI setup succeeds (or fails)
    /// in the render thread, since the WASAPI COM objects aren't `Send` and
    /// have to live on the thread that created them.
    pub fn open(target_name: &str, bus: Arc<MeterMonitorBus>) -> Result<RenderStream, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let died = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::sync_channel::<Result<(), String>>(1);

        let target = target_name.to_string();
        let bus_th = Arc::clone(&bus);
        let stop_th = Arc::clone(&stop);
        let died_th = Arc::clone(&died);
        let tx_th = tx.clone();

        let thread = thread::Builder::new()
            .name("meter-monitor-wasapi".into())
            .spawn(move || {
                if let Err(e) = run(&target, bus_th, stop_th, tx_th) {
                    // Final setup failure: surface to caller and mark dead so
                    // `output_main` rebuilds on next poll.
                    let _ = tx.send(Err(e));
                    died_th.store(true, Ordering::Relaxed);
                } else {
                    died_th.store(true, Ordering::Relaxed);
                }
            })
            .map_err(|e| format!("spawn monitor thread: {e}"))?;

        match rx.recv_timeout(Duration::from_secs(3)) {
            Ok(Ok(())) => Ok(RenderStream {
                thread: Some(thread),
                stop,
                died,
            }),
            Ok(Err(e)) => {
                // Setup failed; thread is about to exit.
                let _ = thread.join();
                Err(e)
            }
            Err(_) => {
                stop.store(true, Ordering::Relaxed);
                let _ = thread.join();
                Err("WASAPI monitor init timed out".into())
            }
        }
    }

    fn run(
        target_name: &str,
        bus: Arc<MeterMonitorBus>,
        stop: Arc<AtomicBool>,
        ready_tx: mpsc::SyncSender<Result<(), String>>,
    ) -> Result<(), String> {
        let _ = initialize_mta().ok();

        let enumerator = DeviceEnumerator::new()
            .map_err(|e| format!("DeviceEnumerator::new: {e}"))?;
        let collection = enumerator
            .get_device_collection(&Direction::Render)
            .map_err(|e| format!("get_device_collection: {e}"))?;
        let count = collection
            .get_nbr_devices()
            .map_err(|e| format!("get_nbr_devices: {e}"))?;
        let mut chosen = None;
        for i in 0..count {
            let dev = match collection.get_device_at_index(i) {
                Ok(d) => d,
                Err(_) => continue,
            };
            if dev
                .get_friendlyname()
                .map(|n| n == target_name)
                .unwrap_or(false)
            {
                chosen = Some(dev);
                break;
            }
        }
        let device = chosen
            .ok_or_else(|| format!("monitor device '{target_name}' not found"))?;
        let mut audio_client = device
            .get_iaudioclient()
            .map_err(|e| format!("get_iaudioclient: {e}"))?;

        // Always render as stereo IEEE float at the source rate. WASAPI's
        // AUTOCONVERTPCM + SRC_DEFAULT_QUALITY flags (set below via
        // StreamMode::EventsShared { autoconvert: true }) let the engine
        // resample/remix to the device mix format in the kernel. Same path
        // OBS's monitor uses on Windows.
        let in_rate = bus.in_rate.load(Ordering::Relaxed).max(1);
        let ch_out: usize = 2;
        let desired = WaveFormat::new(
            32,
            32,
            &SampleType::Float,
            in_rate as usize,
            ch_out,
            None,
        );

        let (_def_period, min_period) = audio_client
            .get_device_period()
            .map_err(|e| format!("get_device_period: {e}"))?;
        // Use the device's minimum period (typically ~10 ms in shared mode).
        // Smaller buffers = lower latency, larger buffers = safer against
        // render-thread jitter. `min_period` is the latency OBS shows for
        // shared-mode monitors.
        let buffer_duration_hns = min_period.max(100_000); // >= 10 ms

        let mode = StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns,
        };
        audio_client
            .initialize_client(&desired, &Direction::Render, &mode)
            .map_err(|e| format!("initialize_client (render): {e}"))?;

        let h_event = audio_client
            .set_get_eventhandle()
            .map_err(|e| format!("set_get_eventhandle: {e}"))?;
        let render_client = audio_client
            .get_audiorenderclient()
            .map_err(|e| format!("get_audiorenderclient: {e}"))?;
        let buffer_frames = audio_client
            .get_bufferframecount()
            .map_err(|e| format!("get_bufferframecount: {e}"))?
            as usize;

        audio_client
            .start_stream()
            .map_err(|e| format!("start_stream (render): {e}"))?;

        // Prime the WASAPI buffer with silence so the device doesn't underrun
        // before our queue is primed. The first real audio arrives once the
        // capture side has pushed ~20 ms worth of samples.
        if let Ok(avail) = audio_client.get_available_space_in_frames() {
            let bytes = avail as usize * ch_out * 4;
            let silence = vec![0u8; bytes];
            let _ = render_client.write_to_device(avail as usize, &silence, None);
        }

        // Only *now* tell the caller setup succeeded — any error above is
        // returned to the caller via the `?` propagation path in `open()`.
        let _ = ready_tx.send(Ok(()));

        log::info!(
            "meter monitor: WASAPI render opened on '{target_name}' \
             (buffer={buffer_frames}f, in_rate={in_rate}, ch_out={ch_out})"
        );

        let bytes_per_frame = ch_out * 4;
        let mut scratch_bytes: Vec<u8> =
            Vec::with_capacity(buffer_frames * bytes_per_frame);
        let mut scratch_l: Vec<f32> = Vec::with_capacity(buffer_frames);
        let mut scratch_r: Vec<f32> = Vec::with_capacity(buffer_frames);

        // Render loop. `wait_for_event` blocks until WASAPI signals it needs
        // more samples — typically once per buffer period. Event-driven
        // rendering is the shared-mode path OBS uses for sub-frame latency.
        while !stop.load(Ordering::Relaxed) {
            if let Err(e) = h_event.wait_for_event(200) {
                // 200 ms timeout: long enough to avoid spinning, short
                // enough that a stop request is honored promptly.
                log::trace!("meter monitor: wait_for_event: {e}");
                continue;
            }
            if stop.load(Ordering::Relaxed) {
                break;
            }

            let avail = match audio_client.get_available_space_in_frames() {
                Ok(a) => a as usize,
                Err(e) => {
                    log::warn!("meter monitor: get_available_space_in_frames: {e}");
                    break;
                }
            };
            if avail == 0 {
                continue;
            }

            // If the bus is disabled, write silence (cheaper than tearing
            // down and reopening when the user toggles listen off+on, and
            // avoids a click at the device).
            let want_audio = bus.enabled.load(Ordering::Relaxed);

            scratch_l.clear();
            scratch_l.resize(avail, 0.0);
            scratch_r.clear();
            scratch_r.resize(avail, 0.0);
            let filled = if want_audio {
                bus.drain_stereo(&mut scratch_l, &mut scratch_r)
            } else {
                0
            };
            let _ = filled; // `drain_stereo` already zero-pads the tail.

            scratch_bytes.clear();
            scratch_bytes.reserve(avail * bytes_per_frame);
            for i in 0..avail {
                scratch_bytes.extend_from_slice(&scratch_l[i].to_le_bytes());
                scratch_bytes.extend_from_slice(&scratch_r[i].to_le_bytes());
            }

            if let Err(e) = render_client.write_to_device(avail, &scratch_bytes, None) {
                log::warn!("meter monitor: write_to_device: {e}");
                break;
            }
        }

        let _ = audio_client.stop_stream();
        Ok(())
    }
}

#[cfg(windows)]
struct PlatformInner {
    stream: wasapi_out::RenderStream,
}

#[cfg(windows)]
impl PlatformInner {
    fn open(target_name: &str, bus: Arc<MeterMonitorBus>) -> Result<Self, String> {
        Ok(Self {
            stream: wasapi_out::open(target_name, bus)?,
        })
    }

    fn is_dead(&self) -> bool {
        self.stream.is_dead()
    }
}

// ------------- Non-Windows: cpal + rubato ----------------------------

#[cfg(not(windows))]
mod cpal_out {
    use super::MeterMonitorBus;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use cpal::SampleFormat;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{Stream, StreamConfig};
    use parking_lot::Mutex;
    use rubato::{
        Resampler, SincFixedOut, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };

    const RESAMPLER_OUT_CHUNK: usize = 128;

    pub struct CpalStream {
        _stream: Stream,
    }

    impl CpalStream {
        pub fn is_dead(&self) -> bool {
            false
        }
    }

    struct MonitorResampler {
        resampler: SincFixedOut<f32>,
        input_l: Vec<f32>,
        input_r: Vec<f32>,
        output_buffers: Vec<Vec<f32>>,
        residual_l: Vec<f32>,
        residual_r: Vec<f32>,
        residual_read: usize,
        residual_len: usize,
    }

    impl MonitorResampler {
        fn new(in_rate: u32, out_rate: u32) -> Option<Self> {
            let ratio = out_rate as f64 / in_rate as f64;
            let params = SincInterpolationParameters {
                sinc_len: 128,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 128,
                window: WindowFunction::BlackmanHarris2,
            };
            let resampler =
                SincFixedOut::<f32>::new(ratio, 2.0, params, RESAMPLER_OUT_CHUNK, 2).ok()?;
            let output_buffers = resampler.output_buffer_allocate(true);
            let max_in = resampler.input_frames_max();
            Some(Self {
                resampler,
                input_l: Vec::with_capacity(max_in),
                input_r: Vec::with_capacity(max_in),
                output_buffers,
                residual_l: Vec::with_capacity(RESAMPLER_OUT_CHUNK),
                residual_r: Vec::with_capacity(RESAMPLER_OUT_CHUNK),
                residual_read: 0,
                residual_len: 0,
            })
        }
    }

    pub fn open(target_name: &str, bus: Arc<MeterMonitorBus>) -> Result<CpalStream, String> {
        let host = cpal::default_host();
        let device = host
            .output_devices()
            .map_err(|e| format!("cpal output_devices failed: {e}"))?
            .find(|d| d.name().map(|n| n == target_name).unwrap_or(false))
            .ok_or_else(|| {
                format!("Monitor device '{target_name}' not found.")
            })?;
        let default_conf = device
            .default_output_config()
            .map_err(|e| format!("no default output config for '{target_name}': {e}"))?;
        let sample_format = default_conf.sample_format();
        let config: StreamConfig = default_conf.config().into();
        let out_ch = config.channels.max(1) as usize;
        let out_rate: u32 = config.sample_rate;

        let resampler: Arc<Mutex<Option<MonitorResampler>>> = {
            let in_rate = bus.in_rate.load(Ordering::Relaxed).max(1);
            Arc::new(Mutex::new(if in_rate == out_rate {
                None
            } else {
                MonitorResampler::new(in_rate, out_rate)
            }))
        };

        let err_cb = |e| log::error!("meter monitor cpal error: {e}");
        let bus_cb = Arc::clone(&bus);
        let rs_cb = Arc::clone(&resampler);
        let stream = match sample_format {
            SampleFormat::F32 => device.build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    let mut rs = rs_cb.lock();
                    fill(&bus_cb, data, out_ch, out_rate, &mut rs);
                },
                err_cb,
                None,
            ),
            SampleFormat::I16 => {
                let tmp = Arc::new(Mutex::new(Vec::<f32>::with_capacity(8192)));
                let tmp_cb = Arc::clone(&tmp);
                device.build_output_stream(
                    &config,
                    move |data: &mut [i16], _| {
                        let mut tmp = tmp_cb.lock();
                        tmp.clear();
                        tmp.resize(data.len(), 0.0);
                        let mut rs = rs_cb.lock();
                        fill(&bus_cb, &mut tmp, out_ch, out_rate, &mut rs);
                        for (d, &s) in data.iter_mut().zip(tmp.iter()) {
                            *d = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        }
                    },
                    err_cb,
                    None,
                )
            }
            SampleFormat::U16 => {
                let tmp = Arc::new(Mutex::new(Vec::<f32>::with_capacity(8192)));
                let tmp_cb = Arc::clone(&tmp);
                device.build_output_stream(
                    &config,
                    move |data: &mut [u16], _| {
                        let mut tmp = tmp_cb.lock();
                        tmp.clear();
                        tmp.resize(data.len(), 0.0);
                        let mut rs = rs_cb.lock();
                        fill(&bus_cb, &mut tmp, out_ch, out_rate, &mut rs);
                        for (d, &s) in data.iter_mut().zip(tmp.iter()) {
                            let v = ((s.clamp(-1.0, 1.0) * 32767.0) as i32 + 32768)
                                .clamp(0, 65535);
                            *d = v as u16;
                        }
                    },
                    err_cb,
                    None,
                )
            }
            other => {
                return Err(format!(
                    "Output device '{target_name}' uses unsupported sample format {other:?}."
                ));
            }
        }
        .map_err(|e| format!("build_output_stream on '{target_name}' failed: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("stream.play on '{target_name}' failed: {e}"))?;

        Ok(CpalStream { _stream: stream })
    }

    fn fill(
        bus: &Arc<MeterMonitorBus>,
        data: &mut [f32],
        out_ch: usize,
        out_rate: u32,
        rs_slot: &mut Option<MonitorResampler>,
    ) {
        data.fill(0.0);
        if !bus.enabled.load(Ordering::Relaxed) {
            return;
        }
        let in_rate = bus.in_rate.load(Ordering::Relaxed).max(1);
        let n_out_frames = data.len() / out_ch.max(1);
        if n_out_frames == 0 {
            return;
        }

        if in_rate == out_rate || rs_slot.is_none() {
            // Matched rates (or resampler construction failed): just drain
            // stereo and splat across the output channels.
            let mut l = vec![0.0f32; n_out_frames];
            let mut r = vec![0.0f32; n_out_frames];
            bus.drain_stereo(&mut l, &mut r);
            for of in 0..n_out_frames {
                let base = of * out_ch;
                if out_ch == 1 {
                    data[base] = 0.5 * (l[of] + r[of]);
                } else {
                    data[base] = l[of];
                    data[base + 1] = r[of];
                    for c in 2..out_ch {
                        data[base + c] = 0.0;
                    }
                }
            }
            return;
        }

        // Resampler active: feed stereo-downmixed source audio into rubato
        // and stream its output through a residual buffer across callbacks.
        let rs = rs_slot.as_mut().unwrap();
        let mut written = 0;
        while written < n_out_frames {
            while written < n_out_frames && rs.residual_read < rs.residual_len {
                let base = written * out_ch;
                let (l, r) = (
                    rs.residual_l[rs.residual_read],
                    rs.residual_r[rs.residual_read],
                );
                if out_ch == 1 {
                    data[base] = 0.5 * (l + r);
                } else {
                    data[base] = l;
                    data[base + 1] = r;
                    for c in 2..out_ch {
                        data[base + c] = 0.0;
                    }
                }
                rs.residual_read += 1;
                written += 1;
            }
            if written >= n_out_frames {
                break;
            }
            let need = rs.resampler.input_frames_next();
            rs.input_l.clear();
            rs.input_l.resize(need, 0.0);
            rs.input_r.clear();
            rs.input_r.resize(need, 0.0);
            let got = bus.drain_stereo(&mut rs.input_l, &mut rs.input_r);
            if got < need {
                // Underrun: pad remaining output with silence (already 0.0)
                // and let the prefill latch re-arm next callback.
                return;
            }
            let inputs: [&[f32]; 2] = [&rs.input_l, &rs.input_r];
            let (_nin, nout) = match rs.resampler.process_into_buffer(
                &inputs,
                &mut rs.output_buffers,
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("monitor resampler: {e}");
                    return;
                }
            };
            rs.residual_l.clear();
            rs.residual_l.extend_from_slice(&rs.output_buffers[0][..nout]);
            rs.residual_r.clear();
            rs.residual_r.extend_from_slice(&rs.output_buffers[1][..nout]);
            rs.residual_read = 0;
            rs.residual_len = nout;
        }
    }
}

#[cfg(not(windows))]
struct PlatformInner {
    stream: cpal_out::CpalStream,
}

#[cfg(not(windows))]
impl PlatformInner {
    fn open(target_name: &str, bus: Arc<MeterMonitorBus>) -> Result<Self, String> {
        Ok(Self {
            stream: cpal_out::open(target_name, bus)?,
        })
    }

    fn is_dead(&self) -> bool {
        self.stream.is_dead()
    }
}
