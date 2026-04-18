//! Real-time spectrum analyzer.
//!
//! * The audio thread writes mono-summed samples into a shared ring
//!   buffer (`SpectrumRing`) — cheap, allocation-free, no analysis.
//! * The UI thread runs the FFT (`SpectrumAnalyzer`) from the latest
//!   N samples each frame, applies a Hann window, then **power-domain** EMA of \|X\|² **per bin**
//!   with **frequency-dependent τ** (slower in the bass for smooth motion, faster in the treble
//!   for sharper peaks).
//!
//! FabFilter Pro-Q’s analyzer is proprietary; this stack (Hann STFT + split-band power EMA +
//! Gaussian width in the painter) is a **close functional analogue** — not a byte-identical clone.

use std::sync::Arc;

use parking_lot::RwLock;
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;

/// FFT size — good balance of low-end resolution (48k / 4096 ≈ 11.7 Hz)
/// and CPU cost.  Power of two for realfft.
/// FFT length (samples). At 48 kHz this is a 170 ms window, giving 5.86 Hz bin resolution —
/// enough to distinguish ~6 Hz infrasonic rumble, the 20 Hz fundamental, and 60 Hz mains hum
/// in separate bins. 4096 (85 ms / 11.7 Hz bins) was too coarse below 20 Hz; 16384 (341 ms)
/// would start to visibly smear percussive transients because the FFT window integrates more
/// history. 8192 is the sweet spot for full-range music.
pub const FFT_SIZE: usize = 8192;

/// Ring length — 2× FFT size so the UI thread can always pull a
/// contiguous tail regardless of where the audio thread last wrote.
pub const RING_LEN: usize = FFT_SIZE * 2;

/// Power EMA τ (seconds) at low FFT bins — smooth, slow crossovers.
const POWER_EMA_TAU_BASS_SEC: f32 = 0.22;
/// Power EMA τ (seconds) at high bins — snappier, more “teeth” on transients.
const POWER_EMA_TAU_TREBLE_SEC: f32 = 0.05;
const POWER_EMA_BLEND_F_LO: f32 = 60.0;
const POWER_EMA_BLEND_F_HI: f32 = 5_000.0;

#[inline]
fn power_ema_tau_hz(f_hz: f32) -> f32 {
    let f = f_hz.max(1.0);
    if f <= POWER_EMA_BLEND_F_LO {
        return POWER_EMA_TAU_BASS_SEC;
    }
    if f >= POWER_EMA_BLEND_F_HI {
        return POWER_EMA_TAU_TREBLE_SEC;
    }
    let u = (f / POWER_EMA_BLEND_F_LO).ln() / (POWER_EMA_BLEND_F_HI / POWER_EMA_BLEND_F_LO).ln();
    let u = u.clamp(0.0, 1.0);
    let t = u * u * (3.0 - 2.0 * u);
    POWER_EMA_TAU_BASS_SEC + (POWER_EMA_TAU_TREBLE_SEC - POWER_EMA_TAU_BASS_SEC) * t
}

pub struct SpectrumRing {
    pub buf: Vec<f32>,
    pub write: usize,
    pub sample_rate: f32,
    pub active: bool,
}

impl Default for SpectrumRing {
    fn default() -> Self {
        Self {
            buf: vec![0.0; RING_LEN],
            write: 0,
            sample_rate: 48_000.0,
            active: false,
        }
    }
}

pub type SharedSpectrum = Arc<RwLock<SpectrumRing>>;

pub fn make_shared_spectrum() -> SharedSpectrum {
    Arc::new(RwLock::new(SpectrumRing::default()))
}

/// Append interleaved multi-channel samples into the ring as mono sums.
/// Called from the audio thread.
pub fn push_samples(shared: &SharedSpectrum, samples: &[f32], channels: usize) {
    if channels == 0 || samples.is_empty() {
        return;
    }
    let mut ring = shared.write();
    let mut w = ring.write;
    let inv = 1.0 / channels as f32;
    for chunk in samples.chunks_exact(channels) {
        let mut sum = 0.0;
        for &s in chunk {
            sum += s;
        }
        ring.buf[w] = sum * inv;
        w += 1;
        if w >= RING_LEN {
            w = 0;
        }
    }
    ring.write = w;
}

pub fn configure(shared: &SharedSpectrum, sample_rate: u32, active: bool) {
    let mut ring = shared.write();
    ring.sample_rate = sample_rate.max(1) as f32;
    ring.active = active;
    if !active {
        for v in &mut ring.buf {
            *v = 0.0;
        }
        ring.write = 0;
    }
}

/// Smoothed, peak-held FFT magnitudes ready for display.
pub struct SpectrumAnalyzer {
    fft: Arc<dyn realfft::RealToComplex<f32>>,
    window: Vec<f32>,
    window_gain: f32,
    input: Vec<f32>,
    scratch: Vec<Complex<f32>>,
    output: Vec<Complex<f32>>,
    /// EMA of \|X[k]\|² (linear power) — used by the painter for Gaussian constant-Q smoothing.
    pub bin_power: Vec<f32>,
    pub magnitudes_db: Vec<f32>,
    pub peak_hold_db: Vec<f32>,
    pub peak_hold_set_at: Vec<f64>,
    last_tick: f64,
    pub sample_rate: f32,
}

impl SpectrumAnalyzer {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        let input = fft.make_input_vec();
        let scratch = fft.make_scratch_vec();
        let output = fft.make_output_vec();

        let window = hann_window(FFT_SIZE);
        // Coherent gain of a Hann window is 0.5 — compensate so a full-
        // scale sine reads near 0 dBFS.
        let window_gain = window.iter().copied().sum::<f32>() / FFT_SIZE as f32;

        let bins = output.len();
        let floor_p = 10_f32.powf(SPECTRUM_FLOOR_DB / 10.0);
        Self {
            fft,
            window,
            window_gain,
            input,
            scratch,
            output,
            bin_power: vec![floor_p; bins],
            magnitudes_db: vec![SPECTRUM_FLOOR_DB; bins],
            peak_hold_db: vec![SPECTRUM_FLOOR_DB; bins],
            peak_hold_set_at: vec![0.0; bins],
            last_tick: 0.0,
            sample_rate: 48_000.0,
        }
    }

    pub fn bin_count(&self) -> usize {
        self.magnitudes_db.len()
    }

    pub fn bin_hz(&self, bin: usize) -> f32 {
        self.sample_rate * bin as f32 / FFT_SIZE as f32
    }

    /// Pull the latest FFT_SIZE samples from the ring, apply the window,
    /// FFT, update power EMA, magnitudes (dB), and peak-hold.
    pub fn process(&mut self, shared: &SharedSpectrum, now: f64) {
        let (active, write, sample_rate) = {
            let ring = shared.read();
            self.sample_rate = ring.sample_rate;
            let start = (ring.write + RING_LEN - FFT_SIZE) % RING_LEN;
            if start + FFT_SIZE <= RING_LEN {
                self.input
                    .copy_from_slice(&ring.buf[start..start + FFT_SIZE]);
            } else {
                let first = RING_LEN - start;
                self.input[..first].copy_from_slice(&ring.buf[start..]);
                self.input[first..].copy_from_slice(&ring.buf[..FFT_SIZE - first]);
            }
            (ring.active, ring.write, ring.sample_rate)
        };
        let _ = write;
        self.sample_rate = sample_rate;

        let dt = if self.last_tick <= 0.0 {
            0.016
        } else {
            (now - self.last_tick).clamp(0.0, 0.25) as f32
        };
        self.last_tick = now;

        let bins = self.magnitudes_db.len();
        let floor_p = 10_f32.powf(SPECTRUM_FLOOR_DB / 10.0);

        if !active {
            let decay = (-14.0_f32 * dt).exp();
            for i in 0..bins {
                self.bin_power[i] = (self.bin_power[i] * decay).max(floor_p);
                let d = 10.0 * self.bin_power[i].max(floor_p).log10();
                self.magnitudes_db[i] = d.clamp(SPECTRUM_FLOOR_DB, 12.0);
            }
            let release_db_per_sec = 120.0_f32;
            for v in &mut self.peak_hold_db {
                *v = (*v - release_db_per_sec * dt).max(SPECTRUM_FLOOR_DB);
            }
            for (i, v) in self.peak_hold_db.iter_mut().enumerate() {
                *v = (*v).min(self.magnitudes_db[i]);
            }
            return;
        }

        for (x, w) in self.input.iter_mut().zip(self.window.iter()) {
            *x *= *w;
        }
        if self
            .fft
            .process_with_scratch(&mut self.input, &mut self.output, &mut self.scratch)
            .is_err()
        {
            return;
        }

        let sr = self.sample_rate.max(1.0);
        let bin_hz = sr / FFT_SIZE as f32;
        let norm = 2.0 / (FFT_SIZE as f32 * self.window_gain.max(1e-6));

        for (i, c) in self.output.iter().enumerate() {
            let mag = (c.re * c.re + c.im * c.im).sqrt() * norm;
            let p_inst = mag * mag;

            if i == 0 {
                self.bin_power[i] = floor_p;
            } else {
                let tau = power_ema_tau_hz(i as f32 * bin_hz);
                let beta = 1.0 - (-dt / tau).exp();
                self.bin_power[i] = self.bin_power[i] * (1.0 - beta) + p_inst * beta;
            }

            let d = 10.0 * self.bin_power[i].max(floor_p).log10();
            let db = d.clamp(SPECTRUM_FLOOR_DB, 12.0);
            self.magnitudes_db[i] = db;

            let hold = self.peak_hold_db[i];
            if db >= hold {
                self.peak_hold_db[i] = db;
                self.peak_hold_set_at[i] = now;
            } else if (now - self.peak_hold_set_at[i]) as f32 > 1.50 {
                self.peak_hold_db[i] = (hold - 10.0 * dt).max(db);
            }
        }
    }
}

/// Noise floor drawn on the spectrum — below this the line flattens.
pub const SPECTRUM_FLOOR_DB: f32 = -120.0;

/// Display compensation toward FabFilter-style “flattened” RTA (pink reference + mild lows trim).
/// Applied only in the painter so analyzer state stays in calibrated dBFS.
#[inline]
pub fn spectrum_pro_q_display_db(f_hz: f32) -> f32 {
    if !f_hz.is_finite() || f_hz <= 0.0 {
        return 0.0;
    }
    // ~3 dB/oct around 500 Hz: pulls sub-bass / bass down vs mids without nuking fundamentals.
    // The tilt runs continuously all the way down to sub-20 Hz and is clamped at −18 dB, so the
    // transition through 20 Hz is smooth instead of stepping from 0 dB → −13.7 dB (which used to
    // manifest as a "wall" right at the 20 Hz ruler).
    const DB_PER_OCT: f32 = 2.95;
    const REF_HZ: f32 = 500.0;
    (DB_PER_OCT * (f_hz / REF_HZ).log2()).clamp(-18.0, 14.0)
}

fn hann_window(n: usize) -> Vec<f32> {
    let mut w = Vec::with_capacity(n);
    let denom = (n - 1).max(1) as f32;
    for i in 0..n {
        let x = i as f32 / denom;
        w.push(0.5 - 0.5 * (std::f32::consts::TAU * x).cos());
    }
    w
}
