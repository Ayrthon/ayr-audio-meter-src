//! Real-time spectrum analyzer.
//!
//! * The audio thread writes mono-summed samples into a shared ring
//!   buffer (`SpectrumRing`) — cheap, allocation-free, no analysis.
//! * The UI thread runs the FFT (`SpectrumAnalyzer`) from the latest
//!   `FFT_SIZE` samples each frame, applies a Hann window, computes \|X[k]\|²
//!   per bin, then **aggregates bins into 251 log-spaced buckets by
//!   mean-square** (ZLEqualizer `kAnalyzerPointNum`). Each bucket dB then
//!   runs through a **fall-only EMA** (attack instant, release exponential)
//!   with a **refresh-rate-normalized decay factor** (`0.95^(23.4375 / fps)`,
//!   ≈ 0.83 s τ at 60 fps, uniform across all frequencies). Peak-hold is
//!   maintained per bucket for the painter.
//!
//! This is a cherry-pick of ZLEqualizer's three most accurate ideas
//! (bucket aggregation, fall-only dB EMA, refresh-normalized decay) grafted
//! onto our existing FFT size, calibration, pink tilt, and peak-hold.
//! FabFilter Pro-Q's analyzer is proprietary; this stack is a **close
//! functional analogue**, not a byte-identical clone.

use std::sync::Arc;

use parking_lot::RwLock;
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;

/// FFT length (samples). At 48 kHz this is a 170 ms window, giving 5.86 Hz bin resolution —
/// enough to distinguish ~6 Hz infrasonic rumble, the 20 Hz fundamental, and 60 Hz mains hum
/// in separate bins. 4096 (85 ms / 11.7 Hz bins) is too coarse below 20 Hz; 16384 (341 ms)
/// would start to visibly smear percussive transients. 8192 is the sweet spot for full-range
/// music.
pub const FFT_SIZE: usize = 8192;

/// Ring length — 2× FFT size so the UI thread can always pull a
/// contiguous tail regardless of where the audio thread last wrote.
pub const RING_LEN: usize = FFT_SIZE * 2;

/// Number of log-spaced display buckets aggregated from FFT bins. Matches ZLEqualizer's
/// `kAnalyzerPointNum = 251`: dense enough to look smooth on a log x-axis, sparse enough
/// that each bucket covers several bins in the mids/highs for proper energy averaging.
pub const POINT_NUM: usize = 251;

/// Lower limit of the bucket grid (Hz). Matches ZL; the display can extend below this.
pub const MIN_FREQ: f32 = 10.0;

/// Fall-only EMA decay factor at the reference step rate, from ZL's
/// `decay_rates_` default of 0.95 evaluated at 23.4375 Hz (≈ one FFT hop / 2048 at 48 kHz).
/// At other refresh rates we rescale: `actual_decay = DECAY_REF ^ (DECAY_REF_HZ / refresh_hz)`.
/// This keeps the perceived release time constant regardless of UI frame rate.
const DECAY_REF: f32 = 0.95;
const DECAY_REF_HZ: f32 = 23.4375;
const REFRESH_HZ_DEFAULT: f32 = 60.0;

/// Noise floor drawn on the spectrum — below this the line flattens.
pub const SPECTRUM_FLOOR_DB: f32 = -120.0;

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

/// Build the log-spaced bucket table. Walks a geometric frequency ladder from `MIN_FREQ` to
/// ~Nyquist, forcing at least one FFT bin per bucket. Matches ZL's `prepareAkima` layout.
///
/// Returns `(bin_start[j], bin_end_exclusive[j], bucket_center_freq_hz[j])`.
fn build_buckets(sample_rate: f32, fft_size: usize) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let max_freq = (sample_rate * 0.5 - 0.1).max(100.0);
    let min_freq = MIN_FREQ.min(max_freq * 0.5).max(1.0);
    let freq_delta = sample_rate / fft_size as f32;
    let freq_mul = (max_freq / min_freq).powf(2.0 / POINT_NUM as f32);
    let limit = fft_size / 2;

    let mut starts: Vec<usize> = Vec::with_capacity(POINT_NUM / 2 + 4);
    let mut ends: Vec<usize> = Vec::with_capacity(POINT_NUM / 2 + 4);

    let mut freq = min_freq * freq_mul.sqrt();
    starts.push(0);
    ends.push(((freq / freq_delta).round() as usize).max(1));

    for _ in 0..(POINT_NUM / 2 + 1) {
        freq *= freq_mul;
        let idx = ((freq / freq_delta).round() as usize).min(limit);
        let last_end = *ends.last().unwrap();
        if idx > last_end {
            starts.push(last_end);
            ends.push(idx);
        }
    }

    if freq > sample_rate * 0.5 {
        let last_start = *starts.last().unwrap();
        let last_end = *ends.last().unwrap();
        let half = (last_end - last_start) / 2;
        starts.push(limit + 1 - half);
        ends.push(limit + 1);
    }

    let freqs: Vec<f32> = starts
        .iter()
        .zip(ends.iter())
        .map(|(&a, &b)| (a + b).saturating_sub(1) as f32 * 0.5 * freq_delta)
        .collect();

    (starts, ends, freqs)
}

/// Smoothed, peak-held FFT bucket magnitudes ready for display.
pub struct SpectrumAnalyzer {
    fft: Arc<dyn realfft::RealToComplex<f32>>,
    window: Vec<f32>,
    window_gain: f32,
    input: Vec<f32>,
    scratch: Vec<Complex<f32>>,
    output: Vec<Complex<f32>>,
    /// Scratch: calibrated per-bin power (|X[k]|² × amplitude-normalisation²).
    ms_buffer: Vec<f32>,

    /// Inclusive-start / exclusive-end FFT bin range for each log bucket.
    bucket_starts: Vec<usize>,
    bucket_ends: Vec<usize>,

    /// Geometric midpoint frequency of each bucket (Hz).
    pub bucket_freqs: Vec<f32>,
    /// Fall-only EMA dB per bucket (calibrated dBFS, pre-tilt). Attack is instant, release
    /// uses the refresh-rate-normalized decay factor.
    pub bucket_dbs: Vec<f32>,
    /// Peak-hold dB per bucket with dwell + linear-dB release.
    pub peak_hold_db: Vec<f32>,
    pub peak_hold_set_at: Vec<f64>,

    last_tick: f64,
    pub sample_rate: f32,
    prepared_sr: f32,
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

        let sr = 48_000.0_f32;
        let (starts, ends, freqs) = build_buckets(sr, FFT_SIZE);
        let nb = freqs.len();

        Self {
            fft,
            window,
            window_gain,
            input,
            scratch,
            output,
            ms_buffer: vec![0.0; FFT_SIZE / 2 + 1],
            bucket_starts: starts,
            bucket_ends: ends,
            bucket_freqs: freqs,
            bucket_dbs: vec![SPECTRUM_FLOOR_DB; nb],
            peak_hold_db: vec![SPECTRUM_FLOOR_DB; nb],
            peak_hold_set_at: vec![0.0; nb],
            last_tick: 0.0,
            sample_rate: sr,
            prepared_sr: sr,
        }
    }

    /// Pull the latest `FFT_SIZE` samples from the ring, apply the window, FFT, aggregate bins
    /// into log buckets, then apply the fall-only dB EMA and peak-hold per bucket.
    pub fn process(&mut self, shared: &SharedSpectrum, now: f64) {
        let (active, _write, sample_rate) = {
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
        self.sample_rate = sample_rate;

        if (sample_rate - self.prepared_sr).abs() > 1.0 {
            let (s, e, f) = build_buckets(sample_rate, FFT_SIZE);
            let nb = f.len();
            self.bucket_starts = s;
            self.bucket_ends = e;
            self.bucket_freqs = f;
            self.bucket_dbs.resize(nb, SPECTRUM_FLOOR_DB);
            self.peak_hold_db.resize(nb, SPECTRUM_FLOOR_DB);
            self.peak_hold_set_at.resize(nb, 0.0);
            self.prepared_sr = sample_rate;
        }

        let dt = if self.last_tick <= 0.0 {
            1.0 / REFRESH_HZ_DEFAULT as f64
        } else {
            (now - self.last_tick).clamp(0.0, 0.25)
        };
        self.last_tick = now;

        // Refresh-normalized decay factor: `0.95 ^ (23.4375 / refresh)`. ZL formulation.
        let refresh = (1.0 / dt.max(1e-6)) as f32;
        let decay = DECAY_REF.powf(DECAY_REF_HZ / refresh.max(1.0));

        let nb = self.bucket_dbs.len();

        if !active {
            // Fast collapse to floor when capture is stopped — matches the pre-bucket
            // behaviour which used an exp(-14·dt) on linear power (≈ 60 dB/s in dB terms).
            let fast_rel_db = (60.0 * dt as f32).max(0.0);
            for d in &mut self.bucket_dbs {
                *d = (*d - fast_rel_db).max(SPECTRUM_FLOOR_DB);
            }
            let peak_rel_db = (120.0 * dt as f32).max(0.0);
            for j in 0..nb {
                let v = (self.peak_hold_db[j] - peak_rel_db).max(SPECTRUM_FLOOR_DB);
                self.peak_hold_db[j] = v.min(self.bucket_dbs[j]);
            }
            return;
        }

        // Window (natural Hann amplitude, not pre-scaled — calibration folded into `p_scale`).
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

        // Amplitude calibration: a full-scale sine reads ≈ 0 dBFS through the peak bin.
        // `norm = 2 / (N · window_gain)` on the magnitude; we need its square for power.
        let norm = 2.0 / (FFT_SIZE as f32 * self.window_gain.max(1e-6));
        let p_scale = norm * norm;

        for (i, c) in self.output.iter().enumerate() {
            self.ms_buffer[i] = (c.re * c.re + c.im * c.im) * p_scale;
        }
        // Kill DC so it never contaminates the 10 Hz bucket.
        if !self.ms_buffer.is_empty() {
            self.ms_buffer[0] = 0.0;
        }

        let bins = self.ms_buffer.len();
        let floor_ln = 1e-30_f32;
        for j in 0..nb {
            let s = self.bucket_starts[j].min(bins);
            let e = self.bucket_ends[j].min(bins);
            if e <= s {
                continue;
            }
            let mut sum = 0.0_f32;
            for i in s..e {
                sum += self.ms_buffer[i];
            }
            let ms = sum / (e - s) as f32;
            let cur_db = 10.0 * ms.max(floor_ln).log10();

            // Fall-only EMA in dB: attack instant, release exponential.
            let new_db = if cur_db < self.bucket_dbs[j] {
                self.bucket_dbs[j] * decay + cur_db * (1.0 - decay)
            } else {
                cur_db
            };
            self.bucket_dbs[j] = new_db.clamp(SPECTRUM_FLOOR_DB, 12.0);

            // Peak-hold: 1.50 s dwell, then 10 dB/s linear release.
            let b = self.bucket_dbs[j];
            let hold = self.peak_hold_db[j];
            if b >= hold {
                self.peak_hold_db[j] = b;
                self.peak_hold_set_at[j] = now;
            } else if (now - self.peak_hold_set_at[j]) as f32 > 1.50 {
                self.peak_hold_db[j] = (hold - 10.0 * dt as f32).max(b);
            }
        }
    }
}

/// Display compensation toward FabFilter-style "flattened" RTA (pink reference + mild lows trim).
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
