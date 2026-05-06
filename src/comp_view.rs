//! Compression analyzer graph — mirrored waveform envelope + peak-range trace.
//!
//! Tells the user at a glance whether the incoming program is **over-compressed**
//! (peaks pinned flat at a single level) or **under-compressed** (peaks tall and
//! varied). Single mono-summed trace.
//!
//! # Data pipeline
//!
//! Audio thread → `push_samples_mono_sum` collapses the interleaved block to mono and
//! folds samples into a running "in-flight" bucket. Every `samples_per_bucket` samples
//! (≈ 5 ms at 200 Hz) it flushes one [`CompBucket`] (`{peak_pos, peak_neg, rms}`, all
//! linear amplitude) into a pre-allocated ring buffer.
//!
//! UI thread → `show_comp` reads the ring once per frame, computes a trailing-window
//! **Peak Range** at every bucket (see [`windowed_peak_range_db`]), then paints a
//! mirrored min/max envelope on top (tinted by that windowed PR), a short-term-RMS
//! line through the middle, and the PR trace underneath with reference bands for
//! `Brickwall / Heavy / Moderate / Natural`.
//!
//! # The PR number
//!
//! One number in the headline: **PR** (Peak Range) — the statistical spread of the
//! peak envelope over the trailing ≈ 3 s window, measured as the 95th percentile of
//! per-bucket peak magnitudes (in dBFS) minus the 10th percentile, after absolute
//! gating of silent buckets. In plain English: **how flat is the top of the waveform**
//! in the last ~3 seconds.
//!
//! Why this and not peak ÷ RMS: a downward limiter (LoudMax, hard brickwall) clamps
//! the peaks but leaves sub-threshold content — breaths, tails, reverb decay — at its
//! natural level. Peak ÷ RMS averages those tails into the denominator and reads a
//! healthy "6-10 dB DR" even when the visible top of the waveform is pinned at a
//! single level. PR ignores the tails by construction (they sit below the gate and
//! don't contribute to either percentile) and reads only the spread of the loud
//! program's peak heights — which is what the eye is reading.
//!
//! ## What it actually says
//!
//! - Peaks all bunch at the same dBFS (within ~3 dB) → **low PR** → the top of the
//!   waveform is flat. Either a brickwall/heavy compressor is at work, or the source
//!   is naturally steady (see failure mode below).
//! - Peaks spread across a wide dBFS range (~15-25 dB) → **high PR** → distinct tall
//!   transients over quieter material. Little or no dynamics processing.
//!
//! ## Honest failure mode
//!
//! You cannot deterministically recover "how much compression was applied" from the
//! output of a compressor without seeing its input. A naturally steady source
//! (sustained organ note, synth pad, continuous tone) has peaks all at the same
//! level too, so PR will read **Brickwall / flat** even with zero processing in the
//! chain. The meter is describing the *observed signal's dynamics*, not the
//! *processing that was applied*. For the kind of material this view is designed
//! for — vocals, dialogue, drums, mixes — this distinction is rarely ambiguous in
//! practice; flat-topped vocal peaks means something is limiting.
//!
//! **Why a window, not per-bucket?** Per 5 ms bucket peak barely moves when a
//! compressor is engaged — you need a macro quantity, "peak spread over ~3 s", that
//! spans multiple phrases / transients. The 3 s window matches EBU R128's short-term
//! loudness window and is long enough to include several consecutive peaks in sparse
//! vocal material without straddling phrase boundaries too coarsely.
//!
//! No audio-thread allocations; no new DSP passes over the PCM.
//!
//! # Time window
//!
//! `WINDOW_SECS = 8.0` at `BUCKETS_PER_SEC = 200.0` → `BUCKETS = 1600`. Short enough
//! to watch individual snare attacks without them being smeared into the 50 ms
//! buckets that [`crate::history_graph`] uses.

use std::sync::Arc;

use egui::epaint::{Mesh, Vertex, WHITE_UV};
use egui::{
    Align2, Color32, FontId, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, pos2,
};
use parking_lot::RwLock;

use crate::meter_widget::{COL_RULER_TEXT, FONT_TICK};
use crate::spectrum_view::{PLOT_BOTTOM_PAD, PLOT_TOP_PAD};

/// Bucket rate (Hz). 200 Hz → 5 ms per bucket — fine enough to resolve snare and vocal
/// transients that a coarser cadence (e.g. 20 Hz like the history graph) would smear.
pub const BUCKETS_PER_SEC: f32 = 200.0;

/// Rolling window (seconds).
pub const WINDOW_SECS: f32 = 8.0;

/// Number of buckets drawn on screen (one slot per bucket in the visible window).
pub const BUCKETS: usize = (WINDOW_SECS * BUCKETS_PER_SEC) as usize;

/// Absolute gate for the peak-range computation (linear peak magnitude). Buckets
/// whose max-abs sample sits below this are treated as silence and **excluded** from
/// the percentile set. Without this, quiet room-tone / tail buckets drag the P10 of
/// the peak-level distribution way down and inflate PR on sparse material.
///
/// ≈ −40 dBFS peak. Below that for a real program is tail / decay / room tone
/// territory — nothing we want colouring "how loud are the loud peaks" statistics.
const PR_PEAK_GATE_LIN: f32 = 0.01;

/// Minimum number of above-gate buckets required in the window before PR is reported.
/// Below this the window is too sparse for a stable percentile — `NaN` is returned
/// and the headline shows `PR --`. 20 buckets ≈ 100 ms of loud content.
const PR_MIN_GATED_BUCKETS: usize = 20;

/// Sliding-window length (buckets) for the PR metric. 3.0 s at 200 Hz → 600 buckets,
/// matching EBU R128's short-term loudness window. Long enough to span multiple
/// phrases of sparse vocal / dialogue material so percentile spreads are stable.
const PR_WINDOW_BUCKETS: usize = (3.0 * BUCKETS_PER_SEC) as usize;

/// Extra buckets kept in the ring BEFORE the visible [`BUCKETS`] region so that the
/// leftmost (oldest) displayed bucket already has a full [`PR_WINDOW_BUCKETS`] of
/// history behind it contributing to its Peak Range value.
///
/// Without this pre-roll, the PR window at display position 0 is only 1 bucket long,
/// at position 1 it's 2 buckets long, … and doesn't become fully populated until
/// ~3 s in. That manifested as a visible discontinuity at the left edge of the Comp
/// graph: the first ~100 ms tinted flat amber (NaN PR from [`PR_MIN_GATED_BUCKETS`]
/// not met), the next ~3 s flickering erratically as the window grew, and only the
/// rightmost ~5 s of the display rendering with its steady-state tint + trace. By
/// keeping [`PR_WINDOW_BUCKETS`] of extra history behind the display we eliminate
/// the ramp-up region; every drawn bucket gets the same-sized window.
const PR_CONTEXT_BUCKETS: usize = PR_WINDOW_BUCKETS;

/// Total ring capacity = displayed region + PR pre-roll context. Only the last
/// [`BUCKETS`] entries are drawn.
const RING_CAPACITY: usize = BUCKETS + PR_CONTEXT_BUCKETS;

/// Left inset for the dB ruler numerals on the lower (PR) band.
const LEFT_PAD: f32 = 28.0;
/// Right gutter — same as history so plot-width discipline matches siblings.
const RIGHT_PAD: f32 = 18.0;

/// Height split between the upper envelope band and the lower PR band.
const UPPER_FRACTION: f32 = 0.68;
/// Vertical gap between the two bands.
const BAND_GAP: f32 = 8.0;

/// Peak Range (dB) reference thresholds — P95 minus P10 of gated per-bucket peak
/// levels (in dBFS) over the trailing [`PR_WINDOW_BUCKETS`]. Maps directly to "how
/// flat is the top of the waveform":
///
///   • `< PR_BRICKWALL_DB` → peaks within ~3 dB → brickwall limited / hard squashed
///   • `< PR_HEAVY_DB`     → peaks mostly similar → heavy compression
///   • `< PR_NATURAL_DB`   → some peak variation → moderate / musical compression
///   • `≥ PR_NATURAL_DB`   → wide peak spread → natural / uncompressed
///
/// Honest caveat: a naturally steady source (organ pad, held tone) will read
/// Brickwall too, because its peaks genuinely are all at the same level. See the
/// module docstring.
const PR_BRICKWALL_DB: f32 = 3.0;
const PR_HEAVY_DB: f32 = 7.0;
const PR_NATURAL_DB: f32 = 12.0;

/// Vertical scale of the lower PR band. 0 dB at the bottom, `PR_AXIS_TOP_DB` at the
/// top. 24 dB is enough headroom for the most dynamic program material (drums,
/// orchestral swells).
const PR_AXIS_TOP_DB: f32 = 24.0;

// ---- Colors ------------------------------------------------------------------------

/// Upper-band backdrop (very faint graphite, matches app palette).
const PANEL_BG_TOP: Color32 = Color32::from_rgb(0x17, 0x15, 0x19);
const PANEL_BG_BOTTOM: Color32 = Color32::from_rgb(0x22, 0x20, 0x24);

/// Envelope outline stroke.
const ENV_OUTLINE: Color32 = Color32::from_rgba_premultiplied(0x20, 0x22, 0x26, 0xa0);

/// RMS overlay line (neutral bright grey — reads over any PR tint).
const RMS_LINE: Color32 = Color32::from_rgb(0xe4, 0xe4, 0xea);

/// PR-zone gradient endpoints. `SQUASHED` tints the fill when peak spread is tight
/// (limited / squashed); `DYNAMIC` when peak spread is wide. Interpolated per-bucket.
const PR_SQUASHED: Color32 = Color32::from_rgb(0xff, 0x4a, 0x3c); // warm red
const PR_MID: Color32 = Color32::from_rgb(0xff, 0xc8, 0x3a);      // amber
const PR_DYNAMIC: Color32 = Color32::from_rgb(0x2f, 0xd6, 0x57);  // green

/// Alpha for the PR-zone filled envelope (uniform; strength is carried by the hue).
const PR_FILL_ALPHA: u8 = 0xc8;

/// Lower-band reference zone fills (faint translucent backgrounds).
const ZONE_BRICKWALL_BG: Color32 = Color32::from_rgba_premultiplied(0x55, 0x16, 0x1a, 0x44);
const ZONE_HEAVY_BG: Color32 = Color32::from_rgba_premultiplied(0x4a, 0x32, 0x10, 0x38);
const ZONE_MODERATE_BG: Color32 = Color32::from_rgba_premultiplied(0x22, 0x3a, 0x14, 0x2c);
const ZONE_NATURAL_BG: Color32 = Color32::from_rgba_premultiplied(0x12, 0x2c, 0x18, 0x1c);

/// Lower-band PR trace stroke.
const PR_TRACE: Color32 = Color32::from_rgb(0xff, 0xd4, 0x3a);

/// Headline text backdrop (small pill behind the readout).
const HEADLINE_PILL_BG: Color32 = Color32::from_rgba_premultiplied(0x0c, 0x0c, 0x10, 0xd0);
const HEADLINE_PILL_BORDER: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x33);

// ---- Ring / bucket -----------------------------------------------------------------

/// One 5 ms window of the mono-summed signal. All values are **linear** amplitude in
/// the [-N, +N] range where N can exceed 1 for hot sum signals; the draw side clamps.
#[derive(Clone, Copy, Default)]
pub struct CompBucket {
    pub peak_pos: f32,
    pub peak_neg: f32,
    pub rms_lin: f32,
}

/// In-flight bucket accumulator — lives inside [`CompEnvelope`] so the audio thread
/// doesn't have to allocate. Reset whenever the sample rate changes.
#[derive(Clone, Copy, Default)]
struct InFlight {
    n: u32,
    pos: f32,
    neg: f32,
    sq: f64,
}

impl InFlight {
    #[inline]
    fn fold(&mut self, m: f32) {
        if m > self.pos {
            self.pos = m;
        }
        if m < self.neg {
            self.neg = m;
        }
        self.sq += (m as f64) * (m as f64);
        self.n += 1;
    }

    #[inline]
    fn reset(&mut self) {
        self.n = 0;
        self.pos = 0.0;
        self.neg = 0.0;
        self.sq = 0.0;
    }

    #[inline]
    fn flush(&mut self, samples_per_bucket: u32) -> CompBucket {
        let n = self.n.max(1);
        let bucket = CompBucket {
            peak_pos: self.pos,
            peak_neg: self.neg,
            rms_lin: (self.sq / n as f64).sqrt() as f32,
        };
        let _ = samples_per_bucket;
        self.reset();
        bucket
    }
}

pub struct CompEnvelope {
    ring: Vec<CompBucket>,
    /// Next write slot.
    write_idx: usize,
    /// How many buckets have been written since the ring was last cleared (saturates at [`RING_CAPACITY`]).
    filled: usize,
    /// Sample rate the in-flight accumulator is configured for. 0 until first `push_samples_mono_sum`.
    sample_rate: u32,
    /// Target samples per bucket = `sample_rate / BUCKETS_PER_SEC`.
    samples_per_bucket: u32,
    /// In-flight (not yet flushed) bucket.
    in_flight: InFlight,
}

pub type SharedComp = Arc<RwLock<CompEnvelope>>;

pub fn make_shared_comp() -> SharedComp {
    Arc::new(RwLock::new(CompEnvelope::new()))
}

impl CompEnvelope {
    pub fn new() -> Self {
        Self {
            ring: vec![CompBucket::default(); RING_CAPACITY],
            write_idx: 0,
            filled: 0,
            sample_rate: 0,
            samples_per_bucket: 0,
            in_flight: InFlight::default(),
        }
    }

    /// Reset the ring + in-flight bucket. Cheap — no deallocation. Called from the
    /// audio stream start/stop paths so each take starts clean.
    pub fn clear(&mut self) {
        for b in &mut self.ring {
            *b = CompBucket::default();
        }
        self.write_idx = 0;
        self.filled = 0;
        self.in_flight.reset();
    }

    /// Reconfigure for a new sample rate (called on stream start / device change).
    /// Safe to call every block — only rewires the accumulator when the rate changes.
    pub fn configure(&mut self, sample_rate: u32) {
        let sr = sample_rate.max(1);
        if sr == self.sample_rate {
            return;
        }
        self.sample_rate = sr;
        self.samples_per_bucket =
            ((sr as f32 / BUCKETS_PER_SEC).round() as u32).max(1);
        self.in_flight.reset();
    }

    /// Append one audio block (interleaved multichannel). The block is folded into
    /// mono-sum frames on the fly, then into the 5 ms accumulator. Called from the
    /// audio thread, so this is the hot path: no allocs, no logs.
    pub fn push_samples_mono_sum(
        &mut self,
        interleaved: &[f32],
        channels: usize,
        sample_rate: u32,
    ) {
        if channels == 0 || interleaved.is_empty() {
            return;
        }
        self.configure(sample_rate);
        let spb = self.samples_per_bucket;
        if spb == 0 {
            return;
        }
        let inv_ch = 1.0 / channels as f32;
        for frame in interleaved.chunks_exact(channels) {
            let mut sum = 0.0_f32;
            for &s in frame {
                sum += s;
            }
            let mono = sum * inv_ch;
            self.in_flight.fold(mono);
            if self.in_flight.n >= spb {
                let b = self.in_flight.flush(spb);
                self.ring[self.write_idx] = b;
                self.write_idx = (self.write_idx + 1) % RING_CAPACITY;
                self.filled = (self.filled + 1).min(RING_CAPACITY);
            }
        }
    }

    /// Copy the ring into oldest→newest order. Called on the UI thread once per frame.
    /// The returned vec has up to [`RING_CAPACITY`] entries; the caller slices off
    /// the last [`BUCKETS`] for display and passes the full vec to
    /// [`windowed_peak_range_db`] so every displayed bucket has its full PR window.
    fn ordered(&self) -> Vec<CompBucket> {
        let mut out = Vec::with_capacity(self.filled);
        if self.filled < RING_CAPACITY {
            out.extend_from_slice(&self.ring[..self.filled]);
        } else {
            out.extend_from_slice(&self.ring[self.write_idx..]);
            out.extend_from_slice(&self.ring[..self.write_idx]);
        }
        out
    }
}

impl Default for CompEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Analysis ---------------------------------------------------------------------

/// Peak-level histogram range / resolution used by the sliding-window PR calc.
/// 0.25 dB bins over [-80, +3] dBFS → 332 bins. Matches EBU R128-style loudness
/// histograms: fine enough that percentile quantization error is under 0.5 dB
/// (well below the ±2 dB wobble caused by content variation itself), coarse
/// enough that sliding cost is negligible.
const PR_HIST_MIN_DB: f32 = -80.0;
const PR_HIST_MAX_DB: f32 = 3.0;
const PR_HIST_BIN_DB: f32 = 0.25;
const PR_HIST_BINS: usize =
    ((PR_HIST_MAX_DB - PR_HIST_MIN_DB) / PR_HIST_BIN_DB) as usize;

/// Map a dBFS peak value to its histogram bin index, clamped to `[0, PR_HIST_BINS)`.
#[inline]
fn pr_hist_bin_of(db: f32) -> usize {
    let t = ((db - PR_HIST_MIN_DB) / PR_HIST_BIN_DB).floor() as isize;
    t.clamp(0, PR_HIST_BINS as isize - 1) as usize
}

/// Given a peak-level histogram (`hist`) with `total` gated buckets in it, return
/// the dBFS value at percentile `pctl ∈ [0, 1]` by scanning cumulative counts.
/// O(PR_HIST_BINS) per call. Returns bin-center dB so results quantize cleanly.
#[inline]
fn pr_percentile_from_hist(hist: &[u32], total: u32, pctl: f32) -> f32 {
    debug_assert!(total > 0);
    // Nearest-rank percentile on the (total-1)-indexed distribution.
    let target = ((total as f32 - 1.0) * pctl).round() as u32;
    let mut cum: u32 = 0;
    for (i, &c) in hist.iter().enumerate() {
        let next = cum + c;
        if next > target {
            return PR_HIST_MIN_DB + (i as f32 + 0.5) * PR_HIST_BIN_DB;
        }
        cum = next;
    }
    PR_HIST_MIN_DB + (PR_HIST_BINS as f32 - 0.5) * PR_HIST_BIN_DB
}

/// Returns the Peak Range (dB) at every bucket position, computed over a trailing
/// window of [`PR_WINDOW_BUCKETS`]. At each `i`:
///
/// 1. The peak magnitude (`max(|peak_pos|, |peak_neg|)`) of the arriving bucket is
///    added to a sliding peak-level histogram, and the departing bucket's entry
///    (if it was above the gate) is removed.
/// 2. Buckets below [`PR_PEAK_GATE_LIN`] are ignored (absolute silence gate — keeps
///    tails and room tone out of the distribution).
/// 3. If fewer than [`PR_MIN_GATED_BUCKETS`] gated buckets are currently in the
///    window → `NaN` (insufficient material).
/// 4. Otherwise `P95(peak_dB) - P10(peak_dB)` is computed from the histogram.
///
/// The percentile spread is what the eye is reading when it looks at the "flatness"
/// of the upper envelope — it goes to zero when all peaks sit at the same level
/// (brickwall limiter) and grows as peaks spread out (dynamic transients).
///
/// Per-frame cost: O(n × PR_HIST_BINS) = 1600 × 332 ≈ 0.5 M bin ops plus 2 percentile
/// scans per bucket. Replaces an earlier O(n · w log w) ≈ 10 M compare formulation
/// that sorted a fresh scratch vec at every bucket and was visibly slow under debug
/// builds / on low-power GPUs.
fn windowed_peak_range_db(ordered: &[CompBucket], window: usize) -> Vec<f32> {
    let n = ordered.len();
    let mut out = vec![f32::NAN; n];
    if n == 0 || window == 0 {
        return out;
    }

    // Precompute per-bucket histogram bin (or `usize::MAX` for sub-gate buckets). One
    // pass over `ordered` so the sliding loop is a tight integer hop.
    let mut peak_bin: Vec<usize> = Vec::with_capacity(n);
    for b in ordered {
        let p = b.peak_pos.abs().max(b.peak_neg.abs());
        peak_bin.push(if p >= PR_PEAK_GATE_LIN {
            pr_hist_bin_of(20.0 * p.log10())
        } else {
            usize::MAX
        });
    }

    let mut hist = vec![0u32; PR_HIST_BINS];
    let mut total: u32 = 0;

    for i in 0..n {
        // Add the arriving bucket if it was above the gate.
        let bin_in = peak_bin[i];
        if bin_in != usize::MAX {
            hist[bin_in] += 1;
            total += 1;
        }
        // Remove the departing bucket (the one that just slid out of the trailing
        // `window` samples) if it was above the gate.
        if i >= window {
            let bin_out = peak_bin[i - window];
            if bin_out != usize::MAX {
                hist[bin_out] -= 1;
                total -= 1;
            }
        }

        if (total as usize) < PR_MIN_GATED_BUCKETS {
            continue; // NaN stays
        }
        let p10 = pr_percentile_from_hist(&hist, total, 0.10);
        let p95 = pr_percentile_from_hist(&hist, total, 0.95);
        out[i] = (p95 - p10).max(0.0);
    }
    out
}

// ---- Drawing ------------------------------------------------------------------------

#[inline]
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(
        mix(a.r(), b.r()),
        mix(a.g(), b.g()),
        mix(a.b(), b.b()),
        mix(a.a(), b.a()),
    )
}

/// Map a PR (Peak Range) dB value to a fill color along the squashed→dynamic gradient.
/// `0 dB → fully squashed (red); ≥ PR_NATURAL_DB → fully dynamic (green)`.
fn pr_color(pr_db: f32) -> Color32 {
    if !pr_db.is_finite() {
        return Color32::from_rgba_unmultiplied(
            PR_MID.r(),
            PR_MID.g(),
            PR_MID.b(),
            PR_FILL_ALPHA,
        );
    }
    let base = if pr_db <= PR_HEAVY_DB {
        let t = (pr_db / PR_HEAVY_DB).clamp(0.0, 1.0);
        lerp_color(PR_SQUASHED, PR_MID, t)
    } else {
        let t = ((pr_db - PR_HEAVY_DB) / (PR_NATURAL_DB - PR_HEAVY_DB))
            .clamp(0.0, 1.0);
        lerp_color(PR_MID, PR_DYNAMIC, t)
    };
    Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), PR_FILL_ALPHA)
}

fn verdict_for(pr_db: f32) -> (&'static str, Color32) {
    if !pr_db.is_finite() {
        return ("--", COL_RULER_TEXT);
    }
    if pr_db < PR_BRICKWALL_DB {
        ("Brickwall", PR_SQUASHED)
    } else if pr_db < PR_HEAVY_DB {
        ("Heavy", Color32::from_rgb(0xff, 0x8a, 0x2a))
    } else if pr_db < PR_NATURAL_DB {
        ("Moderate", PR_TRACE)
    } else {
        ("Natural", PR_DYNAMIC)
    }
}

/// Background gradient + subtle frame for one of the two bands.
fn paint_band_backdrop(painter: &egui::Painter, rect: Rect) {
    let mut mesh = Mesh::default();
    mesh.vertices.push(Vertex {
        pos: rect.left_top(),
        uv: WHITE_UV,
        color: PANEL_BG_TOP,
    });
    mesh.vertices.push(Vertex {
        pos: rect.right_top(),
        uv: WHITE_UV,
        color: PANEL_BG_TOP,
    });
    mesh.vertices.push(Vertex {
        pos: rect.right_bottom(),
        uv: WHITE_UV,
        color: PANEL_BG_BOTTOM,
    });
    mesh.vertices.push(Vertex {
        pos: rect.left_bottom(),
        uv: WHITE_UV,
        color: PANEL_BG_BOTTOM,
    });
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    painter.add(Shape::mesh(mesh));
    painter.rect_stroke(
        rect,
        egui::CornerRadius::same(2),
        Stroke::new(0.5, Color32::from_rgba_premultiplied(0x30, 0x30, 0x3a, 0x60)),
        egui::StrokeKind::Inside,
    );
}

/// Paint the mirrored min/max envelope with per-bucket PR-zone tint + an RMS overlay.
/// `window_pr` is the parallel slice of windowed PR values (dB, NaN where the window
/// doesn't have enough loud content) that colors each bucket by its surrounding 3 s
/// peak-spread state.
fn paint_envelope(
    painter: &egui::Painter,
    ordered: &[CompBucket],
    window_pr: &[f32],
    rect: Rect,
) {
    if ordered.is_empty() {
        return;
    }
    let n = ordered.len();
    let width = rect.width().max(1.0);
    let bucket_w = width / BUCKETS as f32;
    // Right-align the most recent bucket at `rect.right()`; older ones march left.
    let start_x = rect.right() - n as f32 * bucket_w;
    let mid_y = rect.center().y;
    // Normalise peak amplitude to rect half-height. Clamp at 1.0 (values > ±1 are hot sums).
    let half_h = rect.height() * 0.5;

    // --- PR-zone filled envelope (one quad per bucket, color by windowed PR) ---
    let mut env_mesh = Mesh::default();
    for (i, b) in ordered.iter().enumerate() {
        let x0 = start_x + i as f32 * bucket_w;
        let x1 = x0 + bucket_w + 0.5; // slight overdraw to eliminate gaps
        let pos_clamped = b.peak_pos.clamp(-1.0, 1.0);
        let neg_clamped = b.peak_neg.clamp(-1.0, 1.0);
        let y_top = mid_y - pos_clamped * half_h;
        let y_bot = mid_y - neg_clamped * half_h;
        let c = pr_color(window_pr.get(i).copied().unwrap_or(f32::NAN));

        let v = env_mesh.vertices.len() as u32;
        env_mesh.vertices.push(Vertex {
            pos: pos2(x0, y_top),
            uv: WHITE_UV,
            color: c,
        });
        env_mesh.vertices.push(Vertex {
            pos: pos2(x1, y_top),
            uv: WHITE_UV,
            color: c,
        });
        env_mesh.vertices.push(Vertex {
            pos: pos2(x1, y_bot),
            uv: WHITE_UV,
            color: c,
        });
        env_mesh.vertices.push(Vertex {
            pos: pos2(x0, y_bot),
            uv: WHITE_UV,
            color: c,
        });
        env_mesh
            .indices
            .extend_from_slice(&[v, v + 1, v + 2, v, v + 2, v + 3]);
    }
    painter.add(Shape::mesh(env_mesh));

    // --- Envelope outline (top and bottom edges) ---
    let mut pts_top = Vec::with_capacity(n);
    let mut pts_bot = Vec::with_capacity(n);
    for (i, b) in ordered.iter().enumerate() {
        let x = start_x + (i as f32 + 0.5) * bucket_w;
        let pos_clamped = b.peak_pos.clamp(-1.0, 1.0);
        let neg_clamped = b.peak_neg.clamp(-1.0, 1.0);
        pts_top.push(pos2(x, mid_y - pos_clamped * half_h));
        pts_bot.push(pos2(x, mid_y - neg_clamped * half_h));
    }
    painter.add(Shape::line(pts_top, Stroke::new(0.75, ENV_OUTLINE)));
    painter.add(Shape::line(pts_bot, Stroke::new(0.75, ENV_OUTLINE)));

    // --- RMS ± overlay lines ---
    let mut rms_top = Vec::with_capacity(n);
    let mut rms_bot = Vec::with_capacity(n);
    for (i, b) in ordered.iter().enumerate() {
        let x = start_x + (i as f32 + 0.5) * bucket_w;
        let r = b.rms_lin.clamp(0.0, 1.0);
        rms_top.push(pos2(x, mid_y - r * half_h));
        rms_bot.push(pos2(x, mid_y + r * half_h));
    }
    painter.add(Shape::line(rms_top, Stroke::new(1.2, RMS_LINE)));
    painter.add(Shape::line(rms_bot, Stroke::new(1.2, RMS_LINE)));

    // --- Zero line (quiet, so the waveform reads as "around zero") ---
    painter.line_segment(
        [pos2(rect.left(), mid_y), pos2(rect.right(), mid_y)],
        Stroke::new(
            0.5,
            Color32::from_rgba_premultiplied(0x5a, 0x5a, 0x62, 0x60),
        ),
    );
}

/// Paint reference-zone backgrounds + PR trace + dB rulers on the lower band.
/// The polyline is driven by the 3 s-window PR (`window_pr`), which updates smoothly
/// as the window slides.
fn paint_pr_band(
    painter: &egui::Painter,
    ordered: &[CompBucket],
    window_pr: &[f32],
    rect: Rect,
) {
    // --- Zone background bands ---
    let y_for_db = |db: f32| -> f32 {
        let t = (db / PR_AXIS_TOP_DB).clamp(0.0, 1.0);
        rect.bottom() - t * rect.height()
    };
    let zones: [(f32, f32, Color32); 4] = [
        (0.0, PR_BRICKWALL_DB, ZONE_BRICKWALL_BG),
        (PR_BRICKWALL_DB, PR_HEAVY_DB, ZONE_HEAVY_BG),
        (PR_HEAVY_DB, PR_NATURAL_DB, ZONE_MODERATE_BG),
        (PR_NATURAL_DB, PR_AXIS_TOP_DB, ZONE_NATURAL_BG),
    ];
    for (lo, hi, c) in zones {
        let y_lo = y_for_db(lo);
        let y_hi = y_for_db(hi);
        let zone = Rect::from_min_max(pos2(rect.left(), y_hi), pos2(rect.right(), y_lo));
        painter.rect_filled(zone, 0.0, c);
    }

    // --- Zone boundary lines + labels ---
    let label_cfg: [(f32, &str); 3] = [
        (PR_BRICKWALL_DB, "BRK"),
        (PR_HEAVY_DB, "HVY"),
        (PR_NATURAL_DB, "NAT"),
    ];
    for (db, label) in label_cfg {
        let y = y_for_db(db);
        painter.line_segment(
            [pos2(rect.left(), y), pos2(rect.right(), y)],
            Stroke::new(
                0.5,
                Color32::from_rgba_premultiplied(0x4a, 0x4a, 0x52, 0x70),
            ),
        );
        painter.text(
            pos2(rect.left() + 4.0, y - 1.0),
            Align2::LEFT_BOTTOM,
            label,
            FontId::proportional(9.0),
            COL_RULER_TEXT,
        );
    }

    // --- Left-axis tick labels matching zone boundaries ---
    let ticks: [f32; 5] = [
        0.0,
        PR_BRICKWALL_DB,
        PR_HEAVY_DB,
        PR_NATURAL_DB,
        PR_AXIS_TOP_DB,
    ];
    for db in ticks {
        let y = y_for_db(db);
        painter.text(
            pos2(rect.left() - 4.0, y),
            Align2::RIGHT_CENTER,
            format!("{db:.0}"),
            FontId::proportional(FONT_TICK),
            COL_RULER_TEXT,
        );
    }

    // --- PR polyline ---
    if !ordered.is_empty() {
        let n = ordered.len();
        let bucket_w = rect.width() / BUCKETS as f32;
        let start_x = rect.right() - n as f32 * bucket_w;
        let mut pts: Vec<Pos2> = Vec::with_capacity(n);
        for i in 0..n {
            let pr = window_pr.get(i).copied().unwrap_or(f32::NAN);
            if !pr.is_finite() {
                // Break the line across insufficient-content regions so they don't
                // link by interpolation.
                if pts.len() >= 2 {
                    painter.add(Shape::line(
                        std::mem::take(&mut pts),
                        Stroke::new(1.4, PR_TRACE),
                    ));
                } else {
                    pts.clear();
                }
                continue;
            }
            let x = start_x + (i as f32 + 0.5) * bucket_w;
            let y = y_for_db(pr);
            pts.push(pos2(x, y));
        }
        if pts.len() >= 2 {
            painter.add(Shape::line(pts, Stroke::new(1.4, PR_TRACE)));
        }
    }
}

/// Right-aligned headline: "PR 4.2 dB · Heavy".
///
/// `pr_db` is the most recent windowed PR from [`windowed_peak_range_db`]. The
/// verdict is driven by the same number — the upper-band visual and the headline
/// therefore always agree ("looks squashed" and "reads squashed" are the same thing).
fn paint_headline(painter: &egui::Painter, rect: Rect, pr_db: f32) {
    let (verdict, verdict_col) = verdict_for(pr_db);
    let pr_part = if pr_db.is_finite() {
        format!("PR {pr_db:.1} dB")
    } else {
        "PR --".to_string()
    };
    let text = format!("{pr_part}  ·  {verdict}");
    let font = FontId::proportional(12.0);
    let pad_x = 6.0;
    let pad_y = 3.0;

    let galley = painter.layout_no_wrap(text.clone(), font.clone(), verdict_col);
    let size = galley.size();
    let pill = Rect::from_min_size(
        pos2(rect.right() - size.x - pad_x * 2.0 - 4.0, rect.top() + 4.0),
        egui::vec2(size.x + pad_x * 2.0, size.y + pad_y * 2.0),
    );
    painter.rect(
        pill,
        egui::CornerRadius::same(3),
        HEADLINE_PILL_BG,
        Stroke::new(0.75, HEADLINE_PILL_BORDER),
        egui::StrokeKind::Inside,
    );
    painter.galley(
        pos2(pill.left() + pad_x, pill.top() + pad_y),
        galley,
        verdict_col,
    );
}

/// Paint the Comp view. Same allocation discipline as [`crate::history_graph::show_history_graph`]
/// (fill parent `Ui`, no forced minimum width — avoids horizontal scroll affordances).
pub fn show_comp(ui: &mut Ui, shared: &SharedComp, _now: f64) -> Response {
    let mut desired = ui.available_size();
    desired.x = desired.x.max(1.0);
    desired.y = desired.y.max(1.0);
    let (rect, response) = ui.allocate_exact_size(desired, Sense::hover());
    let painter = ui.painter_at(rect);

    let top = rect.top() + PLOT_TOP_PAD;
    let bottom = rect.bottom() - PLOT_BOTTOM_PAD;
    let left = rect.left() + LEFT_PAD;
    let right = rect.right() - RIGHT_PAD;
    let inner_h = (bottom - top - BAND_GAP).max(20.0);
    let upper_h = inner_h * UPPER_FRACTION;
    let upper_rect = Rect::from_min_max(pos2(left, top), pos2(right, top + upper_h));
    let lower_rect = Rect::from_min_max(
        pos2(left, top + upper_h + BAND_GAP),
        pos2(right, bottom),
    );

    let ordered = {
        let g = shared.read();
        g.ordered()
    };
    // One pass across the FULL ring (display region + PR_CONTEXT_BUCKETS of pre-roll
    // history) so every entry in `window_pr` — including the ones at the left edge
    // of the visible display — has a complete trailing 3 s of context behind it.
    let window_pr = windowed_peak_range_db(&ordered, PR_WINDOW_BUCKETS);

    // Slice off the pre-roll. Only the last BUCKETS entries are drawn; the leading
    // PR_CONTEXT_BUCKETS are pure history that already did their job seeding the
    // percentile window for the leftmost display bucket.
    let display_start = ordered.len().saturating_sub(BUCKETS);
    let display_buckets = &ordered[display_start..];
    let display_pr = &window_pr[display_start..];

    // Headline = most recent finite window value in the displayed region; falls back
    // to NaN → "--" if the window doesn't have enough loud buckets yet.
    let pr_db = display_pr
        .iter()
        .rev()
        .copied()
        .find(|v| v.is_finite())
        .unwrap_or(f32::NAN);

    paint_band_backdrop(&painter, upper_rect);
    paint_band_backdrop(&painter, lower_rect);

    paint_envelope(&painter, display_buckets, display_pr, upper_rect);
    paint_pr_band(&painter, display_buckets, display_pr, lower_rect);
    paint_headline(&painter, upper_rect, pr_db);

    // Scroll at ~60 Hz. Audio thread pushes new buckets at 200 Hz, but 60 Hz visual
    // refresh is smooth enough and keeps GPU load modest when the view is open.
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(16));

    // Time axis (same scheme as history graph: "now" at the right, -s labels every 2 s).
    let window = WINDOW_SECS as i32;
    let step = 2;
    for s in (0..=window).step_by(step as usize) {
        let frac = 1.0 - (s as f32 / WINDOW_SECS);
        let x = lower_rect.left() + frac * lower_rect.width();
        painter.line_segment(
            [pos2(x, lower_rect.top()), pos2(x, lower_rect.bottom())],
            Stroke::new(
                0.5,
                Color32::from_rgba_premultiplied(0x4a, 0x4a, 0x52, 0x40),
            ),
        );
        let label = if s == 0 {
            "now".to_string()
        } else {
            format!("-{s}s")
        };
        painter.text(
            pos2(x, lower_rect.bottom() + 2.0),
            Align2::CENTER_TOP,
            label,
            FontId::proportional(FONT_TICK),
            COL_RULER_TEXT,
        );
    }

    response
}
