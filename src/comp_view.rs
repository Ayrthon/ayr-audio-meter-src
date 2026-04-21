//! Compression analyzer graph — mirrored waveform envelope + crest-factor trace.
//!
//! Tells the user at a glance whether the incoming program is **over-compressed**
//! (transients collapsed into a flat bar, tight gap between peak and RMS) or
//! **under-compressed** (tall spikes, wide gap). Single mono-summed trace.
//!
//! # Data pipeline
//!
//! Audio thread → `push_samples_mono_sum` collapses the interleaved block to mono and
//! folds samples into a running "in-flight" bucket. Every `samples_per_bucket` samples
//! (≈ 5 ms at 200 Hz) it flushes one [`CompBucket`] (`{peak_pos, peak_neg, rms}`, all
//! linear amplitude) into a pre-allocated ring buffer.
//!
//! UI thread → `show_comp` reads the ring once per frame, computes a trailing-window
//! gated crest factor (`peak / gated-RMS` over ≈ 1 s — see [`windowed_crests_db`]),
//! then paints a mirrored min/max envelope on top (tinted by that windowed crest), a
//! short-term-RMS line through the middle, and the windowed crest-factor trace
//! underneath with reference bands for `Brickwall / Heavy / Moderate / Natural`.
//!
//! **Why a window, not per-bucket crest?** Inside any single 5 ms bucket crest factor
//! is dominated by waveform shape (≈ 3 dB for a sine, 5-8 dB for a vowel) and barely
//! reacts to program compression. Program-level compression is a macro quantity —
//! "peak over ~1 s vs. gated RMS over the same window" — which is what a broadcast
//! or mastering tool reports, and what correctly separates uncompressed speech
//! (~14-18 dB) from a DBX-style channel-strip compressor (~6-10 dB).
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

/// Ring capacity — one slot per bucket.
pub const BUCKETS: usize = (WINDOW_SECS * BUCKETS_PER_SEC) as usize;

/// Minimum RMS below which crest factor is treated as "silent" (undefined) — avoids
/// blow-up on pure-silence frames where dividing by a near-zero RMS would produce
/// absurdly large crest numbers.
const SILENCE_RMS_LIN: f32 = 1e-5; // ≈ -100 dBFS

/// Absolute gate for the windowed crest computation (linear). Buckets whose RMS sits
/// below this are treated as silence and **excluded** from the window's RMS average —
/// same philosophy as EBU R128's absolute gating. Without this, pauses between words
/// drag the RMS down and artificially inflate crest factor (making uncompressed-looking
/// speech that happens to have long pauses read as "more dynamic" than it really is).
/// ≈ −50 dBFS; above room noise, below any real speech syllable.
const CREST_GATE_LIN: f32 = 3.2e-3;

/// Sliding-window length (buckets) for the "compression crest factor" we display.
/// 1.0 s at 200 Hz → 200 buckets. A per-bucket (5 ms) crest is dominated by
/// waveform-shape micro-dynamics (a sine is ≈ 3 dB crest, a vowel ≈ 5-8 dB) and
/// does *not* meaningfully change when a hardware compressor like the DBX 286s is
/// engaged. Program compression is a macro quantity: "peak over the last ≈1 s
/// vs. the gated RMS over the same window". That's what broadcast / mastering
/// tools report and what this window measures.
const CREST_WINDOW_BUCKETS: usize = (1.0 * BUCKETS_PER_SEC) as usize;

/// Minimum width of the Comp panel (matches history / spectrum right-panel min).
const MIN_WIDTH: f32 = 360.0;
/// Left inset for the dB ruler numerals on the lower (crest) band.
const LEFT_PAD: f32 = 28.0;
/// Right gutter — same as history so plot-width discipline matches siblings.
const RIGHT_PAD: f32 = 18.0;

/// Height split between the upper envelope band and the lower crest band.
const UPPER_FRACTION: f32 = 0.68;
/// Vertical gap between the two bands.
const BAND_GAP: f32 = 8.0;

/// Crest-factor reference thresholds (dB). Anything above `NATURAL` is fully
/// dynamic / music-like; below `BRICKWALL` is effectively brick-walled limiting.
const CREST_BRICKWALL_DB: f32 = 6.0;
const CREST_HEAVY_DB: f32 = 10.0;
const CREST_NATURAL_DB: f32 = 14.0;

/// Vertical scale of the lower crest band. 0 dB at the bottom, `CREST_AXIS_TOP_DB`
/// at the top. 20 dB is enough headroom for orchestral / percussion spikes.
const CREST_AXIS_TOP_DB: f32 = 20.0;

// ---- Colors ------------------------------------------------------------------------

/// Upper-band backdrop (very faint graphite, matches app palette).
const PANEL_BG_TOP: Color32 = Color32::from_rgb(0x17, 0x15, 0x19);
const PANEL_BG_BOTTOM: Color32 = Color32::from_rgb(0x22, 0x20, 0x24);

/// Envelope outline stroke.
const ENV_OUTLINE: Color32 = Color32::from_rgba_premultiplied(0x20, 0x22, 0x26, 0xa0);

/// RMS overlay line (neutral bright grey — reads over any crest tint).
const RMS_LINE: Color32 = Color32::from_rgb(0xe4, 0xe4, 0xea);

/// Crest-zone gradient endpoints. `SQUASHED` tints the fill when peak-RMS is tight;
/// `DYNAMIC` when peak-RMS is wide. Interpolated per-bucket.
const CREST_SQUASHED: Color32 = Color32::from_rgb(0xff, 0x4a, 0x3c); // warm red
const CREST_MID: Color32 = Color32::from_rgb(0xff, 0xc8, 0x3a);      // amber
const CREST_DYNAMIC: Color32 = Color32::from_rgb(0x2f, 0xd6, 0x57);  // green

/// Alpha for the crest-zone filled envelope (uniform; strength is already carried by the hue).
const CREST_FILL_ALPHA: u8 = 0xc8;

/// Lower-band reference zone fills (faint translucent backgrounds).
const ZONE_BRICKWALL_BG: Color32 = Color32::from_rgba_premultiplied(0x55, 0x16, 0x1a, 0x44);
const ZONE_HEAVY_BG: Color32 = Color32::from_rgba_premultiplied(0x4a, 0x32, 0x10, 0x38);
const ZONE_MODERATE_BG: Color32 = Color32::from_rgba_premultiplied(0x22, 0x3a, 0x14, 0x2c);
const ZONE_NATURAL_BG: Color32 = Color32::from_rgba_premultiplied(0x12, 0x2c, 0x18, 0x1c);

/// Lower-band crest-factor trace stroke.
const CREST_TRACE: Color32 = Color32::from_rgb(0xff, 0xd4, 0x3a);

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
    /// How many buckets have been written since the ring was last cleared (saturates at [`BUCKETS`]).
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
            ring: vec![CompBucket::default(); BUCKETS],
            write_idx: 0,
            filled: 0,
            sample_rate: 0,
            samples_per_bucket: 0,
            in_flight: InFlight::default(),
        }
    }

    /// Reset the ring + in-flight bucket. Cheap — no deallocation.
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
                self.write_idx = (self.write_idx + 1) % BUCKETS;
                self.filled = (self.filled + 1).min(BUCKETS);
            }
        }
    }

    /// Copy the ring into oldest→newest order. Called on the UI thread once per frame.
    fn ordered(&self) -> Vec<CompBucket> {
        let mut out = Vec::with_capacity(self.filled);
        if self.filled < BUCKETS {
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

/// Map a crest-factor dB value to a fill color along the squashed→dynamic gradient.
/// `0 dB → fully squashed, ≥ CREST_NATURAL_DB → fully dynamic`.
fn crest_color(crest_db: f32) -> Color32 {
    if !crest_db.is_finite() {
        return Color32::from_rgba_unmultiplied(
            CREST_MID.r(),
            CREST_MID.g(),
            CREST_MID.b(),
            CREST_FILL_ALPHA,
        );
    }
    let base = if crest_db <= CREST_HEAVY_DB {
        let t = (crest_db / CREST_HEAVY_DB).clamp(0.0, 1.0);
        lerp_color(CREST_SQUASHED, CREST_MID, t)
    } else {
        let t = ((crest_db - CREST_HEAVY_DB) / (CREST_NATURAL_DB - CREST_HEAVY_DB))
            .clamp(0.0, 1.0);
        lerp_color(CREST_MID, CREST_DYNAMIC, t)
    };
    Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), CREST_FILL_ALPHA)
}

/// Returns the program crest factor (dB) at every bucket position, computed over a
/// trailing window of [`CREST_WINDOW_BUCKETS`] with absolute silence gating on the
/// RMS term. `f32::NAN` means "silent or insufficient signal" — draw code skips
/// those (tint falls back to neutral; line breaks).
///
/// Sliding-window peak uses a monotonic-decreasing deque (O(n) total). Sliding-window
/// gated RMS uses two running accumulators (sum of squared gated RMS + gated count)
/// that get folded in/out as the window advances — also O(n) total. Entire pass is
/// ~1600 iterations per UI frame, well under the 16 ms budget.
fn windowed_crests_db(ordered: &[CompBucket], window: usize) -> Vec<f32> {
    let n = ordered.len();
    let mut out = vec![f32::NAN; n];
    if n == 0 || window == 0 {
        return out;
    }

    let mut dq: std::collections::VecDeque<(usize, f32)> =
        std::collections::VecDeque::with_capacity(window);
    let mut sum_sq = 0.0_f64;
    let mut gated_count: usize = 0;

    for i in 0..n {
        let p = ordered[i].peak_pos.abs().max(ordered[i].peak_neg.abs());
        while let Some(&(_, back_p)) = dq.back() {
            if back_p <= p {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back((i, p));

        let r = ordered[i].rms_lin;
        if r >= CREST_GATE_LIN {
            sum_sq += (r as f64) * (r as f64);
            gated_count += 1;
        }

        let start = i + 1 - window.min(i + 1);
        while let Some(&(front_i, _)) = dq.front() {
            if front_i < start {
                dq.pop_front();
            } else {
                break;
            }
        }
        // We can only evict from the left of `sum_sq` / `gated_count` when the window
        // is fully wide — before then we're still building it up and keep everything.
        if i + 1 > window {
            let lost = &ordered[i - window];
            if lost.rms_lin >= CREST_GATE_LIN {
                sum_sq -= (lost.rms_lin as f64) * (lost.rms_lin as f64);
                if sum_sq < 0.0 {
                    sum_sq = 0.0;
                }
                gated_count = gated_count.saturating_sub(1);
            }
        }

        let win_peak = dq.front().map(|&(_, p)| p).unwrap_or(0.0);
        if gated_count == 0 || win_peak < SILENCE_RMS_LIN {
            out[i] = f32::NAN;
            continue;
        }
        let rms = (sum_sq / gated_count as f64).sqrt() as f32;
        if rms < SILENCE_RMS_LIN {
            out[i] = f32::NAN;
            continue;
        }
        out[i] = 20.0 * (win_peak / rms).log10();
    }
    out
}

fn verdict_for(crest_db: f32) -> (&'static str, Color32) {
    if !crest_db.is_finite() {
        return ("Silent", COL_RULER_TEXT);
    }
    if crest_db < CREST_BRICKWALL_DB {
        ("Brickwall", CREST_SQUASHED)
    } else if crest_db < CREST_HEAVY_DB {
        ("Heavy", Color32::from_rgb(0xff, 0x8a, 0x2a))
    } else if crest_db < CREST_NATURAL_DB {
        ("Moderate", CREST_TRACE)
    } else {
        ("Natural", CREST_DYNAMIC)
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

/// Paint the mirrored min/max envelope with per-bucket crest-zone tint + an RMS overlay.
/// `window_crest` is the parallel slice of windowed crest-factor values (dB, NaN for
/// silent) that colors each bucket by its surrounding 1 s compression state.
fn paint_envelope(
    painter: &egui::Painter,
    ordered: &[CompBucket],
    window_crest: &[f32],
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

    // --- Crest-zone filled envelope (one quad per bucket, color by windowed crest) ---
    let mut env_mesh = Mesh::default();
    for (i, b) in ordered.iter().enumerate() {
        let x0 = start_x + i as f32 * bucket_w;
        let x1 = x0 + bucket_w + 0.5; // slight overdraw to eliminate gaps
        let pos_clamped = b.peak_pos.clamp(-1.0, 1.0);
        let neg_clamped = b.peak_neg.clamp(-1.0, 1.0);
        let y_top = mid_y - pos_clamped * half_h;
        let y_bot = mid_y - neg_clamped * half_h;
        let c = crest_color(window_crest.get(i).copied().unwrap_or(f32::NAN));

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

/// Paint reference-zone backgrounds + crest-factor line + dB rulers on the lower band.
/// The polyline is driven by the 1 s-window gated crest (`window_crest`), not a
/// per-bucket crest — the latter is dominated by waveform micro-dynamics and barely
/// moves when a hardware compressor is engaged.
fn paint_crest_band(
    painter: &egui::Painter,
    ordered: &[CompBucket],
    window_crest: &[f32],
    rect: Rect,
) {
    // --- Zone background bands ---
    let y_for_db = |db: f32| -> f32 {
        let t = (db / CREST_AXIS_TOP_DB).clamp(0.0, 1.0);
        rect.bottom() - t * rect.height()
    };
    let zones: [(f32, f32, Color32); 4] = [
        (0.0, CREST_BRICKWALL_DB, ZONE_BRICKWALL_BG),
        (CREST_BRICKWALL_DB, CREST_HEAVY_DB, ZONE_HEAVY_BG),
        (CREST_HEAVY_DB, CREST_NATURAL_DB, ZONE_MODERATE_BG),
        (CREST_NATURAL_DB, CREST_AXIS_TOP_DB, ZONE_NATURAL_BG),
    ];
    for (lo, hi, c) in zones {
        let y_lo = y_for_db(lo);
        let y_hi = y_for_db(hi);
        let zone = Rect::from_min_max(pos2(rect.left(), y_hi), pos2(rect.right(), y_lo));
        painter.rect_filled(zone, 0.0, c);
    }

    // --- Zone boundary lines + labels ---
    let label_cfg: [(f32, &str); 3] = [
        (CREST_BRICKWALL_DB, "BRK"),
        (CREST_HEAVY_DB, "HVY"),
        (CREST_NATURAL_DB, "NAT"),
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

    // --- Left-axis tick labels (0, 6, 10, 14, 20 dB) ---
    let ticks: [f32; 5] = [0.0, 6.0, 10.0, 14.0, 20.0];
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

    // --- Crest-factor polyline ---
    if !ordered.is_empty() {
        let n = ordered.len();
        let bucket_w = rect.width() / BUCKETS as f32;
        let start_x = rect.right() - n as f32 * bucket_w;
        let mut pts: Vec<Pos2> = Vec::with_capacity(n);
        for i in 0..n {
            let crest = window_crest.get(i).copied().unwrap_or(f32::NAN);
            if !crest.is_finite() {
                // Break the line across silent regions so they don't link by interpolation.
                if pts.len() >= 2 {
                    painter.add(Shape::line(
                        std::mem::take(&mut pts),
                        Stroke::new(1.4, CREST_TRACE),
                    ));
                } else {
                    pts.clear();
                }
                continue;
            }
            let x = start_x + (i as f32 + 0.5) * bucket_w;
            let y = y_for_db(crest);
            pts.push(pos2(x, y));
        }
        if pts.len() >= 2 {
            painter.add(Shape::line(pts, Stroke::new(1.4, CREST_TRACE)));
        }
    }
}

/// Right-aligned headline: "CF 8.4 dB · Heavy".
fn paint_headline(painter: &egui::Painter, rect: Rect, crest_db: f32) {
    let (verdict, verdict_col) = verdict_for(crest_db);
    let text = if crest_db.is_finite() {
        format!("CF {crest_db:.1} dB  ·  {verdict}")
    } else {
        format!("CF --  ·  {verdict}")
    };
    let font = FontId::proportional(12.0);
    let pad_x = 6.0;
    let pad_y = 3.0;

    // Measure then paint.
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

/// Scroll + paint the Comp view. Mirrors the allocation / layout discipline of
/// [`crate::history_graph::show_history_graph`] so it slots into the right panel
/// without changing the outer `ScrollArea` plumbing in `app.rs`.
pub fn show_comp(ui: &mut Ui, shared: &SharedComp, _now: f64) -> Response {
    let mut desired = ui.available_size();
    desired.x = desired.x.max(MIN_WIDTH);
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
    // One O(n) pass to get the gated, 1 s-windowed crest factor at every bucket.
    let window_crest = windowed_crests_db(&ordered, CREST_WINDOW_BUCKETS);
    // Headline = most recent finite window value; we fall back to NaN → "--" if
    // the window has no gated (non-silent) buckets yet.
    let headline_crest = window_crest
        .iter()
        .rev()
        .copied()
        .find(|c| c.is_finite())
        .unwrap_or(f32::NAN);

    paint_band_backdrop(&painter, upper_rect);
    paint_band_backdrop(&painter, lower_rect);

    paint_envelope(&painter, &ordered, &window_crest, upper_rect);
    paint_crest_band(&painter, &ordered, &window_crest, lower_rect);
    paint_headline(&painter, upper_rect, headline_crest);

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
