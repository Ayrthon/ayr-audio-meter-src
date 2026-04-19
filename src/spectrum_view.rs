//! Spectrum analyzer — rulers aligned with **FabFilter Pro-Q 3**'s EQ spectrum column
//! (inset 20 Hz–20 kHz, majors 20…20k, piecewise 0…−100 dB grid). The trace reads from the
//! analyzer's 251 log-spaced buckets (ZL-style mean-square aggregation + fall-only dB EMA),
//! then display-side applies pink tilt, vertical offset, and a single horizontal smoothing
//! pass. Pro-Q's exact DSP is proprietary; this is a deliberate **approximation**, not a
//! licensed clone.
//!
//! Includes gradient fill, main line, and decade frequency grid.

use egui::epaint::{Mesh, Vertex, WHITE_UV};
use egui::{
    Align2, Button, Color32, CursorIcon, Direction, FontId, Layout, Painter, Pos2, Rect,
    Response, RichText, Sense, Shape, Stroke, Ui, UiBuilder, pos2, vec2,
};

use crate::spectrum::{
    SharedSpectrum, SpectrumAnalyzer, SPECTRUM_FLOOR_DB, spectrum_pro_q_display_db,
};

/// Pro-Q 3 analyzer vertical: 0 dB (top) … −100 dB (bottom), piecewise-linear.
const PRO_Q_ANALYZER_DB_TOP: f32 = 0.0;
const PRO_Q_ANALYZER_DB_BOT: f32 = -100.0;

/// Normalised vertical break points for the piecewise-linear dB axis.
/// `t = 0` is the very top of the plot (above 0 dB), `t = 1` is the very bottom (below −90 dB).
/// Pro-Q 3 leaves small gaps above 0 dB and below −90 dB; we mirror those here.
const PRO_Q_T_TOP_PAD: f32 = 0.01; // 0 dB sits this fraction below the very top
const PRO_Q_T_MINUS12: f32 = 0.12; // −12 dB ~ one third from the top (zoomed EQ band)
const PRO_Q_T_MINUS90: f32 = 0.96; // −90 dB ~ 5 % above the very bottom

#[inline]
fn pro_q_t_from_db(db: f32) -> f32 {
    // Clamp against a slightly wider range than the labelled ticks so the trace can extend into
    // the top and bottom padding bands for Pro-Q's tapered look past 0 dB / −90 dB.
    let d = db.clamp(-120.0, 6.0);
    if d >= 0.0 {
        // Above 0 dB: linear into the top padding band (saturates at +6 dB = t=0).
        let u = ((6.0 - d) / 6.0).clamp(0.0, 1.0);
        u * PRO_Q_T_TOP_PAD
    } else if d >= -12.0 {
        // 0 … −12 dB, uniform 3 dB/grid (zoomed EQ band).
        let u = (0.0 - d) / 12.0;
        PRO_Q_T_TOP_PAD + u * (PRO_Q_T_MINUS12 - PRO_Q_T_TOP_PAD)
    } else if d >= -100.0 {
        // −12 … −90 dB, compressed (all the low-level detail lives here).
        let u = (-12.0 - d) / 88.0;
        PRO_Q_T_MINUS12 + u * (PRO_Q_T_MINUS90 - PRO_Q_T_MINUS12)
    } else {
        // Below −90 dB: linear into the bottom padding band, saturating at −120 dB.
        let u = ((-100.0 - d) / 20.0).clamp(0.0, 1.0);
        PRO_Q_T_MINUS90 + u * (1.0 - PRO_Q_T_MINUS90)
    }
}

/// Log-frequency plot limits. Extended down to 8 Hz so the 10 Hz ruler tick is clearly inset —
/// bin 1 at 48 kHz / 8192 samples sits at 5.86 Hz, so everything above 8 Hz is real FFT data.
const F_DISPLAY_MIN_HZ: f32 = 8.0;
const F_DISPLAY_MAX_CAP_HZ: f32 = 29_000.0;

// Pro-Q 3–inspired panel (dark graphite, soft grid, cool fill).
const BG_TOP: Color32 = Color32::from_rgb(0x13, 0x0e, 0x15);
const BG_BOTTOM: Color32 = Color32::from_rgb(0x23, 0x21, 0x23);
const GRID_MAJOR: Color32 = Color32::from_rgba_premultiplied(0x12, 0x12, 0x14, 0x07);
const GRID_MINOR: Color32 = Color32::from_rgba_premultiplied(0x12, 0x12, 0x14, 0x07);
const LABEL_COL: Color32 = Color32::from_rgb(0xb4, 0xb4, 0xbc);

const FILL_TOP: Color32 = Color32::from_rgba_premultiplied(0x25, 0x27, 0x2c, 0x22);
const FILL_BOT: Color32 = Color32::from_rgba_premultiplied(0x30, 0x32, 0x38, 0x00);
const LINE_COL: Color32 = Color32::from_rgba_premultiplied(0x82, 0x82, 0x86, 0x72);

/// "Pretty mode" palette — warm red at the low end through to cool violet at
/// the top. Anchors are chosen at roughly the centre of each of the five
/// classical spectrum bands (low / low-mid / mid / high-mid / high) so the
/// gradient transitions naturally where band perception transitions. Colours
/// are unpremultiplied RGB; the painter converts them with `premul_alpha` to
/// match the current fill / stroke translucency exactly.
const PRETTY_ANCHORS_HZ: [f32; 5] = [30.0, 180.0, 900.0, 4_000.0, 15_000.0];
const PRETTY_ANCHORS_RGB: [(u8, u8, u8); 5] = [
    (0xff, 0x4a, 0x3c), // low:       warm red
    (0xff, 0x8c, 0x2a), // low-mid:   orange
    (0xff, 0xd4, 0x3a), // mid:       gold / yellow
    (0x46, 0xc8, 0xff), // high-mid:  cyan-blue
    (0xb4, 0x50, 0xff), // high:      violet
];
/// Alpha for the mesh-fill top vertex in pretty mode (matches `FILL_TOP.a`).
const PRETTY_FILL_ALPHA: u8 = 0x22;
/// Alpha for the stroke in pretty mode (matches `LINE_COL.a`).
const PRETTY_LINE_ALPHA: u8 = 0x72;

// ---- Interaction overlay (hover cursor line + user-pinned anchor lines) ----
// Hover line follows the mouse while the cursor is in the plot body. Anchor
// lines are "pins" the user clicked on - each gets a small × to remove it.
// Colour palette is deliberately warm/amber so anchors read as *user data*
// rather than background grid.
const HOVER_LINE_COL: Color32 = Color32::from_rgba_premultiplied(0xcc, 0xcc, 0xd4, 0x78);
const HOVER_PILL_BG: Color32 = Color32::from_rgba_premultiplied(0x18, 0x18, 0x1e, 0xd8);
const HOVER_PILL_FG: Color32 = Color32::from_rgb(0xea, 0xea, 0xf0);

const ANCHOR_LINE_COL: Color32 = Color32::from_rgba_premultiplied(0xff, 0xc8, 0x5a, 0xa8);
const ANCHOR_PILL_BG: Color32 = Color32::from_rgba_premultiplied(0x18, 0x12, 0x0a, 0xe0);
const ANCHOR_PILL_FG: Color32 = Color32::from_rgb(0xff, 0xd8, 0x80);
const ANCHOR_X_FG: Color32 = Color32::from_rgb(0xc8, 0x9a, 0x58);
const ANCHOR_X_HOVER_FG: Color32 = Color32::from_rgb(0xff, 0x82, 0x64);

/// Duplicate-click tolerance in log10(Hz) units. ~0.01 means two anchors within
/// about 2.3 % of each other are treated as the same pin - wide enough that a
/// small wobble on click doesn't stack two lines, narrow enough that users can
/// still place adjacent references (e.g. 440 Hz and 466 Hz) intentionally.
const ANCHOR_DEDUPE_LOG_HZ: f32 = 0.01;

// Purely cosmetic: shift the whole displayed trace down by this many dB. Used to push the
// DAC/ADC idle-noise slope (which sits around -80 to -85 dB after the pink tilt, since the
// tilt adds +14 dB at 20 kHz) below the -100 dB axis bottom so it falls off-screen. Does NOT
// touch analyzer state, peak-hold ballistics, or the raw FFT data - only the pixels.
//
// Tradeoff: real-signal peaks also shift down by this amount on the ruler (a -3 dB peak reads
// at -15 dB on the displayed axis). Tune to taste: -10 to -12 dB usually hides the slope while
// keeping music well inside the visible band.
const DISPLAY_OFFSET_DB: f32 = -16.0;

pub const MIN_WIDTH: f32 = 360.0;
pub const MIN_HEIGHT: f32 = 260.0;

pub fn show_spectrum(
    ui: &mut Ui,
    analyzer: &mut SpectrumAnalyzer,
    _shared: &SharedSpectrum,
    _now: f64,
    pretty_mode: &mut bool,
    anchors: &mut Vec<f32>,
) -> Response {
    // `analyzer` is ticked in `AudioMeterApp::ui` so the spectrum decays when audio stops even
    // if the History tab is visible.

    // Leave a gap between the analyzer's right edge and the window/content edge so the panel
    // doesn't butt up against whatever's next to it.
    const RIGHT_MARGIN: f32 = 16.0;
    let mut desired = ui.available_size();
    desired.x = (desired.x - RIGHT_MARGIN).max(MIN_WIDTH);
    // `Sense::click()` so we get both hover-position reporting AND click events.
    // The hover cursor line, anchor pins, and delete-× hit-testing all ride on
    // this single response; egui resolves overlaps with the Pretty-mode checkbox
    // (registered later via `ui.put`) automatically because later-registered
    // widgets win clicks in their sub-rect.
    let (rect, response) = ui.allocate_exact_size(desired, Sense::click());
    let painter = ui.painter_at(rect);

    // Vertical gradient background: BG_TOP at the top fading to BG_BOTTOM at the bottom.
    // Drawn as a 4-vertex mesh (egui's Shape::Rect only supports a single fill color); rounded
    // corners dropped to 0 so the gradient's square edges match the fill cleanly.
    {
        let mut mesh = Mesh::default();
        mesh.vertices.push(Vertex { pos: rect.left_top(),     uv: WHITE_UV, color: BG_TOP });
        mesh.vertices.push(Vertex { pos: rect.right_top(),    uv: WHITE_UV, color: BG_TOP });
        mesh.vertices.push(Vertex { pos: rect.right_bottom(), uv: WHITE_UV, color: BG_BOTTOM });
        mesh.vertices.push(Vertex { pos: rect.left_bottom(),  uv: WHITE_UV, color: BG_BOTTOM });
        mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        painter.add(Shape::mesh(mesh));
    }

    // Inner plot area: compact left (freq labels below), right gutter for dB ticks like Pro-Q.
    let left_pad = 14.0;
    let right_pad = 10.0;
    let top_pad = 10.0;
    let bottom_pad = 22.0;
    let plot = Rect::from_min_max(
        pos2(rect.left() + left_pad, rect.top() + top_pad),
        pos2(rect.right() - right_pad, rect.bottom() - bottom_pad),
    );

    // Plot the full labelled span even if the sample-rate Nyquist (e.g. 24 kHz @ 48 kHz) is below
    // `F_DISPLAY_MAX_CAP_HZ`: frequencies above Nyquist have no FFT data so the trace naturally
    // tapers into the spectrum floor, producing Pro-Q's visible right-side padding past 20 kHz.
    let _nyquist = (analyzer.sample_rate * 0.5).max(1000.0);
    let f_min = F_DISPLAY_MIN_HZ;
    let f_max = F_DISPLAY_MAX_CAP_HZ;
    let log_fmin = f_min.log10();
    let log_fmax = f_max.log10();

    let x_from_freq = |f: f32| -> f32 {
        let t = (f.max(f_min).log10() - log_fmin) / (log_fmax - log_fmin);
        plot.left() + t.clamp(0.0, 1.0) * plot.width()
    };

    let y_from_db = |db: f32| -> f32 {
        plot.top() + pro_q_t_from_db(db) * plot.height()
    };

    // ----- grid ----------------------------------------------------
    // Minor frequency lines (every integer multiple inside each decade).
    for decade in &[10.0_f32, 100.0, 1_000.0, 10_000.0] {
        for i in 2..10 {
            let f = decade * i as f32;
            if f < f_min || f > f_max {
                continue;
            }
            let x = x_from_freq(f);
            painter.line_segment(
                [pos2(x, plot.top()), pos2(x, plot.bottom())],
                Stroke::new(1.0, GRID_MINOR),
            );
        }
    }
    // Major frequency lines + labels. 10 Hz added as an infrasonic anchor now that the FFT
    // resolves down to ~6 Hz (bin 1 at 48 kHz / 8192 samples = 5.86 Hz).
    let major_freqs: &[f32] = &[
        10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0,
    ];
    for &f in major_freqs {
        if f < f_min || f > f_max {
            continue;
        }
        let x = x_from_freq(f);
        painter.line_segment(
            [pos2(x, plot.top()), pos2(x, plot.bottom())],
            Stroke::new(1.0, GRID_MAJOR),
        );
        let label = if f >= 1_000.0 {
            if (f % 1_000.0).abs() < 1.0 {
                format!("{:.0}k", f / 1_000.0)
            } else {
                format!("{:.1}k", f / 1_000.0)
            }
        } else {
            format!("{:.0}", f)
        };
        painter.text(
            pos2(x, plot.bottom() + 3.0),
            Align2::CENTER_TOP,
            label,
            FontId::proportional(10.0),
            LABEL_COL,
        );
    }
    // Horizontal grid lines at these dB breakpoints (labels intentionally omitted: the ruler was
    // cosmetic - peak meters on the side are the source of truth for real signal level, and after
    // the pink tilt + DISPLAY_OFFSET_DB the numbers here never mapped back to raw dB anyway).
    let db_ticks: &[f32] = &[0.0, -3.0, -6.0, -9.0, -12.0, -30.0, -40.0, -50.0, -60.0, -100.0];
    for &db in db_ticks {
        let y = y_from_db(db);
        if y < plot.top() - 0.5 || y > plot.bottom() + 0.5 {
            continue;
        }
        painter.line_segment(
            [pos2(plot.left(), y), pos2(plot.right(), y)],
            Stroke::new(1.0, GRID_MAJOR),
        );
    }

    // ----- spectrum path ------------------------------------------
    let nb = analyzer.bucket_dbs.len();
    if nb > 2 {
        // Sample the Catmull-Rom curve at **physical-pixel** density so the trace stays
        // crisp at any Windows / egui DPI scale. At 1.0× we get one sample per logical
        // pixel (unchanged); at 1.5× (common on 1440p/4K) we get 1.5× as many samples so
        // each physical pixel has a dedicated curve sample instead of being stretched
        // across 1.5 px of interpolation. Clamped to [1.0, 4.0] to avoid runaway vertex
        // counts on exotic DPI settings.
        let ppp = ui.ctx().pixels_per_point().clamp(1.0, 4.0);
        let pixel_step = 1.0_f32 / ppp;
        let pixel_count = (plot.width() / pixel_step).ceil() as usize;
        if pixel_count > 4 {
            let width = plot.width();
            let log_range = log_fmax - log_fmin;

            let mut mags: Vec<f32> = Vec::with_capacity(pixel_count + 1);
            let mut holds: Vec<f32> = Vec::with_capacity(pixel_count + 1);

            for px in 0..=pixel_count {
                let t = px as f32 / pixel_count as f32;
                let f = 10.0_f32.powf(log_fmin + t * log_range);

                // Catmull-Rom cubic interpolation in log-frequency across the 251 analyzer
                // buckets. Each bucket is a mean-square-averaged power-to-dB value
                // (ZL's `kAnalyzerPointNum` aggregation). A cubic between the four surrounding
                // bucket samples rounds the corners at bucket centers so transient peaks look
                // like Pro-Q's smoothly-shouldered spikes instead of piecewise-linear tents.
                // Conceptually equivalent to the Akima spline ZL draws its 251 points with.
                let m = bucket_catmull_db(f, &analyzer.bucket_freqs, &analyzer.bucket_dbs);
                let h = bucket_catmull_db(f, &analyzer.bucket_freqs, &analyzer.peak_hold_db);

                let comp = spectrum_pro_q_display_db(f) + DISPLAY_OFFSET_DB;
                mags.push((m + comp).clamp(SPECTRUM_FLOOR_DB, 12.0));
                holds.push((h + comp).clamp(SPECTRUM_FLOOR_DB, 12.0));
            }

            // Single light horizontal smoothing pass to soften any pixel-level kinks at bucket
            // boundaries. We drop the old `blend_detail_hz` stage because bucket aggregation
            // already preserves the natural "spikiness" at high frequencies.
            smooth_triple_1d(&mut mags);
            smooth_triple_1d(&mut holds);

            // Inset the path slightly from the plot left/right so 1px strokes are not half-clipped
            // at the edges (which looked like a detached "ghost" line past 20k, especially with a
            // wide halo stroke).
            let x_inset = 1.0_f32;
            let inner_w = (width - 2.0 * x_inset).max(1.0);
            let x_at = |t: f32| plot.left() + x_inset + t.clamp(0.0, 1.0) * inner_w;

            // Rebuild point lists.
            let pts: Vec<Pos2> = mags
                .iter()
                .enumerate()
                .map(|(px, &m)| {
                    let t = px as f32 / pixel_count as f32;
                    pos2(x_at(t), y_from_db(m))
                })
                .collect();
            let _pts_hold: Vec<Pos2> = holds
                .iter()
                .enumerate()
                .map(|(px, &m)| {
                    let t = px as f32 / pixel_count as f32;
                    pos2(x_at(t), y_from_db(m))
                })
                .collect();

            if pts.len() >= 2 {
                // Per-vertex colours for the fill-mesh top row and the stroke, indexed by
                // the same pixel position that drives the trace y-value. In pretty mode the
                // colour is sampled from the warm→cool palette at each pixel's frequency;
                // in default mode every pixel shares `FILL_TOP` / `LINE_COL` so we simply
                // reuse those constants.
                let pretty = *pretty_mode;
                let fill_top_colors: Vec<Color32> = if pretty {
                    (0..pts.len())
                        .map(|i| {
                            let t = i as f32 / pixel_count as f32;
                            let f = 10.0_f32.powf(log_fmin + t * log_range);
                            premul_alpha(pretty_rgb_at_hz(f), PRETTY_FILL_ALPHA)
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                // Gradient-filled mesh under the curve.
                let mut mesh = Mesh::default();
                mesh.vertices.reserve(pts.len() * 2);
                mesh.indices.reserve((pts.len() - 1) * 6);
                let bottom_y = plot.bottom();
                for (i, p) in pts.iter().enumerate() {
                    let (top_c, bot_c) = if pretty {
                        // Fade to fully transparent at the bottom so the hue dissolves into
                        // the gradient background instead of banding against `FILL_BOT`.
                        (fill_top_colors[i], Color32::TRANSPARENT)
                    } else {
                        (FILL_TOP, FILL_BOT)
                    };
                    mesh.vertices.push(Vertex {
                        pos: *p,
                        uv: WHITE_UV,
                        color: top_c,
                    });
                    mesh.vertices.push(Vertex {
                        pos: pos2(p.x, bottom_y),
                        uv: WHITE_UV,
                        color: bot_c,
                    });
                }
                for i in 0..pts.len() - 1 {
                    let a = (i * 2) as u32;
                    let b = a + 1;
                    let c = a + 2;
                    let d = a + 3;
                    mesh.indices.extend_from_slice(&[a, b, c, b, d, c]);
                }
                painter.add(Shape::mesh(mesh));

                // Stroke pass: one `Shape::line` per contiguous "visible run" (stretches
                // of the curve that don't sit on the plot floor). egui's own tessellator
                // feathers the stroke with 1 logical pixel of AA on each side, which is
                // what actually keeps thin lines smooth - my earlier experiments with
                // hand-rolled ribbon meshes bypassed that feathering and ended up worse
                // than the library default. Lesson learnt: let egui do stroke AA.
                //
                // Floor plateaus are still elided: the Catmull-Rom curve clamps to
                // `plot.bottom()` wherever display dB falls below -120, and drawing any
                // stroke there produced a horizontal "ghost line" in silent bands (common
                // in the sub-bass / ~2 kHz regions on machines with mains / ADC leakage).
                // A half-pixel tolerance catches the clamp plateau without clipping real
                // troughs that only graze the floor.
                //
                // Pretty-mode colouring is per-frequency, but `Shape::line` takes a single
                // stroke colour. To keep the warm→cool gradient visible we split each run
                // into fixed-length colour bands (`PRETTY_STROKE_BAND_PX` segments each)
                // and emit one shape per band, sampled at the band's midpoint. Bands share
                // their boundary vertex so the miter normals still line up across the seam
                // and the 1-px feather covers the single-pixel colour step invisibly.
                const STROKE_W: f32 = 0.8;
                const PRETTY_STROKE_BAND_PX: usize = 24;
                let floor_y = plot.bottom();
                let is_floor = |y: f32| y >= floor_y - 0.5;

                let emit_run = |run: &[Pos2], run_start_px: usize| {
                    if run.len() < 2 {
                        return;
                    }
                    if pretty {
                        let mut i = 0;
                        while i + 1 < run.len() {
                            let j = (i + PRETTY_STROKE_BAND_PX).min(run.len() - 1);
                            let mid_px = run_start_px + (i + j) / 2;
                            let t = mid_px as f32 / pixel_count as f32;
                            let f = 10.0_f32.powf(log_fmin + t * log_range);
                            let col = premul_alpha(pretty_rgb_at_hz(f), PRETTY_LINE_ALPHA);
                            painter.add(Shape::line(
                                run[i..=j].to_vec(),
                                Stroke::new(STROKE_W, col),
                            ));
                            i = j;
                        }
                    } else {
                        painter.add(Shape::line(run.to_vec(), Stroke::new(STROKE_W, LINE_COL)));
                    }
                };

                let mut run: Vec<Pos2> = Vec::with_capacity(pts.len());
                let mut run_start_px: usize = 0;
                for (i, p) in pts.iter().enumerate() {
                    if is_floor(p.y) {
                        if !run.is_empty() {
                            emit_run(&run, run_start_px);
                            run.clear();
                        }
                    } else {
                        if run.is_empty() {
                            run_start_px = i;
                        }
                        run.push(*p);
                    }
                }
                if !run.is_empty() {
                    emit_run(&run, run_start_px);
                }
            }
        }
    }

    // Pre-compute the control stack: **Pretty** top-left, **Clear all** top-right.
    // Anchor pills sit in the upper band (see `pill_top` below) — above the
    // densest part of many traces but below the button row.
    //
    // Both rects are calculated up front so the hover-line / anchor-click
    // code below can use their union as a "don't paint / don't click here"
    // guard without caring which button is which. The hover vertical line
    // will still pass through the columns the buttons occupy - the buttons
    // paint after the line so they cover that slice naturally.
    const BTN_W: f32 = 78.0;
    const BTN_H: f32 = 18.0;
    // Single inset from the plot's inner top-left corner so horizontal and
    // vertical spacing to the graph border match.
    const BTN_INSET: f32 = 4.0;
    let pretty_rect = Rect::from_min_size(
        pos2(plot.left() + BTN_INSET, plot.top() + BTN_INSET),
        vec2(BTN_W, BTN_H),
    );
    let clear_rect = Rect::from_min_size(
        pos2(plot.right() - BTN_W - BTN_INSET, plot.top() + BTN_INSET),
        vec2(BTN_W, BTN_H),
    );
    // Everything the hover/click logic must stay out of.
    let overlay_rect = pretty_rect.union(clear_rect);

    // ----- anchor pins + hover cursor line ------------------------
    // Inverse of `x_from_freq`: pixel → Hz along the log axis. Used to resolve
    // both the hover label and the frequency stored on click.
    let freq_from_x = |x: f32| -> f32 {
        let t = ((x - plot.left()) / plot.width().max(1.0)).clamp(0.0, 1.0);
        10.0_f32.powf(log_fmin + t * (log_fmax - log_fmin))
    };

    // Shift = "fine-control" modifier, same convention as knob-dragging in
    // most DAWs. When Shift goes from up → down while the cursor is inside
    // the plot we latch the current pointer x as an anchor; while it stays
    // down the *effective* hover x is computed as
    //     effective_x = anchor_x + (pointer_x - anchor_x) * FINE_SCALE
    // so every physical pixel of mouse motion only moves the hover line
    // `FINE_SCALE` pixels on the log axis. At FINE_SCALE=0.1 that's 10×
    // finer, which at 1 kHz on this plot works out to roughly 1 Hz per
    // physical pixel - enough to land on whole-Hz references (440, 523, ...)
    // without wrestling a sub-pixel hit target.
    //
    // Visually the OS cursor "detaches" from the hover line while Shift is
    // held (same as a DAW knob does). The line + pill are the virtual
    // cursor; the OS cursor stays where your hand is. Releasing Shift
    // immediately drops the anchor and the line snaps back to the pointer.
    //
    // State lives in egui's per-context temp data so we don't need to
    // thread it through the `AudioMeterApp` struct for what is a purely
    // view-internal interaction mode.
    const FINE_SCALE: f32 = 0.1;
    let fine_anchor_id = ui.id().with("spectrum_shift_fine_anchor");
    let shift_held = ui.input(|i| i.modifiers.shift);
    let mut fine_anchor_x: Option<f32> = ui
        .ctx()
        .data(|d| d.get_temp::<f32>(fine_anchor_id));
    if shift_held {
        if fine_anchor_x.is_none() {
            if let Some(p) = response.hover_pos() {
                if plot.contains(p) && !overlay_rect.contains(p) {
                    fine_anchor_x = Some(p.x);
                    ui.ctx()
                        .data_mut(|d| d.insert_temp(fine_anchor_id, p.x));
                }
            }
        }
    } else if fine_anchor_x.is_some() {
        ui.ctx()
            .data_mut(|d| d.remove::<f32>(fine_anchor_id));
        fine_anchor_x = None;
    }
    let effective_x = |raw_x: f32| -> f32 {
        match (shift_held, fine_anchor_x) {
            (true, Some(ax)) => (ax + (raw_x - ax) * FINE_SCALE).clamp(plot.left(), plot.right()),
            _ => raw_x,
        }
    };

    // Lay out the anchor pills first. Each pin owns:
    //   - a full-height vertical line at `x_from_freq(f)`
    //   - a pill ~1/5 down from the plot top (higher than the old 1/3 band)
    //   - a small × hit-rect immediately right of the pill
    // Higher placement keeps titles further from loud low-mid energy and leaves
    // breathing room under the top buttons. The full-height anchor line still
    // spans the plot. Pins in adjacent columns render side-by-side so × rects
    // stay unique per anchor.
    let pill_top = plot.top() + plot.height() * (1.0 / 5.0);
    let anchor_layouts: Vec<AnchorLayout> = anchors
        .iter()
        .copied()
        .map(|f| {
            let x = x_from_freq(f);
            let label = fmt_freq(f);
            let (pill_rect, x_rect) = measure_anchor_pill(&painter, x, pill_top, &label, true);
            AnchorLayout {
                freq_hz: f,
                line_x: x,
                pill_rect,
                x_rect,
            }
        })
        .collect();

    // Hit-test: is the pointer over one of the × buttons right now?
    let hover_pos = response.hover_pos();
    let hovered_delete: Option<usize> = hover_pos.and_then(|p| {
        anchor_layouts
            .iter()
            .position(|a| a.x_rect.contains(p))
    });

    // Click handling: delete takes precedence over "add new anchor".
    // The overlay_rect check is belt-and-braces - `ui.put` / `scope_builder`
    // below normally claim clicks in the Pretty / Clear-all button rects,
    // but we still guard it in case widget stacking order ever changes.
    if response.clicked() {
        if let Some(click_pos) = response.interact_pointer_pos() {
            let delete_idx = anchor_layouts
                .iter()
                .position(|a| a.x_rect.contains(click_pos));
            if let Some(idx) = delete_idx {
                anchors.remove(idx);
            } else if plot.contains(click_pos) && !overlay_rect.contains(click_pos) {
                // `effective_x` folds in the Shift fine-control offset, so a
                // click while holding Shift commits exactly what the hover
                // line was indicating (not the raw OS cursor position).
                let new_f = freq_from_x(effective_x(click_pos.x)).clamp(f_min, f_max);
                let new_log = new_f.log10();
                let duplicate = anchors
                    .iter()
                    .any(|a| (a.log10() - new_log).abs() < ANCHOR_DEDUPE_LOG_HZ);
                if !duplicate {
                    anchors.push(new_f);
                    anchors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                }
            }
        }
    }

    // Cursor feedback: pointing-hand over a delete target, crosshair anywhere
    // else inside the plot body. Outside the plot we leave the cursor alone
    // so neighbouring widgets (scroll area, window edge) behave normally.
    if let Some(p) = hover_pos {
        if hovered_delete.is_some() {
            ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
        } else if plot.contains(p) && !overlay_rect.contains(p) {
            ui.ctx().set_cursor_icon(CursorIcon::Crosshair);
        }
    }

    // Draw anchor lines + pills. Pills go on top of the line so the label
    // stays readable even when the line crosses the trace peak behind it.
    for (i, a) in anchor_layouts.iter().enumerate() {
        painter.line_segment(
            [pos2(a.line_x, plot.top()), pos2(a.line_x, plot.bottom())],
            Stroke::new(1.0, ANCHOR_LINE_COL),
        );
        let label = fmt_freq(a.freq_hz);
        draw_anchor_pill(&painter, a, &label, hovered_delete == Some(i));
    }

    // Draw the hover cursor line. Keep it out of the checkbox footprint and
    // hide it while the cursor is over a delete button (the anchor's own pill
    // already reports the same frequency, drawing a second one on top of it
    // just adds visual noise during the split second before deletion).
    //
    // `effective_x` applies the Shift fine-control scaling so the line reads
    // as a slow-moving virtual cursor while Shift is held.
    if let Some(p) = hover_pos {
        if plot.contains(p) && !overlay_rect.contains(p) && hovered_delete.is_none() {
            let line_x = effective_x(p.x);
            painter.line_segment(
                [pos2(line_x, plot.top()), pos2(line_x, plot.bottom())],
                Stroke::new(1.0, HOVER_LINE_COL),
            );
            let label = fmt_freq(freq_from_x(line_x).clamp(f_min, f_max));
            draw_hover_pill(&painter, line_x, pill_top, &label);
        }
    }

    // Soft outer border tying it in with the rest of the UI.
    painter.rect_stroke(
        rect,
        6.0,
        Stroke::new(1.0, Color32::from_rgb(0x3a, 0x3a, 0x40)),
        egui::StrokeKind::Inside,
    );

    // Top-right button stack: "Pretty" (toggle) above "Clear all" (action).
    // Both are plain `Button`s laid out via `scope_builder` so they share
    // visual treatment (small, same frame, same hit behaviour) - the
    // previous Checkbox widget was the odd one out visually.
    //
    // The Pretty button's label is a `LayoutJob` whose characters are
    // coloured from the pretty-mode waveform palette when the mode is
    // active, giving the button itself a miniature warm→cool gradient
    // that directly echoes the trace it controls. In its off state the
    // label is just the regular `LABEL_COL` so it reads identically to
    // Clear-all until you click it on.
    let pretty_label = if *pretty_mode {
        gradient_pretty_label(11.0)
    } else {
        RichText::new("Pretty").size(11.0).color(LABEL_COL).into()
    };
    let pretty_resp = ui
        .scope_builder(
            UiBuilder::new()
                .max_rect(pretty_rect)
                .layout(Layout::centered_and_justified(Direction::TopDown)),
            |ui| ui.add(Button::new(pretty_label).small()),
        )
        .inner;
    if pretty_resp.clicked() {
        *pretty_mode = !*pretty_mode;
    }

    // "Clear all" sits below Pretty. Always present so users learn the
    // affordance exists, but disabled + greyed by egui when the anchor list
    // is empty. `add_enabled(false, ..)` gives the built-in disabled tint +
    // no-click behaviour without hand-rolling dimmed colours.
    let clear_enabled = !anchors.is_empty();
    let clear_resp = ui
        .scope_builder(
            UiBuilder::new()
                .max_rect(clear_rect)
                .layout(Layout::centered_and_justified(Direction::TopDown)),
            |ui| {
                ui.add_enabled(
                    clear_enabled,
                    Button::new(RichText::new("Clear all").size(11.0).color(LABEL_COL))
                        .small(),
                )
            },
        )
        .inner;
    if clear_resp.clicked() {
        anchors.clear();
    }

    let _ = vec2(0.0, 0.0); // keep `vec2` used.
    response
}

/// Build a `WidgetText` with per-character colouring sampled from the warm→cool
/// pretty-mode palette. Used for the "Pretty" toggle button's active-state
/// label so the control visually echoes the waveform it enables. Characters
/// are distributed linearly across the full palette range so the first
/// character always reads as the low-band red and the last as the high-band
/// violet, regardless of text length.
fn gradient_pretty_label(size: f32) -> egui::WidgetText {
    use egui::text::{LayoutJob, TextFormat};
    const TEXT: &str = "Pretty";
    let mut job = LayoutJob::default();
    let chars: Vec<char> = TEXT.chars().collect();
    let n = chars.len();
    let font = FontId::proportional(size);
    for (i, c) in chars.iter().enumerate() {
        let t = if n <= 1 {
            0.0
        } else {
            i as f32 / (n - 1) as f32
        };
        let (r, g, b) = pretty_rgb_at_t(t);
        job.append(
            &c.to_string(),
            0.0,
            TextFormat {
                font_id: font.clone(),
                color: Color32::from_rgb(r, g, b),
                ..Default::default()
            },
        );
    }
    job.into()
}

/// Sample the pretty palette at a parameterised position `t ∈ [0, 1]` with
/// piecewise-linear interpolation between the five `PRETTY_ANCHORS_RGB`
/// entries. Separate from `pretty_rgb_at_hz` because the button layout uses
/// character index, not frequency, as its x-coordinate - we want the gradient
/// evenly spread across the word rather than compressed to wherever 30 Hz /
/// 15 kHz fall on the log axis.
fn pretty_rgb_at_t(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let segs = PRETTY_ANCHORS_RGB.len() - 1;
    let scaled = t * segs as f32;
    let i = (scaled as usize).min(segs - 1);
    let lt = scaled - i as f32;
    let a = PRETTY_ANCHORS_RGB[i];
    let b = PRETTY_ANCHORS_RGB[i + 1];
    let lerp = |x: u8, y: u8| {
        (x as f32 + (y as f32 - x as f32) * lt)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    (lerp(a.0, b.0), lerp(a.1, b.1), lerp(a.2, b.2))
}

/// Format a frequency for label display. Matches the visual conventions on the
/// bottom ruler (sub-1 kHz in Hz, >= 1 kHz in kHz) but with a little more
/// precision - users pin anchors at musically meaningful values like 440 Hz or
/// 2.50 kHz and want those read back exactly, not rounded to "2k".
fn fmt_freq(hz: f32) -> String {
    if hz >= 10_000.0 {
        format!("{:.1} kHz", hz / 1000.0)
    } else if hz >= 1_000.0 {
        format!("{:.2} kHz", hz / 1000.0)
    } else if hz >= 100.0 {
        format!("{:.0} Hz", hz)
    } else {
        format!("{:.1} Hz", hz)
    }
}

const PILL_PAD_X: f32 = 4.0;
const PILL_PAD_Y: f32 = 2.0;
const PILL_FONT_PX: f32 = 10.0;
const X_BUTTON_SIZE: f32 = 11.0;
const X_BUTTON_GAP: f32 = 3.0;

/// Measure the pill + optional × rects for a frequency label so the caller can
/// hit-test them *before* deciding whether to repaint. Returns rects in screen
/// space, centered horizontally on `anchor_x`, sitting at `top` y. The × rect
/// is `Rect::NOTHING` when `with_x = false`.
fn measure_anchor_pill(
    painter: &Painter,
    anchor_x: f32,
    top: f32,
    label: &str,
    with_x: bool,
) -> (Rect, Rect) {
    let galley = painter.layout_no_wrap(
        label.to_string(),
        FontId::proportional(PILL_FONT_PX),
        ANCHOR_PILL_FG,
    );
    let text_size = galley.size();
    let x_w = if with_x { X_BUTTON_GAP + X_BUTTON_SIZE } else { 0.0 };
    let total_w = text_size.x + PILL_PAD_X * 2.0 + x_w;
    let total_h = text_size.y + PILL_PAD_Y * 2.0;
    let pill = Rect::from_min_size(
        pos2(anchor_x - total_w * 0.5, top),
        vec2(total_w, total_h),
    );
    let x_rect = if with_x {
        let x_center_x = pill.right() - PILL_PAD_X - X_BUTTON_SIZE * 0.5;
        let x_center_y = pill.center().y;
        Rect::from_center_size(
            pos2(x_center_x, x_center_y),
            vec2(X_BUTTON_SIZE, X_BUTTON_SIZE),
        )
    } else {
        Rect::NOTHING
    };
    (pill, x_rect)
}

fn draw_anchor_pill(painter: &Painter, anchor: &AnchorLayout, label: &str, x_hovered: bool) {
    painter.rect_filled(anchor.pill_rect, 3.0, ANCHOR_PILL_BG);
    let galley = painter.layout_no_wrap(
        label.to_string(),
        FontId::proportional(PILL_FONT_PX),
        ANCHOR_PILL_FG,
    );
    painter.galley(
        pos2(
            anchor.pill_rect.left() + PILL_PAD_X,
            anchor.pill_rect.top() + PILL_PAD_Y,
        ),
        galley,
        ANCHOR_PILL_FG,
    );

    // × glyph drawn as two stroke segments. Scales down a hair (`0.3`) from the
    // hit-rect so there's visible padding between the cross arms and the pill
    // edge; hover state just brightens + warms the stroke colour without
    // changing geometry so the layout doesn't shift when the user approaches.
    let c = anchor.x_rect.center();
    let arm = X_BUTTON_SIZE * 0.3;
    let col = if x_hovered { ANCHOR_X_HOVER_FG } else { ANCHOR_X_FG };
    let sw = if x_hovered { 1.4 } else { 1.2 };
    painter.line_segment(
        [pos2(c.x - arm, c.y - arm), pos2(c.x + arm, c.y + arm)],
        Stroke::new(sw, col),
    );
    painter.line_segment(
        [pos2(c.x + arm, c.y - arm), pos2(c.x - arm, c.y + arm)],
        Stroke::new(sw, col),
    );
}

fn draw_hover_pill(painter: &Painter, anchor_x: f32, top: f32, label: &str) {
    let galley = painter.layout_no_wrap(
        label.to_string(),
        FontId::proportional(PILL_FONT_PX),
        HOVER_PILL_FG,
    );
    let text_size = galley.size();
    let total_w = text_size.x + PILL_PAD_X * 2.0;
    let total_h = text_size.y + PILL_PAD_Y * 2.0;
    let pill = Rect::from_min_size(
        pos2(anchor_x - total_w * 0.5, top),
        vec2(total_w, total_h),
    );
    painter.rect_filled(pill, 3.0, HOVER_PILL_BG);
    painter.galley(
        pos2(pill.left() + PILL_PAD_X, pill.top() + PILL_PAD_Y),
        galley,
        HOVER_PILL_FG,
    );
}

struct AnchorLayout {
    freq_hz: f32,
    line_x: f32,
    pill_rect: Rect,
    x_rect: Rect,
}

/// Sample the warm→cool pretty-mode palette at `f_hz`. Interpolates linearly in log-frequency
/// between the five anchor colours so transitions between bands are smooth gradients instead
/// of hard stripes.
fn pretty_rgb_at_hz(f_hz: f32) -> (u8, u8, u8) {
    let anchors = &PRETTY_ANCHORS_HZ;
    let cols = &PRETTY_ANCHORS_RGB;
    if !f_hz.is_finite() || f_hz <= anchors[0] {
        return cols[0];
    }
    if f_hz >= anchors[4] {
        return cols[4];
    }
    for i in 0..4 {
        if f_hz >= anchors[i] && f_hz < anchors[i + 1] {
            let ln_f = f_hz.ln();
            let ln_a = anchors[i].ln();
            let ln_b = anchors[i + 1].ln();
            let t = ((ln_f - ln_a) / (ln_b - ln_a).max(1e-9)).clamp(0.0, 1.0);
            let a = cols[i];
            let b = cols[i + 1];
            let lerp8 =
                |x: u8, y: u8| (x as f32 * (1.0 - t) + y as f32 * t).round().clamp(0.0, 255.0) as u8;
            return (lerp8(a.0, b.0), lerp8(a.1, b.1), lerp8(a.2, b.2));
        }
    }
    cols[4]
}

/// Apply premultiplied alpha: `rgb` is the visible unpremultiplied colour, the returned
/// `Color32` is the correctly-premultiplied form egui needs for its gradient meshes.
#[inline]
fn premul_alpha(rgb: (u8, u8, u8), alpha: u8) -> Color32 {
    let af = alpha as f32 / 255.0;
    Color32::from_rgba_premultiplied(
        (rgb.0 as f32 * af).round() as u8,
        (rgb.1 as f32 * af).round() as u8,
        (rgb.2 as f32 * af).round() as u8,
        alpha,
    )
}

/// Uniform Catmull-Rom cubic interpolation of `bucket_vals` in log-frequency at `f_hz`.
///
/// Treats the four surrounding bucket samples as control points and evaluates the standard
/// Catmull-Rom spline (tension 0.5) on the local log-frequency parameter `t ∈ [0, 1]` between
/// the two central samples. Buckets are geometrically spaced so `t` in log-frequency is very
/// close to uniform; this is the same family of C¹-continuous cubics that ZLEqualizer's Akima
/// path produces, without the monotonicity guard (spectrum traces already average in power so
/// overshoot is visually negligible here).
///
/// Outside the bucket range the function clamps to the first/last sample.
fn bucket_catmull_db(f_hz: f32, bucket_freqs: &[f32], bucket_vals: &[f32]) -> f32 {
    let n = bucket_freqs.len().min(bucket_vals.len());
    if n == 0 {
        return SPECTRUM_FLOOR_DB;
    }
    if n == 1 {
        return bucket_vals[0];
    }
    let last = n - 1;
    if f_hz <= bucket_freqs[0] {
        return bucket_vals[0];
    }
    if f_hz >= bucket_freqs[last] {
        return bucket_vals[last];
    }
    // Binary search for `i` with bucket_freqs[i] <= f_hz < bucket_freqs[i+1].
    let mut lo = 0usize;
    let mut hi = last;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if bucket_freqs[mid] <= f_hz {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let i = lo;

    let ln1 = bucket_freqs[i].max(1e-6).ln();
    let ln2 = bucket_freqs[i + 1].max(1e-6).ln();
    let denom = (ln2 - ln1).abs().max(1e-9);
    let t = ((f_hz.max(1e-6).ln() - ln1) / denom).clamp(0.0, 1.0);

    let p1 = bucket_vals[i];
    let p2 = bucket_vals[i + 1];
    let p0 = if i > 0 { bucket_vals[i - 1] } else { p1 };
    let p3 = if i + 2 <= last {
        bucket_vals[i + 2]
    } else {
        p2
    };

    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

fn smooth_triple_1d(v: &mut [f32]) {
    if v.len() < 3 {
        return;
    }
    let src: Vec<f32> = v.to_vec();
    for i in 1..v.len() - 1 {
        v[i] = 0.2 * src[i - 1] + 0.6 * src[i] + 0.2 * src[i + 1];
    }
}

