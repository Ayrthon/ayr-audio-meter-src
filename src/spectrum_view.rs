//! Spectrum analyzer — rulers aligned with **FabFilter Pro-Q 3**’s EQ spectrum column (inset
//! 20 Hz–20 kHz, majors 20…20k, 0…−120 dB grid). The trace uses **Gaussian constant-Q** smoothing
//! in log-frequency on **linear power** (FabFilter-style silk). Pro-Q’s exact DSP is proprietary;
//! this is a deliberate **approximation**, not a licensed clone.
//!
//! Includes gradient fill, main line, peak-hold outline, and decade frequency grid.

use egui::epaint::{Mesh, Vertex, WHITE_UV};
use egui::{
    Align2, Color32, FontId, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, pos2, vec2,
};

use crate::spectrum::{
    SharedSpectrum, SpectrumAnalyzer, SPECTRUM_FLOOR_DB, spectrum_pro_q_display_db,
};

/// Pro-Q 3 on-screen **spectrum analyzer** ruler: 0 … −120 dB.
const PRO_Q_ANALYZER_DB_TOP: f32 = 0.0;
const PRO_Q_ANALYZER_DB_BOT: f32 = -120.0;

/// Log-frequency plot limits: extend a little **before** 20 Hz and **after** 20 kHz so those
/// ticks sit slightly inset (FabFilter default full-span look), without new numeric labels.
const F_DISPLAY_MIN_HZ: f32 = 17.5;
const F_DISPLAY_MAX_CAP_HZ: f32 = 22_000.0;

// Pro-Q 3–inspired panel (dark graphite, soft grid, cool fill).
const BG_COL: Color32 = Color32::from_rgb(0x2c, 0x2c, 0x2f);
const GRID_MAJOR: Color32 = Color32::from_rgba_premultiplied(0x62, 0x62, 0x68, 0x28);
const GRID_MINOR: Color32 = Color32::from_rgba_premultiplied(0x4a, 0x4a, 0x50, 0x1c);
const LABEL_COL: Color32 = Color32::from_rgb(0xb4, 0xb4, 0xbc);
const ZERO_DB_LINE: Color32 = Color32::from_rgb(0xf0, 0x7a, 0x28);

const FILL_TOP: Color32 = Color32::from_rgba_premultiplied(0x4a, 0x4e, 0x58, 0x45);
const FILL_BOT: Color32 = Color32::from_rgba_premultiplied(0x30, 0x32, 0x38, 0x00);
const LINE_COL: Color32 = Color32::from_rgba_premultiplied(0xe4, 0xe4, 0xea, 0xb8);
const HOLD_COL: Color32 = Color32::from_rgba_premultiplied(0xb0, 0xb0, 0xb8, 0x62);

/// σ in ln(Hz); ~0.11 ≈ 1/6-oct Gaussian — smooth Pro-Q-like curve without raw FFT teeth.
const GAUSS_SIGMA_LN: f32 = 0.108;

pub const MIN_WIDTH: f32 = 360.0;
pub const MIN_HEIGHT: f32 = 260.0;

pub fn show_spectrum(
    ui: &mut Ui,
    analyzer: &mut SpectrumAnalyzer,
    _shared: &SharedSpectrum,
    _now: f64,
) -> Response {
    // `analyzer` is ticked in `AudioMeterApp::ui` so the spectrum decays when audio stops even
    // if the History tab is visible.

    // Claim exactly what the container gives us — the outer window
    // already enforces a sensible minimum size (see
    // `WINDOW_W_MIN_WITH_GRAPH`), so if the user has somehow shrunk
    // beyond the recommended width we still render inside our bounds
    // instead of painting past the right edge.
    let desired = ui.available_size();
    let (rect, response) = ui.allocate_exact_size(desired, Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 6.0, BG_COL);

    // Inner plot area: compact left (freq labels below), right gutter for dB ticks like Pro-Q.
    let left_pad = 14.0;
    let right_pad = 36.0;
    let top_pad = 10.0;
    let bottom_pad = 22.0;
    let plot = Rect::from_min_max(
        pos2(rect.left() + left_pad, rect.top() + top_pad),
        pos2(rect.right() - right_pad, rect.bottom() - bottom_pad),
    );

    let nyquist = (analyzer.sample_rate * 0.5).max(1000.0);
    let f_min = F_DISPLAY_MIN_HZ;
    let f_max = nyquist.min(F_DISPLAY_MAX_CAP_HZ);
    let log_fmin = f_min.log10();
    let log_fmax = f_max.log10();

    let x_from_freq = |f: f32| -> f32 {
        let t = (f.max(f_min).log10() - log_fmin) / (log_fmax - log_fmin);
        plot.left() + t.clamp(0.0, 1.0) * plot.width()
    };

    let y_from_db = |db: f32| -> f32 {
        let d = db.clamp(PRO_Q_ANALYZER_DB_BOT, PRO_Q_ANALYZER_DB_TOP);
        let t = (PRO_Q_ANALYZER_DB_TOP - d) / (PRO_Q_ANALYZER_DB_TOP - PRO_Q_ANALYZER_DB_BOT);
        plot.top() + t * plot.height()
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
    // Major frequency lines + labels (matches Pro-Q 4 default full-range lettering).
    let major_freqs: &[f32] = &[
        20.0, 50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0,
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
    // dB grid — 10 dB steps (Pro-Q spectrum column: 0 … −120).
    for step in 0..=12 {
        let db = -(step * 10);
        let y = y_from_db(db as f32);
        if y < plot.top() || y > plot.bottom() {
            continue;
        }
        let stroke = if db == 0 {
            Stroke::new(1.25, ZERO_DB_LINE)
        } else {
            Stroke::new(1.0, GRID_MAJOR)
        };
        painter.line_segment(
            [pos2(plot.left(), y), pos2(plot.right(), y)],
            stroke,
        );
        painter.text(
            pos2(plot.right() + 3.0, y),
            Align2::LEFT_CENTER,
            format!("{db}"),
            FontId::proportional(10.0),
            LABEL_COL,
        );
    }

    // ----- spectrum path ------------------------------------------
    let bins = analyzer.bin_count();
    if bins > 2 {
        let pixel_step = 2.0_f32;
        let pixel_count = (plot.width() / pixel_step).ceil() as usize;
        if pixel_count > 4 {
            let width = plot.width();
            let log_range = log_fmax - log_fmin;
            let bin_hz = analyzer.sample_rate.max(1.0) / crate::spectrum::FFT_SIZE as f32;
            let floor_p = 10_f32.powf(SPECTRUM_FLOOR_DB / 10.0);

            let mut mags: Vec<f32> = Vec::with_capacity(pixel_count + 1);
            let mut holds: Vec<f32> = Vec::with_capacity(pixel_count + 1);

            for px in 0..=pixel_count {
                let t = px as f32 / pixel_count as f32;
                let f = 10.0_f32.powf(log_fmin + t * log_range);
                let (m, h) = gaussian_constant_q_db(
                    f,
                    bin_hz,
                    bins,
                    &analyzer.bin_power,
                    &analyzer.peak_hold_db,
                    GAUSS_SIGMA_LN,
                    floor_p,
                );
                let comp = spectrum_pro_q_display_db(f);
                mags.push((m + comp).clamp(SPECTRUM_FLOOR_DB, 12.0));
                holds.push((h + comp).clamp(SPECTRUM_FLOOR_DB, 12.0));
            }

            smooth_triple_1d(&mut mags);
            smooth_triple_1d(&mut mags);
            smooth_triple_1d(&mut holds);
            smooth_triple_1d(&mut holds);

            // Inset the path slightly from the plot left/right so 1px strokes are not half-clipped
            // at the edges (which looked like a detached “ghost” line past 20k, especially with a
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
            let pts_hold: Vec<Pos2> = holds
                .iter()
                .enumerate()
                .map(|(px, &m)| {
                    let t = px as f32 / pixel_count as f32;
                    pos2(x_at(t), y_from_db(m))
                })
                .collect();

            if pts.len() >= 2 {
                // Gradient-filled mesh under the curve.
                let mut mesh = Mesh::default();
                mesh.vertices.reserve(pts.len() * 2);
                mesh.indices.reserve((pts.len() - 1) * 6);
                let bottom_y = plot.bottom();
                for p in &pts {
                    mesh.vertices.push(Vertex {
                        pos: *p,
                        uv: WHITE_UV,
                        color: FILL_TOP,
                    });
                    mesh.vertices.push(Vertex {
                        pos: pos2(p.x, bottom_y),
                        uv: WHITE_UV,
                        color: FILL_BOT,
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

                // Single thin stroke (dual wide+halo layers caused a doubled edge on the right).
                let line_stroke = Stroke::new(1.05, LINE_COL);
                for w in pts.windows(2) {
                    painter.line_segment([w[0], w[1]], line_stroke);
                }

                // Peak-hold trail — thinner, dimmer.
                let hold_stroke = Stroke::new(0.75, HOLD_COL);
                for w in pts_hold.windows(2) {
                    painter.line_segment([w[0], w[1]], hold_stroke);
                }
            }
        }
    }

    // Soft outer border tying it in with the rest of the UI.
    painter.rect_stroke(
        rect,
        6.0,
        Stroke::new(1.0, Color32::from_rgb(0x3a, 0x3a, 0x40)),
        egui::StrokeKind::Inside,
    );

    let _ = vec2(0.0, 0.0); // keep `vec2` used.
    response
}

/// Gaussian band in log-frequency on **linear power** (main trace) and on power derived from
/// peak-hold dB (outline), matching how polished RTAs avoid jagged bin-to-bin dB averaging.
fn gaussian_constant_q_db(
    f_center: f32,
    bin_hz: f32,
    bins: usize,
    bin_power: &[f32],
    peak_db: &[f32],
    sigma_ln: f32,
    floor_p: f32,
) -> (f32, f32) {
    let ln_fc = f_center.max(22.0).ln();
    let f_lo = (ln_fc - 4.2 * sigma_ln).exp();
    let f_hi = (ln_fc + 4.2 * sigma_ln).exp();
    let i0 = (f_lo / bin_hz).floor() as isize;
    let i1 = (f_hi / bin_hz).ceil() as isize;
    let i0 = i0.max(1) as usize;
    let i1 = (i1 as usize).min(bins.saturating_sub(1));

    let mut sum_wp = 0.0_f32;
    let mut sum_wph = 0.0_f32;
    let mut sum_w = 0.0_f32;

    for i in i0..=i1 {
        let fi = i as f32 * bin_hz;
        if fi < 18.0 {
            continue;
        }
        let ln_f = fi.ln();
        let u = (ln_f - ln_fc) / sigma_ln;
        let w = (-0.5 * u * u).exp();
        if i < bin_power.len() {
            sum_wp += w * bin_power[i];
        }
        let ph = if i < peak_db.len() {
            10_f32.powf(peak_db[i].clamp(SPECTRUM_FLOOR_DB, 12.0) / 10.0)
        } else {
            floor_p
        };
        sum_wph += w * ph;
        sum_w += w;
    }

    if sum_w < 1e-20 {
        return (SPECTRUM_FLOOR_DB, SPECTRUM_FLOOR_DB);
    }

    let m = 10.0 * (sum_wp / sum_w).max(floor_p).log10();
    let h = 10.0 * (sum_wph / sum_w).max(floor_p).log10();
    (
        m.clamp(SPECTRUM_FLOOR_DB, 12.0),
        h.clamp(SPECTRUM_FLOOR_DB, 12.0),
    )
}

fn smooth_triple_1d(v: &mut [f32]) {
    if v.len() < 3 {
        return;
    }
    let src: Vec<f32> = v.to_vec();
    for i in 1..v.len() - 1 {
        v[i] = 0.22 * src[i - 1] + 0.56 * src[i] + 0.22 * src[i + 1];
    }
}
