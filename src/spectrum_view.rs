//! Spectrum analyzer â€” rulers aligned with **FabFilter Pro-Q 3**â€™s EQ spectrum column (inset
//! 20 Hzâ€“20 kHz, majors 20â€¦20k, 0â€¦âˆ’120 dB grid). The trace uses **Gaussian** smoothing in log-frequency
//! on **linear power** with **frequency-dependent width** (smooth bass, increasingly defined / spikey
//! treble) plus an Hz-based blend toward raw columns after horizontal smooth. Pro-Qâ€™s exact DSP is
//! proprietary; this is a deliberate **approximation**, not a licensed clone.
//!
//! Includes gradient fill, main line, peak-hold outline, and decade frequency grid.

use egui::epaint::{Mesh, Vertex, WHITE_UV};
use egui::{
    Align2, Color32, FontId, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, pos2, vec2,
};

use crate::spectrum::{
    SharedSpectrum, SpectrumAnalyzer, SPECTRUM_FLOOR_DB, spectrum_pro_q_display_db,
};

/// Pro-Q 3 analyzer vertical: 0 dB (top) â€¦ âˆ’90 dB (bottom).
///
/// The axis is **piecewise-linear**: the top 0 â€¦ âˆ’12 dB band is zoomed (3 dB per major grid) and
/// the remainder compresses from âˆ’12 dB down to âˆ’90 dB, matching Pro-Q 3â€™s ruler.
const PRO_Q_ANALYZER_DB_TOP: f32 = 0.0;
const PRO_Q_ANALYZER_DB_BOT: f32 = -90.0;

/// Normalised vertical break points for the piecewise-linear dB axis.
/// `t = 0` is the very top of the plot (above 0 dB), `t = 1` is the very bottom (below âˆ’90 dB).
/// Pro-Q 3 leaves small gaps above 0 dB and below âˆ’90 dB; we mirror those here.
const PRO_Q_T_TOP_PAD: f32 = 0.03; // 0 dB sits this fraction below the very top
const PRO_Q_T_MINUS12: f32 = 0.33; // âˆ’12 dB ~ one third from the top (zoomed EQ band)
const PRO_Q_T_MINUS90: f32 = 0.95; // âˆ’90 dB ~ 5 % above the very bottom

#[inline]
fn pro_q_t_from_db(db: f32) -> f32 {
    // Clamp against a slightly wider range than the labelled ticks so the trace can extend into
    // the top and bottom padding bands for Pro-Qâ€™s tapered look past 0 dB / âˆ’90 dB.
    let d = db.clamp(-120.0, 6.0);
    if d >= 0.0 {
        // Above 0 dB: linear into the top padding band (saturates at +6 dB = t=0).
        let u = ((6.0 - d) / 6.0).clamp(0.0, 1.0);
        u * PRO_Q_T_TOP_PAD
    } else if d >= -12.0 {
        // 0 â€¦ âˆ’12 dB, uniform 3 dB/grid (zoomed EQ band).
        let u = (0.0 - d) / 12.0;
        PRO_Q_T_TOP_PAD + u * (PRO_Q_T_MINUS12 - PRO_Q_T_TOP_PAD)
    } else if d >= -90.0 {
        // âˆ’12 â€¦ âˆ’90 dB, compressed (all the low-level detail lives here).
        let u = (-12.0 - d) / 78.0;
        PRO_Q_T_MINUS12 + u * (PRO_Q_T_MINUS90 - PRO_Q_T_MINUS12)
    } else {
        // Below âˆ’90 dB: linear into the bottom padding band, saturating at âˆ’120 dB.
        let u = ((-90.0 - d) / 30.0).clamp(0.0, 1.0);
        PRO_Q_T_MINUS90 + u * (1.0 - PRO_Q_T_MINUS90)
    }
}

/// Log-frequency plot limits. Extended down to 8 Hz so the 10 Hz ruler tick is clearly inset â€”
/// bin 1 at 48 kHz / 8192 samples sits at 5.86 Hz, so everything above 8 Hz is real FFT data.
const F_DISPLAY_MIN_HZ: f32 = 8.0;
const F_DISPLAY_MAX_CAP_HZ: f32 = 29_000.0;

// Pro-Q 3â€“inspired panel (dark graphite, soft grid, cool fill).
const BG_COL: Color32 = Color32::from_rgb(0x2c, 0x2c, 0x2f);
const GRID_MAJOR: Color32 = Color32::from_rgba_premultiplied(0x62, 0x62, 0x68, 0x28);
const GRID_MINOR: Color32 = Color32::from_rgba_premultiplied(0x4a, 0x4a, 0x50, 0x1c);
const LABEL_COL: Color32 = Color32::from_rgb(0xb4, 0xb4, 0xbc);

const FILL_TOP: Color32 = Color32::from_rgba_premultiplied(0x4a, 0x4e, 0x58, 0x45);
const FILL_BOT: Color32 = Color32::from_rgba_premultiplied(0x30, 0x32, 0x38, 0x00);
const LINE_COL: Color32 = Color32::from_rgba_premultiplied(0xe4, 0xe4, 0xea, 0xb8);
const HOLD_COL: Color32 = Color32::from_rgba_premultiplied(0xb0, 0xb0, 0xb8, 0x62);

/// Mid-band Ïƒ reference in ln(Hz); [`gauss_sigma_ln_for_freq`] widens toward bass, narrows toward treble.
const GAUSS_SIGMA_LN_REF: f32 = 0.025;

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

    // Claim exactly what the container gives us â€” the outer window
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

    // Plot the full labelled span even if the sample-rate Nyquist (e.g. 24 kHz @ 48 kHz) is below
    // `F_DISPLAY_MAX_CAP_HZ`: frequencies above Nyquist have no FFT data so the trace naturally
    // tapers into the spectrum floor, producing Pro-Qâ€™s visible right-side padding past 20 kHz.
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
    // dB grid â€” Pro-Q 3 ruler: zoomed Â±12 dB EQ band at 3 dB/grid, then compressed below.
    // Negative values are drawn without an explicit sign; positive values keep the leading '+'.
    let db_ticks: &[(f32, &str)] = &[
        (0.0, "0"),
        (-3.0, "-3"),
        (-6.0, "-6"),
        (-9.0, "-9"),
        (-12.0, "-12"),
        (-30.0, "-30"),
        (-40.0, "-40"),
        (-50.0, "-50"),
        (-60.0, "-60"),
        (-90.0, "-90"),
    ];
    for &(db, label) in db_ticks {
        let y = y_from_db(db);
        if y < plot.top() - 0.5 || y > plot.bottom() + 0.5 {
            continue;
        }
        let stroke = Stroke::new(1.0, GRID_MAJOR);
        painter.line_segment(
            [pos2(plot.left(), y), pos2(plot.right(), y)],
            stroke,
        );
        painter.text(
            pos2(plot.right() + 3.0, y),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(10.0),
            LABEL_COL,
        );
    }

    // ----- spectrum path ------------------------------------------
    let bins = analyzer.bin_count();
    if bins > 2 {
        let pixel_step = 1.0_f32;
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
                // Adaptive minimum Ïƒ (in ln-Hz) that guarantees the Gaussian's Â±2Ïƒ main lobe always
                // spans â‰¥ ~1.5 bins in Hz terms. Below ~40 Hz the FFT has too few bins per octave
                // for the static bass Ïƒ to produce a smooth curve â€” without this floor you get
                // stair-steps where each pixel reads mostly one bin. Mids/highs are unaffected
                // because `gauss_sigma_ln_for_freq` already returns Ïƒ wider than this minimum.
                let min_sigma_ln = (0.75 * bin_hz / f.max(bin_hz)).min(0.9);
                let sigma_ln = gauss_sigma_ln_for_freq(f).max(min_sigma_ln);
                let (m, h) = gaussian_constant_q_db(
                    f,
                    bin_hz,
                    bins,
                    &analyzer.bin_power,
                    &analyzer.peak_hold_db,
                    sigma_ln,
                    floor_p,
                );

                let comp = spectrum_pro_q_display_db(f);
                mags.push((m + comp).clamp(SPECTRUM_FLOOR_DB, 12.0));
                holds.push((h + comp).clamp(SPECTRUM_FLOOR_DB, 12.0));
            }

            let raw_mags = mags.clone();
            let raw_holds = holds.clone();
            smooth_triple_1d(&mut mags);
            smooth_triple_1d(&mut holds);
            blend_detail_hz(&mut mags, &raw_mags, log_fmin, log_fmax);
            blend_detail_hz(&mut holds, &raw_holds, log_fmin, log_fmax);

            // Inset the path slightly from the plot left/right so 1px strokes are not half-clipped
            // at the edges (which looked like a detached â€œghostâ€ line past 20k, especially with a
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

                // Peak-hold trail â€” thinner, dimmer.
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

/// Wider kernel (smooth lows) â†’ narrower kernel (defined highs), uniform in log-frequency.
/// Curve is super-linear in `u` so Ïƒ drops fast through the low-mids (200 Hz â€“ 1 kHz),
/// matching Pro-Qâ€™s behaviour where individual partials start resolving well before 1 kHz.
#[inline]
fn gauss_sigma_ln_for_freq(f_hz: f32) -> f32 {
    const F_LO: f32 = 8.0;
    const F_HI: f32 = 22_000.0;
    let f = f_hz.clamp(F_LO, F_HI);
    let u = (f / F_LO).log10() / (F_HI / F_LO).log10();
    let u = u.clamp(0.0, 1.0);
    // `u.powf(0.35)` rises VERY fast early so the kernel tightens sharply through the low-mids.
    // Bass scale (5.5Ã—) keeps the sub-bass hump smooth by blurring across the FFT's coarse low bins
    // (~11.7 Hz @ 48 kHz / 4096). The Ïƒ floor is tight (0.004) so the 8â€“20 kHz band shows very
    // defined individual partials â€” matching Pro-Qâ€™s crisp treble spikes.
    let t = u.powf(0.35);
    let scale = 5.5 - 5.3 * t;
    (GAUSS_SIGMA_LN_REF * scale).clamp(0.004, 0.22)
}

/// 0 = fully smoothed (deep sub), 1 = mostly raw columns. Ramps in across the **low-mids**
/// (~70 Hz â†’ 2.5 kHz) so partials start resolving around 200-300 Hz and mids/highs are
/// fully detailed â€” matching Pro-Qâ€™s RTA where lows are a smooth hump but 500 Hz already
/// shows individual peaks.
#[inline]
fn detail_blend_for_freq(f_hz: f32) -> f32 {
    const F0: f32 = 70.0;
    const F1: f32 = 2_500.0;
    if f_hz <= F0 {
        return 0.0;
    }
    if f_hz >= F1 {
        return 1.0;
    }
    let u = (f_hz / F0).ln() / (F1 / F0).ln();
    let t = u.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Mix pre-smooth column values back in toward high frequencies so the trace **opens up** into spikes.
fn blend_detail_hz(out: &mut [f32], raw: &[f32], log_fmin: f32, log_fmax: f32) {
    let n = out.len();
    if n == 0 || raw.len() != n {
        return;
    }
    let lr = (log_fmax - log_fmin).max(1e-6);
    let nm1 = (n.saturating_sub(1)).max(1) as f32;
    for i in 0..n {
        let t = i as f32 / nm1;
        let f = 10.0_f32.powf(log_fmin + t * lr);
        let k = detail_blend_for_freq(f).min(1.0);
        out[i] = out[i] * (1.0 - k) + raw[i] * k;
    }
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
    // Clamp the kernel center to just above DC so the sub-20 Hz display pixels follow their own
    // log-frequency position instead of collapsing onto a fixed 22 Hz center (which used to
    // produce a flat plateau between 8 Hz and the FFTâ€™s first usable bin).
    let ln_fc = f_center.max(bin_hz.max(1.0)).ln();
    let f_lo = (ln_fc - 4.2 * sigma_ln).exp();
    let f_hi = (ln_fc + 4.2 * sigma_ln).exp();
    let i0 = (f_lo / bin_hz).floor() as isize;
    let i1 = (f_hi / bin_hz).ceil() as isize;
    // Bin 0 is DC and is force-floored inside the analyzer; start at bin 1 (â‰ˆ 5.86 Hz at 48 kHz /
    // 8192). Do NOT add a Hz-based lower cutoff â€” we want real FFT content from 6 Hz up so that
    // sub-20 Hz transients (kick tails, room rumble) roll smoothly into the 20 Hz region.
    let i0 = i0.max(1) as usize;
    let i1 = (i1 as usize).min(bins.saturating_sub(1));

    let mut sum_wp = 0.0_f32;
    let mut sum_wph = 0.0_f32;
    let mut sum_w = 0.0_f32;

    for i in i0..=i1 {
        let fi = i as f32 * bin_hz;
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
        v[i] = 0.2 * src[i - 1] + 0.6 * src[i] + 0.2 * src[i + 1];
    }
}
