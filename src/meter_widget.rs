//! Vertical level meter widget drawn with egui's painter.

use egui::{Color32, FontId, Painter, Pos2, Rect, Response, Sense, Stroke, Ui, Vec2, pos2, vec2};

use crate::dsp::INF_FLOOR_DB;
use crate::meter::{
    ColorPalette, FillSource, MeterConfig, MeterMode, MeterScale, MeterScaleKind, approach, config_for,
};

pub use crate::meter::{MAX_DB, MIN_DB};
pub const CLIP_DB: f32 = 0.0;

pub const COL_BG: Color32 = Color32::from_rgb(0x14, 0x14, 0x18);
/// dB ruler numerals (left / right of the bar).
pub const COL_RULER_TEXT: Color32 = Color32::from_rgb(0xa3, 0xa3, 0xa7);
pub const COL_TRACK: Color32 = Color32::from_rgb(0x0f, 0x0f, 0x12);
pub const COL_BORDER: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x42);
pub const COL_TICK: Color32 = Color32::from_rgb(0x6b, 0x6b, 0x78);
/// Legacy name — meter ruler ticks use [`COL_RULER_TEXT`] (`#a3a3a7`).
pub const COL_TICK_MAJOR: Color32 = COL_RULER_TEXT;
pub const COL_TEXT: Color32 = Color32::from_rgb(0xe6, 0xe6, 0xee);
pub const COL_TEXT_DIM: Color32 = Color32::from_rgb(0x8a, 0x8a, 0x94);
pub const COL_OBS_TICK_MAJOR: Color32 = COL_RULER_TEXT;
// RMS / "pressure" line — drawn inside the filled bar to show the
// averaged energy level below the peak.  Near-black with high alpha
// so it reads clearly against bright green / orange / olive fills.
pub const COL_RMS_LINE: Color32 = Color32::from_rgba_premultiplied(0x08, 0x0a, 0x0e, 0xe0);

pub const COL_HOLD: Color32 = Color32::WHITE;
pub const COL_CLIP_OFF: Color32 = Color32::from_rgb(0x33, 0x14, 0x1a);
pub const COL_CLIP_ON: Color32 = Color32::from_rgb(0xff, 0x3b, 0x2e);
pub const COL_REFERENCE: Color32 = Color32::from_rgb(0xff, 0xcc, 0x33);
/// OBS Yami theme: dashed alignment line at -20 dBFS should read as a warm
/// mustard dash (not the generic Pro Tools amber used by `COL_REFERENCE`).
pub const COL_OBS_REFERENCE: Color32 = Color32::from_rgb(0xe5, 0xaf, 0x24);
pub const COL_READOUT_BG: Color32 = Color32::from_rgb(0x0c, 0x0c, 0x0f);
pub const COL_READOUT_BORDER: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x33);
pub const COL_READOUT_TEXT: Color32 = Color32::from_rgb(0xd6, 0xd6, 0xdf);
pub const COL_READOUT_MAX: Color32 = Color32::from_rgb(0xff, 0xdd, 0x55);
pub const COL_MODE_TEXT: Color32 = Color32::from_rgb(0x7f, 0x7f, 0x8a);

pub const METER_GREEN: Color32 = Color32::from_rgb(0x2f, 0xd6, 0x57);
pub const METER_YELLOW: Color32 = Color32::from_rgb(0xff, 0xd6, 0x33);
pub const METER_RED: Color32 = Color32::from_rgb(0xff, 0x3b, 0x2e);

pub const METER_GREEN_BG: Color32 = Color32::from_rgb(0x0d, 0x2a, 0x17);
pub const METER_YELLOW_BG: Color32 = Color32::from_rgb(0x2d, 0x27, 0x10);
pub const METER_RED_BG: Color32 = Color32::from_rgb(0x2a, 0x11, 0x0d);

// Pro Tools palette — dim forest green for the "low" band, bright
// lime for the "mid" band, and orange for the "high" band.  Matches
// the Avid Pro Tools 2024/2025 Sample Peak look where the low band is
// a same-hue dimmer green (not olive) running from ~-60 to -20 dBFS.
pub const PT_LOW:       Color32 = Color32::from_rgb(0x1c, 0xa3, 0x3b);
pub const PT_LOW_BG:    Color32 = Color32::from_rgb(0x08, 0x1d, 0x10);
pub const PT_ORANGE:    Color32 = Color32::from_rgb(0xf7, 0x8a, 0x12);
pub const PT_ORANGE_BG: Color32 = Color32::from_rgb(0x2e, 0x1c, 0x08);

// Standard palette "low" band (dimmer green).  Only drawn when a
// mode opts in; stays invisible otherwise.
pub const STD_DIM_GREEN:    Color32 = Color32::from_rgb(0x1b, 0x97, 0x3c);
pub const STD_DIM_GREEN_BG: Color32 = Color32::from_rgb(0x07, 0x1e, 0x0e);

// OBS mode uses the same meter band fills and dim tracks as [`ColorPalette::Standard`]
// so green / yellow / red match the other palettes.

#[derive(Clone, Copy)]
pub struct ZoneStyle {
    pub fill: Color32,
    pub bg: Color32,
}

/// 4 bands per palette, ordered bottom→top:
/// `low` → `mid` → `high` → `over`.
pub fn zone_styles(palette: ColorPalette) -> [ZoneStyle; 4] {
    match palette {
        ColorPalette::Obs => [
            // `zone_low_top_db` is MIN_DB in OBS mode, so this first slot
            // collapses to zero height; keep it defined anyway for consistency.
            ZoneStyle { fill: STD_DIM_GREEN, bg: STD_DIM_GREEN_BG },
            ZoneStyle { fill: METER_GREEN,   bg: METER_GREEN_BG },
            ZoneStyle { fill: METER_YELLOW,  bg: METER_YELLOW_BG },
            ZoneStyle { fill: METER_RED,     bg: METER_RED_BG },
        ],
        ColorPalette::ProTools => [
            ZoneStyle { fill: PT_LOW,      bg: PT_LOW_BG },
            ZoneStyle { fill: METER_GREEN, bg: METER_GREEN_BG },
            ZoneStyle { fill: PT_ORANGE,   bg: PT_ORANGE_BG },
            ZoneStyle { fill: METER_RED,   bg: METER_RED_BG },
        ],
        ColorPalette::Standard => [
            ZoneStyle { fill: STD_DIM_GREEN, bg: STD_DIM_GREEN_BG },
            ZoneStyle { fill: METER_GREEN,   bg: METER_GREEN_BG },
            ZoneStyle { fill: METER_YELLOW,  bg: METER_YELLOW_BG },
            ZoneStyle { fill: METER_RED,     bg: METER_RED_BG },
        ],
    }
}

/// Monotonically-increasing zone tops (dBFS), bottom → top:
/// `[low_top, mid_top, high_top, over_top=0]`.
pub fn zone_tops_db(cfg: &MeterConfig) -> [f32; 4] {
    let low = cfg.zone_low_top_db.clamp(MIN_DB, MAX_DB);
    let mid = cfg.zone_mid_top_db.clamp(low, MAX_DB);
    let high = cfg.zone_high_top_db.clamp(mid, MAX_DB);
    [low, mid, high, MAX_DB]
}

/// Peak-hold segment colour for OBS-style meters. Mirrors the logic in
/// `obs-studio/frontend/components/VolumeMeter.cpp` (`getPeakColor` /
/// `paintEvent`): the short horizontal hold bar takes the **foreground**
/// colour of whichever band its dB level currently sits in, not a flat white.
#[inline]
fn obs_peak_hold_color(hold_db: f32, tops_db: &[f32; 4], styles: &[ZoneStyle; 4]) -> Color32 {
    const MIN_INPUT_DB: f32 = -50.0;
    if hold_db < MIN_INPUT_DB {
        return styles[1].bg;
    }
    if hold_db >= tops_db[2] {
        styles[3].fill
    } else if hold_db >= tops_db[1] {
        styles[2].fill
    } else {
        styles[1].fill
    }
}

pub const BAR_TOP_PAD: f32 = 26.0;
/// Bottom padding when a channel label is shown. Must cover:
///   live strip + gap + readout + gap + label.
pub const BAR_BOTTOM_PAD_WITH_LABEL: f32 = 58.0;
/// Bottom padding when no channel label (still need readout + strip).
pub const BAR_BOTTOM_PAD_NO_LABEL: f32 = 40.0;

/// Same bottom inset as [`show_meter`] uses from `has_label = !args.label.is_empty()`.
#[inline]
pub fn meter_bar_bottom_pad_from_label_nonempty(has_label: bool) -> f32 {
    if has_label {
        BAR_BOTTOM_PAD_WITH_LABEL
    } else {
        BAR_BOTTOM_PAD_NO_LABEL
    }
}

/// Bottom inset for a strip with `channels` meters, matching each [`show_meter`] call
/// that uses [`channel_label`]`(channels, i)` for `i in 0..channels`.
pub fn meter_bar_bottom_pad_for_strip(channels: usize) -> f32 {
    if channels == 0 {
        return BAR_BOTTOM_PAD_WITH_LABEL;
    }
    let any_labeled = (0..channels).any(|i| !channel_label(channels, i).is_empty());
    meter_bar_bottom_pad_from_label_nonempty(any_labeled)
}
/// Thin status strip beneath the bar (OBS-style input indicator concept).
/// Its colour tracks the currently "active" band in real time.
pub const BAR_INPUT_INDICATOR_H: f32 = 6.0;
pub const BAR_INPUT_INDICATOR_GAP: f32 = 3.0;
pub const CLIP_H: f32 = 9.0;
pub const MODE_H: f32 = 14.0;
pub const READOUT_H: f32 = 22.0;
pub const LABEL_H: f32 = 18.0;
pub const READOUT_GAP: f32 = 2.0;

pub const METER_WIDTH: f32 = 92.0;
pub const METER_MIN_HEIGHT: f32 = 300.0;

/// Horizontal inset before the first channel meter (inside the strip `Ui`).
pub const METER_STRIP_MARGIN_START: f32 = 2.0;
/// Space between adjacent channel meters (L/R). The strip forces horizontal
/// `item_spacing` to **0**, so this is the only inter-meter gap.
pub const METER_STRIP_GAP_X: f32 = 2.0;
/// Space between the last meter and the reference column (hairline gap).
pub const METER_STRIP_GAP_TO_REF: f32 = 2.0;
/// Width passed to [`show_reference_label`]; must stay in sync with its
/// `allocate_exact_size` width.
pub const METER_STRIP_REF_COLUMN_W: f32 = 54.0;
/// Trailing margin after the reference column.
pub const METER_STRIP_MARGIN_END: f32 = 2.0;

/// Total width for `channels` meters plus reference column and strip margins.
#[inline]
pub fn meter_strip_content_width(channels: usize) -> f32 {
    let n = channels.max(1) as f32;
    let inter = (channels.max(1).saturating_sub(1) as f32) * METER_STRIP_GAP_X;
    METER_STRIP_MARGIN_START
        + n * METER_WIDTH
        + inter
        + METER_STRIP_GAP_TO_REF
        + METER_STRIP_REF_COLUMN_W
        + METER_STRIP_MARGIN_END
}

// Typography — sized so readouts and scale labels stay legible.
pub const FONT_TICK: f32 = 11.75;
pub const FONT_MODE: f32 = 10.5;
pub const FONT_READOUT: f32 = 12.0;
pub const FONT_CHANNEL: f32 = 12.0;
pub const FONT_REF_LABEL: f32 = 11.0;

/// Retained state for one channel meter. Ballistics + peak-hold +
/// clip + max-peak.
pub struct MeterState {
    pub peak_db: f32,
    pub rms_db: f32,
    pub target_peak: f32,
    pub target_rms: f32,
    pub hold_db: f32,
    pub max_peak_db: f32,
    pub hold_set_at: f64,
    pub clip_until: f64,
    pub last_tick: f64,
}

impl Default for MeterState {
    fn default() -> Self {
        Self {
            peak_db: INF_FLOOR_DB,
            rms_db: INF_FLOOR_DB,
            target_peak: INF_FLOOR_DB,
            target_rms: INF_FLOOR_DB,
            hold_db: INF_FLOOR_DB,
            max_peak_db: INF_FLOOR_DB,
            hold_set_at: 0.0,
            clip_until: 0.0,
            last_tick: 0.0,
        }
    }
}

impl MeterState {
    pub fn set_target(&mut self, peak_db: f32, rms_db: f32) {
        self.target_peak = peak_db.clamp(INF_FLOOR_DB, 6.0);
        self.target_rms = rms_db.clamp(INF_FLOOR_DB, 6.0);
    }

    pub fn clear(&mut self) {
        self.hold_db = INF_FLOOR_DB;
        self.max_peak_db = INF_FLOOR_DB;
        self.clip_until = 0.0;
    }

    pub fn tick(&mut self, cfg: &MeterConfig, now: f64) {
        let dt = if self.last_tick <= 0.0 {
            0.016
        } else {
            (now - self.last_tick).clamp(0.0, 0.2) as f32
        };
        self.last_tick = now;

        // Peak with instant attack, controlled release.
        if self.target_peak >= self.peak_db {
            // Fast exponential attack
            self.peak_db = approach(
                self.peak_db,
                self.target_peak,
                dt,
                cfg.peak_attack_tau,
                cfg.peak_attack_tau,
            );
        } else {
            // Linear release
            self.peak_db -= cfg.peak_release_db_per_sec * dt;
            if self.peak_db < self.target_peak {
                self.peak_db = self.target_peak;
            }
        }
        if self.peak_db < INF_FLOOR_DB {
            self.peak_db = INF_FLOOR_DB;
        }

        if let Some(tau) = cfg.rms_linear_blend_tau {
            // OBS `VolumeMeter::calculateBallisticsForChannel` magnitude path:
            // `attack = (current - display) * (dt / magnitudeIntegrationTime) * 0.99`
            // then clamp to [-60, 0] dB on the widget side.
            let step = (dt / tau) * 0.99;
            self.rms_db += (self.target_rms - self.rms_db) * step;
            self.rms_db = self.rms_db.clamp(MIN_DB, MAX_DB);
        } else {
            self.rms_db = approach(
                self.rms_db,
                self.target_rms,
                dt,
                cfg.rms_attack_tau,
                cfg.rms_release_tau,
            );
        }
        if self.rms_db < INF_FLOOR_DB {
            self.rms_db = INF_FLOOR_DB;
        }

        // Peak-hold sticks to whichever source the meter is filling
        // from, so it always lines up with the top of the coloured
        // fill (important for RMS-based modes like VU / K-System).
        let hold_source = match cfg.fill_source {
            FillSource::Peak => self.peak_db,
            FillSource::Rms => self.rms_db,
        };
        let hold_target = match cfg.fill_source {
            FillSource::Peak => self.target_peak,
            FillSource::Rms => self.target_rms,
        };
        if hold_source >= self.hold_db {
            self.hold_db = hold_source;
            self.hold_set_at = now;
        } else if (now - self.hold_set_at) as f32 > cfg.hold_time {
            if cfg.hold_snap_after_timeout {
                self.hold_db = hold_source;
                self.hold_set_at = now;
            } else {
                self.hold_db -= cfg.hold_release_db_per_sec * dt;
                if self.hold_db < hold_source {
                    self.hold_db = hold_source;
                }
            }
        }
        if self.hold_db < INF_FLOOR_DB {
            self.hold_db = INF_FLOOR_DB;
        }

        if hold_target > self.max_peak_db {
            self.max_peak_db = hold_target;
        }

        if self.target_peak >= 0.0 {
            self.clip_until = now + 1.5;
        }
    }

    pub fn is_clipping(&self, now: f64) -> bool {
        self.clip_until > now
    }
}

pub struct MeterPaintArgs<'a> {
    pub label: &'a str,
    pub mode: MeterMode,
    pub state: &'a MeterState,
}

pub fn show_meter(ui: &mut Ui, args: MeterPaintArgs<'_>, now: f64) -> Response {
    let available = ui.available_height().max(METER_MIN_HEIGHT);
    let desired = vec2(METER_WIDTH, available);
    let (rect, response) = ui.allocate_exact_size(desired, Sense::click());
    let painter = ui.painter_at(rect);

    let cfg = config_for(args.mode);
    let scale = MeterScale::for_mode(args.mode);
    let w = rect.width();

    let scale_w = 22.0_f32;
    let bar_x = rect.left() + scale_w;
    let bar_w = (w - scale_w - scale_w).max(1.0);
    let bottom_pad = meter_bar_bottom_pad_from_label_nonempty(!args.label.is_empty());
    let bar_top = rect.top() + BAR_TOP_PAD;
    // `bar_bottom` is the **physical bottom edge of the meter bar** — the
    // dB scale maps linearly between `bar_top` (0 dBFS) and `bar_bottom`
    // (MIN_DB). The live indicator strip + readout live *below* this line in
    // `bottom_pad` space; shrinking `bar_bottom` to make room for them was
    // what broke tick alignment against the coloured bar.
    let bar_bottom = rect.bottom() - bottom_pad;

    // No full-rect panel fill — meters sit directly on the app backdrop.

    // --- clip indicator ------------------------------------------
    let clip_rect = Rect::from_min_size(
        pos2(bar_x, rect.top() + 1.0),
        vec2(bar_w, CLIP_H),
    );
    let clip_color = if args.state.is_clipping(now) {
        COL_CLIP_ON
    } else {
        COL_CLIP_OFF
    };
    painter.rect_filled(clip_rect, 2.0, clip_color);

    // --- mode label above bar ------------------------------------
    painter.text(
        pos2(bar_x + bar_w * 0.5, rect.top() + CLIP_H + 2.0),
        egui::Align2::CENTER_TOP,
        args.mode.short_label(),
        FontId::proportional(FONT_MODE),
        COL_MODE_TEXT,
    );

    // --- zone backgrounds ---------------------------------------
    let styles = zone_styles(cfg.palette);
    let tops = zone_tops_db(&cfg);
    paint_zone_backgrounds(
        &painter,
        bar_x,
        bar_w,
        bar_top,
        bar_bottom,
        &tops,
        &styles,
        &scale,
    );

    // --- foreground fill ----------------------------------------
    let fill_db = match cfg.fill_source {
        FillSource::Peak => args.state.peak_db,
        FillSource::Rms => args.state.rms_db,
    };
    if fill_db > INF_FLOOR_DB + 0.5 {
        let fill_y = scale.dbfs_to_y(fill_db, bar_top, bar_bottom);
        paint_zone_fill(
            &painter,
            bar_x,
            bar_w,
            bar_top,
            bar_bottom,
            fill_y,
            &tops,
            &styles,
            &scale,
        );
    }

    // --- bottom live indicator strip ----------------------------
    // Small always-visible strip directly below the bar, coloured by the
    // current meter level's active zone. Mirrors the "tiny always-on colour
    // piece" users expect from OBS-like meters and gives us extra breathing
    // room for the bottom ruler labels.
    let indicator_col = if fill_db < -50.0 {
        // Very quiet/silent -> show the zone background, not lit fill.
        styles[1].bg
    } else if fill_db >= tops[2] {
        styles[3].fill
    } else if fill_db >= tops[1] {
        styles[2].fill
    } else {
        styles[1].fill
    };
    let indicator_rect = Rect::from_min_max(
        pos2(bar_x, bar_bottom + BAR_INPUT_INDICATOR_GAP),
        pos2(
            bar_x + bar_w,
            bar_bottom + BAR_INPUT_INDICATOR_GAP + BAR_INPUT_INDICATOR_H,
        ),
    );
    painter.rect_filled(indicator_rect, 0.5, indicator_col);

    // --- reference dashed line ----------------------------------
    if cfg.reference_db > MIN_DB && cfg.reference_db <= 0.0 {
        let ref_y = scale.dbfs_to_y(cfg.reference_db, bar_top, bar_bottom);
        let ref_col = if cfg.palette == ColorPalette::Obs {
            COL_OBS_REFERENCE
        } else {
            COL_REFERENCE
        };
        draw_dashed_h(&painter, bar_x, bar_x + bar_w, ref_y, ref_col, 4.0, 3.0, 1.1);
    }

    // --- RMS "pressure" line (inside the fill) -------------------
    if cfg.show_rms_line
        && cfg.fill_source == FillSource::Peak
        && args.state.rms_db > INF_FLOOR_DB + 3.0
    {
        let rms_y = scale.dbfs_to_y(args.state.rms_db, bar_top, bar_bottom);
        let fill_y = scale.dbfs_to_y(args.state.peak_db, bar_top, bar_bottom);
        if rms_y > fill_y + 1.0 && rms_y < bar_bottom {
            let inset = 2.0_f32.min(bar_w * 0.12);
            painter.hline(
                (bar_x + inset)..=(bar_x + bar_w - inset),
                rms_y,
                Stroke::new(2.0, COL_RMS_LINE),
            );
        }
    }

    // --- peak-hold --------------------------------------------
    if args.state.hold_db > INF_FLOOR_DB + 0.5 {
        let hold_y = scale.dbfs_to_y(args.state.hold_db, bar_top, bar_bottom);
        let hold_col = if cfg.palette == ColorPalette::Obs {
            obs_peak_hold_color(args.state.hold_db, &tops, &styles)
        } else {
            COL_HOLD
        };
        painter.hline(bar_x..=bar_x + bar_w, hold_y, Stroke::new(1.5, hold_col));
    }

    // --- border ----------------------------------------------
    painter.rect_stroke(
        Rect::from_min_max(pos2(bar_x, bar_top), pos2(bar_x + bar_w, bar_bottom)),
        0.0,
        Stroke::new(1.0, COL_BORDER),
        egui::StrokeKind::Inside,
    );

    // --- scale ticks + labels ---------------------------------
    // Tick marks extend outward from the bar (longer than before); same #a3a3a7 as numerals.
    // Minor (unlabeled) ticks are half the length and use the dimmer COL_TICK
    // so they read as subdivisions without fighting the labeled majors.
    const RULER_TICK_INSET: f32 = 1.0;
    const RULER_TICK_OUTER: f32 = 8.0;
    const RULER_MINOR_TICK_OUTER: f32 = 4.0;

    let tick_to_y = |disp: f32| -> f32 {
        let dbfs = match scale.kind {
            MeterScaleKind::Dbf => disp,
            MeterScaleKind::Vu | MeterScaleKind::K => scale.reference_dbfs + disp,
        };
        scale.dbfs_to_y(dbfs, bar_top, bar_bottom)
    };

    // Minor ticks: subdivide each adjacent major-major gap. The per-gap count
    // keeps a useful density across modes with very different spacings:
    //   * gap >= 10 dB (e.g. K-20's -40..-30..-20, or dBFS's -40..-50..-60)
    //     → 3 quarter-points so 10 dB isn't a visually empty span.
    //   * gap  in 4..10 (OBS's 6 dB, PPM's 5 dB, K-20's 4 dB) → 1 midpoint.
    //   * gap  < 4 (VU's 1/2/3 dB near the top) → no minors; it's already
    //     fine-grained enough that more marks only add clutter.
    let majors = scale.tick_display_values();
    for pair in majors.windows(2) {
        let a = pair[0];
        let b = pair[1];
        let gap = (a - b).abs();
        let n: usize = if gap >= 10.0 {
            3
        } else if gap >= 4.0 {
            1
        } else {
            0
        };
        for i in 1..=n {
            let frac = i as f32 / (n as f32 + 1.0);
            let disp = a + (b - a) * frac;
            if disp > scale.disp_max || disp < scale.disp_min {
                continue;
            }
            let y = tick_to_y(disp);
            if y < bar_top - 1.0 || y > bar_bottom + 1.0 {
                continue;
            }
            painter.line_segment(
                [
                    pos2(bar_x - RULER_MINOR_TICK_OUTER, y),
                    pos2(bar_x - RULER_TICK_INSET, y),
                ],
                Stroke::new(1.0, COL_TICK),
            );
            painter.line_segment(
                [
                    pos2(bar_x + bar_w + RULER_TICK_INSET, y),
                    pos2(bar_x + bar_w + RULER_MINOR_TICK_OUTER, y),
                ],
                Stroke::new(1.0, COL_TICK),
            );
        }
    }

    for &disp in majors {
        if disp > scale.disp_max || disp < scale.disp_min {
            continue;
        }
        let y = tick_to_y(disp);
        if y < bar_top - 1.0 || y > bar_bottom + 1.0 {
            continue;
        }
        painter.line_segment(
            [
                pos2(bar_x - RULER_TICK_OUTER, y),
                pos2(bar_x - RULER_TICK_INSET, y),
            ],
            Stroke::new(1.0, COL_RULER_TEXT),
        );
        painter.line_segment(
            [
                pos2(bar_x + bar_w + RULER_TICK_INSET, y),
                pos2(bar_x + bar_w + RULER_TICK_OUTER, y),
            ],
            Stroke::new(1.0, COL_RULER_TEXT),
        );
        let label = scale.tick_label(disp);
        painter.text(
            pos2(bar_x - 4.0, y),
            egui::Align2::RIGHT_CENTER,
            label,
            FontId::proportional(FONT_TICK),
            COL_RULER_TEXT,
        );
    }

    // --- readout pill -----------------------------------------
    // The pill spans (almost) the full meter width rather than just the
    // bar area so the two values have enough room at the larger font.
    // Keep the readout below the live strip; otherwise the panel covers it.
    let readout_top = indicator_rect.bottom() + READOUT_GAP;
    let readout_rect = Rect::from_min_max(
        pos2(rect.left() + 3.0, readout_top),
        pos2(rect.right() - 3.0, readout_top + READOUT_H),
    );
    painter.rect_filled(readout_rect, 3.0, COL_READOUT_BG);
    painter.rect_stroke(
        readout_rect,
        3.0,
        Stroke::new(1.0, COL_READOUT_BORDER),
        egui::StrokeKind::Inside,
    );
    let half_x = readout_rect.left() + readout_rect.width() * 0.5;
    painter.line_segment(
        [
            pos2(half_x, readout_rect.top() + 3.0),
            pos2(half_x, readout_rect.bottom() - 3.0),
        ],
        Stroke::new(1.0, COL_READOUT_BORDER),
    );
    // Anchor each number against the outer edges of its half, with a
    // small inner margin, so the two values never touch the divider.
    let inner_margin = 4.0;
    // Show the live value of whichever source the meter is filling
    // from — peak for sample-peak meters, RMS for VU / K-System.
    let live_db = match cfg.fill_source {
        FillSource::Peak => args.state.peak_db,
        FillSource::Rms => args.state.rms_db,
    };
    let cur_text = scale.format_readout_dbfs(live_db);
    let max_text = scale.format_readout_dbfs(args.state.max_peak_db);
    painter.text(
        pos2(half_x - inner_margin, readout_rect.center().y),
        egui::Align2::RIGHT_CENTER,
        cur_text,
        FontId::proportional(FONT_READOUT),
        COL_READOUT_TEXT,
    );
    painter.text(
        pos2(half_x + inner_margin, readout_rect.center().y),
        egui::Align2::LEFT_CENTER,
        max_text,
        FontId::proportional(FONT_READOUT),
        COL_READOUT_MAX,
    );
    if let Some(pos) = response.hover_pos() {
        if readout_rect.contains(pos) {
            ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
        }
    }
    // --- channel label ----------------------------------------
    if !args.label.is_empty() {
        painter.text(
            pos2(rect.center().x, rect.bottom() - 2.0),
            egui::Align2::CENTER_BOTTOM,
            args.label,
            FontId::proportional(FONT_CHANNEL),
            COL_TEXT,
        );
    }

    response
}

/// Paint the dim "track" backgrounds for each zone, bottom-to-top.
pub fn paint_zone_backgrounds(
    painter: &Painter,
    x: f32,
    w: f32,
    top: f32,
    bottom: f32,
    tops_db: &[f32; 4],
    styles: &[ZoneStyle; 4],
    scale: &MeterScale,
) {
    let mut prev_y = bottom;
    for (i, &top_db) in tops_db.iter().enumerate() {
        let top_y = scale.dbfs_to_y(top_db, top, bottom).min(prev_y);
        if top_y < prev_y {
            painter.rect_filled(
                Rect::from_min_max(pos2(x, top_y), pos2(x + w, prev_y)),
                0.0,
                styles[i].bg,
            );
        }
        prev_y = top_y;
    }
    // Anything above the highest zone falls back to the neutral track
    // color (only visible if the highest zone doesn't reach bar_top).
    if prev_y > top {
        painter.rect_filled(
            Rect::from_min_max(pos2(x, top), pos2(x + w, prev_y)),
            0.0,
            COL_TRACK,
        );
    }
}

/// Vertical blend width (dB) where lit fill lerps between adjacent zone colors.
const FILL_GRADIENT_BLEND_DB: f32 = 2.5;

#[inline]
fn lerp_color32(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgba_unmultiplied(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t).round() as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t).round() as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t).round() as u8,
        (a.a() as f32 + (b.a() as f32 - a.a() as f32) * t).round() as u8,
    )
}

#[inline]
fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Color for the lit bar at `db` (dBFS), with soft transitions at internal zone boundaries.
fn fill_color_at_dbfs(db: f32, tops_db: &[f32; 4], styles: &[ZoneStyle; 4]) -> Color32 {
    let half = FILL_GRADIENT_BLEND_DB * 0.5;
    for k in 0..3 {
        let c = tops_db[k];
        let t = (db - (c - half)) / FILL_GRADIENT_BLEND_DB;
        if t > 0.0 && t < 1.0 {
            let s = smoothstep01(t);
            return lerp_color32(styles[k].fill, styles[k + 1].fill, s);
        }
    }
    if db <= tops_db[0] {
        styles[0].fill
    } else if db <= tops_db[1] {
        styles[1].fill
    } else if db <= tops_db[2] {
        styles[2].fill
    } else {
        styles[3].fill
    }
}

/// Paint the lit portion of each zone up to `fill_y` (pixel-y of the
/// current meter reading — lower y = higher level).
pub fn paint_zone_fill(
    painter: &Painter,
    x: f32,
    w: f32,
    top: f32,
    bottom: f32,
    fill_y: f32,
    tops_db: &[f32; 4],
    styles: &[ZoneStyle; 4],
    scale: &MeterScale,
) {
    let y0 = fill_y.max(top);
    let y1 = bottom;
    if y0 >= y1 {
        return;
    }
    let mut y = y0;
    while y + 1e-4 < y1 {
        let y_next = (y + 1.0_f32).min(y1);
        let y_mid = (y + y_next) * 0.5;
        let db = scale.y_to_dbfs(y_mid, top, bottom);
        let col = fill_color_at_dbfs(db, tops_db, styles);
        painter.rect_filled(
            Rect::from_min_max(pos2(x, y), pos2(x + w, y_next)),
            0.0,
            col,
        );
        y = y_next;
    }
}

fn draw_dashed_h(
    painter: &egui::Painter,
    x0: f32,
    x1: f32,
    y: f32,
    color: Color32,
    dash: f32,
    gap: f32,
    width: f32,
) {
    let stroke = Stroke::new(width, color);
    let mut x = x0;
    while x < x1 {
        let x_end = (x + dash).min(x1);
        painter.line_segment([pos2(x, y), pos2(x_end, y)], stroke);
        x += dash + gap;
    }
}

/// Free-standing widget that shows the reference label (0 dBFS / 0 VU / 0 K)
/// aligned with the dashed reference line of the adjacent meter.
pub fn show_reference_label(ui: &mut Ui, mode: MeterMode, strip_channels: usize) -> Response {
    let available = ui.available_height().max(METER_MIN_HEIGHT);
    let (rect, response) =
        ui.allocate_exact_size(vec2(METER_STRIP_REF_COLUMN_W, available), Sense::hover());
    let painter = ui.painter_at(rect);
    let cfg = config_for(mode);
    let scale = MeterScale::for_mode(mode);
    let bar_top = rect.top() + BAR_TOP_PAD;
    let bar_bottom = rect.bottom() - meter_bar_bottom_pad_for_strip(strip_channels);
    if cfg.reference_db > MIN_DB && cfg.reference_db <= 0.0 {
        let y = scale.dbfs_to_y(cfg.reference_db, bar_top, bar_bottom);
        let ref_col = if cfg.palette == ColorPalette::Obs {
            COL_OBS_REFERENCE
        } else {
            COL_REFERENCE
        };
        painter.text(
            pos2(rect.left() + 4.0, y),
            egui::Align2::LEFT_CENTER,
            scale.reference_label_text(),
            FontId::proportional(FONT_REF_LABEL),
            ref_col,
        );
    }
    response
}

/// Convenience: channel labels for common channel counts (matches the
/// Python app).
pub fn channel_label(channels: usize, idx: usize) -> &'static str {
    match (channels, idx) {
        (1, 0) => "M",
        (2, 0) => "L",
        (2, 1) => "R",
        (4, 0) => "FL",
        (4, 1) => "FR",
        (4, 2) => "RL",
        (4, 3) => "RR",
        (6, 0) => "FL",
        (6, 1) => "FR",
        (6, 2) => "C",
        (6, 3) => "LFE",
        (6, 4) => "RL",
        (6, 5) => "RR",
        (8, 0) => "FL",
        (8, 1) => "FR",
        (8, 2) => "C",
        (8, 3) => "LFE",
        (8, 4) => "RL",
        (8, 5) => "RR",
        (8, 6) => "SL",
        (8, 7) => "SR",
        _ => "",
    }
}

#[allow(dead_code)]
pub fn make_pos(x: f32, y: f32) -> Pos2 {
    pos2(x, y)
}

#[allow(dead_code)]
pub fn make_vec(x: f32, y: f32) -> Vec2 {
    vec2(x, y)
}
