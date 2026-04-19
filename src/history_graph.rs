//! Scrolling real-time history graph drawn with egui's painter.
//!
//! Uses a ring buffer so no array shifting happens on every sample.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{Color32, FontId, Pos2, Rect, Response, Sense, Stroke, Ui, pos2, vec2};
use parking_lot::RwLock;

use crate::clip_capture::SharedClipStatus;
use crate::dsp::INF_FLOOR_DB;
use crate::meter::{MeterMode, MeterScale, MeterScaleKind, MIN_DB, config_for};
use crate::meter_widget::{
    BAR_BOTTOM_PAD_WITH_LABEL, BAR_TOP_PAD, COL_BG, COL_BORDER, COL_REFERENCE, COL_TEXT_DIM,
    paint_zone_backgrounds, zone_styles, zone_tops_db,
};

const SAMPLES: usize = 600; // 30 s @ 20 Hz
pub const SAMPLE_HZ: f32 = 20.0;
const SMOOTH_WIN: usize = 5;
const WINDOW_SECS: f32 = SAMPLES as f32 / SAMPLE_HZ;
/// Red overlay on the history plot when clip capture triggers.
const TRIGGER_FLASH_DURATION: Duration = Duration::from_millis(520);

const CHANNEL_COLORS: [Color32; 8] = [
    Color32::from_rgb(0x4c, 0xaf, 0x50),
    Color32::from_rgb(0x29, 0xb6, 0xf6),
    Color32::from_rgb(0xff, 0xb3, 0x00),
    Color32::from_rgb(0xe5, 0x73, 0xc9),
    Color32::from_rgb(0xff, 0x7a, 0x59),
    Color32::from_rgb(0x7c, 0xe3, 0xd9),
    Color32::from_rgb(0xc8, 0xd1, 0x2d),
    Color32::from_rgb(0x9f, 0x9f, 0xaa),
];

pub struct HistoryGraph {
    channels: usize,
    ring: Vec<[f32; crate::audio::MAX_CHANNELS]>,
    write_idx: usize,
    filled: usize,
    latest_peak: [f32; crate::audio::MAX_CHANNELS],
    last_tick_secs: f64,
}

pub type SharedHistory = Arc<RwLock<HistoryGraph>>;

pub fn make_shared_history() -> SharedHistory {
    Arc::new(RwLock::new(HistoryGraph::new()))
}

impl HistoryGraph {
    pub fn new() -> Self {
        Self {
            channels: 0,
            ring: vec![[INF_FLOOR_DB; crate::audio::MAX_CHANNELS]; SAMPLES],
            write_idx: 0,
            filled: 0,
            latest_peak: [INF_FLOOR_DB; crate::audio::MAX_CHANNELS],
            last_tick_secs: 0.0,
        }
    }

    pub fn set_channels(&mut self, channels: usize) {
        self.channels = channels.min(crate::audio::MAX_CHANNELS);
        self.clear();
    }

    pub fn clear(&mut self) {
        for slot in &mut self.ring {
            *slot = [INF_FLOOR_DB; crate::audio::MAX_CHANNELS];
        }
        self.latest_peak = [INF_FLOOR_DB; crate::audio::MAX_CHANNELS];
        self.write_idx = 0;
        self.filled = 0;
        self.last_tick_secs = 0.0;
    }

    /// Reset only the internal tick clock without clearing the ring.
    /// Used by the audio thread when it starts up so its fresh
    /// `Instant` clock aligns with `last_tick_secs`.
    pub fn reset_tick_clock(&mut self) {
        self.last_tick_secs = 0.0;
    }

    pub fn push_latest(&mut self, peak_db: &[f32]) {
        for (i, v) in peak_db.iter().enumerate().take(crate::audio::MAX_CHANNELS) {
            self.latest_peak[i] = *v;
        }
    }

    pub fn tick(&mut self, now_secs: f64) {
        if self.last_tick_secs <= 0.0 {
            self.last_tick_secs = now_secs;
            self.ring[self.write_idx] = self.latest_peak;
            self.write_idx = (self.write_idx + 1) % SAMPLES;
            self.filled = (self.filled + 1).min(SAMPLES);
            return;
        }
        let dt = now_secs - self.last_tick_secs;
        let step = 1.0 / SAMPLE_HZ as f64;
        let n = (dt / step).floor().max(0.0) as usize;
        if n == 0 {
            return;
        }
        for _ in 0..n.min(SAMPLES) {
            self.ring[self.write_idx] = self.latest_peak;
            self.write_idx = (self.write_idx + 1) % SAMPLES;
            self.filled = (self.filled + 1).min(SAMPLES);
        }
        self.last_tick_secs += n as f64 * step;
    }

    fn ordered(&self) -> Vec<[f32; crate::audio::MAX_CHANNELS]> {
        let mut out = Vec::with_capacity(self.filled);
        if self.filled < SAMPLES {
            out.extend_from_slice(&self.ring[..self.filled]);
        } else {
            out.extend_from_slice(&self.ring[self.write_idx..]);
            out.extend_from_slice(&self.ring[..self.write_idx]);
        }
        out
    }
}

/// Horizontal **dotted** line (short dashes) for clip trigger overlay.
fn draw_dotted_h(
    painter: &egui::Painter,
    x0: f32,
    x1: f32,
    y: f32,
    color: Color32,
    dot: f32,
    gap: f32,
    width: f32,
) {
    let stroke = Stroke::new(width, color);
    let mut x = x0;
    while x < x1 {
        let x_end = (x + dot).min(x1);
        painter.line_segment([pos2(x, y), pos2(x_end, y)], stroke);
        x += dot + gap;
    }
}

pub fn show_history_graph(
    ui: &mut Ui,
    shared: &SharedHistory,
    mode: MeterMode,
    _now: f64,
    clip_armed: bool,
    clip_threshold_db: f32,
    clip_status: &SharedClipStatus,
) -> Response {
    let graph = shared.read();
    let available = ui.available_size();
    let desired = vec2(
        available.x.max(200.0),
        available.y.max(crate::meter_widget::METER_MIN_HEIGHT),
    );
    let (rect, response) = ui.allocate_exact_size(desired, Sense::hover());
    let painter = ui.painter_at(rect);

    let cfg = config_for(mode);
    let scale = MeterScale::for_mode(mode);
    painter.rect_filled(rect, 0.0, COL_BG);

    let left_pad = 34.0;
    // Right pad leaves room for the "now" label to sit fully inside
    // the graph area (it's center-aligned on the rightmost time tick).
    let right_pad = 22.0;
    let top = rect.top() + BAR_TOP_PAD;
    let bottom = rect.bottom() - BAR_BOTTOM_PAD_WITH_LABEL;
    let left = rect.left() + left_pad;
    let right = rect.right() - right_pad;
    let plot = Rect::from_min_max(pos2(left, top), pos2(right, bottom));

    let styles = zone_styles(cfg.palette);
    let tops = zone_tops_db(&cfg);
    paint_zone_backgrounds(&painter, left, right - left, top, bottom, &tops, &styles, &scale);

    // Vertical grid (same mapping as the level meters for this mode).
    for &disp in scale.tick_display_values() {
        if disp > scale.disp_max || disp < scale.disp_min {
            continue;
        }
        let dbfs = match scale.kind {
            MeterScaleKind::Dbf => disp,
            MeterScaleKind::Vu | MeterScaleKind::K => scale.reference_dbfs + disp,
        };
        let y = scale.dbfs_to_y(dbfs, top, bottom);
        if y < top - 1.0 || y > bottom + 1.0 {
            continue;
        }
        painter.line_segment(
            [pos2(left, y), pos2(right, y)],
            Stroke::new(0.5, Color32::from_rgba_premultiplied(0x4a, 0x4a, 0x52, 0x55)),
        );
        let txt = scale.tick_label(disp);
        painter.text(
            pos2(left - 4.0, y),
            egui::Align2::RIGHT_CENTER,
            txt,
            FontId::proportional(10.5),
            COL_TEXT_DIM,
        );
    }

    // Time grid
    for s in (0..=(WINDOW_SECS as i32)).step_by(5) {
        let frac = 1.0 - (s as f32 / WINDOW_SECS);
        let x = left + frac * (right - left);
        painter.line_segment(
            [pos2(x, top), pos2(x, bottom)],
            Stroke::new(0.5, Color32::from_rgba_premultiplied(0x4a, 0x4a, 0x52, 0x44)),
        );
        let label = if s == 0 { "now".to_string() } else { format!("-{s}s") };
        painter.text(
            pos2(x, bottom + 2.0),
            egui::Align2::CENTER_TOP,
            label,
            FontId::proportional(10.5),
            COL_TEXT_DIM,
        );
    }

    // Reference line
    if cfg.reference_db > MIN_DB && cfg.reference_db <= 0.0 {
        let y = scale.dbfs_to_y(cfg.reference_db, top, bottom);
        draw_dashed_h(&painter, left, right, y, COL_REFERENCE, 4.0, 3.0, 1.1);
    }

    // Per-channel smoothed line — "now" is anchored to the right edge and
    // older samples stream leftward.  When the ring is not yet full, the
    // left side of the plot is simply empty instead of pinning the newest
    // sample somewhere in the middle.
    if graph.channels > 0 && graph.filled > 1 {
        let ordered = graph.ordered();
        let samples = ordered.len();
        let dx = (right - left) / (SAMPLES - 1) as f32;
        for ch in 0..graph.channels {
            let color = CHANNEL_COLORS[ch % CHANNEL_COLORS.len()];
            let mut raw = Vec::with_capacity(samples);
            for row in &ordered {
                raw.push(row[ch]);
            }
            let first_valid = raw
                .iter()
                .position(|&v| v > INF_FLOOR_DB + 0.5)
                .unwrap_or(samples);
            if first_valid >= samples {
                continue;
            }
            let smoothed = smooth_with_edge_pad(&raw[first_valid..], SMOOTH_WIN);
            let mut pts: Vec<Pos2> = Vec::with_capacity(smoothed.len());
            for (k, v) in smoothed.iter().enumerate() {
                let ordered_idx = first_valid + k;
                let age = (samples - 1).saturating_sub(ordered_idx) as f32;
                let x = right - age * dx;
                if x < left - 0.5 {
                    continue;
                }
                let y = scale.dbfs_to_y(*v, top, bottom);
                pts.push(pos2(x, y));
            }
            if pts.len() >= 2 {
                painter.add(egui::Shape::line(pts, Stroke::new(1.6, color)));
            }
        }
    }

    // Clip capture threshold (dotted red) + label when armed
    const COL_CLIP_TRIGGER: Color32 = Color32::from_rgb(0xe8, 0x4a, 0x55);
    if clip_armed && clip_threshold_db.is_finite() && clip_threshold_db > INF_FLOOR_DB + 0.5 {
        let y = scale.dbfs_to_y(clip_threshold_db, top, bottom);
        if y >= top - 0.5 && y <= bottom + 0.5 {
            draw_dotted_h(&painter, left, right, y, COL_CLIP_TRIGGER, 2.0, 5.0, 1.25);
            let (text_y, align) = if y - 14.0 > top + 2.0 {
                (y - 4.0, egui::Align2::LEFT_BOTTOM)
            } else {
                (y + 4.0, egui::Align2::LEFT_TOP)
            };
            painter.text(
                pos2(left + 6.0, text_y),
                align,
                "Armed Record Trigger",
                FontId::proportional(9.0),
                COL_CLIP_TRIGGER,
            );
        }
    }

    // Clip trigger flash — brief red tint over the plot (set from audio thread on threshold).
    if let Some(t0) = clip_status.lock().graph_trigger_flash_at {
        let elapsed = Instant::now().saturating_duration_since(t0);
        if elapsed < TRIGGER_FLASH_DURATION {
            let t = elapsed.as_secs_f32() / TRIGGER_FLASH_DURATION.as_secs_f32();
            let alpha = (1.0 - t).powi(2);
            let a = (alpha * 110.0).round().clamp(0.0, 255.0) as u8;
            painter.rect_filled(
                plot,
                0.0,
                Color32::from_rgba_unmultiplied(0xff, 0x30, 0x38, a),
            );
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(16));
        }
    }

    // Plot border
    painter.rect_stroke(
        plot,
        0.0,
        Stroke::new(1.0, COL_BORDER),
        egui::StrokeKind::Inside,
    );

    response
}

fn smooth_with_edge_pad(src: &[f32], win: usize) -> Vec<f32> {
    if src.len() < 2 || win < 2 {
        return src.to_vec();
    }
    let half = win / 2;
    let mut padded = Vec::with_capacity(src.len() + 2 * half);
    for _ in 0..half {
        padded.push(src[0]);
    }
    padded.extend_from_slice(src);
    for _ in 0..half {
        padded.push(*src.last().unwrap());
    }
    let mut out = Vec::with_capacity(src.len());
    let inv = 1.0 / win as f32;
    for i in 0..src.len() {
        let mut sum = 0.0f32;
        for j in 0..win {
            sum += padded[i + j];
        }
        out.push(sum * inv);
    }
    out
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
