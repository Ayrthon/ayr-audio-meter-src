//! Scrolling real-time history graph drawn with egui's painter.
//!
//! Uses a ring buffer so no array shifting happens on every sample.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{
    Color32, CursorIcon, FontId, Pos2, Rect, Response, Sense, Shape, Stroke, StrokeKind, Ui,
    pos2, vec2,
};
use egui::epaint::{Mesh, Vertex, WHITE_UV};
use parking_lot::RwLock;

use crate::clip_capture::{
    SharedClipSettings, SharedClipStatus, CLIP_THRESHOLD_SLIDER_MAX_DB,
    CLIP_THRESHOLD_SLIDER_MIN_DB,
};
use crate::dsp::INF_FLOOR_DB;
use crate::meter::{ColorPalette, MeterMode, MeterScale, MeterScaleKind, MIN_DB, config_for};
use crate::meter_widget::{
    COL_REFERENCE, COL_RULER_TEXT, FONT_TICK, paint_zone_backgrounds, zone_styles, zone_tops_db,
};
use crate::spectrum_view::{PLOT_BOTTOM_PAD, PLOT_TOP_PAD};

const SAMPLES: usize = 600; // 30 s @ 20 Hz
pub const SAMPLE_HZ: f32 = 20.0;
/// No averaging — points are the same per-sample dB values fed from the meters (`history_trace`),
/// so the trace lines up with the live bar in every meter mode.
const SMOOTH_WIN: usize = 1;
const WINDOW_SECS: f32 = SAMPLES as f32 / SAMPLE_HZ;
/// Red overlay on the history plot when clip capture triggers.
const TRIGGER_FLASH_DURATION: Duration = Duration::from_millis(520);

/// Vertical opacity of the fill under each history trace (plot bottom → top).
const HISTORY_TRACE_FILL_OPACITY_BOTTOM: f32 = 0.15;
const HISTORY_TRACE_FILL_OPACITY_TOP: f32 = 0.80;

/// Min scroll width (right-edge gutter is applied in `app.rs` around the whole column).
const MIN_WIDTH: f32 = 360.0;
/// Left inset for dB ruler numerals (wider than spectrum’s freq axis — different label geometry).
const HISTORY_LEFT_PAD: f32 = 32.0;
/// Match spectrum’s right gutter so plot width discipline is the same between panels.
const HISTORY_RIGHT_PAD: f32 = 18.0;

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

#[inline]
fn history_trace_fill_color_at_y(base: Color32, y: f32, top: f32, bottom: f32) -> Color32 {
    let y = y.clamp(top, bottom);
    let h = bottom - top;
    let t = if h.abs() < 1e-6 {
        0.0
    } else {
        ((y - bottom) / (top - bottom)).clamp(0.0, 1.0)
    };
    let a = HISTORY_TRACE_FILL_OPACITY_BOTTOM
        + (HISTORY_TRACE_FILL_OPACITY_TOP - HISTORY_TRACE_FILL_OPACITY_BOTTOM) * t;
    let alpha = (a * 255.0).round().clamp(0.0, 255.0) as u8;
    Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), alpha)
}

/// Fills the region from each trace segment down to `bottom` with a vertical opacity gradient.
fn paint_history_trace_fill_under(
    painter: &egui::Painter,
    pts: &[Pos2],
    base_color: Color32,
    top: f32,
    bottom: f32,
) {
    if pts.len() < 2 {
        return;
    }
    let mut mesh = Mesh::default();
    for i in 0..pts.len() - 1 {
        let p0 = pts[i];
        let p1 = pts[i + 1];
        let b0 = pos2(p0.x, bottom);
        let b1 = pos2(p1.x, bottom);
        let v0 = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex {
            pos: p0,
            uv: WHITE_UV,
            color: history_trace_fill_color_at_y(base_color, p0.y, top, bottom),
        });
        mesh.vertices.push(Vertex {
            pos: p1,
            uv: WHITE_UV,
            color: history_trace_fill_color_at_y(base_color, p1.y, top, bottom),
        });
        mesh.vertices.push(Vertex {
            pos: b1,
            uv: WHITE_UV,
            color: history_trace_fill_color_at_y(base_color, bottom, top, bottom),
        });
        mesh.vertices.push(Vertex {
            pos: b0,
            uv: WHITE_UV,
            color: history_trace_fill_color_at_y(base_color, bottom, top, bottom),
        });
        mesh.indices.extend_from_slice(&[v0, v0 + 1, v0 + 2, v0, v0 + 2, v0 + 3]);
    }
    painter.add(Shape::mesh(mesh));
}

fn trace_points_for_channel(
    ordered: &[[f32; crate::audio::MAX_CHANNELS]],
    ch: usize,
    graph_channels: usize,
    scale: &MeterScale,
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
) -> Option<(Vec<Pos2>, Color32)> {
    if ch >= graph_channels {
        return None;
    }
    let samples = ordered.len();
    let color = CHANNEL_COLORS[ch % CHANNEL_COLORS.len()];
    let dx = (right - left) / (SAMPLES - 1) as f32;
    let mut raw = Vec::with_capacity(samples);
    for row in ordered {
        raw.push(row[ch]);
    }
    let first_valid = raw
        .iter()
        .position(|&v| v > INF_FLOOR_DB + 0.5)
        .unwrap_or(samples);
    if first_valid >= samples {
        return None;
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
    if pts.len() < 2 {
        return None;
    }
    Some((pts, color))
}

pub fn show_history_graph(
    ui: &mut Ui,
    shared: &SharedHistory,
    mode: MeterMode,
    _now: f64,
    clip_settings: &SharedClipSettings,
    clip_status: &SharedClipStatus,
) -> Response {
    let graph = shared.read();
    let mut desired = ui.available_size();
    desired.x = desired.x.max(MIN_WIDTH);
    // Same sizing as `spectrum_view::show_spectrum`: use whatever height the right column
    // gives (no extra minimum vs the spectrum analyzer).
    let (rect, response) = ui.allocate_exact_size(desired, Sense::hover());
    let painter = ui.painter_at(rect);

    let cfg = config_for(mode);
    let scale = MeterScale::for_mode(mode);

    let left_pad = HISTORY_LEFT_PAD;
    let right_pad = HISTORY_RIGHT_PAD;
    let top = rect.top() + PLOT_TOP_PAD;
    let bottom = rect.bottom() - PLOT_BOTTOM_PAD;
    let left = rect.left() + left_pad;
    let right = rect.right() - right_pad;
    let plot = Rect::from_min_max(pos2(left, top), pos2(right, bottom));

    let traces: Vec<Option<(Vec<Pos2>, Color32)>> = if graph.channels > 0 && graph.filled > 1 {
        let ordered = graph.ordered();
        let mut traces: Vec<Option<(Vec<Pos2>, Color32)>> =
            (0..graph.channels).map(|_| None).collect();
        for ch in 0..graph.channels {
            traces[ch] = trace_points_for_channel(
                &ordered,
                ch,
                graph.channels,
                &scale,
                left,
                right,
                top,
                bottom,
            );
        }
        traces
    } else {
        Vec::new()
    };

    let styles = zone_styles(cfg.palette);
    let tops = zone_tops_db(&cfg);
    paint_zone_backgrounds(&painter, left, right - left, top, bottom, &tops, &styles, &scale);

    // Faded band above the clip trigger line (louder region): fills the visual gap vs zones below.
    // Vertical opacity gradient: stronger at the plot top, softer at the threshold line.
    const ARMED_TRIGGER_OPACITY_TOP: f32 = 0.80;
    const ARMED_TRIGGER_OPACITY_BOTTOM: f32 = 0.35;
    let clip_thr_db = clip_settings.threshold_db();
    if clip_thr_db.is_finite() {
        let y_thr = scale
            .dbfs_to_y(clip_thr_db, top, bottom)
            .clamp(top, bottom);
        if y_thr > top + 0.5 {
            let a_top = (ARMED_TRIGGER_OPACITY_TOP * 255.0).round().clamp(0.0, 255.0) as u8;
            let a_bot = (ARMED_TRIGGER_OPACITY_BOTTOM * 255.0).round().clamp(0.0, 255.0) as u8;
            let (c_top, c_bot) = if clip_settings.enabled() {
                (
                    Color32::from_rgba_unmultiplied(0x72, 0x18, 0x22, a_top),
                    Color32::from_rgba_unmultiplied(0x72, 0x18, 0x22, a_bot),
                )
            } else {
                (
                    Color32::from_rgba_unmultiplied(0x38, 0x3a, 0x40, a_top),
                    Color32::from_rgba_unmultiplied(0x38, 0x3a, 0x40, a_bot),
                )
            };
            let mut mesh = Mesh::default();
            mesh.vertices.push(Vertex {
                pos: pos2(left, top),
                uv: WHITE_UV,
                color: c_top,
            });
            mesh.vertices.push(Vertex {
                pos: pos2(right, top),
                uv: WHITE_UV,
                color: c_top,
            });
            mesh.vertices.push(Vertex {
                pos: pos2(right, y_thr),
                uv: WHITE_UV,
                color: c_bot,
            });
            mesh.vertices.push(Vertex {
                pos: pos2(left, y_thr),
                uv: WHITE_UV,
                color: c_bot,
            });
            mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
            painter.add(Shape::mesh(mesh));
        }
    }

    // Area under each history trace — before grids so dB/time ticks stay legible.
    for ch in 0..traces.len() {
        if let Some((pts, color)) = &traces[ch] {
            paint_history_trace_fill_under(&painter, pts, *color, top, bottom);
        }
    }

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
            pos2(left - 8.0, y),
            egui::Align2::RIGHT_CENTER,
            txt,
            FontId::proportional(FONT_TICK),
            COL_RULER_TEXT,
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
            FontId::proportional(FONT_TICK),
            COL_RULER_TEXT,
        );
    }

    // Reference line
    if cfg.reference_db > MIN_DB && cfg.reference_db <= 0.0 {
        let y = scale.dbfs_to_y(cfg.reference_db, top, bottom);
        draw_dashed_h(&painter, left, right, y, COL_REFERENCE, 4.0, 3.0, 1.1);
    }

    // Per-channel line — same geometry as the fill above; `traces` was built once.
    for ch in 0..traces.len() {
        if let Some((pts, color)) = &traces[ch] {
            painter.add(egui::Shape::line(pts.clone(), Stroke::new(1.6, *color)));
        }
    }

    // Clip capture threshold + label — always visible. Armed state draws a
    // solid line (reads as a committed trigger); disarmed keeps the dotted
    // stroke so it clearly looks inert. OBS palette uses a slightly thicker
    // stroke / dash spacing so the line stays legible on busy meter colours.
    const COL_CLIP_TRIGGER: Color32 = Color32::from_rgb(0xe8, 0x4a, 0x55);
    const COL_CLIP_TRIGGER_INACTIVE: Color32 = Color32::from_rgb(0x6a, 0x6e, 0x78);
    let clip_threshold_db = clip_settings.threshold_db();
    let mut response = response;
    if clip_threshold_db.is_finite() {
        let armed = clip_settings.enabled();
        let line_color = if armed {
            COL_CLIP_TRIGGER
        } else {
            COL_CLIP_TRIGGER_INACTIVE
        };
        let y_raw = scale.dbfs_to_y(clip_threshold_db, top, bottom);
        let y = y_raw.clamp(top, bottom);
        let obs = cfg.palette == ColorPalette::Obs;
        let (stroke_w, dot, gap) = if obs {
            (2.15_f32, 3.5_f32, 4.0_f32)
        } else {
            (1.25_f32, 2.0_f32, 5.0_f32)
        };
        if armed {
            painter.line_segment(
                [pos2(left, y), pos2(right, y)],
                Stroke::new(stroke_w, line_color),
            );
        } else {
            draw_dotted_h(&painter, left, right, y, line_color, dot, gap, stroke_w);
        }
        let (text_y, align) = if y - 14.0 > top + 2.0 {
            (y - 4.0, egui::Align2::LEFT_BOTTOM)
        } else {
            (y + 4.0, egui::Align2::LEFT_TOP)
        };
        let font = if obs {
            FontId::proportional(10.0)
        } else {
            FontId::proportional(9.0)
        };
        let trigger_caption = format!("{:.1} dBFS - Armed Record Trigger", clip_threshold_db);
        painter.text(
            pos2(left + 6.0, text_y),
            align,
            &trigger_caption,
            font,
            line_color,
        );

        const HIT_PAD: f32 = 10.0;
        let hit_rect = Rect::from_min_max(pos2(left, y - HIT_PAD), pos2(right, y + HIT_PAD))
            .intersect(plot);
        let drag_id = ui.id().with("history_clip_threshold_drag");
        let drag_resp = ui.interact(hit_rect, drag_id, Sense::click_and_drag());
        if drag_resp.hovered() || drag_resp.dragged() {
            ui.ctx().set_cursor_icon(CursorIcon::ResizeVertical);
        }
        if drag_resp.drag_started() || drag_resp.dragged() {
            if let Some(pos) = drag_resp.interact_pointer_pos() {
                let new_thr = scale
                    .y_to_dbfs(pos.y, top, bottom)
                    .clamp(CLIP_THRESHOLD_SLIDER_MIN_DB, CLIP_THRESHOLD_SLIDER_MAX_DB);
                clip_settings.set_threshold_db(new_thr);
            }
            ui.ctx().request_repaint();
        }
        if drag_resp.is_pointer_button_down_on() {
            if let Some(ptr) = drag_resp.interact_pointer_pos() {
                let readout_db = clip_settings.threshold_db();
                let readout = format!("{:.1} dBFS", readout_db);
                let readout_font = FontId::proportional(if obs { 15.0 } else { 14.0 });
                const READOUT_PAD_X: f32 = 10.0;
                const READOUT_PAD_Y: f32 = 6.0;
                let readout_bg = Color32::from_rgba_unmultiplied(0x12, 0x13, 0x18, 235);
                let fg = Color32::WHITE;
                let galley = painter.layout_no_wrap(readout, readout_font, fg);
                let tw = galley.size().x + READOUT_PAD_X * 2.0;
                let th = galley.size().y + READOUT_PAD_Y * 2.0;
                let mut pill_min = pos2(ptr.x - tw * 0.5, ptr.y - th - 14.0);
                pill_min.x = pill_min
                    .x
                    .clamp(plot.left() + 4.0, plot.right() - tw - 4.0);
                pill_min.y = pill_min
                    .y
                    .clamp(plot.top() + 4.0, plot.bottom() - th - 4.0);
                let pill = Rect::from_min_size(pill_min, vec2(tw, th));
                painter.rect_filled(pill, 4.0, readout_bg);
                painter.rect_stroke(
                    pill,
                    4.0,
                    Stroke::new(1.0, line_color),
                    StrokeKind::Inside,
                );
                painter.galley(
                    pos2(
                        pill.left() + READOUT_PAD_X,
                        pill.top() + READOUT_PAD_Y,
                    ),
                    galley,
                    fg,
                );
            }
        }
        response = response.union(drag_resp);
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
