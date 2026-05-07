//! Clips library playlist: flat sections with thin separators, rows as `Clip N` + casual
//! date/time + optional **NEWEST** pill, round play control, and waveform preview.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use egui::{
    Align, Align2, Button, Color32, CornerRadius, FontFamily, FontId, Frame, Key, Margin, Modal,
    PointerButton, RichText, Sense, Stroke, Ui, Vec2,
};

use crate::clip_capture::{
    clip_file_display_name, format_clip_captured_casual, list_clip_wavs_from_settings, ClipFileInfo,
    POST_ROLL_SECS,
};
use crate::clip_playback::{
    load_wav_f32, waveform_peaks, ClipPlaybackHandle, LoadedClip, MAX_PLAYBACK_RATE_MUL,
};
use crate::user_settings::SharedUserSettings;

const PEAK_BUCKETS_MAX: usize = 1200;
const WAVE_H: f32 = 76.0;
const LISTEN_AFTER_WAVE_SPACER: f32 = 4.0;
/// Timeline labels below the waveform (placeholder + loaded — keep identical layout).
const CLIP_TIMELINE_LABEL_SIZE: f32 = 9.5;
const CLIP_TIMELINE_LABEL_COLOR: Color32 = Color32::from_gray(140);
/// Playhead nudge for ArrowLeft / ArrowRight when a clip is loaded.
const ARROW_SCRUB_STEP_SEC: f64 = 0.05;
/// Hold Shift for larger jumps (faster repositioning).
const ARROW_SCRUB_SHIFT_FAST_SEC: f64 = 0.25;
/// Half-width of the trigger highlight on the waveform (seconds each side of the crossing).
const TRIGGER_HIGHLIGHT_PAD_SEC: f64 = 0.5;

// ── Waveform scratch (inspired by Mixxx `PositionScratchController` + `WWaveformViewer`) ──
// Closed-loop: playback rate follows **position error** (finger vs engine), not raw dx/dt.
// See mixxx `src/engine/positionscratchcontroller.cpp`.
const MIXXX_SCRATCH_INTERVAL: f64 = 0.016;
const MIXXX_ERR_LP: f64 = 0.38;
const MIXXX_KP: f64 = 42.0;
const MIXXX_KD: f64 = 16.0;

fn split_casual_when(casual: &str) -> Option<(&str, &str)> {
    casual.split_once(" · ")
}

fn draw_clip_timeline_footer(ui: &mut Ui, left_sec: f64, right_sec: f64) {
    ui.add_space(LISTEN_AFTER_WAVE_SPACER);
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{:.2} s", left_sec))
                .size(CLIP_TIMELINE_LABEL_SIZE)
                .color(CLIP_TIMELINE_LABEL_COLOR)
                .family(FontFamily::Monospace),
        );
        ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
            ui.label(
                RichText::new(format!("{:.2} s", right_sec))
                    .size(CLIP_TIMELINE_LABEL_SIZE)
                    .color(CLIP_TIMELINE_LABEL_COLOR)
                    .family(FontFamily::Monospace),
            );
        });
    });
}

/// Vertical gap between clip rows so rounded strokes do not read as one continuous frame.
const CLIP_ROW_GAP: f32 = 12.0;

const CLIP_ROW_H: f32 = 38.0;

/// One clip row: `Clip N`, casual date/time, optional **NEWEST** pill; returns row click response.
/// Uses an explicit click hit‑test (`Frame` responses are hover‑only in egui 0.34).
fn draw_clip_row(
    ui: &mut Ui,
    slot: usize,
    casual_when: &str,
    selected: bool,
    is_newest: bool,
) -> egui::Response {
    let full_w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(full_w, CLIP_ROW_H), Sense::hover());
    let hovered = ui.rect_contains_pointer(rect);
    let (fill, stroke) = if selected {
        (
            Color32::from_rgba_unmultiplied(38, 72, 120, 140),
            Stroke::new(1.5, Color32::from_rgb(100, 160, 245)),
        )
    } else {
        (
            Color32::from_rgb(0x20, 0x22, 0x2c),
            Stroke::new(1.0, Color32::from_rgb(0x38, 0x3a, 0x48)),
        )
    };
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 6.0, fill);
    if hovered && !selected {
        // Translucent white overlay on hover; keeps selected style dominant.
        painter.rect_filled(
            rect,
            6.0,
            Color32::from_rgba_unmultiplied(255, 255, 255, 18),
        );
    }
    painter.rect_stroke(rect, 6.0, stroke, egui::StrokeKind::Inside);

    let inner = rect.shrink2(Vec2::new(10.0, 8.0));
    ui.allocate_ui_at_rect(inner, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
            ui.label(
                RichText::new(format!("Clip {slot}"))
                    .size(11.5)
                    .strong()
                    .color(Color32::from_gray(240)),
            );
            if let Some((day, time)) = split_casual_when(casual_when) {
                ui.label(RichText::new("·").size(11.0).color(Color32::from_gray(95)));
                ui.label(RichText::new(day).size(11.0).color(Color32::from_gray(155)));
                ui.label(RichText::new("·").size(11.0).color(Color32::from_gray(95)));
                ui.label(
                    RichText::new(time)
                        .size(11.0)
                        .color(Color32::from_gray(215)),
                );
            } else {
                ui.label(RichText::new(" · ").size(11.0).color(Color32::from_gray(95)));
                ui.label(
                    RichText::new(casual_when)
                        .size(11.0)
                        .color(Color32::from_gray(165)),
                );
            }
            if is_newest {
                ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                    Frame::new()
                        .fill(Color32::from_rgb(0x3a, 0x3c, 0x4e))
                        .inner_margin(Margin::symmetric(8, 3))
                        .corner_radius(CornerRadius::same(4))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new("NEWEST")
                                    .size(8.5)
                                    .strong()
                                    .color(Color32::from_gray(210)),
                            );
                        });
                });
            }
        });
    });

    let resp = ui.interact(rect, ui.id().with("clip_row").with(slot), Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

#[derive(Clone)]
enum DeletePrompt {
    Batch { paths: Vec<PathBuf> },
}

fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|s| clip_file_display_name(&s.to_string_lossy()))
        .unwrap_or_else(|| path.display().to_string())
}

/// 1-based index in the sorted list (newest first), if this path is a current capture.
fn clip_slot_1_based(entries: &[ClipFileInfo], path: &Path) -> Option<usize> {
    entries
        .iter()
        .position(|e| e.path.as_path() == path)
        .map(|i| i + 1)
}

pub struct ClipPlaylistState {
    clip_dir_settings: SharedUserSettings,
    entries: Vec<ClipFileInfo>,
    last_known_total_clips: u64,
    /// Selected clip files (preview loads only when exactly one path is selected).
    selected_paths: BTreeSet<PathBuf>,
    loaded: Option<LoadedClip>,
    loaded_path: Option<PathBuf>,
    peaks: Vec<f32>,
    last_peak_w_px: i32,
    /// Playhead used when scrubbing or when preview is paused (kept in sync while playing).
    scrub_sec: f64,
    /// Previous frame had primary drag on the waveform (for scratch release).
    wave_scrub_drag_prev: bool,
    /// Low-pass Mixxx-style filtered position error (normalized).
    scratch_err_filtered: f64,
    scratch_err_prev: f64,
    scratch_interval_accum: f64,
    scratch_last_cmd_rate: f64,
    player: Option<ClipPlaybackHandle>,
    player_err: Option<String>,
    delete_prompt: Option<DeletePrompt>,
    /// Last clicked clip used as the shift-range anchor.
    range_anchor_path: Option<PathBuf>,
    /// Keyboard navigation endpoint (moves with arrow keys).
    keyboard_focus_path: Option<PathBuf>,
    /// While true, keep keyboard-focused row in view across held-key repeats.
    follow_keyboard_focus: bool,
    /// One-shot scroll target used to keep keyboard navigation in view.
    scroll_to_path_once: Option<PathBuf>,
    /// Edge alignment for one-shot keyboard scroll.
    scroll_align_once: Option<Align>,
}

impl ClipPlaylistState {
    pub fn new(clip_dir_settings: SharedUserSettings) -> Self {
        let entries = list_clip_wavs_from_settings(&*clip_dir_settings.lock());
        Self {
            clip_dir_settings,
            entries,
            last_known_total_clips: 0,
            selected_paths: BTreeSet::new(),
            loaded: None,
            loaded_path: None,
            peaks: Vec::new(),
            last_peak_w_px: 0,
            scrub_sec: 0.0,
            wave_scrub_drag_prev: false,
            scratch_err_filtered: 0.0,
            scratch_err_prev: 0.0,
            scratch_interval_accum: 0.0,
            scratch_last_cmd_rate: 0.0,
            player: None,
            player_err: None,
            delete_prompt: None,
            range_anchor_path: None,
            keyboard_focus_path: None,
            follow_keyboard_focus: false,
            scroll_to_path_once: None,
            scroll_align_once: None,
        }
    }

    pub fn sync_saved_count(&mut self, total_clips_saved: u64) {
        if total_clips_saved != self.last_known_total_clips {
            self.last_known_total_clips = total_clips_saved;
            self.refresh_list();
        }
    }

    /// Rescan the clips folder from the current [`UserSettingsData::clips_save_dir`].
    pub fn reload_clip_list(&mut self) {
        self.refresh_list();
    }

    fn refresh_list(&mut self) {
        self.entries = list_clip_wavs_from_settings(&*self.clip_dir_settings.lock());
        let valid: BTreeSet<PathBuf> = self.entries.iter().map(|e| e.path.clone()).collect();
        self.selected_paths.retain(|p| valid.contains(p));
        if let Some(ref p) = self.loaded_path {
            if !p.exists() {
                self.clear_loaded();
            }
        }
    }

    fn clear_loaded(&mut self) {
        self.loaded = None;
        self.loaded_path = None;
        self.peaks.clear();
        self.last_peak_w_px = 0;
        self.scrub_sec = 0.0;
        self.wave_scrub_drag_prev = false;
        self.scratch_err_filtered = 0.0;
        self.scratch_err_prev = 0.0;
        self.scratch_interval_accum = 0.0;
        self.scratch_last_cmd_rate = 0.0;
        self.player = None;
        self.player_err = None;
    }

    fn ensure_player(&mut self) -> Result<&ClipPlaybackHandle, String> {
        if self.player.is_none() {
            match ClipPlaybackHandle::spawn() {
                Ok(h) => {
                    self.player = Some(h);
                    self.player_err = None;
                }
                Err(e) => {
                    self.player_err = Some(e.clone());
                    return Err(e);
                }
            }
        }
        self.player
            .as_ref()
            .ok_or_else(|| "Clip player failed to start.".into())
    }

    /// Seconds from file start where the clip threshold fired (end of pre-roll / start of post).
    fn trigger_time_from_span(samples_len: usize, ch: usize, sr: u32) -> f64 {
        let ch = ch.max(1);
        let sr = sr.max(1);
        let post_samples_needed =
            ((POST_ROLL_SECS * sr as f64) * ch as f64).ceil() as usize;
        let post_samples = post_samples_needed.min(samples_len);
        let pre_samples = samples_len.saturating_sub(post_samples);
        (pre_samples as f64 / ch as f64) / sr as f64
    }

    fn toggle_clip_preview_playback(&mut self) {
        let scrub = self.scrub_sec;
        let loaded_duration = self.loaded.as_ref().map(|l| l.duration_sec());
        match self.ensure_player() {
            Ok(p) => {
                p.set_scrub_active(false);
                p.set_playback_rate_mul(1.0);

                let dur_engine = p.duration_sec().max(1e-9);
                let dur_loaded = loaded_duration
                    .unwrap_or(dur_engine)
                    .max(1e-9);
                let dur = dur_engine.max(dur_loaded);
                let ph = p.playhead_sec();
                // Generous: float drift, UI vs engine duration, and “tail” of the file.
                let end_eps = (0.35_f64).max(dur * 0.02).min(dur * 0.25);
                let at_end = ph + end_eps >= dur
                    || scrub + end_eps >= dur
                    || (dur > 0.0 && ph / dur >= 0.985);

                // Stopped (or DSP-stopped) at EOF but `is_playing()` can still read true until the
                // player thread processes `SetPlaying(false)`. If the user hits Play in that window,
                // we'd only send pause — feels like "nothing happens". Near-end always means restart.
                // Do not assign `self.scrub_sec` here: `p` borrows `self.player` and would conflict
                // with `&mut self`; playhead mutex is updated by `seek_and_set_playing`, then the
                // waveform pass sets `scrub_sec` from `playhead_sec()`.
                if at_end {
                    p.seek_and_set_playing(0.0, true);
                    return;
                }

                let playing_now = p.is_playing();
                if playing_now {
                    p.set_playing(false);
                } else {
                    p.seek_and_set_playing(scrub, true);
                }
            }
            Err(e) => self.player_err = Some(e),
        }
    }

    fn load_selection_pcm(&mut self, path: &std::path::Path) {
        self.player_err = None;
        match load_wav_f32(path) {
            Ok(lc) => {
                self.scrub_sec = 0.0;
                self.loaded_path = Some(path.to_path_buf());
                let clip_for_player = lc.clone();
                self.loaded = Some(lc);
                self.rebuild_peaks_for_width(self.last_peak_w_px.max(400));
                if let Ok(p) = self.ensure_player() {
                    let _ = p.load_clip(&clip_for_player);
                }
            }
            Err(e) => {
                self.clear_loaded();
                self.player_err = Some(e);
            }
        }
    }

    fn sync_preview_for_selection(&mut self) {
        if self.selected_paths.len() == 1 {
            let path = self.selected_paths.first().unwrap().clone();
            if self.loaded_path.as_ref() != Some(&path) {
                self.load_selection_pcm(&path);
            }
        } else {
            self.clear_loaded();
        }
    }

    fn rebuild_peaks_for_width(&mut self, width_px: i32) {
        let Some(lc) = &self.loaded else {
            self.peaks.clear();
            return;
        };
        let buckets = (width_px as usize).clamp(50, PEAK_BUCKETS_MAX);
        self.peaks = waveform_peaks(&lc.samples, lc.channels, buckets);
        self.last_peak_w_px = width_px;
    }

    fn perform_delete_batch(&mut self, paths: &[PathBuf]) {
        self.player_err = None;
        let mut err = None;
        for p in paths {
            if let Err(e) = std::fs::remove_file(p) {
                err = Some((p.clone(), e.to_string()));
                break;
            }
        }
        let loaded_hit = self
            .loaded_path
            .as_ref()
            .is_some_and(|lp| paths.contains(lp));
        if loaded_hit {
            self.clear_loaded();
        }
        for p in paths {
            self.selected_paths.remove(p);
        }
        self.refresh_list();
        if let Some((p, e)) = err {
            self.player_err = Some(format!("Stopped at {}: {e}", p.display()));
        }
    }

    fn draw_delete_prompts(&mut self, ctx: &egui::Context) {
        let Some(kind) = self.delete_prompt.clone() else {
            return;
        };
        let DeletePrompt::Batch { paths } = kind;
        let n = paths.len();
        let paths_exec = paths.clone();

        let title = if n == 1 {
            let p0 = &paths[0];
            let label = clip_slot_1_based(&self.entries, p0)
                .map(|k| format!("Clip {k}"))
                .unwrap_or_else(|| file_label(p0));
            format!("Delete {label}?")
        } else {
            format!("Delete {n} clips?")
        };

        let dlg = Modal::new(egui::Id::new("clip_delete_modal")).show(ctx, |ui| {
            // Keep this confirm dialog compact so it doesn't dominate the left panel.
            ui.set_max_width(320.0);
            ui.set_min_width(320.0);
            ui.heading(RichText::new(&title).size(16.0));
            ui.add_space(8.0);
            ui.label(
                RichText::new("This removes the file(s) from disk. You can’t undo this.")
                    .size(11.0)
                    .color(Color32::from_gray(210)),
            );
            ui.add_space(8.0);
            const MAX_LIST: usize = 8;
            for p in paths.iter().take(MAX_LIST) {
                let label = clip_slot_1_based(&self.entries, p)
                    .map(|k| format!("Clip {k}"))
                    .unwrap_or_else(|| file_label(p));
                ui.label(
                    RichText::new(format!("• {label}"))
                        .size(10.0)
                        .color(Color32::from_gray(150)),
                );
            }
            if n > MAX_LIST {
                ui.label(
                    RichText::new(format!("… and {} more.", n - MAX_LIST))
                        .size(10.0)
                        .color(Color32::from_gray(130)),
                );
            }
            ui.add_space(16.0);
            let mut confirm_delete = false;
            if ui.input(|i| i.key_pressed(Key::Enter)) {
                confirm_delete = true;
                ui.close();
            }
            ui.horizontal(|ui| {
                if ui
                    .add(egui::Button::new("Cancel").min_size(Vec2::new(88.0, 28.0)))
                    .clicked()
                {
                    ui.close();
                }
                ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add(
                            egui::Button::new(RichText::new("Delete").color(Color32::from_rgb(
                                0xff, 0xb4, 0xb4,
                            )))
                            .min_size(Vec2::new(88.0, 28.0)),
                        )
                        .clicked()
                    {
                        confirm_delete = true;
                        ui.close();
                    }
                });
            });
            confirm_delete
        });

        if dlg.inner {
            self.perform_delete_batch(&paths_exec);
        }
        if dlg.should_close() {
            self.delete_prompt = None;
        }
    }

    fn draw_preview_placeholder(&self, ui: &mut Ui) {
        // Match `draw_loaded_waveform_preview` vertical layout (play row, gap, wave, timeline).
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(10.0, 0.0);
            ui.add_enabled_ui(false, |ui| {
                ui.add_sized(
                    Vec2::splat(34.0),
                    Button::new(RichText::new("▶").size(15.0))
                        .fill(Color32::from_rgb(0x30, 0x32, 0x3a))
                        .corner_radius(CornerRadius::same(17)),
                );
            });
            ui.label(
                RichText::new("—  ·  —  ·  —")
                    .size(9.5)
                    .color(Color32::from_gray(75)),
            );
        });
        ui.add_space(6.0);

        let full_w = ui.available_width();
        let (wave_rect, _) = ui.allocate_exact_size(Vec2::new(full_w, WAVE_H), Sense::hover());
        let painter = ui.painter_at(wave_rect);
        let bg = Color32::from_rgb(0x0e, 0x0f, 0x14);
        let border = Color32::from_rgb(0x34, 0x36, 0x44);
        painter.rect_filled(wave_rect, 2.0, bg);
        painter.rect_stroke(
            wave_rect,
            2.0,
            Stroke::new(1.0, border),
            egui::StrokeKind::Inside,
        );
        let cy = wave_rect.center().y;
        painter.line_segment(
            [
                egui::pos2(wave_rect.left() + 12.0, cy),
                egui::pos2(wave_rect.right() - 12.0, cy),
            ],
            Stroke::new(1.0, Color32::from_rgb(0x18, 0x1a, 0x24)),
        );
        // Paint hint with the painter only (no `allocate_ui_at_rect`) so vertical flow matches
        // `draw_loaded_waveform_preview`, which ends the wave row with `allocate_exact_size` + `interact`.
        let inner_hint = wave_rect.shrink2(Vec2::new(6.0, 6.0));
        painter.text(
            inner_hint.center(),
            Align2::CENTER_CENTER,
            "Select a clip to listen.",
            FontId::proportional(14.0),
            Color32::from_gray(120),
        );

        draw_clip_timeline_footer(ui, 0.0, 0.0);
    }

    fn draw_loaded_waveform_preview(
        &mut self,
        ui: &mut Ui,
        ctx: &egui::Context,
        dur: f64,
        ch: usize,
        sr: u32,
        samples_len: usize,
    ) {
        let playing_now = self
            .player
            .as_ref()
            .is_some_and(|p| p.is_playing());

        if ui.input(|i| i.key_pressed(Key::Space)) && !ctx.egui_wants_keyboard_input() {
            self.toggle_clip_preview_playback();
            ctx.request_repaint();
        }

        // Move playhead with ArrowLeft / ArrowRight (Shift = faster / larger steps).
        if !ctx.egui_wants_keyboard_input() {
            let step = if ctx.input(|i| i.modifiers.shift) {
                ARROW_SCRUB_SHIFT_FAST_SEC
            } else {
                ARROW_SCRUB_STEP_SEC
            };
            let mut delta: Option<f64> = None;
            if ui.input(|i| i.key_pressed(Key::ArrowRight)) {
                delta = Some(step);
            } else if ui.input(|i| i.key_pressed(Key::ArrowLeft)) {
                delta = Some(-step);
            }
            if let Some(d) = delta {
                let t = (self.scrub_sec + d).clamp(0.0, dur);
                self.scrub_sec = t;
                if let Ok(p) = self.ensure_player() {
                    p.set_scrub_active(false);
                    p.set_playback_rate_mul(1.0);
                    p.seek_sec(t);
                }
                ctx.request_repaint();
            }
        }

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(10.0, 0.0);
            let symbol = if playing_now { "⏸" } else { "▶" };
            if ui
                .add_sized(
                    Vec2::splat(34.0),
                    Button::new(RichText::new(symbol).size(15.0))
                        .fill(Color32::from_rgb(0x38, 0x3a, 0x48))
                        .corner_radius(CornerRadius::same(17)),
                )
                .clicked()
            {
                self.toggle_clip_preview_playback();
            }
            ui.label(
                RichText::new(format!("{:.1} s  ·  {} ch  ·  {} Hz", dur, ch, sr))
                    .size(9.5)
                    .color(Color32::from_gray(130)),
            );
        });

        ui.add_space(6.0);

        let full_w = ui.available_width();
        let wave_rect = ui
            .allocate_exact_size(Vec2::new(full_w, WAVE_H), Sense::hover())
            .0;
        let painter = ui.painter_at(wave_rect);

        let w_px = wave_rect.width().floor() as i32;
        if (w_px - self.last_peak_w_px).abs() > 3 && w_px >= 50 {
            self.rebuild_peaks_for_width(w_px);
        }

        let trigger_sec = Self::trigger_time_from_span(samples_len, ch, sr);
        let inner = wave_rect.shrink2(Vec2::new(6.0, 6.0));
        let w = inner.width().max(1.0);

        let bg = Color32::from_rgb(0x0e, 0x0f, 0x14);
        let border = Color32::from_rgb(0x34, 0x36, 0x44);
        painter.rect_filled(wave_rect, 2.0, bg);
        painter.rect_stroke(
            wave_rect,
            2.0,
            Stroke::new(1.0, border),
            egui::StrokeKind::Inside,
        );

        let mid_y = inner.center().y;
        let h2 = inner.height() * 0.42;

        if !self.peaks.is_empty() {
            let max_pk = self.peaks.iter().cloned().fold(1e-9_f32, f32::max);
            let n = self.peaks.len();
            let step = w / n as f32;
            let col = Color32::from_rgb(0x5c, 0x9e, 0xd4);
            for (i, &pk) in self.peaks.iter().enumerate() {
                let t = (pk / max_pk).clamp(0.0, 1.0);
                let half_h = h2 * t;
                let x0 = inner.left() + i as f32 * step;
                let x1 = (inner.left() + (i + 1) as f32 * step).min(inner.right());
                let xc = (x0 + x1) * 0.5;
                painter.line_segment(
                    [egui::pos2(xc, mid_y - half_h), egui::pos2(xc, mid_y + half_h)],
                    Stroke::new(1.25, col),
                );
            }
        }

        if trigger_sec.is_finite() && dur > 1e-9 {
            let t0 = (trigger_sec - TRIGGER_HIGHLIGHT_PAD_SEC).clamp(0.0, dur);
            let t1 = (trigger_sec + TRIGGER_HIGHLIGHT_PAD_SEC).clamp(0.0, dur);
            if t1 > t0 {
                let x0 = inner.left() + (t0 / dur) as f32 * w;
                let x1 = inner.left() + (t1 / dur) as f32 * w;
                let band = egui::Rect::from_min_max(
                    egui::pos2(x0, inner.top()),
                    egui::pos2(x1.max(x0 + 1.0), inner.bottom()),
                );
                painter.rect_filled(
                    band,
                    0.0,
                    Color32::from_rgba_unmultiplied(220, 55, 65, 72),
                );
            }
        }

        let frac = (self.scrub_sec / dur).clamp(0.0, 1.0) as f32;
        let play_x = inner.left() + frac * w;
        painter.line_segment(
            [egui::pos2(play_x, inner.top()), egui::pos2(play_x, inner.bottom())],
            Stroke::new(1.5, Color32::from_rgb(0xff, 0xd5, 0x4a)),
        );

        let interact = ui.interact(
            wave_rect,
            ui.id().with("clip_wave_scrub"),
            Sense::click_and_drag(),
        );
        let wave_drag = interact.dragged_by(PointerButton::Primary);
        let dt = ctx.input(|i| i.stable_dt as f64).max(1.0 / 1000.0);

        if self.wave_scrub_drag_prev && !wave_drag {
            let seek_to = self.scrub_sec;
            if let Ok(p) = self.ensure_player() {
                p.set_scrub_active(false);
                p.set_playback_rate_mul(1.0);
                p.seek_sec(seek_to);
            }
            self.scratch_err_filtered = 0.0;
            self.scratch_err_prev = 0.0;
            self.scratch_interval_accum = 0.0;
            self.scratch_last_cmd_rate = 0.0;
        }
        self.wave_scrub_drag_prev = wave_drag;

        // `interact_pointer_pos` can be None during drag on some platforms; fall back to global pointer.
        let pointer_pos = interact
            .interact_pointer_pos()
            .or_else(|| ctx.input(|i| i.pointer.interact_pos()));

        if let Some(pos) = pointer_pos {
            let xf = ((pos.x - inner.left()) / w).clamp(0.0, 1.0) as f64;
            let t = (xf * dur).clamp(0.0, dur);

            if interact.clicked_by(PointerButton::Primary) {
                self.scrub_sec = t;
                if let Ok(p) = self.ensure_player() {
                    p.set_scrub_active(false);
                    p.set_playback_rate_mul(1.0);
                    p.seek_sec(t);
                }
            } else if interact.drag_started_by(PointerButton::Primary) {
                self.scrub_sec = t;
                if let Ok(p) = self.ensure_player() {
                    p.seek_sec(t);
                    p.set_scrub_active(true);
                    self.scratch_err_filtered = 0.0;
                    self.scratch_err_prev = 0.0;
                    self.scratch_interval_accum = MIXXX_SCRATCH_INTERVAL;
                    self.scratch_last_cmd_rate = 0.0;
                }
            } else if wave_drag {
                // Closed-loop scratch only: do not gate on pointer “movement” heuristics — those
                // mis-fire on Windows (delta/position quantization) and mute playback every frame.
                self.scrub_sec = t;
                let nf = (samples_len / ch.max(1)).max(1) as f64;
                let sr_f = sr as f64;
                let engine_f = self
                    .player
                    .as_ref()
                    .map(|pl| pl.playhead_sec() * sr_f)
                    .unwrap_or_else(|| xf * nf);
                let target_f = xf * nf;
                let err = (target_f - engine_f) / nf;

                self.scratch_interval_accum += dt;
                if self.scratch_interval_accum >= MIXXX_SCRATCH_INTERVAL {
                    self.scratch_interval_accum -= MIXXX_SCRATCH_INTERVAL;
                    self.scratch_err_filtered = MIXXX_ERR_LP * err
                        + (1.0 - MIXXX_ERR_LP) * self.scratch_err_filtered;
                    let d = self.scratch_err_filtered - self.scratch_err_prev;
                    self.scratch_err_prev = self.scratch_err_filtered;
                    self.scratch_last_cmd_rate = (MIXXX_KP * self.scratch_err_filtered
                        + MIXXX_KD * d)
                        .clamp(-MAX_PLAYBACK_RATE_MUL, MAX_PLAYBACK_RATE_MUL);
                }
                let cmd = self.scratch_last_cmd_rate;
                if let Ok(p) = self.ensure_player() {
                    p.set_playback_rate_mul(cmd);
                }
                ctx.request_repaint();
            }
        }

        if let Some(p) = &self.player {
            if !wave_drag {
                self.scrub_sec = p.playhead_sec();
            }
        }

        draw_clip_timeline_footer(ui, self.scrub_sec, dur);
    }

    fn begin_frame(&mut self, ctx: &egui::Context, total_clips_saved: u64) {
        self.sync_saved_count(total_clips_saved);

        if self.player.as_ref().is_some_and(|p| p.is_playing() || p.is_scrub_active()) {
            ctx.request_repaint();
        }
    }

    fn draw_clips_toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 0.0);
            if ui
                .add_sized(
                    Vec2::splat(32.0),
                    Button::new(RichText::new("↻").size(18.0))
                        .fill(Color32::from_rgb(0x38, 0x3a, 0x48))
                        .corner_radius(CornerRadius::same(6)),
                )
                .clicked()
            {
                self.refresh_list();
            }
            let n = self.entries.len();
            let all_selected = n > 0 && self.selected_paths.len() == n;
            let sel_tool_label = if all_selected {
                "Deselect all"
            } else {
                "Select all"
            };
            if ui
                .add_enabled(
                    n > 0,
                    Button::new(RichText::new(sel_tool_label).size(12.0)),
                )
                .clicked()
            {
                if all_selected {
                    self.selected_paths.clear();
                } else {
                    self.selected_paths = self.entries.iter().map(|e| e.path.clone()).collect();
                }
                self.sync_preview_for_selection();
            }
            let sel_n_tb = self.selected_paths.len();
            let del = ui.add_enabled(
                sel_n_tb > 0,
                Button::new(RichText::new("Delete").size(12.0).color(if sel_n_tb > 0 {
                    Color32::from_rgb(0xff, 0xb8, 0xb8)
                } else {
                    Color32::from_gray(90)
                }))
                .fill(if sel_n_tb > 0 {
                    Color32::from_rgb(0x52, 0x28, 0x2c)
                } else {
                    Color32::from_rgb(0x30, 0x30, 0x36)
                })
                .corner_radius(CornerRadius::same(6)),
            );
            if del.clicked() {
                let paths: Vec<PathBuf> = self.selected_paths.iter().cloned().collect();
                self.delete_prompt = Some(DeletePrompt::Batch { paths });
            }

            let rest_h = ui.spacing().interact_size.y;
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width().max(0.0), rest_h),
                egui::Layout::right_to_left(Align::Center),
                |ui| {
                    let clip_lbl = if n == 1 {
                        "1 clip".to_string()
                    } else {
                        format!("{n} clips")
                    };
                    ui.label(
                        RichText::new(clip_lbl)
                            .size(12.0)
                            .color(Color32::from_gray(130)),
                    );
                },
            );
        });
    }

    fn select_range_to(&mut self, to_path: &Path) {
        if self.entries.is_empty() {
            return;
        }
        let anchor_path = self
            .range_anchor_path
            .as_ref()
            .cloned()
            .or_else(|| self.selected_paths.first().cloned())
            .unwrap_or_else(|| to_path.to_path_buf());
        let Some(from_idx) = self.entries.iter().position(|e| e.path == anchor_path) else {
            return;
        };
        let Some(to_idx) = self.entries.iter().position(|e| e.path == to_path) else {
            return;
        };
        let (a, b) = if from_idx <= to_idx {
            (from_idx, to_idx)
        } else {
            (to_idx, from_idx)
        };
        self.selected_paths.clear();
        for ent in &self.entries[a..=b] {
            self.selected_paths.insert(ent.path.clone());
        }
    }

    fn nudge_selection_by(&mut self, delta: isize, extend_range: bool) {
        if self.entries.is_empty() {
            return;
        }
        let current_idx = self
            .keyboard_focus_path
            .as_ref()
            .and_then(|p| self.entries.iter().position(|e| &e.path == p))
            .or_else(|| {
                self.entries
                    .iter()
                    .position(|e| self.selected_paths.contains(&e.path))
            })
            .unwrap_or(0);
        let next_idx = (current_idx as isize + delta)
            .clamp(0, self.entries.len().saturating_sub(1) as isize) as usize;
        let next_path = self.entries[next_idx].path.clone();
        if extend_range {
            if self.range_anchor_path.is_none() {
                self.range_anchor_path = Some(self.entries[current_idx].path.clone());
            }
            self.select_range_to(&next_path);
        } else {
            self.selected_paths.clear();
            self.selected_paths.insert(next_path.clone());
            self.range_anchor_path = Some(next_path.clone());
        }
        self.keyboard_focus_path = Some(next_path);
        self.follow_keyboard_focus = true;
        self.scroll_to_path_once = self.keyboard_focus_path.clone();
        self.scroll_align_once = Some(if delta >= 0 { Align::BOTTOM } else { Align::TOP });
        self.sync_preview_for_selection();
    }

    /// Fixed top block: player / waveform preview.
    pub fn show_player(&mut self, ui: &mut Ui, ctx: &egui::Context, total_clips_saved: u64) {
        self.begin_frame(ctx, total_clips_saved);
        ui.add_space(2.0);

        if let Some(ref e) = self.player_err {
            ui.label(
                RichText::new(format!("Could not load clip: {e}"))
                    .size(9.5)
                    .color(Color32::from_rgb(220, 120, 120)),
            );
            ui.add_space(6.0);
        }

        let sel_n_listen = self.selected_paths.len();
        let show_waveform = self.loaded.is_some() && sel_n_listen == 1;
        if show_waveform {
            let (dur, ch, sr, samples_len) = {
                let l = self.loaded.as_ref().unwrap();
                (
                    l.duration_sec().max(1e-6),
                    l.channels,
                    l.sample_rate,
                    l.samples.len(),
                )
            };
            self.draw_loaded_waveform_preview(ui, ctx, dur, ch, sr, samples_len);
        } else {
            self.draw_preview_placeholder(ui);
        }
        ui.add_space(12.0);
        self.draw_clips_toolbar(ui);
    }

    /// Ever-extending rows list + row actions (no inner scroller, no boxed card).
    pub fn show_clips_list(&mut self, ui: &mut Ui, ctx: &egui::Context, total_clips_saved: u64) {
        self.begin_frame(ctx, total_clips_saved);
        ui.add_space(2.0);

        // Keep follow mode active only while arrow navigation is physically held.
        // This avoids snap-back when the user later scrolls manually.
        self.follow_keyboard_focus = ctx.input(|i| i.key_down(Key::ArrowUp) || i.key_down(Key::ArrowDown));

        if !ctx.egui_wants_keyboard_input() {
            let shift_held = ctx.input(|i| i.modifiers.shift);
            if ctx.input(|i| i.key_pressed(Key::ArrowDown)) {
                self.nudge_selection_by(1, shift_held);
            } else if ctx.input(|i| i.key_pressed(Key::ArrowUp)) {
                self.nudge_selection_by(-1, shift_held);
            }
            if ctx.input(|i| i.key_pressed(Key::Escape)) {
                self.selected_paths.clear();
                self.keyboard_focus_path = None;
                self.follow_keyboard_focus = false;
                self.range_anchor_path = None;
                self.scroll_to_path_once = None;
                self.scroll_align_once = None;
                self.sync_preview_for_selection();
            }
            if ctx.input(|i| i.key_pressed(Key::Delete)) && !self.selected_paths.is_empty() {
                let paths: Vec<PathBuf> = self.selected_paths.iter().cloned().collect();
                self.delete_prompt = Some(DeletePrompt::Batch { paths });
            }
        }

        // Inline list — caller controls scrolling at the tab/panel level.
        let mut row_action: Option<(PathBuf, bool, bool)> = None;
        // Any pointer press should cancel pending keyboard follow jumps.
        if ctx.input(|i| i.pointer.any_pressed()) {
            self.follow_keyboard_focus = false;
            self.scroll_to_path_once = None;
            self.scroll_align_once = None;
        }
        // `left_panel_collapsing` zeros item_spacing on nested uis, so rely on explicit
        // `add_space` calls to separate the stroked row frames instead of item_spacing.
        if self.entries.is_empty() {
            ui.label(
                RichText::new("No clips yet — arm capture and let the meter cross the threshold.")
                    .size(9.5)
                    .color(Color32::from_gray(125)),
            );
        } else {
            for (idx, ent) in self.entries.iter().enumerate() {
                if idx > 0 {
                    ui.add_space(CLIP_ROW_GAP);
                }
                let selected = self.selected_paths.contains(&ent.path);
                let casual = format_clip_captured_casual(ent.modified);
                let r = draw_clip_row(ui, idx + 1, &casual, selected, idx == 0);
                let should_follow = self
                    .scroll_to_path_once
                    .as_ref()
                    .or_else(|| {
                        if self.follow_keyboard_focus {
                            self.keyboard_focus_path.as_ref()
                        } else {
                            None
                        }
                    })
                    .is_some_and(|p| p == &ent.path);
                if should_follow {
                    let clip = ui.clip_rect();
                    let should_scroll = match self.scroll_align_once {
                        Some(Align::BOTTOM) => r.rect.bottom() > clip.bottom(),
                        Some(Align::TOP) => r.rect.top() < clip.top(),
                        _ => !clip.contains_rect(r.rect),
                    };
                    if should_scroll {
                        ui.scroll_to_rect(r.rect, self.scroll_align_once);
                    }
                }
                if r.clicked() {
                    let mods = r.ctx.input(|i| i.modifiers);
                    row_action = Some((ent.path.clone(), mods.command, mods.shift));
                }
            }
        }
        self.scroll_to_path_once = None;
        self.scroll_align_once = None;

        if let Some((path, cmd_multi, shift_range)) = row_action {
            if shift_range {
                self.select_range_to(&path);
            } else if cmd_multi {
                if self.selected_paths.contains(&path) {
                    self.selected_paths.remove(&path);
                } else {
                    self.selected_paths.insert(path.clone());
                }
            } else if self.selected_paths.contains(&path) {
                self.selected_paths.remove(&path);
            } else {
                self.selected_paths.clear();
                self.selected_paths.insert(path.clone());
            }
            if !shift_range {
                self.range_anchor_path = Some(path.clone());
            } else if self.range_anchor_path.is_none() {
                self.range_anchor_path = Some(path.clone());
            }
            self.keyboard_focus_path = Some(path.clone());
            self.follow_keyboard_focus = false;
            self.sync_preview_for_selection();
        }

        let sel_n = self.selected_paths.len();
        if sel_n > 1 {
            ui.add_space(6.0);
            ui.label(
                RichText::new(format!(
                    "{sel_n} selected — preview listens to one clip at a time."
                ))
                .size(9.0)
                .color(Color32::from_rgb(0xc8, 0xb8, 0x90)),
            );
        }

        self.draw_delete_prompts(ctx);
    }

    /// Backward-compatible full layout.
    pub fn show(&mut self, ui: &mut Ui, ctx: &egui::Context, total_clips_saved: u64) {
        self.show_player(ui, ctx, total_clips_saved);
        ui.add_space(14.0);
        ui.separator();
        ui.add_space(14.0);
        self.show_clips_list(ui, ctx, total_clips_saved);
    }
}
