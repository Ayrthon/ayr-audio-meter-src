//! Playlist of trigger clips + waveform preview with scrub, Space play/pause, and a short
//! translucent band around the threshold-crossing instant on the wave.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use egui::{
    Align, Color32, CursorIcon, Key, Modal, PointerButton, RichText, ScrollArea, Sense, Stroke,
    Ui, Vec2,
};

use crate::clip_capture::{list_clip_wavs, ClipFileInfo, POST_ROLL_SECS};
use crate::clip_playback::{load_wav_f32, waveform_peaks, ClipPlaybackHandle, LoadedClip};

const PEAK_BUCKETS_MAX: usize = 1200;
const WAVE_H: f32 = 76.0;
const LIST_MAX_H: f32 = 160.0;
/// Half-width of the trigger highlight on the waveform (seconds each side of the crossing).
const TRIGGER_HIGHLIGHT_PAD_SEC: f64 = 0.5;

#[derive(Clone)]
enum DeletePrompt {
    Batch { paths: Vec<PathBuf> },
}

fn file_label(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

pub struct ClipPlaylistState {
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
    player: Option<ClipPlaybackHandle>,
    player_err: Option<String>,
    delete_prompt: Option<DeletePrompt>,
}

impl ClipPlaylistState {
    pub fn new() -> Self {
        Self {
            entries: list_clip_wavs(),
            last_known_total_clips: 0,
            selected_paths: BTreeSet::new(),
            loaded: None,
            loaded_path: None,
            peaks: Vec::new(),
            last_peak_w_px: 0,
            scrub_sec: 0.0,
            player: None,
            player_err: None,
            delete_prompt: None,
        }
    }

    pub fn sync_saved_count(&mut self, total_clips_saved: u64) {
        if total_clips_saved != self.last_known_total_clips {
            self.last_known_total_clips = total_clips_saved;
            self.refresh_list();
        }
    }

    fn refresh_list(&mut self) {
        self.entries = list_clip_wavs();
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
        let playing_now = self
            .player
            .as_ref()
            .is_some_and(|p| p.is_playing());
        let scrub = self.scrub_sec;
        match self.ensure_player() {
            Ok(p) => {
                if playing_now {
                    p.set_playing(false);
                } else {
                    p.seek_sec(scrub);
                    p.set_playing(true);
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
            format!("Delete “{}”?", file_label(&paths[0]))
        } else {
            format!("Delete {n} clips?")
        };

        let dlg = Modal::new(egui::Id::new("clip_delete_modal")).show(ctx, |ui| {
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
                ui.label(
                    RichText::new(format!("• {}", file_label(p)))
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

    /// Draw playlist + waveform. `total_clips_saved` keeps the list in sync after captures.
    pub fn show(&mut self, ui: &mut Ui, ctx: &egui::Context, total_clips_saved: u64) {
        self.sync_saved_count(total_clips_saved);

        if self
            .player
            .as_ref()
            .is_some_and(|p| p.is_playing())
        {
            ctx.request_repaint();
        }

        ui.label(
            RichText::new("Recorded clips")
                .size(12.0)
                .strong()
                .color(crate::app::TITLE_COL),
        );
        ui.add_space(2.0);
        ui.label(
            RichText::new("Click a row to select · ⌘/Ctrl+click to add or remove from the selection")
                .size(9.0)
                .color(Color32::from_gray(125)),
        );
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            if ui.button("Refresh list").clicked() {
                self.refresh_list();
            }
            ui.add_space(6.0);
            let n = self.entries.len();
            if ui
                .add_enabled(n > 0, egui::Button::new("Select all"))
                .on_hover_text("Select every clip in the list")
                .clicked()
            {
                self.selected_paths = self.entries.iter().map(|e| e.path.clone()).collect();
                self.sync_preview_for_selection();
            }
            ui.add_space(6.0);
            let sel_n = self.selected_paths.len();
            let del = ui
                .add_enabled(
                    sel_n > 0,
                    egui::Button::new(RichText::new("Delete").color(if sel_n > 0 {
                        Color32::from_rgb(0xff, 0xa8, 0xa8)
                    } else {
                        Color32::from_gray(90)
                    })),
                )
                .on_hover_text("Delete all selected clips (asks for confirmation)");
            if del.clicked() {
                let paths: Vec<PathBuf> = self.selected_paths.iter().cloned().collect();
                self.delete_prompt = Some(DeletePrompt::Batch { paths });
            }
        });

        ui.add_space(4.0);

        if self.entries.is_empty() {
            ui.label(
                RichText::new("No WAV clips yet — arm capture and let the meter cross the threshold.")
                    .size(9.5)
                    .color(Color32::from_gray(125)),
            );
            self.draw_delete_prompts(ctx);
            return;
        }

        let mut row_action: Option<(PathBuf, bool)> = None;
        ScrollArea::vertical()
            .max_height(LIST_MAX_H)
            .id_salt("clip_playlist_scroll")
            .show(ui, |ui| {
                for ent in &self.entries {
                    let selected = self.selected_paths.contains(&ent.path);
                    let cmd = ui.input(|i| i.modifiers.command);
                    let resp = ui
                        .add(egui::Button::selectable(
                            selected,
                            RichText::new(&ent.file_name).size(10.5),
                        ))
                        .on_hover_text(if cmd {
                            "Click to remove from selection"
                        } else {
                            "Click to select only this clip · hold ⌘/Ctrl to add to selection"
                        });
                    if resp.clicked() {
                        row_action = Some((ent.path.clone(), cmd));
                    }
                }
            });
        if let Some((path, cmd_multi)) = row_action {
            if cmd_multi {
                if self.selected_paths.contains(&path) {
                    self.selected_paths.remove(&path);
                } else {
                    self.selected_paths.insert(path);
                }
            } else {
                self.selected_paths.clear();
                self.selected_paths.insert(path);
            }
            self.sync_preview_for_selection();
        }

        let sel_n = self.selected_paths.len();
        if sel_n > 0 {
            ui.add_space(3.0);
            ui.label(
                RichText::new(format!(
                    "{sel_n} clip{} selected",
                    if sel_n == 1 { "" } else { "s" }
                ))
                .size(9.5)
                .color(Color32::from_gray(135)),
            );
        }

        if let Some(ref e) = self.player_err {
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!("Clip panel: {e}"))
                    .size(9.5)
                    .color(Color32::from_rgb(220, 120, 120)),
            );
        }

        if self.loaded.is_none() {
            ui.add_space(6.0);
            if sel_n == 0 {
                ui.label(
                    RichText::new("Select one clip to show its waveform and preview.")
                        .size(9.5)
                        .color(Color32::from_gray(120)),
                );
            } else {
                ui.label(
                    RichText::new("Waveform preview is available when exactly one clip is selected.")
                        .size(9.5)
                        .color(Color32::from_gray(120)),
                );
            }
            self.draw_delete_prompts(ctx);
            return;
        }
        let (dur, ch, sr, samples_len) = {
            let l = self.loaded.as_ref().unwrap();
            (
                l.duration_sec().max(1e-6),
                l.channels,
                l.sample_rate,
                l.samples.len(),
            )
        };
        let playing_now = self
            .player
            .as_ref()
            .is_some_and(|p| p.is_playing());

        if ui.input(|i| i.key_pressed(Key::Space)) && !ctx.egui_wants_keyboard_input() {
            self.toggle_clip_preview_playback();
            ctx.request_repaint();
        }

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let label = if playing_now { "Pause" } else { "Play" };
            if ui
                .button(label)
                .on_hover_text("Space — play / pause")
                .clicked()
            {
                self.toggle_clip_preview_playback();
            }
            ui.label(
                RichText::new(format!(
                    "{:.1} s  ·  {} ch  ·  {} Hz",
                    dur, ch, sr
                ))
                .size(9.5)
                .color(Color32::from_gray(130)),
            );
        });

        ui.add_space(4.0);

        let full_w = ui.available_width();
        let wave_rect = ui.allocate_exact_size(Vec2::new(full_w, WAVE_H), Sense::hover()).0;
        let painter = ui.painter_at(wave_rect);

        let w_px = wave_rect.width().floor() as i32;
        if self.loaded.is_some() && (w_px - self.last_peak_w_px).abs() > 3 && w_px >= 50 {
            self.rebuild_peaks_for_width(w_px);
        }

        let bg = Color32::from_rgb(0x0e, 0x0f, 0x14);
        let border = Color32::from_rgb(0x34, 0x36, 0x44);
        painter.rect_filled(wave_rect, 2.0, bg);
        painter.rect_stroke(
            wave_rect,
            2.0,
            Stroke::new(1.0, border),
            egui::StrokeKind::Inside,
        );

        let inner = wave_rect.shrink2(Vec2::new(6.0, 6.0));
        let mid_y = inner.center().y;
        let w = inner.width().max(1.0);
        let h2 = inner.height() * 0.42;

        if !self.peaks.is_empty() {
            let max_pk = self
                .peaks
                .iter()
                .cloned()
                .fold(1e-9_f32, f32::max);
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

        // Instant the clip capture tripped (pre/post boundary), ± pad for visibility.
        let trigger_sec = Self::trigger_time_from_span(samples_len, ch, sr);
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
        if interact.hovered() || interact.dragged() {
            ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
        }
        // With `Sense::click_and_drag`, egui can set `dragged()` after a long press with almost
        // no movement. Seeking every frame then fights the output callback and sounds robotic.
        // Only commit seeks on click, drag start, or when the pointer actually moves (DJ-style).
        if let Some(pos) = interact.interact_pointer_pos() {
            let primary_drag = interact.dragged_by(PointerButton::Primary);
            let primary_click = interact.clicked_by(PointerButton::Primary);
            if primary_drag || primary_click {
                let xf = ((pos.x - inner.left()) / w).clamp(0.0, 1.0) as f64;
                let t = (xf * dur).clamp(0.0, dur);
                self.scrub_sec = t;
                let drag_moved = interact.drag_delta().length_sq() > 1e-6;
                let should_seek = primary_click
                    || interact.drag_started_by(PointerButton::Primary)
                    || (primary_drag && drag_moved);
                if should_seek {
                    if let Ok(p) = self.ensure_player() {
                        p.seek_sec(t);
                    }
                }
            }
        }

        if let Some(p) = &self.player {
            if !interact.dragged_by(PointerButton::Primary) {
                self.scrub_sec = p.playhead_sec();
            }
        }

        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("{:.2} s", self.scrub_sec))
                    .size(9.5)
                    .color(Color32::from_gray(140)),
            );
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("{:.2} s", dur))
                        .size(9.5)
                        .color(Color32::from_gray(140)),
                );
            });
        });

        self.draw_delete_prompts(ctx);
    }
}
