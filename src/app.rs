//! Main application: device picker, meter mode selector, level meters
//! and scrolling history graph.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use anyhow::Result;
use eframe::egui;
use egui::{
    Align, Color32, CornerRadius, CursorIcon, Frame, Label, Layout, Margin, OpenUrl, RichText,
    ScrollArea, Sense, Stroke,
};
use parking_lot::Mutex;

use crate::audio::{
    AudioEngine, DeviceInfo, SharedClipTriggerStates, SharedHistoryTrace, SharedLevels, SourceKind,
    default_input_name, default_output_name, list_devices, make_shared_clip_trigger_states,
    make_shared_history_trace, make_shared_levels,
};
use crate::history_graph::{SharedHistory, make_shared_history, show_history_graph};
use crate::loudness::{NEG_INF_LUFS, SharedLoudness, make_shared_loudness};
use crate::meter::{MeterMode, config_for};
use crate::meter_widget::{
    MeterPaintArgs, MeterState, channel_label, show_meter, show_reference_label,
};
use crate::clip_capture::{
    clips_directory_display, make_shared_clip_capture, make_shared_clip_settings,
    make_shared_clip_status, SharedClipCapture, SharedClipSettings, SharedClipStatus,
};
use crate::spectrum::{SharedSpectrum, SpectrumAnalyzer, make_shared_spectrum};
use crate::spectrum_view::show_spectrum;
use crate::weighting::{FreqWeighting, MeterScratch};

const BG: Color32 = Color32::from_rgb(0x0c, 0x0c, 0x10);
const PANEL_BG: Color32 = Color32::from_rgb(0x15, 0x15, 0x1b);
const DIVIDER: Color32 = Color32::from_rgb(0x2b, 0x2b, 0x35);
const TITLE_COL: Color32 = Color32::from_rgb(0xf0, 0xf0, 0xf6);
const ACCENT: Color32 = Color32::from_rgb(0x4c, 0xaf, 0x50);
const AYRTHON_URL: &str = "https://ayrthon.com";

/// Optional CI/build stamp: `AUDIO_METER_BUILD=123 cargo build` — otherwise crate version.
#[inline]
fn audio_meter_build_label() -> &'static str {
    option_env!("AUDIO_METER_BUILD").unwrap_or(env!("CARGO_PKG_VERSION"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphView {
    Off,
    History,
    Spectrum,
}

pub struct AudioMeterApp {
    devices: Vec<DeviceInfo>,
    selected_device: Option<usize>,
    mode: MeterMode,
    panel: GraphView,
    last_panel: GraphView,
    // Width we last auto-snapped the window to.  If the current width
    // diverges from this the user has manually resized (or maximized),
    // so we stop forcing new sizes until the user switches panels and
    // we explicitly re-adopt the snap.
    last_auto_width: Option<f32>,
    // Stability tracking for the viewport inner rect.  We only send
    // InnerSize / MinInnerSize commands once the window has been sitting
    // still for a few frames — otherwise an in-flight drag / DPI change
    // / monitor move can collide with a viewport command and cause the
    // platform layer to act weird or crash.
    last_inner_rect: Option<egui::Rect>,
    stable_rect_frames: u32,
    // Whether we've sent the initial panel-aware MinInnerSize after
    // startup.  We defer it until the window has settled so it doesn't
    // collide with winit's own initial layout work.
    initial_min_sent: bool,
    /// IEC A/C curves for peak/RMS/history only (`AtomicU8` = [`FreqWeighting`] discriminant).
    weighting_atomic: Arc<AtomicU8>,
    /// [`MeterMode`] discriminant for clip trigger ballistics (mirrors `self.mode` while capturing).
    clip_meter_mode: Arc<AtomicU8>,
    /// Ballistics + peak-hold state driven on the audio thread for clip threshold (matches graph).
    clip_trigger_states: SharedClipTriggerStates,
    meter_scratch: Arc<Mutex<MeterScratch>>,
    clip_settings: SharedClipSettings,
    clip_capture: SharedClipCapture,
    clip_status: SharedClipStatus,
    /// [`ClipStatus::total_clips_saved`] when the user last turned **Arm Record Trigger** on.
    clip_session_baseline: u64,
    /// Peak-hold levels for the history graph (same as the white line on each meter).
    history_trace: SharedHistoryTrace,
    levels: SharedLevels,
    loudness: SharedLoudness,
    spectrum: SharedSpectrum,
    analyzer: SpectrumAnalyzer,
    /// "Pretty mode" paints the spectrum trace with a warm→cool palette across
    /// the five frequency bands (low / low-mid / mid / high-mid / high) instead
    /// of the default monochrome translucent look. Toggled via the checkbox in
    /// the top-right corner of the spectrum view.
    spectrum_pretty_mode: bool,
    /// User-pinned frequency reference lines, as Hz. Kept sorted ascending so the
    /// draw order and duplicate-click heuristics are deterministic. Lives in
    /// memory for the duration of the app session; there's no config file yet,
    /// adding one here would be a cross-cutting change for a list most users
    /// rebuild per listening session anyway. Managed entirely inside
    /// `spectrum_view::show_spectrum` - `app` only owns the storage.
    spectrum_anchors: Vec<f32>,
    meter_states: Vec<MeterState>,
    history: SharedHistory,
    engine: Option<AudioEngine>,
    channels: usize,
    status: String,
}

/// Target window widths for the two UI modes.  The app snaps to one
/// of these whenever the user toggles the history graph.
const WINDOW_W_WITH_GRAPH: f32 = 1140.0;

/// Minimum window width when the graph / spectrum panel is visible —
/// enough to keep the right-hand panel usable (in particular the
/// spectrum analyzer needs a few hundred pixels for its log-frequency
/// axis to be readable).  Matches the graph snap width so the user
/// always sees a full-width graph when it's on.
const WINDOW_W_MIN_WITH_GRAPH: f32 = 1140.0;

/// Compute how wide the window needs to be to fit the left panel + the
/// N meters + the "0 dB ref" label column (no graph).  Keeping this
/// dynamic means mono / stereo / 5.1 / 7.1 all snap to the right size.
fn window_w_without_graph(channels: usize) -> f32 {
    // Left panel: inner max_width (~304) + Frame inner_margin (~10 per side).
    const LEFT_PANEL: f32 = 324.0;
    // 6 px gap + vertical separator (~2 px + item_spacing) + 6 px gap.
    const GUTTERS: f32 = 20.0;
    // CentralPanel / window-frame padding on each side.
    const OUTER_PAD: f32 = 18.0;
    // Allocated meters-column width inside `meters_and_graph`.
    let meter_col = (channels.max(1) as f32)
        * (crate::meter_widget::METER_WIDTH + 4.0)
        + 60.0;
    LEFT_PANEL + GUTTERS + meter_col + OUTER_PAD
}

impl AudioMeterApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_theme(&cc.egui_ctx);
        log_gl_caps(cc);
        let devices = list_devices().unwrap_or_default();

        // Prefer the system default output (loopback) so the user sees
        // something move the first time they hit Start.  Fall back to
        // the default input, then to any device.
        let selected_device = default_output_name()
            .as_ref()
            .and_then(|n| {
                devices
                    .iter()
                    .position(|d| d.kind == SourceKind::Loopback && d.name == *n)
            })
            .or_else(|| {
                default_input_name().as_ref().and_then(|n| {
                    devices
                        .iter()
                        .position(|d| d.kind == SourceKind::Input && d.name == *n)
                })
            })
            .or_else(|| (0..devices.len()).next());

        let mut meter_states = Vec::with_capacity(2);
        for _ in 0..2 {
            meter_states.push(MeterState::default());
        }
        let history = make_shared_history();
        history.write().set_channels(2);

        Self {
            devices,
            selected_device,
            mode: MeterMode::SamplePeak,
            panel: GraphView::History,
            last_panel: GraphView::History,
            // Seed with the History snap width (matches initial
            // `with_inner_size`) so the first panel switch doesn't
            // mistake the startup size for a user resize.
            last_auto_width: Some(WINDOW_W_WITH_GRAPH),
            last_inner_rect: None,
            stable_rect_frames: 0,
            initial_min_sent: false,
            weighting_atomic: Arc::new(AtomicU8::new(FreqWeighting::Z as u8)),
            clip_meter_mode: Arc::new(AtomicU8::new(MeterMode::SamplePeak.to_discriminant())),
            clip_trigger_states: make_shared_clip_trigger_states(),
            meter_scratch: Arc::new(Mutex::new(MeterScratch::new())),
            clip_settings: make_shared_clip_settings(),
            clip_capture: make_shared_clip_capture(),
            clip_status: make_shared_clip_status(),
            clip_session_baseline: 0,
            history_trace: make_shared_history_trace(),
            levels: make_shared_levels(),
            loudness: make_shared_loudness(),
            spectrum: make_shared_spectrum(),
            analyzer: SpectrumAnalyzer::new(),
            spectrum_pretty_mode: false,
            spectrum_anchors: Vec::new(),
            meter_states,
            history,
            engine: None,
            channels: 2,
            status: "Idle".into(),
        }
    }

    fn refresh_devices(&mut self) {
        let prev_name = self
            .selected_device
            .and_then(|i| self.devices.get(i))
            .map(|d| (d.kind, d.name.clone()));
        self.devices = list_devices().unwrap_or_default();
        // Try to preserve selection by matching (kind, name).
        self.selected_device = prev_name
            .as_ref()
            .and_then(|(k, n)| {
                self.devices
                    .iter()
                    .position(|d| d.kind == *k && d.name == *n)
            })
            .or_else(|| (0..self.devices.len()).next());
    }

    fn start_stream(&mut self) -> Result<()> {
        let Some(idx) = self.selected_device else {
            self.status = "No device selected".into();
            return Ok(());
        };
        let Some(dev) = self.devices.get(idx).cloned() else {
            self.status = "Device out of range".into();
            return Ok(());
        };
        let cfg = config_for(self.mode);
        self.clip_meter_mode
            .store(self.mode.to_discriminant(), Ordering::Relaxed);
        *self.clip_trigger_states.lock() = std::array::from_fn(|_| MeterState::default());
        let engine = AudioEngine::start(
            &dev,
            cfg.true_peak,
            self.levels.clone(),
            self.loudness.clone(),
            self.spectrum.clone(),
            self.history.clone(),
            self.history_trace.clone(),
            self.weighting_atomic.clone(),
            self.meter_scratch.clone(),
            self.clip_meter_mode.clone(),
            self.clip_trigger_states.clone(),
            self.clip_settings.clone(),
            self.clip_capture.clone(),
            self.clip_status.clone(),
        )?;
        self.engine = Some(engine);
        let kind_label = match dev.kind {
            SourceKind::Input => "input",
            SourceKind::Loopback => "output (loopback)",
        };
        self.status = format!("Capturing {kind_label}: {}", dev.name);
        Ok(())
    }

    fn active_kind(&self) -> SourceKind {
        self.selected_device
            .and_then(|i| self.devices.get(i))
            .map(|d| d.kind)
            .unwrap_or(SourceKind::Input)
    }

    fn stop_stream(&mut self) {
        if let Some(mut e) = self.engine.take() {
            e.stop();
        }
        *self.clip_trigger_states.lock() = std::array::from_fn(|_| MeterState::default());
        for s in &mut self.meter_states {
            s.clear();
        }
        self.history.write().clear();
        *self.loudness.write() = Default::default();
        self.status = "Stopped".into();
    }

    /// Track whether the viewport's inner rect has been sitting still
    /// for a few consecutive frames.  Viewport commands are only safe
    /// to emit once the platform layer has settled — otherwise they
    /// can collide with an in-flight monitor move / DPI transition /
    /// resize drag, which at best looks jittery and at worst crashes.
    fn update_rect_stability(&mut self, ctx: &egui::Context) -> bool {
        let current = ctx.input(|i| i.viewport().inner_rect);
        let Some(cur) = current else {
            self.stable_rect_frames = 0;
            return false;
        };
        let stable = match self.last_inner_rect {
            Some(prev) => {
                (cur.width() - prev.width()).abs() < 0.5
                    && (cur.height() - prev.height()).abs() < 0.5
                    && (cur.left() - prev.left()).abs() < 0.5
                    && (cur.top() - prev.top()).abs() < 0.5
            }
            None => false,
        };
        if stable {
            self.stable_rect_frames = self.stable_rect_frames.saturating_add(1);
        } else {
            self.stable_rect_frames = 0;
        }
        self.last_inner_rect = Some(cur);
        self.stable_rect_frames >= 6
    }

    /// Snap the OS window to the appropriate width whenever the user
    /// switches the right panel between Off / History / Spectrum —
    /// but only when the user hasn't manually resized or maximized
    /// the window, and only when the viewport has been sitting still
    /// long enough that we're sure we aren't mid-drag / mid-DPI-change.
    fn maybe_resize_for_graph(&mut self, ctx: &egui::Context) {
        let rect_stable = self.update_rect_stability(ctx);

        // On the first stable frame, push a panel-aware MinInnerSize so
        // the window has the right minimum before the user has
        // interacted with any panel toggle.  Without this, the startup
        // min from `main.rs` (conservative 600 px) lets the user shrink
        // the window enough to cut off the graph.
        if !self.initial_min_sent
            && rect_stable
            && !ctx.input(|i| i.pointer.any_down() || i.pointer.any_pressed())
        {
            let min_w = match self.panel {
                GraphView::Off => window_w_without_graph(self.meter_states.len()),
                GraphView::History | GraphView::Spectrum => WINDOW_W_MIN_WITH_GRAPH,
            };
            ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
                min_w, 520.0,
            )));
            self.initial_min_sent = true;
        }

        if self.panel == self.last_panel {
            return;
        }

        // Don't fire ANY viewport commands while the user is interacting
        // with the window chrome or while the window hasn't settled
        // after a recent move / resize.  We keep `last_panel` unchanged
        // so the next idle frame re-attempts the update.
        let pointer_busy = ctx.input(|i| i.pointer.any_down() || i.pointer.any_pressed());
        if pointer_busy || !rect_stable {
            return;
        }

        // Update the platform min size so the user can't shrink below
        // what's needed for the current layout.  Safe to send only
        // once we're sure the platform layer is idle.
        let min_w = match self.panel {
            GraphView::Off => window_w_without_graph(self.meter_states.len()),
            GraphView::History | GraphView::Spectrum => WINDOW_W_MIN_WITH_GRAPH,
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
            min_w, 520.0,
        )));

        // If the user has maximized or fullscreened, treat their size
        // as authoritative and just update bookkeeping — except when they
        // switch to "Hide panel": then we restore the window and snap to the
        // compact width (otherwise the frame stays full-screen wide).
        let (maximized, fullscreen, inner_rect) = ctx.input(|i| {
            let vp = i.viewport();
            (
                vp.maximized.unwrap_or(false),
                vp.fullscreen.unwrap_or(false),
                vp.inner_rect,
            )
        });
        if self.panel == GraphView::Off
            && self.last_panel != GraphView::Off
            && maximized
            && !fullscreen
        {
            let compact_w = window_w_without_graph(self.meter_states.len());
            let h = inner_rect.map(|r| r.height()).unwrap_or(640.0);
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                compact_w, h,
            )));
            self.last_auto_width = Some(compact_w);
            self.last_panel = self.panel;
            return;
        }
        if maximized || fullscreen {
            self.last_panel = self.panel;
            self.last_auto_width = None;
            return;
        }

        let target_w = match self.panel {
            GraphView::Off => window_w_without_graph(self.meter_states.len()),
            GraphView::History | GraphView::Spectrum => WINDOW_W_WITH_GRAPH,
        };

        if let Some(rect) = inner_rect {
            let cur_w = rect.width();
            // Consider the current width "user-owned" if it doesn't
            // match the last auto-snap (within a small tolerance).  We
            // still always snap when hiding the panel, or when *opening*
            // the right panel from "Hide panel" — otherwise `last_auto_width`
            // is `None` (e.g. after maximize) and we'd skip widening, so the
            // graph would appear selected but stay clipped until a manual resize.
            let user_resized = match self.last_auto_width {
                Some(last) => (cur_w - last).abs() > 4.0,
                None => true,
            };
            let hide_panel = self.panel == GraphView::Off;
            let opening_right_panel = self.last_panel == GraphView::Off
                && matches!(
                    self.panel,
                    GraphView::History | GraphView::Spectrum
                );
            let snap_to_target = hide_panel || opening_right_panel || !user_resized;
            if snap_to_target {
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                    target_w,
                    rect.height(),
                )));
                self.last_auto_width = Some(target_w);
            } else {
                self.last_auto_width = None;
            }
        }
        self.last_panel = self.panel;
    }

    fn sync_channels(&mut self) {
        let lv = self.levels.read();
        let active = lv.active;
        let reported = lv.channels;
        drop(lv);
        // Only trust the channel count once a stream is actually
        // running.  Before start — or after stop — keep the last
        // known layout (default stereo) so the UI always shows a
        // full pair of meters instead of collapsing to one.
        if !active || reported == 0 {
            return;
        }
        if reported != self.channels {
            self.channels = reported;
            self.meter_states.resize_with(reported, MeterState::default);
            self.history.write().set_channels(reported);
        }
    }

    fn tick_meters(&mut self, now: f64) {
        self.sync_channels();
        let lv = self.levels.read();
        let peaks = lv.peak_db;
        let rms = lv.rms_db;
        let active = lv.active;
        let ch = lv.channels.min(self.meter_states.len());
        drop(lv);

        let cfg = config_for(self.mode);
        for i in 0..self.meter_states.len() {
            let target_peak = if active && i < ch {
                peaks[i]
            } else {
                crate::dsp::INF_FLOOR_DB
            };
            let target_rms = if active && i < ch {
                rms[i]
            } else {
                crate::dsp::INF_FLOOR_DB
            };
            self.meter_states[i].set_target(target_peak, target_rms);
            self.meter_states[i].tick(&cfg, now);
        }

        // History graph tracks **peak hold** (`hold_db`) — the white horizontal line —
        // not the moving bar top. The bar can read lower while hold is still up.
        let mut ht = self.history_trace.lock();
        for i in 0..crate::audio::MAX_CHANNELS {
            ht[i] = if active && i < ch && i < self.meter_states.len() {
                self.meter_states[i].hold_db
            } else {
                crate::dsp::INF_FLOOR_DB
            };
        }

        let _ = now;
    }

}

impl eframe::App for AudioMeterApp {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = ctx.input(|i| i.time);
        self.tick_meters(now);
        // Spectrum: run the analyzer while the tab is open, and keep decaying after Stop even if
        // another panel is selected (otherwise FFT state never reaches the noise floor).
        let spectrum_capturing = self.spectrum.read().active;
        if self.panel == GraphView::Spectrum || !spectrum_capturing {
            self.analyzer.process(&self.spectrum, now);
        }
        self.maybe_resize_for_graph(&ctx);

        ui.vertical(|ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                // Do not nest `ui.horizontal` for the title: each `horizontal()` allocates the
                // full row width, so an inner horizontal would eat all space and the credits
                // `with_layout` below would get zero width (invisible in release builds).
                ui.add_space(14.0);
                ui.label(
                    RichText::new("Audio Meter")
                        .color(TITLE_COL)
                        .size(18.0)
                        .strong(),
                );
                ui.label(
                    RichText::new(format!(
                        "real-time dBFS metering{}",
                        FreqWeighting::from_u8(self.weighting_atomic.load(Ordering::Relaxed))
                            .short_suffix()
                    ))
                    .color(Color32::from_gray(120))
                    .size(11.0),
                );

                // Right-align credits. In `right_to_left`, the first child is at the far right, so
                // add chunks as: margin, ` - Build (…)`, `Ayrthon`, `Created by ` → reads LTR as
                // `Created by Ayrthon - Build (…)`.
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.add_space(14.0);
                    ui.label(
                        RichText::new(format!(
                            " - Build ({})",
                            audio_meter_build_label()
                        ))
                        .color(Color32::from_gray(130))
                        .size(11.0),
                    );
                    let ayrthon = ui.add(
                        Label::new(
                            RichText::new("Ayrthon")
                                .color(Color32::WHITE)
                                .size(11.0),
                        )
                        .sense(Sense::click()),
                    );
                    if ayrthon.hovered() {
                        ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
                    }
                    if ayrthon.clicked() {
                        ui.ctx().open_url(OpenUrl::new_tab(AYRTHON_URL));
                    }
                    ui.label(
                        RichText::new("Created by ")
                            .color(Color32::from_gray(130))
                            .size(11.0),
                    );
                });
            });
            ui.add_space(4.0);
            ui.separator();

            ui.horizontal_top(|ui| {
                self.left_controls(ui, now);
                ui.add_space(6.0);
                ui.separator();
                ui.add_space(6.0);
                self.meters_and_graph(ui, now);
            });

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_space(14.0);
                ui.label(
                    RichText::new(&self.status)
                        .color(if self.engine.is_some() { ACCENT } else { Color32::from_gray(140) })
                        .size(11.0),
                );
            });
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

impl AudioMeterApp {
    fn left_controls(&mut self, ui: &mut egui::Ui, now: f64) {
        Frame::new()
            .fill(PANEL_BG)
            .inner_margin(egui::Margin::symmetric(10, 10))
            .show(ui, |ui| {
                ui.set_min_width(280.0);
                ui.set_max_width(304.0);
                let avail_h = ui.available_height();
                ScrollArea::vertical()
                    .id_salt("left_controls_scroll")
                    .max_height(avail_h)
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                    ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                    // Extra gutter on the right so the floating scrollbar
                    // never overlaps combo boxes / checkboxes.
                    let width = (ui.available_width() - 20.0).clamp(228.0, 270.0);

                    // ── AUDIO SOURCE ───────────────────────────────────────
                    section_header(ui, "AUDIO SOURCE");
                    ui.add_space(3.0);
                    self.device_dropdown(ui, "device_combo", width);

                    ui.add_space(3.0);
                    ui.horizontal(|ui| {
                        let (badge, hint, badge_bg) = match self.active_kind() {
                            SourceKind::Input => (
                                "INPUT",
                                "microphone / line-in",
                                Color32::from_rgb(0x26, 0x49, 0x78),
                            ),
                            SourceKind::Loopback => (
                                "OUTPUT",
                                "speakers (loopback)",
                                Color32::from_rgb(0x2a, 0x56, 0x3c),
                            ),
                        };
                        draw_badge(ui, badge, badge_bg);
                        ui.label(
                            RichText::new(hint)
                                .color(Color32::from_gray(150))
                                .size(10.5),
                        );
                    });

                    ui.add_space(8.0);

                    // ── PRIMARY START/STOP ────────────────────────────────
                    let playing = self.engine.is_some();
                    let btn_size = egui::vec2(width, 36.0);
                    if playing {
                        if breathing_stop_button(ui, now, btn_size).clicked() {
                            self.stop_stream();
                        }
                    } else if start_meter_button(ui, btn_size).clicked() {
                        self.start_or_restart();
                    }

                    ui.add_space(5.0);
                    if ui
                        .add_sized([width, 24.0], egui::Button::new("↻  Refresh devices"))
                        .clicked()
                    {
                        self.refresh_devices();
                    }

                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(5.0);

                    // ── CLIP CAPTURE ───────────────────────────────────────
                    section_header(ui, "CLIP CAPTURE");
                    ui.add_space(3.0);

                    let clip_detail = format!(
                        "When enabled, saves {:.0} s of audio before and {:.0} s after a block's peak first reaches the threshold. Files are 32-bit float WAV.\n\n\
                         Trigger uses the same per-callback peak as the meters: frequency weighting, true-peak vs sample-peak mode, and the hottest channel in that block. \
                         The moving meter can still read a little lower between blocks or because of ballistics.",
                        crate::clip_capture::PRE_ROLL_SECS,
                        crate::clip_capture::POST_ROLL_SECS
                    );

                    Frame::new()
                        .fill(Color32::from_rgb(0x11, 0x12, 0x18))
                        .stroke(Stroke::new(1.0, Color32::from_rgb(0x34, 0x36, 0x44)))
                        .corner_radius(CornerRadius::same(6))
                        .inner_margin(Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.set_width(width);
                            let rec_on = self.clip_settings.enabled();
                            let arm_h = 34.0_f32;
                            let arm_resp = arm_record_trigger_button(
                                ui,
                                now,
                                rec_on,
                                egui::vec2(width, arm_h),
                            );
                            if arm_resp.clicked() {
                                let new_on = !rec_on;
                                self.clip_settings.set_enabled(new_on);
                                if new_on {
                                    self.clip_session_baseline =
                                        self.clip_status.lock().total_clips_saved;
                                }
                            }
                            arm_resp.on_hover_text(&clip_detail);

                            ui.add_space(6.0);

                            let mut thr = self.clip_settings.threshold_db();
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new("Threshold")
                                        .size(12.0)
                                        .color(TITLE_COL)
                                        .strong(),
                                );
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    ui.label(
                                        RichText::new(format!("{thr:.1} dBFS"))
                                            .size(13.0)
                                            .color(ACCENT)
                                            .strong(),
                                    );
                                });
                            });
                            ui.add_space(4.0);

                            let slider_w = ui.available_width();
                            let prev_slider_w = ui.style().spacing.slider_width;
                            ui.style_mut().spacing.slider_width = slider_w;
                            let rail_h = 20.0_f32;
                            let slider_resp = ui.add_sized(
                                egui::vec2(slider_w, rail_h),
                                egui::Slider::new(&mut thr, -48.0_f32..=0.0_f32)
                                    .fixed_decimals(1)
                                    .show_value(false)
                                    .trailing_fill(true),
                            );
                            ui.style_mut().spacing.slider_width = prev_slider_w;
                            if slider_resp.changed() {
                                self.clip_settings.set_threshold_db(thr);
                            }

                            ui.add_space(2.0);
                            ui.label(
                                RichText::new("Peaks at or above this level start a capture.")
                                    .color(Color32::from_gray(120))
                                    .size(9.5),
                            );

                            ui.add_space(5.0);
                            ui.label(
                                RichText::new(format!("Save folder: {}", clips_directory_display()))
                                    .color(Color32::from_gray(115))
                                    .size(9.5),
                            );
                            #[cfg(windows)]
                            if ui
                                .add_sized([ui.available_width(), 24.0], egui::Button::new("Open clips folder"))
                                .clicked()
                            {
                                if let Some(p) = crate::clip_capture::clips_dir_path() {
                                    let _ = std::process::Command::new("explorer").arg(p).spawn();
                                }
                            }
                            {
                                let st = self.clip_status.lock();
                                let armed_now = self.clip_settings.enabled();
                                if armed_now {
                                    let n = st
                                        .total_clips_saved
                                        .saturating_sub(self.clip_session_baseline);
                                    let (counter_text, counter_color) = if n == 0 {
                                        (
                                            "Clips saved this arm: 0".into(),
                                            Color32::from_gray(130),
                                        )
                                    } else {
                                        (format!("Clips saved this arm: {n}"), ACCENT)
                                    };
                                    ui.add_space(5.0);
                                    ui.label(
                                        RichText::new(counter_text)
                                            .size(11.0)
                                            .strong()
                                            .color(counter_color),
                                    );
                                }
                                if let Some(ref e) = st.last_error {
                                    ui.add_space(3.0);
                                    ui.label(
                                        RichText::new(format!("Error: {e}"))
                                            .color(Color32::from_rgb(200, 100, 100))
                                            .size(9.5),
                                    );
                                }
                            }
                        });

                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(5.0);

                    section_header(ui, "RIGHT PANEL");
                    ui.add_space(3.0);
                    graph_panel_checkboxes(ui, &mut self.panel);

                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(6.0);

                    // ── METER MODE ─────────────────────────────────────────
                    section_header(ui, "METER MODE");
                    ui.add_space(3.0);
                    let cfg = config_for(self.mode);
                    let (mid_name, high_name) = match cfg.palette {
                        crate::meter::ColorPalette::ProTools => ("Green", "Orange"),
                        crate::meter::ColorPalette::Obs => ("Green", "Yellow"),
                        crate::meter::ColorPalette::Standard => ("Green", "Yellow"),
                    };
                    let mode_zone_hint = format!(
                        "Reference line: {:+.0} dBFS\n\
                         Color zones: {} from {:+.0} dBFS, {} from {:+.0} dBFS (up to 0 dBFS).",
                        cfg.reference_db,
                        mid_name,
                        cfg.zone_mid_top_db,
                        high_name,
                        cfg.zone_high_top_db,
                    );
                    let mode_combo = egui::ComboBox::from_id_salt("mode_combo")
                        .width(width)
                        .selected_text(self.mode.display_name())
                        .show_ui(ui, |ui| {
                            for mode in MeterMode::ALL {
                                if ui
                                    .selectable_label(self.mode == mode, mode.display_name())
                                    .clicked()
                                {
                                    self.mode = mode;
                                    self.clip_meter_mode
                                        .store(mode.to_discriminant(), Ordering::Relaxed);
                                    *self.clip_trigger_states.lock() =
                                        std::array::from_fn(|_| MeterState::default());
                                    for s in &mut self.meter_states {
                                        s.clear();
                                    }
                                    // Leave `self.history` alone — the
                                    // user wants the scrolling graph to
                                    // carry over when switching modes.
                                }
                            }
                        });
                    mode_combo.response.on_hover_text(mode_zone_hint);

                    ui.add_space(5.0);
                    section_header(ui, "FREQUENCY WEIGHTING");
                    ui.add_space(3.0);
                    let mut w = FreqWeighting::from_u8(
                        self.weighting_atomic.load(Ordering::Relaxed),
                    );
                    let w_combo = egui::ComboBox::from_id_salt("freq_weight_combo")
                        .width(width)
                        .selected_text(w.label())
                        .show_ui(ui, |ui| {
                            for ww in FreqWeighting::ALL {
                                if ui
                                    .selectable_label(w == ww, ww.list_entry())
                                    .clicked()
                                {
                                    w = ww;
                                }
                            }
                        });
                    w_combo.response.on_hover_text(
                        "Applies to peak, RMS, and level history only.\n\
                         LUFS and the spectrum use full-band BS.1770 K-weighting.",
                    );
                    self.weighting_atomic.store(w as u8, Ordering::Relaxed);

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(6.0);

                    // ── LOUDNESS (EBU R128 / BS.1770) ──────────────────────
                    section_header(ui, "LOUDNESS  (EBU R128)");
                    ui.add_space(3.0);
                    self.loudness_panel(ui, width);
                    });
                });
            });
    }

    fn device_dropdown(&mut self, ui: &mut egui::Ui, id: &str, width: f32) {
        let current_label = self
            .selected_device
            .and_then(|i| self.devices.get(i))
            .map(|d| {
                let tag = match d.kind {
                    SourceKind::Input => "IN",
                    SourceKind::Loopback => "OUT",
                };
                format!("[{tag}] {}", d.name)
            })
            .unwrap_or_else(|| "<no devices>".into());

        // Lock the picker while capture is live so the user can't swap devices mid-stream
        // (which would otherwise drop the current engine, allocate a new one, and reset the
        // spectrum/meter state without an explicit intent). Stop metering first to change
        // devices.
        let metering_active = self.engine.is_some();
        let combo_resp = ui.add_enabled_ui(!metering_active, |ui| {
            egui::ComboBox::from_id_salt(id)
                .width(width)
                .selected_text(truncate(&current_label, 36))
                .show_ui(ui, |ui| {
                let has_inputs = self.devices.iter().any(|d| d.kind == SourceKind::Input);
                let has_outputs = self
                    .devices
                    .iter()
                    .any(|d| d.kind == SourceKind::Loopback);

                if has_outputs {
                    ui.label(
                        RichText::new("OUTPUT  (speakers / loopback)")
                            .color(Color32::from_gray(150))
                            .size(10.0)
                            .strong(),
                    );
                    for (i, d) in self.devices.iter().enumerate() {
                        if d.kind != SourceKind::Loopback {
                            continue;
                        }
                        let selected = self.selected_device == Some(i);
                        if ui.selectable_label(selected, &d.name).clicked() {
                            self.selected_device = Some(i);
                        }
                    }
                }

                if has_inputs {
                    if has_outputs {
                        ui.add_space(4.0);
                        ui.separator();
                    }
                    ui.label(
                        RichText::new("INPUT  (microphones)")
                            .color(Color32::from_gray(150))
                            .size(10.0)
                            .strong(),
                    );
                    for (i, d) in self.devices.iter().enumerate() {
                        if d.kind != SourceKind::Input {
                            continue;
                        }
                        let selected = self.selected_device == Some(i);
                        if ui.selectable_label(selected, &d.name).clicked() {
                            self.selected_device = Some(i);
                        }
                    }
                }

                if !has_inputs && !has_outputs {
                    ui.label(
                        RichText::new("no devices found").color(Color32::from_gray(130)),
                    );
                }
                })
        }).response;

        if metering_active {
            combo_resp.on_hover_text("Stop metering to change the audio source.");
        }
    }

    fn start_or_restart(&mut self) {
        if self.engine.is_some() {
            self.stop_stream();
        }
        if let Err(err) = self.start_stream() {
            self.status = format!("Error: {err}");
        }
    }

    fn loudness_panel(&mut self, ui: &mut egui::Ui, width: f32) {
        let state = *self.loudness.read();
        // Each box's Frame adds 8 px inner_margin on left + right (16 px
        // total).  Two boxes share a 6 px gap.  The *content* width we
        // ask the inner ui to lock to must leave room for both, or the
        // right box overflows the panel's clip rect and gets its rounded
        // corner sliced by the clip — which reads as a phantom shadow.
        let cell_w = (((width - 6.0) * 0.5) - 16.0).floor().max(60.0);

        ui.horizontal(|ui| {
            lufs_box(
                ui,
                "Integrated (LONG)",
                state.integrated_lufs,
                "LUFS",
                cell_w,
                true,
            );
            ui.add_space(6.0);
            lufs_box(
                ui,
                "Short-term (S)",
                state.short_lufs,
                "LUFS",
                cell_w,
                false,
            );
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            lufs_box(
                ui,
                "Momentary (M)",
                state.momentary_lufs,
                "LUFS",
                cell_w,
                false,
            );
            ui.add_space(6.0);
            lufs_box(
                ui,
                "True Peak",
                state.true_peak_dbtp,
                "dBTP",
                cell_w,
                state.true_peak_dbtp > -1.0,
            );
        });
        ui.add_space(6.0);
        ui.vertical_centered(|ui| {
            if ui
                .add_sized([width, 22.0], egui::Button::new("↺  Reset integrated / peak"))
                .clicked()
            {
                self.loudness.write().reset_request = true;
            }
        });
    }

    fn meters_and_graph(&mut self, ui: &mut egui::Ui, now: f64) {
        // Leave a little breathing room between the meter bottoms and
        // the window / status bar so the readouts don't feel crammed
        // against the edge.
        const METERS_BOTTOM_PAD: f32 = 18.0;
        let row_h = (ui.available_height() - METERS_BOTTOM_PAD)
            .max(crate::meter_widget::METER_MIN_HEIGHT);
        ui.allocate_ui_with_layout(
            egui::Vec2::new(ui.available_width(), row_h),
            Layout::left_to_right(Align::TOP),
            |ui| {
            ui.allocate_ui_with_layout(
                egui::Vec2::new(
                    (self.meter_states.len() as f32) * (crate::meter_widget::METER_WIDTH + 4.0) + 60.0,
                    ui.available_height(),
                ),
                Layout::left_to_right(Align::TOP),
                |ui| {
                    ui.add_space(6.0);
                    for (i, state) in self.meter_states.iter().enumerate() {
                        let label = channel_label(self.channels, i).to_string();
                        show_meter(
                            ui,
                            MeterPaintArgs {
                                label: &label,
                                mode: self.mode,
                                state,
                            },
                            now,
                        );
                        ui.add_space(2.0);
                    }
                    show_reference_label(ui, self.mode);
                },
            );

            match self.panel {
                GraphView::Off => {}
                GraphView::History => {
                    ui.add_space(4.0);
                    ScrollArea::horizontal()
                        .id_salt("graph_scroll")
                        .show(ui, |ui| {
                            show_history_graph(
                                ui,
                                &self.history,
                                self.mode,
                                now,
                                self.clip_settings.enabled(),
                                self.clip_settings.threshold_db(),
                                &self.clip_status,
                            );
                        });
                }
                GraphView::Spectrum => {
                    ui.add_space(4.0);
                    ScrollArea::horizontal()
                        .id_salt("spectrum_scroll")
                        .show(ui, |ui| {
                            show_spectrum(
                                ui,
                                &mut self.analyzer,
                                &self.spectrum,
                                now,
                                &mut self.spectrum_pretty_mode,
                                &mut self.spectrum_anchors,
                            );
                        });
                }
            }
        });
    }
}

/// One-shot GL capability probe. We ask for 8× MSAA in `main.rs`, but the driver is
/// free to give us fewer samples (or none) - NVIDIA Optimus and some Intel Arc
/// profiles force "application-controlled MSAA" off regardless of what the app
/// requests. Logging the actual `GL_SAMPLES` count makes the difference between
/// "the trace is stair-stepped because the anti-aliasing plan is wrong" and "the
/// trace is stair-stepped because MSAA silently isn't running" trivially debuggable.
fn log_gl_caps(cc: &eframe::CreationContext<'_>) {
    let Some(gl) = cc.gl.as_ref() else {
        log::warn!("No glow context available at startup (wgpu renderer?)");
        return;
    };
    use eframe::glow::HasContext;
    unsafe {
        let samples = gl.get_parameter_i32(eframe::glow::SAMPLES);
        let sample_buffers = gl.get_parameter_i32(eframe::glow::SAMPLE_BUFFERS);
        let renderer = gl.get_parameter_string(eframe::glow::RENDERER);
        let version = gl.get_parameter_string(eframe::glow::VERSION);
        log::info!(
            "GL renderer='{renderer}' version='{version}' MSAA samples={samples} sample_buffers={sample_buffers}"
        );
        if samples < 2 {
            log::warn!(
                "MSAA not active (GL_SAMPLES={samples}). Requested 8× in main.rs but the driver declined. \
                 Check NVIDIA/Intel control-panel 'Antialiasing - Mode' and set it to 'Application-controlled'."
            );
        }
    }
}

fn apply_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG;
    visuals.window_fill = BG;
    visuals.extreme_bg_color = Color32::from_rgb(0x08, 0x08, 0x0c);
    visuals.widgets.noninteractive.bg_fill = PANEL_BG;
    visuals.widgets.inactive.bg_fill = PANEL_BG;
    visuals.widgets.inactive.weak_bg_fill = PANEL_BG;
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(0x22, 0x22, 0x2a);
    visuals.widgets.active.bg_fill = Color32::from_rgb(0x2c, 0x2c, 0x36);
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, DIVIDER);
    visuals.override_text_color = Some(Color32::from_rgb(0xd8, 0xd8, 0xe0));
    ctx.set_visuals(visuals);
    // Widen egui's stroke feathering from the default 1.0 px to 1.5 px. egui's
    // tessellator adds a fade-to-transparent skirt that wide on each side of every
    // stroke; it's what keeps thin `Shape::line` calls smooth. On 1.5× DPI Windows
    // scaling the default 1.0 px feather is only 1.5 physical px wide, which on
    // shallow diagonals in the spectrum trace still shows faint stair-stepping even
    // with 8× MSAA on top. Bumping to 1.5 logical px = 2.25 physical px on the same
    // display gives the gradient a meaningful foothold across two full pixel rows,
    // which is what actually kills the residual aliasing. Tiny softness penalty on
    // text/vector widgets, imperceptible at normal viewing distances.
    ctx.tessellation_options_mut(|opts| {
        opts.feathering = true;
        opts.feathering_size_in_pixels = 1.5;
    });
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    // Keep the default thin floating scrollbar, but make the hovered
    // width match the dormant width so it doesn't balloon when the
    // cursor is near it.
    let mut scroll = egui::style::ScrollStyle::floating();
    scroll.bar_width = scroll.floating_width;
    scroll.interact_handle_opacity = scroll.active_handle_opacity;
    scroll.interact_background_opacity = scroll.active_background_opacity;
    style.spacing.scroll = scroll;
    ctx.set_style(style);
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n - 1).collect();
        out.push('…');
        out
    }
}

/// Small readout box used in the Loudness panel.
fn lufs_box(
    ui: &mut egui::Ui,
    label: &str,
    value: f32,
    unit: &str,
    width: f32,
    accent: bool,
) {
    let fill = Color32::from_rgb(0x12, 0x14, 0x1c);
    let stroke_col = if accent {
        Color32::from_rgb(0xd4, 0x76, 0x2a)
    } else {
        DIVIDER
    };
    Frame::new()
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, stroke_col))
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.set_min_width(width);
            ui.set_max_width(width);
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(label)
                        .color(Color32::from_gray(140))
                        .size(9.5)
                        .strong(),
                );
                let shown = if value <= NEG_INF_LUFS + 1.0 || !value.is_finite() {
                    "—".to_string()
                } else {
                    format!("{:+.1}", value)
                };
                let num_color = if accent {
                    Color32::from_rgb(0xff, 0xbf, 0x7d)
                } else {
                    TITLE_COL
                };
                ui.label(
                    RichText::new(shown)
                        .color(num_color)
                        .size(16.0)
                        .strong(),
                );
                ui.label(
                    RichText::new(unit)
                        .color(Color32::from_gray(115))
                        .size(9.5),
                );
            });
        });
}

fn section_header(ui: &mut egui::Ui, text: &str) {
    ui.label(
        RichText::new(text)
            .color(Color32::from_rgb(0xb4, 0xb4, 0xcc))
            .size(10.0)
            .strong(),
    );
}

/// Right-hand graph / spectrum panel — one of three mutually exclusive views.
fn graph_panel_checkboxes(ui: &mut egui::Ui, panel: &mut GraphView) {
    let mut hidden = *panel == GraphView::Off;
    if ui.checkbox(&mut hidden, "Hidden").changed() {
        if hidden {
            *panel = GraphView::Off;
        } else if *panel == GraphView::Off {
            *panel = GraphView::History;
        }
    }

    let mut hist = *panel == GraphView::History;
    if ui.checkbox(&mut hist, "Level history").changed() {
        if hist {
            *panel = GraphView::History;
        } else if *panel == GraphView::History {
            *panel = GraphView::Off;
        }
    }

    let mut spec = *panel == GraphView::Spectrum;
    if ui.checkbox(&mut spec, "Spectrum analyzer").changed() {
        if spec {
            *panel = GraphView::Spectrum;
        } else if *panel == GraphView::Spectrum {
            *panel = GraphView::Off;
        }
    }
}

fn start_meter_button(ui: &mut egui::Ui, size: egui::Vec2) -> egui::Response {
    let text = RichText::new("▶  Start meter")
        .size(14.0)
        .color(Color32::from_rgb(0xe6, 0xf5, 0xe6))
        .strong();
    let btn = egui::Button::new(text)
        .fill(Color32::from_rgb(0x2a, 0x6a, 0x36))
        .stroke(egui::Stroke::new(
            1.0,
            Color32::from_rgb(0x4c, 0xaf, 0x50),
        ))
        .min_size(size);
    ui.add(btn)
}

/// Toggle for clip capture: muted red when disarmed, breathing red glow when armed
/// (same pulse feel as the stop meter button).
fn arm_record_trigger_button(
    ui: &mut egui::Ui,
    now: f64,
    armed: bool,
    size: egui::Vec2,
) -> egui::Response {
    if armed {
        let t = now as f32;
        let s = 0.5 + 0.5 * (t * std::f32::consts::TAU * 0.9).sin();

        let base = Color32::from_rgb(0x5a, 0x1c, 0x20);
        let peak = Color32::from_rgb(0xd0, 0x38, 0x40);
        let fill = lerp_color(base, peak, s);

        let stroke_base = Color32::from_rgb(0xe0, 0x58, 0x60);
        let stroke_peak = Color32::from_rgb(0xff, 0x98, 0xa0);
        let stroke_col = lerp_color(stroke_base, stroke_peak, s);

        let text = RichText::new("●  Arm Record Trigger")
            .size(13.0)
            .color(Color32::from_rgb(0xff, 0xf4, 0xf4))
            .strong();
        let btn = egui::Button::new(text)
            .fill(fill)
            .stroke(egui::Stroke::new(1.5, stroke_col))
            .min_size(size);

        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(16));
        ui.add(btn)
    } else {
        let text = RichText::new("○  Arm Record Trigger")
            .size(13.0)
            .color(Color32::from_rgb(0xee, 0xcc, 0xd0))
            .strong();
        let btn = egui::Button::new(text)
            .fill(Color32::from_rgb(0x34, 0x1a, 0x1e))
            .stroke(egui::Stroke::new(
                1.0,
                Color32::from_rgb(0x72, 0x3c, 0x42),
            ))
            .min_size(size);
        ui.add(btn)
    }
}

fn breathing_stop_button(ui: &mut egui::Ui, now: f64, size: egui::Vec2) -> egui::Response {
    // Gentle ~0.9 Hz sine pulse — similar feel to a recording / live light.
    let t = now as f32;
    let s = 0.5 + 0.5 * (t * std::f32::consts::TAU * 0.9).sin();

    let base = Color32::from_rgb(0x58, 0x18, 0x18);
    let peak = Color32::from_rgb(0xc0, 0x2f, 0x2f);
    let fill = lerp_color(base, peak, s);

    let stroke_base = Color32::from_rgb(0xd8, 0x58, 0x58);
    let stroke_peak = Color32::from_rgb(0xff, 0x8a, 0x8a);
    let stroke_col = lerp_color(stroke_base, stroke_peak, s);

    let text = RichText::new("■  Stop meter")
        .size(14.0)
        .color(Color32::from_rgb(0xff, 0xee, 0xee))
        .strong();
    let btn = egui::Button::new(text)
        .fill(fill)
        .stroke(egui::Stroke::new(1.5, stroke_col))
        .min_size(size);

    // Keep the animation smooth while the meter is running.
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(16));
    ui.add(btn)
}

fn draw_badge(ui: &mut egui::Ui, label: &str, bg: Color32) {
    let text = RichText::new(label)
        .size(9.5)
        .color(Color32::from_rgb(0xf0, 0xf0, 0xf6))
        .strong();
    let btn = egui::Button::new(text)
        .fill(bg)
        .stroke(egui::Stroke::NONE)
        .sense(egui::Sense::hover())
        .min_size(egui::vec2(46.0, 16.0));
    ui.add(btn);
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| -> u8 {
        (x as f32 + (y as f32 - x as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}
