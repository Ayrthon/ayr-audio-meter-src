//! Main application: device picker, meter mode selector, level meters
//! and scrolling history graph.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use eframe::egui;
use egui::epaint::{Mesh, Vertex, WHITE_UV};
use egui::{
    Align, Align2, Color32, CornerRadius, CursorIcon, FontId, Frame, Label, Layout, Margin,
    OpenUrl, RichText, ScrollArea, Sense, Shape, Stroke, StrokeKind, TextureHandle, TextureOptions,
    Widget,
};
use egui::collapsing_header::CollapsingState;
use egui::containers::menu::menu_style;
use egui::style::StyleModifier;
use parking_lot::Mutex;

use crate::audio::{
    AudioEngine, DeviceInfo, SharedClipTriggerStates, SharedHistoryTrace, SharedLevels, SourceKind,
    default_input_name, default_output_name, list_devices, make_shared_clip_trigger_states,
    make_shared_history_trace, make_shared_levels,
};
use crate::history_graph::{SharedHistory, make_shared_history, show_history_graph};
use crate::loudness::{NEG_INF_LUFS, SharedLoudness, make_shared_loudness};
use crate::meter::{FillSource, MeterMode, config_for, wall_clock_secs};
use crate::meter_widget::{
    MeterPaintArgs, MeterState, channel_label, show_meter, show_reference_label,
};
use crate::clip_capture::{
    make_shared_clip_capture, make_shared_clip_settings, make_shared_clip_status, SharedClipCapture,
    SharedClipSettings, SharedClipStatus,
};
use crate::clip_playlist_ui::ClipPlaylistState;
use crate::user_settings::{
    clips_path_display, load_settings_file, make_shared_user_settings, resolve_clips_dir,
    save_settings_file, SharedUserSettings,
};
use crate::spectrum::{SharedSpectrum, SpectrumAnalyzer, make_shared_spectrum};
use crate::spectrum_view::show_spectrum;
use crate::weighting::{FreqWeighting, MeterScratch};

/// Darkest edge of the vignetted metal background (`#232328`, matches `paint_metallic_background` edge).
const BG: Color32 = Color32::from_rgb(0x23, 0x23, 0x28);
const PANEL_BG: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x30);
/// Outer “card” fill for each left-panel section (reference: ~`#18181B`).
const LEFT_SECTION_CARD_BG: Color32 = Color32::from_rgb(0x18, 0x18, 0x1b);
/// 1 px outer card outline (reference: black edge).
const LEFT_SECTION_CARD_OUTER_STROKE: Color32 = Color32::BLACK;
/// Subtle inset highlight along the top of stacked bands (title row, content row).
const LEFT_SECTION_INNER_HIGHLIGHT: Color32 =
    Color32::from_rgba_unmultiplied_const(255, 255, 255, 0x22);
/// Upper line of the header / body separator (dark, above the highlight).
const LEFT_SECTION_DIVIDER_DARK: Color32 = Color32::from_rgb(0x12, 0x12, 0x14);
/// 1 px stroke for nested panels (clips path, etc.) — slightly softer than pure black.
const LEFT_SECTION_INNER_BOX_STROKE: Color32 = Color32::from_rgb(0x2d, 0x2d, 0x30);
/// Accordion title strip (matches card top; same as card bg for a flat header).
const LEFT_SECTION_HEADER_BG: Color32 = Color32::from_rgb(0x18, 0x18, 0x1b);
/// Body fill under each section header (`#2C2B30`).
const LEFT_SECTION_BODY_BG: Color32 = Color32::from_rgb(0x2c, 0x2b, 0x30);
/// Outer corner radius for section cards (reference ~8–12 px).
const LEFT_SECTION_CARD_RADIUS: u8 = 10;
/// ComboBox closed button and popup menu fill (`#1E1C22`).
const LEFT_COMBO_BG: Color32 = Color32::from_rgb(0x1e, 0x1c, 0x22);
/// Small caps / secondary labels in the left column (`#A0A0A0`).
const LEFT_LABEL_SECONDARY: Color32 = Color32::from_rgb(0xa0, 0xa0, 0xa0);
/// Vertical gap between stacked accordion cards in the left column.
const LEFT_SECTION_STACK_GAP: f32 = 8.0;
const DIVIDER: Color32 = Color32::from_rgb(0x2b, 0x2b, 0x35);
pub(crate) const TITLE_COL: Color32 = Color32::from_rgb(0xf0, 0xf0, 0xf6);
const ACCENT: Color32 = Color32::from_rgb(0x4c, 0xaf, 0x50);
const AYRTHON_URL: &str = "https://ayrthon.com";
/// Inset from the main window’s right edge (toolbar, graphs, title credits — same visual weight).
const RIGHT_PANEL_GUTTER: f32 = 32.0;
/// Vertical gap between the transport toolbar and the history/spectrum graph (tighter than `RIGHT_PANEL_GUTTER`).
const RIGHT_PANEL_TOOLBAR_TO_GRAPH_GAP: f32 = 8.0;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LeftPanelTab {
    #[default]
    Controls,
    Loudness,
    Clips,
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
    /// True-peak vs sample-peak for `process_weighted_levels` on the audio thread (follows mode).
    meter_true_peak: Arc<AtomicBool>,
    /// Ballistics + peak-hold on the audio thread for clip threshold (matches the white hold line).
    clip_trigger_states: SharedClipTriggerStates,
    meter_scratch: Arc<Mutex<MeterScratch>>,
    clip_settings: SharedClipSettings,
    clip_capture: SharedClipCapture,
    clip_status: SharedClipStatus,
    /// [`ClipStatus::total_clips_saved`] when the user last turned **Arm Record Trigger** on.
    clip_session_baseline: u64,
    clip_playlist: ClipPlaylistState,
    user_settings: SharedUserSettings,
    user_settings_path: Option<PathBuf>,
    /// Which section of the left column is visible (Controls / Loudness / Clips).
    left_tab: LeftPanelTab,
    /// Per-channel levels fed to the history ring (same as each meter **bar fill**).
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
    /// Wall-clock start of the current metering session (`None` when stopped).
    metering_runtime_start: Option<Instant>,
    channels: usize,
    status: String,
    /// Tiled fine grain for the main window backdrop (cool charcoal noise).
    bg_noise: TextureHandle,
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

/// Fixed outer width of the left column (reserved in the main horizontal layout).
const LEFT_COLUMN_OUTER_W: f32 = 340.0;
/// Inner width inside the left column’s padded frame (`inner_margin` 10 px each side).
const LEFT_COLUMN_INNER_W: f32 = LEFT_COLUMN_OUTER_W - 20.0;
/// Reserve space for the vertical scrollbar so combo/slider width does not depend on tab content.
const LEFT_COLUMN_SCROLL_GUTTER: f32 = 20.0;
/// Fixed width for Controls / Loudness / Clips content (must not use `available_width()` — it changes per tab).
const LEFT_COLUMN_CONTROL_W: f32 = (LEFT_COLUMN_INNER_W - LEFT_COLUMN_SCROLL_GUTTER).clamp(228.0, 310.0);
/// Horizontal padding inside `left_panel_collapsing` section body (must match `Frame::inner_margin` X there).
const LEFT_SECTION_CARD_BODY_PAD_X: i8 = 14;

#[inline]
fn left_panel_section_content_w(column_w: f32) -> f32 {
    (column_w - 2.0 * LEFT_SECTION_CARD_BODY_PAD_X as f32).max(1.0)
}

/// LUFS 2×2 grid uses this fraction of the section body width so tiles sit inside the accordion comfortably.
const LUFS_GRID_WIDTH_FRAC: f32 = 0.86;

/// Compute how wide the window needs to be to fit the left panel + the
/// N meters + the "0 dB ref" label column (no graph).  Keeping this
/// dynamic means mono / stereo / 5.1 / 7.1 all snap to the right size.
fn window_w_without_graph(channels: usize) -> f32 {
    // Left column outer width (see `LEFT_COLUMN_OUTER_W` + frame margins in layout).
    const LEFT_PANEL: f32 = LEFT_COLUMN_OUTER_W;
    // 6 px gap + vertical separator (~2 px + item_spacing) + 6 px gap.
    const GUTTERS: f32 = 20.0;
    // CentralPanel / window-frame padding on each side.
    const OUTER_PAD: f32 = 18.0;
    // Allocated meters-column width inside `meters_and_graph` (must match
    // `meter_strip_content_width` + any in-strip extras).
    let meter_col = crate::meter_widget::meter_strip_content_width(channels);
    LEFT_PANEL + GUTTERS + meter_col + OUTER_PAD
}

impl AudioMeterApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_theme(&cc.egui_ctx);
        let bg_noise = cc.egui_ctx.load_texture(
            "am_metallic_noise",
            build_metallic_noise_image(),
            TextureOptions::LINEAR_REPEAT,
        );
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

        let (settings_data, settings_path) = load_settings_file();
        let user_settings = make_shared_user_settings(settings_data);

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
            meter_true_peak: Arc::new(AtomicBool::new(config_for(MeterMode::SamplePeak).true_peak)),
            clip_trigger_states: make_shared_clip_trigger_states(),
            meter_scratch: Arc::new(Mutex::new(MeterScratch::new())),
            clip_settings: make_shared_clip_settings(),
            clip_capture: make_shared_clip_capture(),
            clip_status: make_shared_clip_status(),
            clip_session_baseline: 0,
            clip_playlist: ClipPlaylistState::new(user_settings.clone()),
            user_settings,
            user_settings_path: settings_path,
            left_tab: LeftPanelTab::default(),
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
            metering_runtime_start: None,
            channels: 2,
            status: "Idle".into(),
            bg_noise,
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
        self.meter_true_peak
            .store(cfg.true_peak, Ordering::Relaxed);
        *self.clip_trigger_states.lock() = std::array::from_fn(|_| MeterState::default());
        let engine = AudioEngine::start(
            &dev,
            self.meter_true_peak.clone(),
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
            self.user_settings.clone(),
        )?;
        self.engine = Some(engine);
        self.metering_runtime_start = Some(Instant::now());
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
        self.metering_runtime_start = None;
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

    fn tick_meters(&mut self) {
        let now = wall_clock_secs();
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

        // History graph follows the same dB as the coloured bar (peak or RMS per mode).
        let mut ht = self.history_trace.lock();
        for i in 0..crate::audio::MAX_CHANNELS {
            ht[i] = if active && i < ch && i < self.meter_states.len() {
                let s = &self.meter_states[i];
                match cfg.fill_source {
                    FillSource::Peak => s.peak_db,
                    FillSource::Rms => s.rms_db,
                }
            } else {
                crate::dsp::INF_FLOOR_DB
            };
        }

    }

}

impl eframe::App for AudioMeterApp {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        paint_metallic_background(&ui.painter(), ui.max_rect(), &self.bg_noise);
        let ctx = ui.ctx().clone();
        let now = ctx.input(|i| i.time);
        self.tick_meters();
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
                    RichText::new("AYR Audio Meter")
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
                    ui.add_space(RIGHT_PANEL_GUTTER);
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
            inset_horizontal_divider(ui);
            ui.add_space(4.0);

            ui.horizontal_top(|ui| {
                // Fixed width so Controls / Loudness / Clips never change the column width.
                ui.allocate_ui_with_layout(
                    egui::vec2(LEFT_COLUMN_OUTER_W, ui.available_height()),
                    Layout::top_down(Align::Min),
                    |ui| {
                        ui.set_min_width(LEFT_COLUMN_OUTER_W);
                        ui.set_max_width(LEFT_COLUMN_OUTER_W);
                        self.left_controls(ui);
                    },
                );
                // A bit of clearance on the left for the left-column's scrollbar,
                // tighter on the meter side so the divider hugs the meters column.
                ui.add_space(3.0);
                inset_vertical_divider(ui);
                ui.add_space(2.0);
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
    fn persist_user_settings(&self) {
        if let Some(ref p) = self.user_settings_path {
            let data = self.user_settings.lock().clone();
            if let Err(e) = save_settings_file(p, &data) {
                log::warn!("failed to save settings: {e}");
            }
        }
    }

    /// Full clips library: folder chooser, “How capture works”, waveform list.
    fn clips_library_panel(&mut self, ui: &mut egui::Ui, width: f32) {
        let clip_detail = format!(
            "When enabled, saves {:.0} s of audio before and {:.0} s after a block's peak first reaches the threshold. Files are 32-bit float WAV.\n\n\
             Trigger uses the same per-callback peak as the meters: frequency weighting, true-peak vs sample-peak mode, and the hottest channel in that block. \
             The moving meter can still read a little lower between blocks or because of ballistics.",
            crate::clip_capture::PRE_ROLL_SECS,
            crate::clip_capture::POST_ROLL_SECS
        );
        let total_clips = self.clip_status.lock().total_clips_saved;

        left_panel_collapsing(ui, "clips_tab_library", "CLIP LIBRARY", width, LEFT_SECTION_CARD_BODY_PAD_X, |ui| {
            ui.add_space(4.0);
            let ctx_clips = ui.ctx().clone();
            self.clip_playlist.show(ui, &ctx_clips, total_clips);
        });
        ui.add_space(LEFT_SECTION_STACK_GAP);

        left_panel_collapsing(ui, "clips_tab_save_folder", "SAVE FOLDER", width, LEFT_SECTION_CARD_BODY_PAD_X, |ui| {
            ui.add_space(4.0);
            ui.label(
                RichText::new("Current location:")
                    .color(LEFT_LABEL_SECONDARY)
                    .size(9.5),
            );
            ui.add_space(5.0);
            let clips_path_label = clips_path_display(&*self.user_settings.lock());
            Frame::new()
                .fill(LEFT_COMBO_BG)
                .stroke(Stroke::new(1.0, LEFT_SECTION_INNER_BOX_STROKE))
                .corner_radius(CornerRadius::same(6))
                .inner_margin(Margin::symmetric(10, 10))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(clips_path_label)
                            .size(9.5)
                            .color(Color32::from_gray(210))
                            .family(egui::FontFamily::Monospace),
                    );
                });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 0.0);
                if ui
                    .add_sized(
                        [140.0, 28.0],
                        egui::Button::new(RichText::new("Choose folder…").size(11.5)),
                    )
                    .clicked()
                {
                    let start = resolve_clips_dir(&*self.user_settings.lock());
                    let mut dlg = rfd::FileDialog::new();
                    if let Some(ref p) = start {
                        dlg = dlg.set_directory(p);
                    }
                    if let Some(folder) = dlg.pick_folder() {
                        self.user_settings.lock().clips_save_dir = Some(folder);
                        self.persist_user_settings();
                        self.clip_playlist.reload_clip_list();
                    }
                }
                let custom = self
                    .user_settings
                    .lock()
                    .clips_save_dir
                    .is_some();
                if ui
                    .add_enabled(
                        custom,
                        egui::Button::new(RichText::new("Use default folder").size(11.5)),
                    )
                    .clicked()
                {
                    self.user_settings.lock().clips_save_dir = None;
                    self.persist_user_settings();
                    self.clip_playlist.reload_clip_list();
                }
            });
            ui.add_space(10.0);
            if ui
                .add_sized(
                    [ui.available_width(), 34.0],
                    egui::Button::new(RichText::new("📂  Open clips folder").size(12.0)),
                )
                .clicked()
            {
                if let Some(p) = resolve_clips_dir(&*self.user_settings.lock()) {
                    #[cfg(windows)]
                    let _ = std::process::Command::new("explorer").arg(&p).spawn();
                    #[cfg(target_os = "macos")]
                    let _ = std::process::Command::new("open").arg(&p).spawn();
                    #[cfg(all(not(windows), not(target_os = "macos")))]
                    let _ = std::process::Command::new("xdg-open").arg(&p).spawn();
                }
            }
            ui.add_space(6.0);
            ui.label(
                RichText::new("Peaks at or above the threshold start a capture.")
                    .color(LEFT_LABEL_SECONDARY)
                    .size(9.5),
            );
        });
        ui.add_space(LEFT_SECTION_STACK_GAP);

        left_panel_collapsing_with_default(
            ui,
            "clips_tab_how_capture",
            "HOW CAPTURE WORKS",
            width,
            LEFT_SECTION_CARD_BODY_PAD_X,
            false,
            |ui| {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(clip_detail.as_str())
                        .size(11.0)
                        .color(crate::meter_widget::COL_RULER_TEXT),
                );
            },
        );
    }

    fn left_controls_controls_tab(&mut self, ui: &mut egui::Ui, width: f32) {
        let cw = left_panel_section_content_w(width);
        left_panel_collapsing(ui, "left_tab_ctrl_audio", "AUDIO SOURCE", width, LEFT_SECTION_CARD_BODY_PAD_X, |ui| {
            ui.add_space(4.0);
            self.device_dropdown(ui, "device_combo", cw);

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
                        .color(LEFT_LABEL_SECONDARY)
                        .size(10.5),
                );
            });

            ui.add_space(8.0);

            if ui
                .add_sized([cw, 24.0], egui::Button::new("↻  Refresh devices"))
                .clicked()
            {
                self.refresh_devices();
            }
        });

        ui.add_space(LEFT_SECTION_STACK_GAP);

        left_panel_collapsing(ui, "left_tab_ctrl_meter_mode", "METER MODE", width, LEFT_SECTION_CARD_BODY_PAD_X, |ui| {
            ui.add_space(4.0);
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
            let mode_combo_hover = format!(
                "{}\n\n{}",
                self.mode.display_name(),
                mode_zone_hint
            );
            let mode_combo = with_left_panel_combo_style(ui, |ui| {
                egui::ComboBox::from_id_salt("mode_combo")
                    .width(cw)
                    .selected_text(self.mode.name_outside_paren())
                    .popup_style(left_panel_combo_popup_style())
                    .show_ui(ui, |ui| {
                        for mode in MeterMode::ALL {
                            if ui
                                .selectable_label(self.mode == mode, mode.display_name())
                                .clicked()
                            {
                                self.mode = mode;
                                self.clip_meter_mode
                                    .store(mode.to_discriminant(), Ordering::Relaxed);
                                self.meter_true_peak.store(
                                    config_for(mode).true_peak,
                                    Ordering::Relaxed,
                                );
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
                    })
            });
            mode_combo.response.on_hover_text(mode_combo_hover);
        });

        ui.add_space(LEFT_SECTION_STACK_GAP);

        left_panel_collapsing(ui, "left_tab_ctrl_weight", "FREQUENCY WEIGHTING", width, LEFT_SECTION_CARD_BODY_PAD_X, |ui| {
            ui.add_space(4.0);
            let mut w =
                FreqWeighting::from_u8(self.weighting_atomic.load(Ordering::Relaxed));
            let w_combo = with_left_panel_combo_style(ui, |ui| {
                egui::ComboBox::from_id_salt("freq_weight_combo")
                    .width(cw)
                    .selected_text(w.label())
                    .popup_style(left_panel_combo_popup_style())
                    .show_ui(ui, |ui| {
                        for ww in FreqWeighting::ALL {
                            if ui
                                .selectable_label(w == ww, ww.list_entry())
                                .clicked()
                            {
                                w = ww;
                            }
                        }
                    })
            });
            w_combo.response.on_hover_text(
                "Applies to peak, RMS, and level history only.\n\
                 LUFS and the spectrum use full-band BS.1770 K-weighting.",
            );
            self.weighting_atomic.store(w as u8, Ordering::Relaxed);
        });

        ui.add_space(LEFT_SECTION_STACK_GAP);

        let clip_detail_short = format!(
            "Saves {:.0} s before + {:.0} s after the threshold is crossed (32-bit float WAV). Use the Clips tab for the list, waveform, and folder.",
            crate::clip_capture::PRE_ROLL_SECS,
            crate::clip_capture::POST_ROLL_SECS
        );

        left_panel_collapsing(ui, "left_tab_ctrl_clip_cap", "CLIP CAPTURE", width, LEFT_SECTION_CARD_BODY_PAD_X, |ui| {
            ui.add_space(4.0);
            let content_w = ui.available_width();
            ui.set_width(content_w);

            let mut thr = self.clip_settings.threshold_db();
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Threshold")
                        .size(11.5)
                        .color(TITLE_COL)
                        .strong(),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("{thr:.1} dBFS"))
                            .size(12.5)
                            .color(ACCENT)
                            .strong(),
                    );
                });
            });
            ui.add_space(3.0);

            let slider_w = ui.available_width();
            let prev_slider_w = ui.style().spacing.slider_width;
            ui.style_mut().spacing.slider_width = slider_w;
            let rail_h = 18.0_f32;
            let slider_resp = ui.add_sized(
                egui::vec2(slider_w, rail_h),
                egui::Slider::new(
                    &mut thr,
                    crate::clip_capture::CLIP_THRESHOLD_SLIDER_MIN_DB
                        ..=crate::clip_capture::CLIP_THRESHOLD_SLIDER_MAX_DB,
                )
                    .fixed_decimals(1)
                    .show_value(false)
                    .trailing_fill(true),
            );
            ui.style_mut().spacing.slider_width = prev_slider_w;
            if slider_resp.changed() {
                self.clip_settings.set_threshold_db(thr);
            }
            slider_resp.on_hover_text(&clip_detail_short);

            ui.add_space(6.0);
            let lib = ui
                .add_sized(
                    egui::vec2(ui.available_width(), 26.0),
                    egui::Button::new(if self.left_tab == LeftPanelTab::Clips {
                        "Clips library (open)…"
                    } else {
                        "Clips library…"
                    }),
                )
                .on_hover_text(
                    "List, preview, folder, delete — Space plays/pauses the waveform",
                );
            if lib.clicked() {
                self.left_tab = LeftPanelTab::Clips;
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
                            .size(10.5)
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
    }

    fn left_controls(&mut self, ui: &mut egui::Ui) {
        Frame::new()
            .inner_margin(egui::Margin::symmetric(10, 10))
            .show(ui, |ui| {
                ui.set_width(LEFT_COLUMN_INNER_W);
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
                        for (tab, label) in [
                            (LeftPanelTab::Controls, "CONTROLS"),
                            (LeftPanelTab::Loudness, "LOUDNESS"),
                            (LeftPanelTab::Clips, "CLIPS"),
                        ] {
                            let selected = self.left_tab == tab;
                            let text = RichText::new(label).size(12.0).color(if selected {
                                TITLE_COL
                            } else {
                                LEFT_LABEL_SECONDARY
                            });
                            if ui.selectable_label(selected, text).clicked() {
                                self.left_tab = tab;
                            }
                        }
                    });
                    ui.add_space(4.0);
                    inset_horizontal_divider(ui);
                    ui.add_space(4.0);

                    let avail_h = ui.available_height();
                    ScrollArea::vertical()
                        .id_salt("left_controls_scroll")
                        .max_height(avail_h)
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                                // Fixed width — same as `clips_library_panel` (differs by tab / scrollbar if
                                // derived from `available_width()` only).
                                let width = LEFT_COLUMN_CONTROL_W;
                                ui.set_min_width(width);
                                ui.set_max_width(width);
                                match self.left_tab {
                                    LeftPanelTab::Controls => {
                                        self.left_controls_controls_tab(ui, width);
                                    }
                                    LeftPanelTab::Loudness => {
                                        left_panel_collapsing(
                                            ui,
                                            "left_tab_loudness_r128",
                                            "LOUDNESS  (EBU R128)",
                                            width,
                                            LEFT_SECTION_CARD_BODY_PAD_X,
                                            |ui| {
                                                ui.add_space(4.0);
                                                self.loudness_panel(ui);
                                            },
                                        );
                                    }
                                    LeftPanelTab::Clips => {
                                        self.clips_library_panel(ui, width);
                                    }
                                }
                            });
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
            with_left_panel_combo_style(ui, |ui| {
            egui::ComboBox::from_id_salt(id)
                .width(width)
                .selected_text(truncate(&current_label, 36))
                .popup_style(left_panel_combo_popup_style())
                .show_ui(ui, |ui| {
                let has_inputs = self.devices.iter().any(|d| d.kind == SourceKind::Input);
                let has_outputs = self
                    .devices
                    .iter()
                    .any(|d| d.kind == SourceKind::Loopback);

                if has_outputs {
                    ui.label(
                        RichText::new("OUTPUT  (speakers / loopback)")
                            .color(LEFT_LABEL_SECONDARY)
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
                            .color(LEFT_LABEL_SECONDARY)
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
            })
        }).response;

        if metering_active {
            combo_resp.on_hover_text("Stop metering to change the audio source.");
        }
    }

    fn loudness_panel(&mut self, ui: &mut egui::Ui) {
        let state = *self.loudness.read();
        let body_w = ui.available_width().max(1.0);
        // Narrower than full body so the grid matches the collapsible tab width visually.
        let grid_w = (body_w * LUFS_GRID_WIDTH_FRAC).max(1.0);
        // Each box's Frame adds 4 px inner_margin on left + right (8 px
        // total per box).  Two boxes share a 4 px gap.  The *content* width we
        // ask the inner ui to lock to must leave room for both, or the
        // right box overflows the panel's clip rect and gets its rounded
        // corner sliced by the clip — which reads as a phantom shadow.
        const LUFS_PAIR_GAP: f32 = 4.0;
        const LUFS_FRAME_PAD_X: f32 = 4.0;
        let outer_h = 2.0 * LUFS_FRAME_PAD_X;
        let cell_w = (((grid_w - LUFS_PAIR_GAP) * 0.5) - outer_h).floor().max(52.0);
        let pair_outer_w = 2.0 * (cell_w + outer_h) + LUFS_PAIR_GAP;

        let draw_meters = |ui: &mut egui::Ui| {
            ui.set_min_width(pair_outer_w);
            ui.set_max_width(pair_outer_w);
            ui.horizontal(|ui| {
                lufs_box(
                    ui,
                    "Integrated (LONG)",
                    state.integrated_lufs,
                    "LUFS",
                    cell_w,
                    true,
                );
                ui.add_space(LUFS_PAIR_GAP);
                lufs_box(
                    ui,
                    "Short-term (S)",
                    state.short_lufs,
                    "LUFS",
                    cell_w,
                    false,
                );
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                lufs_box(
                    ui,
                    "Momentary (M)",
                    state.momentary_lufs,
                    "LUFS",
                    cell_w,
                    false,
                );
                ui.add_space(LUFS_PAIR_GAP);
                lufs_box(
                    ui,
                    "True Peak",
                    state.true_peak_dbtp,
                    "dBTP",
                    cell_w,
                    state.true_peak_dbtp > -1.0,
                );
            });
        };

        ui.vertical_centered(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(pair_outer_w, 0.0),
                Layout::top_down(Align::Min),
                draw_meters,
            );
        });

        ui.add_space(4.0);

        ui.vertical_centered(|ui| {
            let prev_pad = ui.style().spacing.button_padding;
            ui.style_mut().spacing.button_padding = egui::vec2(15.0, 4.0);
            if ui
                .add(egui::Button::new(
                    RichText::new("↺  Reset integrated / peak").size(11.5),
                ))
                .clicked()
            {
                self.loudness.write().reset_request = true;
            }
            ui.style_mut().spacing.button_padding = prev_pad;
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
                        crate::meter_widget::meter_strip_content_width(self.meter_states.len()),
                        ui.available_height(),
                    ),
                    Layout::left_to_right(Align::TOP),
                    |ui| {
                        let prev_spacing = ui.spacing().item_spacing;
                        ui.spacing_mut().item_spacing = egui::vec2(0.0, prev_spacing.y);

                        ui.add_space(crate::meter_widget::METER_STRIP_MARGIN_START);
                        let n = self.meter_states.len();
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
                            if i + 1 < n {
                                ui.add_space(crate::meter_widget::METER_STRIP_GAP_X);
                            }
                        }
                        ui.add_space(crate::meter_widget::METER_STRIP_GAP_TO_REF);
                        show_reference_label(ui, self.mode, self.channels);
                        ui.add_space(crate::meter_widget::METER_STRIP_MARGIN_END);

                        ui.spacing_mut().item_spacing = prev_spacing;
                    },
                );

                ui.add_space(3.0);
                inset_vertical_divider(ui);
                ui.add_space(3.0);
                let right_col_w = ui.available_width().max(1.0);
                ui.allocate_ui_with_layout(
                    egui::vec2(right_col_w, row_h),
                    Layout::top_down(Align::Min),
                    |ui| {
                        // Same right/bottom inset for transport toolbar (play / stop / arm / runtime)
                        // and for the history or spectrum area below — not skipped.
                        Frame::new()
                            .inner_margin(Margin {
                                left: 0,
                                right: RIGHT_PANEL_GUTTER as i8,
                                top: 0,
                                bottom: RIGHT_PANEL_GUTTER as i8,
                            })
                            .show(ui, |ui| {
                                ui.set_min_height(row_h);
                                ui.set_min_width(ui.available_width());
                                self.right_panel_toolbar(ui);
                                ui.add_space(RIGHT_PANEL_TOOLBAR_TO_GRAPH_GAP);

                                let rest_h = ui.available_height().max(1.0);
                                match self.panel {
                                    GraphView::Off => {}
                                    GraphView::History => {
                                        ScrollArea::horizontal()
                                            .id_salt("graph_scroll")
                                            .max_height(rest_h)
                                            .show(ui, |ui| {
                                                show_history_graph(
                                                    ui,
                                                    &self.history,
                                                    self.mode,
                                                    now,
                                                    &self.clip_settings,
                                                    &self.clip_status,
                                                );
                                            });
                                    }
                                    GraphView::Spectrum => {
                                        ScrollArea::horizontal()
                                            .id_salt("spectrum_scroll")
                                            .max_height(rest_h)
                                            .show(ui, |ui| {
                                                show_spectrum(
                                                    ui,
                                                    &mut self.analyzer,
                                                    &self.spectrum,
                                                    now,
                                                    &mut self.spectrum_pretty_mode,
                                                    &mut self.spectrum_anchors,
                                                    FreqWeighting::from_u8(
                                                        self.weighting_atomic
                                                            .load(Ordering::Relaxed),
                                                    ),
                                                );
                                            });
                                    }
                                }
                            });
                    },
                );
            },
        );
    }

    /// Transport, view mode, and runtime clock above the history / spectrum area.
    fn right_panel_toolbar(&mut self, ui: &mut egui::Ui) {
        const BAR_H: f32 = 34.0;
        const ICON_BTN: egui::Vec2 = egui::vec2(30.0, 26.0);
        // Square: circular record ARM control matches transport row height.
        const ARM_RECORD: egui::Vec2 = egui::vec2(26.0, 26.0);

        let full_w = ui.available_width().max(1.0);
        ui.set_min_width(full_w);

        Frame::new()
            .fill(PANEL_BG)
            .stroke(Stroke::new(1.0, DIVIDER))
            .corner_radius(CornerRadius::same(4))
            .inner_margin(Margin::symmetric(8, 4))
            .show(ui, |ui| {
                ui.set_width(full_w);
                ui.set_min_height(BAR_H);
                // Transport/record pulse ~30fps; runtime clock with ms uses ~60fps when metering.
                if self.metering_runtime_start.is_some() {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(16));
                } else if self.engine.is_some() || self.clip_settings.enabled() {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(33));
                }

                let ctx = ui.ctx().clone();
                let hover_fg_w = ctx.global_style().visuals.widgets.hovered.fg_stroke.width;
                let active_fg_w = ctx.global_style().visuals.widgets.active.fg_stroke.width;
                ctx.global_style_mut(|s| {
                    s.visuals.widgets.hovered.fg_stroke.width = 1.0;
                    s.visuals.widgets.active.fg_stroke.width = 1.0;
                });

                let row_inner_h = (BAR_H - 8.0).max(18.0);
                // Play + gap + stop + gap + arm (matches `ICON_BTN` / `ARM_RECORD` + 6 px gaps).
                const TRANSPORT_CLUSTER_W: f32 =
                    ICON_BTN.x + 6.0 + ICON_BTN.x + 6.0 + ARM_RECORD.x;
                const RUNTIME_COL_W: f32 = 140.0;
                const VIEW_TO_CENTER_GAP: f32 = 8.0;

                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);

                    let total_bar = ui.available_width().max(1.0);

                    // Left: view mode (width measured so transport can be centered on the full bar).
                    let view_lr = ui.vertical(|ui| {
                        ui.label(
                            RichText::new("View")
                                .size(10.5)
                                .color(crate::meter_widget::COL_RULER_TEXT),
                        );
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
                            if ui
                                .selectable_label(self.panel == GraphView::History, "History")
                                .clicked()
                            {
                                self.panel = GraphView::History;
                            }
                            if ui
                                .selectable_label(self.panel == GraphView::Spectrum, "Spectrum")
                                .clicked()
                            {
                                self.panel = GraphView::Spectrum;
                            }
                        });
                    });
                    let view_w = view_lr.response.rect.width();

                    ui.add_space(VIEW_TO_CENTER_GAP);

                    // Symmetric spacers so play/stop/arm sit at horizontal center of the toolbar.
                    let gap_left = (total_bar / 2.0 - TRANSPORT_CLUSTER_W / 2.0)
                        - view_w
                        - VIEW_TO_CENTER_GAP;
                    ui.add_space(gap_left.max(0.0));

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(6.0, 0.0);
                        let playing = self.engine.is_some();

                        let play = ui.add_enabled_ui(!playing, |ui| {
                            ui.add_sized(ICON_BTN, ToolbarTransportIcon::Play)
                        });
                        if play.inner.clicked() {
                            if let Err(e) = self.start_stream() {
                                self.status = format!("Error: {e}");
                            }
                        }

                        let stop = ui.add_enabled_ui(playing, |ui| {
                            ui.add_sized(
                                ICON_BTN,
                                ToolbarTransportIcon::Stop {
                                    meter_active: playing,
                                },
                            )
                        });
                        if stop.inner.clicked() {
                            self.stop_stream();
                        }

                        let rec_on = self.clip_settings.enabled();
                        let arm = ui.add_sized(ARM_RECORD, RecordArmButton { armed: rec_on });
                        if arm.clicked() {
                            let new_on = !rec_on;
                            self.clip_settings.set_enabled(new_on);
                            if new_on {
                                self.clip_session_baseline =
                                    self.clip_status.lock().total_clips_saved;
                            }
                        }
                    });

                    let gap_right = (total_bar - RUNTIME_COL_W)
                        - (total_bar / 2.0 + TRANSPORT_CLUSTER_W / 2.0);
                    ui.add_space(gap_right.max(0.0));

                    // Right: runtime — fixed column, text right-aligned to the panel edge.
                    ui.allocate_ui_with_layout(
                        egui::vec2(RUNTIME_COL_W, row_inner_h),
                        Layout::right_to_left(Align::Center),
                        |ui| {
                            ui.vertical(|ui| {
                                ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                                    ui.label(
                                        RichText::new("Current runtime")
                                            .size(10.5)
                                            .color(crate::meter_widget::COL_RULER_TEXT),
                                    );
                                });
                                ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                                    ui.label(
                                        RichText::new(format_runtime_hms(
                                            self.metering_runtime_start,
                                        ))
                                        .size(12.5)
                                        .family(egui::FontFamily::Monospace)
                                        .color(TITLE_COL)
                                        .strong(),
                                    );
                                });
                            });
                        },
                    );
                });

                ctx.global_style_mut(|s| {
                    s.visuals.widgets.hovered.fg_stroke.width = hover_fg_w;
                    s.visuals.widgets.active.fg_stroke.width = active_fg_w;
                });
            });
    }
}

/// Transport play / stop — custom paint so hover never changes stroke width or layout
/// (unlike [`egui::Button`], which uses interactive widget styles).
#[derive(Clone, Copy)]
enum ToolbarTransportIcon {
    Play,
    /// When enabled, `meter_active` is true and the stop control “breathes” (meter running).
    Stop { meter_active: bool },
}

impl ToolbarTransportIcon {
    const GLYPH_PX: f32 = 14.0;
    const RADIUS: CornerRadius = CornerRadius::same(4);

    fn glyph(self) -> &'static str {
        match self {
            ToolbarTransportIcon::Play => "▶",
            ToolbarTransportIcon::Stop { .. } => "■",
        }
    }

    /// Original slate — used when play/stop is **disabled** (wrong mode for that control).
    fn inactive_slate_colors(hovered: bool) -> (Color32, Color32, Color32) {
        let fill = Color32::from_rgb(0x36, 0x3e, 0x4a);
        let stroke = if hovered {
            Color32::from_rgb(0x72, 0x7c, 0x8a)
        } else {
            Color32::from_rgb(0x52, 0x5c, 0x6a)
        };
        let fg = Color32::from_rgb(0xe8, 0xee, 0xf6);
        (fill, stroke, fg)
    }

    /// Enabled play — static green (stop uses breathing green instead; vice versa).
    fn play_colors(hovered: bool) -> (Color32, Color32, Color32) {
        let fill = Color32::from_rgb(0x2a, 0x6a, 0x36);
        let stroke = if hovered {
            Color32::from_rgb(0x5c, 0xc8, 0x68)
        } else {
            Color32::from_rgb(0x4c, 0xaf, 0x50)
        };
        let fg = Color32::from_rgb(0xe6, 0xf5, 0xe6);
        (fill, stroke, fg)
    }

    /// Enabled stop while metering — pulses **green** (deeper than play).
    fn stop_colors(hovered: bool, meter_active: bool, breath: f32) -> (Color32, Color32, Color32) {
        let fill_lo = Color32::from_rgb(0x1f, 0x52, 0x2e);
        let fill_hi = Color32::from_rgb(0x32, 0x7a, 0x44);
        let fill = if meter_active {
            lerp_color32(fill_lo, fill_hi, breath)
        } else {
            fill_lo
        };

        let stroke_lo = Color32::from_rgb(0x38, 0x8e, 0x4c);
        let stroke_hi = Color32::from_rgb(0x52, 0xc4, 0x6a);
        let stroke = if hovered {
            Color32::from_rgb(0x7e, 0xea, 0x90)
        } else if meter_active {
            lerp_color32(stroke_lo, stroke_hi, breath)
        } else {
            stroke_lo
        };

        let fg = Color32::from_rgb(0xe6, 0xf5, 0xe6);
        (fill, stroke, fg)
    }
}

impl Widget for ToolbarTransportIcon {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, Sense::click());
        if !ui.is_rect_visible(rect) {
            return response;
        }
        let painter = ui.painter_at(rect);
        let over = response.hovered();
        let enabled = ui.is_enabled();
        let time = ui.ctx().input(|i| i.time) as f32;
        let (fill, stroke_col, fg) = if !enabled {
            Self::inactive_slate_colors(over)
        } else {
            match self {
                ToolbarTransportIcon::Play => Self::play_colors(over),
                ToolbarTransportIcon::Stop { meter_active } => {
                    let breath = if meter_active {
                        (time * 2.25_f32).sin() * 0.5 + 0.5
                    } else {
                        0.0
                    };
                    Self::stop_colors(over, meter_active, breath)
                }
            }
        };
        painter.rect(
            rect,
            Self::RADIUS,
            fill,
            Stroke::new(1.0, stroke_col),
            StrokeKind::Inside,
        );
        let font = FontId::proportional(Self::GLYPH_PX);
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            self.glyph(),
            font,
            fg,
        );
        response
    }
}

/// Conventional record control: outer ring + center dot (clip arm).
struct RecordArmButton {
    armed: bool,
}

impl Widget for RecordArmButton {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, Sense::click());
        if !ui.is_rect_visible(rect) {
            return response;
        }
        let painter = ui.painter_at(rect);
        let center = rect.center();
        // Inscribed circle: leave margin so stroke is not clipped by the widget rect.
        let side = rect.width().min(rect.height());
        let r_outer = (side * 0.5 - 2.0).max(1.0);
        let r_dot = r_outer * 0.42;
        let hovered = response.hovered() && ui.is_enabled();
        let time = ui.ctx().input(|i| i.time) as f32;
        let breath = if self.armed {
            (time * 2.4_f32).sin() * 0.5 + 0.5
        } else {
            0.0
        };

        if self.armed {
            let fill_lo = Color32::from_rgb(0x72, 0x14, 0x1c);
            let fill_hi = Color32::from_rgb(0xe8, 0x22, 0x34);
            let fill = lerp_color32(fill_lo, fill_hi, breath);
            let ring_col = lerp_color32(
                Color32::from_rgb(0xc0, 0x38, 0x48),
                Color32::from_rgb(0xff, 0x78, 0x88),
                breath,
            );
            let dot_lo = Color32::from_rgb(0x4a, 0x0a, 0x12);
            let dot_hi = Color32::from_rgb(0xff, 0xa8, 0xb0);
            let dot = lerp_color32(dot_lo, dot_hi, breath);
            let ring = Stroke::new(1.0, ring_col);
            painter.circle_filled(center, r_outer, fill);
            painter.circle_stroke(center, r_outer, ring);
            painter.circle_filled(center, r_dot, dot);
        } else {
            let fill = Color32::from_rgb(0x22, 0x18, 0x1a);
            let ring_col = if hovered {
                Color32::from_rgb(0x78, 0x5a, 0x60)
            } else {
                Color32::from_rgb(0x5c, 0x44, 0x4a)
            };
            let ring = Stroke::new(1.0, ring_col);
            let dot = Color32::from_rgb(0x5a, 0x48, 0x4c);
            painter.circle_filled(center, r_outer, fill);
            painter.circle_stroke(center, r_outer, ring);
            painter.circle_filled(center, r_dot, dot);
        }

        response
    }
}

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
fn format_runtime_hms(start: Option<Instant>) -> String {
    let Some(t0) = start else {
        return "00:00:00.00".to_string();
    };
    let elapsed = t0.elapsed();
    let total_secs = elapsed.as_secs();
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    // Two-digit centiseconds (0–99), not three-digit milliseconds.
    let cs = (elapsed.subsec_millis() / 10).min(99);
    format!("{h:02}:{m:02}:{s:02}.{cs:02}")
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

/// Deterministic hash for procedural noise (no `rand` dependency).
#[inline]
fn hash_bg_noise_u8(x: u32, y: u32) -> u8 {
    let mut h = x
        .wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    ((h ^ (h >> 16)) & 0xFF) as u8
}

/// Cool charcoal grain — tiled over the window for a sandblasted / anodized-metal feel.
fn build_metallic_noise_image() -> egui::ColorImage {
    const W: usize = 128;
    const H: usize = 128;
    let mut pixels = Vec::with_capacity(W * H);
    for y in 0..H {
        for x in 0..W {
            let h = hash_bg_noise_u8(x as u32, y as u32);
            let h2 = hash_bg_noise_u8(x as u32 / 3, y as u32 / 3);
            let h3 = hash_bg_noise_u8(x as u32 / 11, y as u32 / 11);
            let mut v = 0x2cu8
                .saturating_add(h % 20)
                .saturating_add(h2 % 12)
                .saturating_add(h3 % 8);
            v = v.min(0x58);
            // Slight cool slate bias (pro audio UI reference).
            let r = v.saturating_sub(5);
            let g = v.saturating_sub(2);
            let b = v.saturating_add(6);
            pixels.push(Color32::from_rgb(r, g, b));
        }
    }
    egui::ColorImage::new([W, H], pixels)
}

/// Pro-Q / hardware-style backdrop: radial-ish vignette (light pool upper-center), soft sheen,
/// then tiled fine grain (reference: dark anodized panel + subtle noise).
fn paint_metallic_background(
    painter: &egui::Painter,
    rect: egui::Rect,
    noise: &TextureHandle,
) {
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }

    let cx = rect.center().x;
    let cy = rect.top() + rect.height() * 0.20;
    let center = egui::pos2(cx, cy);
    // Vignette: lighter center `#36353b` → darker edge `#232328`.
    let c_col = Color32::from_rgb(0x36, 0x35, 0x3b);
    let edge = Color32::from_rgb(0x23, 0x23, 0x28);

    let tl = rect.left_top();
    let tm = egui::pos2(cx, rect.top());
    let tr = rect.right_top();
    let rm = egui::pos2(rect.right(), cy);
    let br = rect.right_bottom();
    let bm = egui::pos2(cx, rect.bottom());
    let bl = rect.left_bottom();
    let lm = egui::pos2(rect.left(), cy);

    let ring = [tl, tm, tr, rm, br, bm, bl, lm];

    let mut fan = Mesh::default();
    fan.vertices.push(Vertex {
        pos: center,
        uv: WHITE_UV,
        color: c_col,
    });
    for p in ring {
        fan.vertices.push(Vertex {
            pos: p,
            uv: WHITE_UV,
            color: edge,
        });
    }
    for i in 0..8 {
        let i0 = 0_u32;
        let i1 = 1 + i as u32;
        let i2 = 1 + ((i + 1) % 8) as u32;
        fan.indices.extend_from_slice(&[i0, i1, i2]);
    }
    painter.add(Shape::mesh(fan));

    // Soft band under title bar — same palette as the main gradient (lighter → darker).
    let h = rect.height() * 0.28;
    let y1 = rect.top() + h;
    let sheen_tl = Color32::from_rgba_unmultiplied(0x36, 0x35, 0x3b, 0x1a);
    let sheen_tr = Color32::from_rgba_unmultiplied(0x23, 0x23, 0x28, 0x0c);
    let sheen_b = Color32::from_rgba_unmultiplied(0, 0, 0, 0);
    let mut sheen = Mesh::default();
    sheen.vertices.push(Vertex {
        pos: egui::pos2(rect.left(), rect.top()),
        uv: WHITE_UV,
        color: sheen_tl,
    });
    sheen.vertices.push(Vertex {
        pos: egui::pos2(rect.right(), rect.top()),
        uv: WHITE_UV,
        color: sheen_tr,
    });
    sheen.vertices.push(Vertex {
        pos: egui::pos2(rect.right(), y1),
        uv: WHITE_UV,
        color: sheen_b,
    });
    sheen.vertices.push(Vertex {
        pos: egui::pos2(rect.left(), y1),
        uv: WHITE_UV,
        color: sheen_b,
    });
    sheen.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    painter.add(Shape::mesh(sheen));

    // Tiled grain — breaks banding and reads like matte metal.
    let [tw, th] = noise.size();
    let tw = tw.max(1) as f32;
    let th = th.max(1) as f32;
    let tiles_x = rect.width() / tw * 2.2;
    let tiles_y = rect.height() / th * 2.2;
    let uv = egui::Rect::from_min_max(
        egui::pos2(0.0, 0.0),
        egui::pos2(tiles_x, tiles_y),
    );
    // Tint only (noise texels unchanged) — neutral grey between the two gradient stops.
    painter.image(
        noise.id(),
        rect,
        uv,
        Color32::from_rgba_unmultiplied(0x2c, 0x2c, 0x32, 0x24),
    );
}

fn apply_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG;
    visuals.window_fill = BG;
    visuals.extreme_bg_color = Color32::from_rgb(0x23, 0x23, 0x28);
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
    const LABEL_PT: f32 = 10.75;
    const VALUE_PT: f32 = 19.0;
    const UNIT_PT: f32 = 9.0;

    let fill = Color32::from_rgb(0x12, 0x14, 0x1c);
    let stroke = if accent {
        Stroke::new(1.0, Color32::from_rgb(0xd4, 0x76, 0x2a))
    } else {
        Stroke::new(1.0, DIVIDER)
    };
    Frame::new()
        .fill(fill)
        .stroke(stroke)
        .corner_radius(egui::CornerRadius::same(3))
        .inner_margin(egui::Margin::symmetric(4, 2))
        .show(ui, |ui| {
            ui.set_min_width(width);
            ui.set_max_width(width);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
                ui.label(
                    RichText::new(label)
                        .color(Color32::from_gray(150))
                        .size(LABEL_PT)
                        .strong()
                        .line_height(Some(13.5)),
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
                        .size(VALUE_PT)
                        .strong(),
                );
                ui.label(
                    RichText::new(unit)
                        .color(Color32::from_gray(115))
                        .size(UNIT_PT),
                );
            });
        });
}

/// Thin chevron on the right of each accordion header (open = points up, closed = points down).
/// Paints the accordion chevron without its own click sense; the entire header bar handles the click.
fn left_panel_chevron_paint(ui: &mut egui::Ui, openness: f32) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(14.0, 14.0), Sense::hover());
    let open = openness > 0.5;
    let c = rect.center();
    let s = 3.5;
    let stroke = Stroke::new(1.5, Color32::WHITE);
    let p = ui.painter();
    if open {
        let tip = egui::pos2(c.x, c.y - s * 0.45);
        let l = egui::pos2(c.x - s, c.y + s * 0.35);
        let r = egui::pos2(c.x + s, c.y + s * 0.35);
        p.line_segment([l, tip], stroke);
        p.line_segment([tip, r], stroke);
    } else {
        let tip = egui::pos2(c.x, c.y + s * 0.45);
        let l = egui::pos2(c.x - s, c.y - s * 0.35);
        let r = egui::pos2(c.x + s, c.y - s * 0.35);
        p.line_segment([l, tip], stroke);
        p.line_segment([tip, r], stroke);
    }
}

/// Paints a full-width 1 px horizontal band (used for inset highlights and dividers).
fn left_panel_1px_band(ui: &mut egui::Ui, color: Color32) {
    let w = ui.available_width().max(1.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 1.0), Sense::hover());
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, color);
}

/// Two-tone inset horizontal divider: 1 px dark shadow stacked above a 1 px light highlight.
/// Drawn inside a single 2 px rect so the two bands always sit flush (bypasses `item_spacing`).
fn inset_horizontal_divider(ui: &mut egui::Ui) {
    let w = ui.available_width().max(1.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 2.0), Sense::hover());
    let dark = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), 1.0));
    let light = egui::Rect::from_min_size(
        egui::pos2(rect.min.x, rect.min.y + 1.0),
        egui::vec2(rect.width(), 1.0),
    );
    let p = ui.painter();
    p.rect_filled(dark, CornerRadius::ZERO, LEFT_SECTION_DIVIDER_DARK);
    p.rect_filled(light, CornerRadius::ZERO, LEFT_SECTION_INNER_HIGHLIGHT);
}

/// Two-tone inset vertical divider: 1 px dark shadow on the left, 1 px light highlight on the right.
/// Fills the available height of the surrounding horizontal layout.
fn inset_vertical_divider(ui: &mut egui::Ui) {
    let h = ui.available_height().max(1.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(2.0, h), Sense::hover());
    let dark = egui::Rect::from_min_size(rect.min, egui::vec2(1.0, rect.height()));
    let light = egui::Rect::from_min_size(
        egui::pos2(rect.min.x + 1.0, rect.min.y),
        egui::vec2(1.0, rect.height()),
    );
    let p = ui.painter();
    p.rect_filled(dark, CornerRadius::ZERO, LEFT_SECTION_DIVIDER_DARK);
    p.rect_filled(light, CornerRadius::ZERO, LEFT_SECTION_INNER_HIGHLIGHT);
}

/// Collapsible block: one rounded card (black stroke + fill) with stacked inset highlights.
/// `body_pad_x` is horizontal inner padding for the section body (use `0` for full-bleed edge-to-edge content).
fn left_panel_collapsing<R>(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    title: &str,
    section_w: f32,
    body_pad_x: i8,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) {
    left_panel_collapsing_with_default(
        ui,
        id_salt,
        title,
        section_w,
        body_pad_x,
        true,
        add_contents,
    );
}

fn left_panel_collapsing_with_default<R>(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    title: &str,
    section_w: f32,
    body_pad_x: i8,
    default_open: bool,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) {
    let id = ui.make_persistent_id(id_salt);
    let mut state = CollapsingState::load_with_default_open(ui.ctx(), id, default_open);

    // Fixed width matches `LEFT_COLUMN_CONTROL_W` from the scroll body — do not use
    // `available_width()` alone or collapsed headers shrink-wrap via egui's horizontal layout.
    let card_w = section_w.max(1.0);
    let o = state.openness(ui.ctx());
    let show_divider = o > 0.05;
    let card_radius = CornerRadius::same(LEFT_SECTION_CARD_RADIUS);

    ui.set_min_width(card_w);
    ui.set_max_width(card_w);
    ui.set_width(card_w);
    Frame::new()
        .fill(LEFT_SECTION_CARD_BG)
        .stroke(Stroke::new(1.0, LEFT_SECTION_CARD_OUTER_STROKE))
        .corner_radius(card_radius)
        .inner_margin(Margin::ZERO)
        .show(ui, |ui| {
            ui.set_min_width(card_w);
            ui.set_max_width(card_w);
            ui.set_width(card_w);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);

                // Inset highlight along the top of the card (inside the black stroke).
                left_panel_1px_band(ui, LEFT_SECTION_INNER_HIGHLIGHT);

                let header_frame = Frame::new()
                    .fill(LEFT_SECTION_HEADER_BG)
                    .stroke(Stroke::NONE)
                    .corner_radius(CornerRadius::ZERO)
                    .inner_margin(Margin::symmetric(LEFT_SECTION_CARD_BODY_PAD_X, 10))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(6.0, 0.0);
                            ui.label(
                                RichText::new(title.to_uppercase())
                                    .color(TITLE_COL)
                                    .size(12.5)
                                    .strong(),
                            );
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.add_space(8.0);
                                let o = state.openness(ui.ctx());
                                left_panel_chevron_paint(ui, o);
                            });
                        });
                    });
                // Make the full header bar toggle the section (not just the chevron).
                let header_click = ui.interact(
                    header_frame.response.rect,
                    id.with("header_click"),
                    Sense::click(),
                );
                if header_click.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                if header_click.clicked() {
                    state.toggle(ui);
                }

                if show_divider {
                    // Stacked band: dark line, then light highlight (reference “inset” divider).
                    left_panel_1px_band(ui, LEFT_SECTION_DIVIDER_DARK);
                    left_panel_1px_band(ui, LEFT_SECTION_INNER_HIGHLIGHT);
                }

                let _ = state.show_body_unindented(ui, |ui| {
                    ui.set_width(ui.available_width());
                    Frame::new()
                        .fill(LEFT_SECTION_BODY_BG)
                        .stroke(Stroke::NONE)
                        .corner_radius(CornerRadius::ZERO)
                        .inner_margin(Margin::ZERO)
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            // Same top treatment as the title band: light edge on the content block.
                            left_panel_1px_band(ui, LEFT_SECTION_INNER_HIGHLIGHT);
                            Frame::new()
                                .inner_margin(Margin::symmetric(body_pad_x, 12))
                                .show(ui, |ui| add_contents(ui))
                                .inner
                        })
                        .inner
                });

                // Guarantees the card's min_rect stays `card_w` wide when the body is hidden
                // (horizontal layouts otherwise shrink-wrap to the header row).
                ui.allocate_space(egui::vec2(card_w, 0.0));
            });
        });
}

/// Popup list styling for left-panel [`egui::ComboBox`] menus (`Frame::menu` uses [`Visuals::window_fill`]).
fn left_panel_combo_popup_style() -> StyleModifier {
    StyleModifier::new(|s| {
        menu_style(s);
        s.visuals.window_fill = LEFT_COMBO_BG;
    })
}

/// Closed combo button uses [`WidgetVisuals::weak_bg_fill`]; tint hover/active slightly.
fn with_left_panel_combo_style<R>(
    ui: &mut egui::Ui,
    content: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    ui.scope(|ui| {
        let v = &mut ui.style_mut().visuals.widgets;
        let f = LEFT_COMBO_BG;
        let hover = Color32::from_rgb(0x2a, 0x28, 0x2e);
        let active = Color32::from_rgb(0x32, 0x30, 0x36);
        v.inactive.weak_bg_fill = f;
        v.inactive.bg_fill = f;
        v.hovered.weak_bg_fill = hover;
        v.hovered.bg_fill = hover;
        v.active.weak_bg_fill = active;
        v.active.bg_fill = active;
        v.open.weak_bg_fill = f;
        v.open.bg_fill = f;
        let stroke = Stroke::new(1.0, Color32::from_rgb(0x40, 0x3e, 0x48));
        v.inactive.bg_stroke = stroke;
        v.hovered.bg_stroke = stroke;
        v.active.bg_stroke = stroke;
        v.open.bg_stroke = stroke;
        let r = CornerRadius::same(5);
        v.inactive.corner_radius = r;
        v.hovered.corner_radius = r;
        v.active.corner_radius = r;
        v.open.corner_radius = r;
        content(ui)
    })
    .inner
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

