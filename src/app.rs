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
    Align, Align2, Color32, CornerRadius, CursorIcon, FontId, Frame, Label, Layout, Margin, Modal,
    OpenUrl, ProgressBar, RichText, ScrollArea, Sense, Shape, Stroke, StrokeKind, TextureHandle,
    TextureOptions, UiBuilder, Widget,
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
use crate::comp_view::{SharedComp, make_shared_comp, show_comp};
use crate::history_graph::{SharedHistory, make_shared_history, show_history_graph};
use crate::loudness::{NEG_INF_LUFS, SharedLoudness, make_shared_loudness};
use crate::meter::{FillSource, MeterMode, config_for, wall_clock_secs};
use crate::meter_monitor::MeterMonitorBus;
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
    Comp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LeftPanelTab {
    #[default]
    Controls,
    Network,
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
    /// Clips-tab list, waveform preview, and playback scratch state.
    clip_playlist: ClipPlaylistState,
    user_settings: SharedUserSettings,
    user_settings_path: Option<PathBuf>,
    /// Which section of the left column is visible (Controls / Loudness / Clips).
    left_tab: LeftPanelTab,
    /// When true, the full left column is hidden and only a thin collapse/expand
    /// rail remains at the far left. Persisted in `user_settings`.
    left_collapsed: bool,
    /// Mirrors [`Self::left_collapsed`] on the last frame the auto-resize logic saw;
    /// a difference triggers a width snap + min-inner-size update next stable frame
    /// (same pattern as `last_panel`).
    last_left_collapsed: bool,
    /// Last-frame value of the left-column open animation (`0.0` = fully collapsed,
    /// `1.0` = fully expanded). Written by `ui()`, read by `maybe_resize_for_graph`
    /// so it can *defer* the OS window shrink until the slide-out animation has
    /// actually finished (otherwise the meters briefly overflow the shrinking
    /// window edge mid-animation).
    left_open_anim_t: f32,
    /// Per-channel levels fed to the history ring (same as each meter **bar fill**).
    history_trace: SharedHistoryTrace,
    levels: SharedLevels,
    loudness: SharedLoudness,
    spectrum: SharedSpectrum,
    /// Mono-summed envelope + RMS ring feeding the Comp (compression analyzer) view.
    comp: SharedComp,
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
    /// Tap-to-listen: plays the same PCM the meters see through the default output device.
    meter_monitor: Arc<MeterMonitorBus>,
    engine: Option<AudioEngine>,
    /// Wall-clock start of the current metering session (`None` when stopped).
    metering_runtime_start: Option<Instant>,
    channels: usize,
    status: String,
    /// Tiled fine grain for the main window backdrop (cool charcoal noise).
    bg_noise: TextureHandle,
    /// Wall-clock time at which `AudioMeterApp::new` returned. Used on
    /// Windows to decide when to teleport the main window from its
    /// off-screen parking spot to the primary-monitor center (see
    /// `main.rs` for why we no longer block the main thread at startup).
    #[cfg(windows)]
    launched_at: Instant,
    /// Windows-only: set to `true` the first time we teleport the main
    /// window on-screen, so we only send the `OuterPosition` + `Focus`
    /// viewport commands once rather than fighting any user drag later.
    #[cfg(windows)]
    main_window_revealed: bool,
    /// After first reveal, do a few extra recenter passes to account for
    /// late-applied decorated-window sizing on some multi-display setups.
    #[cfg(windows)]
    startup_recenter_passes_left: u8,
    /// Shared state flipped by the background updater thread (see
    /// `updater::spawn_check` in `AudioMeterApp::new`). The UI reads this every
    /// frame in `maybe_show_update_modal` to drive the "Update available" flow.
    update: crate::updater::SharedUpdateStatus,
    /// Set when the user clicks "Later" (or dismisses the modal via Esc / click
    /// outside). Session-only — next launch re-runs the check and the modal can
    /// reappear. We intentionally do not persist a "skipped version" because the
    /// user explicitly asked for a prompt every startup.
    update_dismissed: bool,
    /// Trial / license / activation state, populated by `crate::license::load_or_init`
    /// and subsequently mutated by the background LS activate/validate/deactivate
    /// threads. Read each frame by `maybe_show_license_gate` and
    /// `render_license_status_chip`.
    license: crate::license::SharedLicense,
    /// Background thread (see `license::spawn_activate`) drops per-attempt
    /// activation errors into this slot so the UI can render them inline
    /// without having to mutate the global `license` state. Drained into
    /// `license_activation_error` once per frame.
    license_activation_error_shared: crate::license::SharedActivationError,
    /// User clicked the license chip (or a "Manage License" button) and wants
    /// the modal to pop even when the app isn't in a locked state. Cleared
    /// when the modal is dismissed.
    license_modal_forced: bool,
    /// Scratch buffer for the key-entry `TextEdit` inside the license modal.
    /// Trimmed + uppercased at submit time.
    license_key_input: String,
    /// Human-readable error shown under the key field on a failed activation
    /// attempt. Cleared as soon as the user edits the key again.
    license_activation_error: Option<String>,
    /// Discovery handle for AYR Audio Link senders on the LAN. Browser
    /// thread is owned here so it lives for the app's lifetime. Polled
    /// every `refresh_devices` to splice senders into `self.devices`.
    network_sources: std::sync::Arc<crate::network_source::NetworkSources>,
    /// Last wall-clock moment we reconciled `self.devices` against the
    /// mDNS browser's view of the LAN. Poll every second so senders that
    /// come online appear in the dropdown without the user having to hit
    /// "Refresh" — but throttle so we don't allocate a fresh `Vec` on
    /// every frame.
    network_last_reconcile: Instant,
}

/// Target window widths for the two UI modes.  The app snaps to one
/// of these whenever the user toggles the history graph.
const WINDOW_W_WITH_GRAPH: f32 = 1220.0;

/// Minimum window width when the graph / spectrum panel is visible —
/// enough to keep the right-hand panel usable (in particular the
/// spectrum analyzer needs a few hundred pixels for its log-frequency
/// axis to be readable).  Matches the graph snap width so the user
/// always sees a full-width graph when it's on.
const WINDOW_W_MIN_WITH_GRAPH: f32 = 1220.0;

/// Fixed outer width of the left column (reserved in the main horizontal layout).
const LEFT_COLUMN_OUTER_W: f32 = 340.0;
/// Width of the always-visible collapse/expand rail to the LEFT of the left column.
/// A thin strip with a chevron toggle; clicking hides the full left column and gives
/// the meters / graph its width back.
const LEFT_COLLAPSE_RAIL_W: f32 = 26.0;
/// Horizontal spacer between the rail and whatever sits to its right (the left column
/// when expanded, or the meters divider when collapsed).
const LEFT_COLLAPSE_RAIL_GAP: f32 = 2.0;
/// Duration of the slide animation when the user toggles the left collapse rail.
/// Fast enough to feel snappy (matches egui's default `animation_time`) while still
/// being perceptible as "sliding under the rail" rather than a teleport.
const LEFT_COLLAPSE_ANIM_SECS: f32 = 0.22;
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
///
/// `left_collapsed` accounts for the user having hidden the full 340 px left column
/// via the collapse rail; when true the window can safely shrink by that amount while
/// keeping the rail + dividers + meters in view.
fn window_w_without_graph(channels: usize, left_collapsed: bool) -> f32 {
    // Rail (always visible) + its trailing spacer to the left of everything else.
    const RAIL_BLOCK: f32 = LEFT_COLLAPSE_RAIL_W + LEFT_COLLAPSE_RAIL_GAP;
    // Left column (only contributes when expanded) + its trailing spacer.
    let left_column_block = if left_collapsed {
        0.0
    } else {
        LEFT_COLUMN_OUTER_W + 3.0
    };
    // Vertical divider (~2 px) + small leading spacer before the meters column.
    const DIVIDER_BLOCK: f32 = 4.0;
    // CentralPanel / window-frame padding on each side.
    const OUTER_PAD: f32 = 18.0;
    // Allocated meters-column width inside `meters_and_graph` (must match
    // `meter_strip_content_width` + any in-strip extras).
    let meter_col = crate::meter_widget::meter_strip_content_width(channels);
    RAIL_BLOCK + left_column_block + DIVIDER_BLOCK + meter_col + OUTER_PAD
}

impl AudioMeterApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        crate::startup_log::log("AudioMeterApp::new: begin");
        apply_theme(&cc.egui_ctx);
        crate::startup_log::log("AudioMeterApp::new: theme applied");
        let bg_noise = cc.egui_ctx.load_texture(
            "am_metallic_noise",
            build_metallic_noise_image(),
            TextureOptions::LINEAR_REPEAT,
        );
        crate::startup_log::log("AudioMeterApp::new: bg noise texture uploaded");
        log_gl_caps(cc);
        crate::startup_log::log("AudioMeterApp::new: GL caps logged, calling list_devices");
        let devices = list_devices().unwrap_or_default();
        crate::startup_log::log(format!(
            "AudioMeterApp::new: list_devices returned ({} devices)",
            devices.len()
        ));

        // Prefer the system default output (loopback) so the user sees
        // something move the first time they hit Start.  Fall back to
        // the default input, then to any device.
        crate::startup_log::log("AudioMeterApp::new: resolving default_output_name");
        let default_out = default_output_name();
        crate::startup_log::log(format!(
            "AudioMeterApp::new: default_output_name = {:?}",
            default_out
        ));
        crate::startup_log::log("AudioMeterApp::new: resolving default_input_name");
        let default_in = default_input_name();
        crate::startup_log::log(format!(
            "AudioMeterApp::new: default_input_name = {:?}",
            default_in
        ));
        let selected_device = default_out
            .as_ref()
            .and_then(|n| {
                devices
                    .iter()
                    .position(|d| d.kind == SourceKind::Loopback && d.name == *n)
            })
            .or_else(|| {
                default_in.as_ref().and_then(|n| {
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
        let left_collapsed_from_disk = settings_data.left_panel_collapsed;
        let monitor_device_from_disk = settings_data.monitor_device_name.clone();
        let user_settings = make_shared_user_settings(settings_data);

        let meter_monitor = MeterMonitorBus::new();
        // Restore the persisted monitor-device choice before any capture
        // starts so the first enable opens on the user's previous pick.
        meter_monitor.set_target_device(monitor_device_from_disk);

        let _app = Self {
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
            clip_playlist: ClipPlaylistState::new(user_settings.clone()),
            user_settings,
            user_settings_path: settings_path,
            left_tab: LeftPanelTab::default(),
            left_collapsed: left_collapsed_from_disk,
            last_left_collapsed: left_collapsed_from_disk,
            // Seed the animation state to match the persisted collapse state so the
            // very first frame lays out without a spurious slide animation.
            left_open_anim_t: if left_collapsed_from_disk { 0.0 } else { 1.0 },
            history_trace: make_shared_history_trace(),
            levels: make_shared_levels(),
            loudness: make_shared_loudness(),
            spectrum: make_shared_spectrum(),
            comp: make_shared_comp(),
            analyzer: SpectrumAnalyzer::new(),
            spectrum_pretty_mode: false,
            spectrum_anchors: Vec::new(),
            meter_states,
            history,
            meter_monitor,
            engine: None,
            metering_runtime_start: None,
            channels: 2,
            status: "Idle".into(),
            bg_noise,
            #[cfg(windows)]
            launched_at: Instant::now(),
            #[cfg(windows)]
            main_window_revealed: false,
            #[cfg(windows)]
            startup_recenter_passes_left: 0,
            update: crate::updater::make_shared_status(),
            update_dismissed: false,
            license: crate::license::make_shared_license(),
            license_activation_error_shared:
                crate::license::make_shared_activation_error(),
            license_modal_forced: false,
            license_key_input: String::new(),
            license_activation_error: None,
            network_sources: crate::network_source::NetworkSources::start(),
            network_last_reconcile: Instant::now()
                .checked_sub(std::time::Duration::from_secs(60))
                .unwrap_or_else(Instant::now),
        };
        // Fire the GitHub release check on a background thread. The result
        // flows back into `_app.update` and is picked up by the UI loop next
        // frame. Intentionally done after the struct is constructed so the
        // shared handle is cloned from the same `Arc` the UI will read.
        crate::updater::spawn_check(
            crate::updater::UPDATE_REPO,
            env!("CARGO_PKG_VERSION"),
            _app.update.clone(),
        );
        // Resolve the trial anchor + cached license grant *before* the first
        // frame. If we had a key on disk and its last-validate is older than
        // REVALIDATE_EVERY_DAYS, this call also kicks a background thread to
        // re-hit Lemon Squeezy so a refunded / deactivated key doesn't stay
        // trusted forever.
        crate::license::load_or_init(_app.license.clone());
        crate::startup_log::log("AudioMeterApp::new: end");
        #[allow(clippy::let_and_return)]
        return _app;
    }

    /// Reconciles the current `devices` list with what the mDNS browser
    /// currently sees on the LAN. Preserves any physical (input/loopback)
    /// entries and the user's selection — only the NETWORK entries are
    /// swapped out. Intentionally cheap so it can run once a second from
    /// the UI loop without a visible hiccup.
    fn reconcile_network_sources(&mut self) {
        let discovered = self.network_sources.discovered();

        // Fast path: if the set of service_ids hasn't changed, skip the
        // rebuild. Display names can drift (sender user edits) but the
        // service_id is stable, so that's what we key on.
        let current_ids: Vec<_> = self
            .devices
            .iter()
            .filter(|d| d.kind == SourceKind::Network)
            .filter_map(|d| d.network.as_ref().map(|n| n.service_id.clone()))
            .collect();
        let new_ids: Vec<_> = discovered
            .iter()
            .filter_map(|d| d.network.as_ref().map(|n| n.service_id.clone()))
            .collect();
        if current_ids == new_ids {
            return;
        }

        // Remember selection by service_id / (kind,name) so indices don't
        // silently shift under the user.
        let prev_selected = self
            .selected_device
            .and_then(|i| self.devices.get(i))
            .map(|d| match (&d.kind, &d.network) {
                (SourceKind::Network, Some(n)) => (SourceKind::Network, n.service_id.clone()),
                _ => (d.kind, d.name.clone()),
            });

        self.devices.retain(|d| d.kind != SourceKind::Network);
        self.devices.extend(discovered);

        self.selected_device = prev_selected.and_then(|(k, key)| {
            self.devices.iter().position(|d| match k {
                SourceKind::Network => {
                    d.kind == SourceKind::Network
                        && d.network.as_ref().map(|n| &n.service_id) == Some(&key)
                }
                _ => d.kind == k && d.name == key,
            })
        });
    }

    fn refresh_devices(&mut self) {
        // Remember the user's previous selection so we can restore it after
        // the list is rebuilt. For network sources, the display name can
        // collide across machines, so we key on the mDNS service_id (stable
        // per-sender) instead of the human-readable name.
        #[derive(Clone)]
        enum PrevSel {
            Local(SourceKind, String),
            Network(String),
        }
        let prev_sel = self
            .selected_device
            .and_then(|i| self.devices.get(i))
            .map(|d| match (&d.kind, &d.network) {
                (SourceKind::Network, Some(n)) => PrevSel::Network(n.service_id.clone()),
                _ => PrevSel::Local(d.kind, d.name.clone()),
            });

        let mut devices = list_devices().unwrap_or_default();
        // Splice in any senders the mDNS browser has turned up. The
        // browser is a background thread that lives for the lifetime
        // of the app, so repeated `refresh_devices` calls still see
        // whatever set of senders is currently visible.
        devices.extend(self.network_sources.discovered());
        self.devices = devices;

        self.selected_device = match prev_sel {
            Some(PrevSel::Network(ref sid)) => self.devices.iter().position(|d| {
                d.kind == SourceKind::Network
                    && d.network.as_ref().map(|n| &n.service_id) == Some(sid)
            }),
            Some(PrevSel::Local(k, ref n)) => self
                .devices
                .iter()
                .position(|d| d.kind == k && d.name == *n),
            None => None,
        }
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
            self.comp.clone(),
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
            self.meter_monitor.clone(),
        )?;
        self.engine = Some(engine);
        self.metering_runtime_start = Some(Instant::now());
        let kind_label = match dev.kind {
            SourceKind::Input => "input",
            SourceKind::Loopback => "output (loopback)",
            SourceKind::Network => "network sender",
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
    /// switches the right panel between Off / History / Spectrum — or
    /// toggles the left column collapse/expand rail — but only when
    /// the user hasn't manually resized or maximized the window, and
    /// only when the viewport has been sitting still long enough that
    /// we're sure we aren't mid-drag / mid-DPI-change.
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
                GraphView::Off => {
                    window_w_without_graph(self.meter_states.len(), self.left_collapsed)
                }
                GraphView::History | GraphView::Spectrum | GraphView::Comp => {
                    WINDOW_W_MIN_WITH_GRAPH
                }
            };
            ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
                min_w, 520.0,
            )));
            self.initial_min_sent = true;
        }

        let panel_changed = self.panel != self.last_panel;
        let collapse_changed = self.left_collapsed != self.last_left_collapsed;
        if !panel_changed && !collapse_changed {
            return;
        }

        // Don't fire ANY viewport commands while the user is interacting
        // with the window chrome or while the window hasn't settled
        // after a recent move / resize.  We keep `last_panel` /
        // `last_left_collapsed` unchanged so the next idle frame
        // re-attempts the update.
        let pointer_busy = ctx.input(|i| i.pointer.any_down() || i.pointer.any_pressed());
        if pointer_busy || !rect_stable {
            return;
        }

        // When the user is COLLAPSING the left column, the OS window must only
        // shrink AFTER the slide-out animation has finished — otherwise the
        // meters (pinned to the right of the still-animating column slot) would
        // briefly overflow the shrinking window edge. Expanding is fine to fire
        // immediately: the window widens first, then the column slides in from
        // under the rail into the newly-available space.
        let just_collapsed = collapse_changed && self.left_collapsed;
        if just_collapsed && self.left_open_anim_t > 0.01 {
            return;
        }

        // Update the platform min size so the user can't shrink below
        // what's needed for the current layout.  Safe to send only
        // once we're sure the platform layer is idle.
        let min_w = match self.panel {
            GraphView::Off => {
                window_w_without_graph(self.meter_states.len(), self.left_collapsed)
            }
            GraphView::History | GraphView::Spectrum | GraphView::Comp => {
                WINDOW_W_MIN_WITH_GRAPH
            }
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
            let compact_w =
                window_w_without_graph(self.meter_states.len(), self.left_collapsed);
            let h = inner_rect.map(|r| r.height()).unwrap_or(640.0);
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                compact_w, h,
            )));
            self.last_auto_width = Some(compact_w);
            self.last_panel = self.panel;
            self.last_left_collapsed = self.left_collapsed;
            return;
        }
        if maximized || fullscreen {
            self.last_panel = self.panel;
            self.last_left_collapsed = self.left_collapsed;
            self.last_auto_width = None;
            return;
        }

        // Target width for the current layout. When the right graph panel is on we
        // keep the window at its standard 1140 px regardless of collapse state — the
        // meters + graph simply get more horizontal room when the left column is
        // hidden. When the graph is off we snap to the exact compact width, which
        // changes by ~320 px across the collapse toggle.
        let target_w = match self.panel {
            GraphView::Off => {
                window_w_without_graph(self.meter_states.len(), self.left_collapsed)
            }
            GraphView::History | GraphView::Spectrum | GraphView::Comp => {
                WINDOW_W_WITH_GRAPH
            }
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
                    GraphView::History | GraphView::Spectrum | GraphView::Comp
                );
            // When the user toggles the left column collapse rail while the graph
            // is off, always snap to the new compact width — the old
            // `last_auto_width` was for the *previous* collapse state and would
            // otherwise leave the window 320 px wider than it needs to be
            // (or 320 px too narrow) until the next panel switch.
            let collapse_snap =
                collapse_changed && self.panel == GraphView::Off;
            let snap_to_target =
                hide_panel || opening_right_panel || collapse_snap || !user_resized;
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
        self.last_left_collapsed = self.left_collapsed;
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
        let ctx = ui.ctx().clone();

        // Log the first frame so we can tell from launch.log whether the
        // event loop is actually pumping at all (vs eframe getting wedged
        // during window / GL creation before the first repaint).
        static FIRST_UI: once_cell::sync::OnceCell<()> = once_cell::sync::OnceCell::new();
        let _ = FIRST_UI.get_or_init(|| {
            crate::startup_log::log("ui: first frame");
        });

        // On Windows we launch with the main viewport parked far off-screen
        // so the splash icon (a separate always-on-top Win32 layered window)
        // has the screen to itself during its solo phase. After that phase
        // we teleport the window to the primary-monitor center and pull it
        // to the foreground. Driving the reveal from here — rather than a
        // `thread::sleep` in `main` — keeps the launch within Windows'
        // foreground-privilege window so we can still steal focus even if
        // the user launched us from a fullscreen game.
        #[cfg(windows)]
        if !self.main_window_revealed {
            let elapsed = self.launched_at.elapsed();
            if elapsed >= crate::SPLASH_SOLO_DURATION {
                if let Some(cmd) = egui::ViewportCommand::center_on_screen(&ctx) {
                    ctx.send_viewport_cmd(cmd);
                } else {
                    let (inner_w, inner_h) = ctx
                        .input(|i| i.viewport().inner_rect.map(|r| (r.width(), r.height())))
                        .unwrap_or((crate::MAIN_W, crate::MAIN_H));
                    let center = crate::centered_launch_position(inner_w, inner_h);
                    ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(center));
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                self.main_window_revealed = true;
                self.startup_recenter_passes_left = 8;
                crate::startup_log::log("ui: main window revealed + centered");
            } else {
                // Tick the event loop so we come back right after the solo
                // phase ends rather than sleeping until the next user input.
                let remaining = crate::SPLASH_SOLO_DURATION
                    .saturating_sub(elapsed)
                    .saturating_add(std::time::Duration::from_millis(16));
                ctx.request_repaint_after(remaining);
            }
        } else if self.startup_recenter_passes_left > 0 {
            if let Some(cmd) = egui::ViewportCommand::center_on_screen(&ctx) {
                ctx.send_viewport_cmd(cmd);
            }
            self.startup_recenter_passes_left -= 1;
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }

        paint_metallic_background(&ui.painter(), ui.max_rect(), &self.bg_noise);
        // Render the GitHub-release update modal over the main UI (if any). The
        // modal is drawn as an egui top layer, so calling it here — before the
        // main layout — just means it's composed correctly on top once the rest
        // of the UI has been laid out. No-op while the updater is Idle /
        // Checking / UpToDate / NetworkError / dismissed.
        self.maybe_show_update_modal(&ctx);
        // Same pattern for the paid-license / trial gate. When the user is in
        // TrialExpired / Invalid the modal is un-dismissable until they
        // activate or quit — see `maybe_show_license_gate` for state mapping.
        self.maybe_show_license_gate(&ctx);
        let now = ctx.input(|i| i.time);
        // Freeze meter animation when the license gate is blocking the UI so
        // the dim overlay doesn't sit on top of still-dancing meters. Audio
        // capture keeps running in the background (we don't stop the engine)
        // so unlocking the app resumes instantly from the live signal.
        let license_grants_access = self.license.lock().grants_full_access();
        if license_grants_access {
            self.tick_meters();
        }
        // Reconcile network sources into the device dropdown periodically.
        // Only do this while we're not metering — if the user has already
        // kicked off a stream, mutating `self.devices` would shift indices
        // under the active `selected_device`. The browser thread keeps
        // running regardless, so the next Stop → reconcile picks up any
        // senders that appeared mid-stream.
        if self.engine.is_none()
            && self.network_last_reconcile.elapsed() >= std::time::Duration::from_secs(1)
        {
            self.network_last_reconcile = Instant::now();
            self.reconcile_network_sources();
        }
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
                // Dot separator + update-status chip. The chip rotates between
                // "Checking for updates…" / "Running the latest release (X.Y.Z)" /
                // "Update available — Update now" (green, clickable; reopens the
                // modal) / "Downloading update…" / etc., based on the shared
                // `UpdateStatus` the background thread writes to. See
                // `render_update_status_chip` for the full state mapping.
                ui.label(
                    RichText::new("·")
                        .color(Color32::from_gray(80))
                        .size(11.0),
                );
                self.render_update_status_chip(ui);
                // Second separator + license status chip. Clicking it pops the
                // license gate modal even when the app isn't in a locked state
                // (useful while licensed, so the user can get to Manage
                // License / Deactivate without digging into the left panel).
                ui.label(
                    RichText::new("·")
                        .color(Color32::from_gray(80))
                        .size(11.0),
                );
                self.render_license_status_chip(ui);

                // Right-align credits. In `right_to_left`, the first child is at the far right,
                // so add chunks tail-first: margin → `Ayrthon` (clickable link) → leading
                // `AYR Audio, a brand by ` copy. Reads LTR as `AYR Audio, a brand by Ayrthon`.
                // The version number used to live here ("- Build (…)") but now shows on the
                // left next to the weighted subtitle so it can do double duty as the
                // update-status indicator.
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.add_space(RIGHT_PANEL_GUTTER);
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
                        RichText::new("AYR Audio, a brand by ")
                            .color(Color32::from_gray(130))
                            .size(11.0),
                    );
                });
            });
            ui.add_space(4.0);
            inset_horizontal_divider(ui);
            ui.add_space(4.0);

            ui.horizontal_top(|ui| {
                // Always-on collapse / expand rail at the far left. Clicking toggles
                // `self.left_collapsed`, which on the next stable frame triggers
                // a window width snap via `maybe_resize_for_graph` (gated on the
                // slide animation below so the window doesn't shrink until the
                // column has finished sliding out).
                if paint_left_collapse_rail(ui, self.left_collapsed) {
                    self.left_collapsed = !self.left_collapsed;
                    self.user_settings.lock().left_panel_collapsed = self.left_collapsed;
                    self.persist_user_settings();
                }
                ui.add_space(LEFT_COLLAPSE_RAIL_GAP);

                // Smooth 0..1 slide factor. `animate_bool_with_time` interpolates
                // linearly and internally calls `request_repaint` while the
                // animation is in flight, so we don't need to drive frames manually.
                // We apply a `smoothstep` on top for a gentler ease-in-out curve.
                let open_linear = ui.ctx().animate_bool_with_time(
                    egui::Id::new("left_column_open_anim"),
                    !self.left_collapsed,
                    LEFT_COLLAPSE_ANIM_SECS,
                );
                let open_t = {
                    let x = open_linear.clamp(0.0, 1.0);
                    x * x * (3.0 - 2.0 * x) // smoothstep
                };
                self.left_open_anim_t = open_t;

                // Column slot width animates between 0 and `LEFT_COLUMN_OUTER_W`.
                // The content is rendered at its NATURAL full 340 px width inside
                // a clipped sub-ui whose right edge is pinned to the slot's right
                // edge — so as the slot shrinks, the column appears to slide left
                // *under* the rail rather than reflowing / squeezing its widgets.
                let avail_h = ui.available_height();
                let slot_w = LEFT_COLUMN_OUTER_W * open_t;
                if slot_w > 0.5 {
                    let (slot_rect, _) = ui.allocate_exact_size(
                        egui::vec2(slot_w, avail_h),
                        Sense::hover(),
                    );
                    let content_rect = egui::Rect::from_min_size(
                        egui::pos2(slot_rect.max.x - LEFT_COLUMN_OUTER_W, slot_rect.min.y),
                        egui::vec2(LEFT_COLUMN_OUTER_W, avail_h),
                    );
                    ui.scope_builder(
                        UiBuilder::new()
                            .max_rect(content_rect)
                            .layout(Layout::top_down(Align::Min)),
                        |ui| {
                            // Hide any content that extends past the slot's left
                            // edge while the animation is in progress.
                            ui.set_clip_rect(slot_rect);
                            ui.set_min_width(LEFT_COLUMN_OUTER_W);
                            ui.set_max_width(LEFT_COLUMN_OUTER_W);
                            self.left_controls(ui);
                        },
                    );
                }
                // Trailing scrollbar clearance also animates so it doesn't leave a
                // 3 px dead gap next to the rail once the column is fully collapsed.
                ui.add_space(3.0 * open_t);
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
    /// Inline status chip rendered in the title row (next to the "real-time
    /// dBFS metering" subtitle). Doubles as both the build-version indicator
    /// (it always shows the current version when idle / up-to-date) and the
    /// dismissed-update nag: when an update is available but the user clicked
    /// "Later" on the modal, this chip shows "Update available — Update now"
    /// in green and clicking it re-arms `update_dismissed = false` so the
    /// modal pops back up next frame.
    fn render_update_status_chip(&mut self, ui: &mut egui::Ui) {
        // Prefer the build-label env override (set by CI / `launch-release.ps1`
        // to a git hash) so dev builds show "Version abc1234" rather than
        // claiming to be the Cargo.toml version. Falls back to CARGO_PKG_VERSION.
        let current = audio_meter_build_label();
        let status = { self.update.lock().clone() };

        // Text + color for the non-clickable states. For Available / LaunchFailed
        // we fall through below to render a clickable label instead.
        let (text, color): (String, Color32) = match &status {
            crate::updater::UpdateStatus::Idle
            | crate::updater::UpdateStatus::Checking => (
                "Checking for updates…".to_string(),
                Color32::from_gray(120),
            ),
            crate::updater::UpdateStatus::UpToDate { github_latest } => {
                let msg = if github_latest == current {
                    format!(
                        "Updates checked — v{github_latest} is the latest public release"
                    )
                } else {
                    format!(
                        "Updates checked — you're on {current}; GitHub latest is v{github_latest}"
                    )
                };
                (msg, Color32::from_gray(140))
            }
            crate::updater::UpdateStatus::Downloading { progress, .. } => (
                format!("Downloading update… {:.0}%", (progress * 100.0).clamp(0.0, 100.0)),
                Color32::from_gray(170),
            ),
            crate::updater::UpdateStatus::Downloaded(_)
            | crate::updater::UpdateStatus::Launching => (
                "Launching installer…".to_string(),
                Color32::from_gray(170),
            ),
            // Offline / parse error: quiet chip — confirms we're not claiming up-to-date.
            crate::updater::UpdateStatus::NetworkError(_) => (
                format!("Couldn't reach GitHub for updates — you're on {current}"),
                Color32::from_gray(120),
            ),
            // Clickable green call-to-action states (fall through).
            crate::updater::UpdateStatus::Available(_)
            | crate::updater::UpdateStatus::LaunchFailed(_) => (String::new(), Color32::TRANSPARENT),
        };

        match &status {
            crate::updater::UpdateStatus::Available(info) => {
                let label = format!("Update available ({}) — Update now", info.version);
                let resp = ui.add(
                    Label::new(RichText::new(label).color(ACCENT).strong().size(11.0))
                        .sense(Sense::click()),
                );
                if resp.hovered() {
                    ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
                }
                if resp.clicked() {
                    // Re-arm the modal flow: clear the session-level dismissal
                    // so `maybe_show_update_modal` picks it up next frame.
                    self.update_dismissed = false;
                }
            }
            crate::updater::UpdateStatus::LaunchFailed(_) => {
                // A previous download/launch attempt errored out. Invite a retry
                // with the same chip (clicking re-opens the "Update available"
                // modal via the same `LaunchFailed` state, which the modal
                // renders with an OK button — clicking that dismisses and the
                // next launch will re-check).
                let resp = ui.add(
                    Label::new(
                        RichText::new("Update failed — retry")
                            .color(Color32::from_rgb(0xff, 0xb4, 0xb4))
                            .strong()
                            .size(11.0),
                    )
                    .sense(Sense::click()),
                );
                if resp.hovered() {
                    ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
                }
                if resp.clicked() {
                    self.update_dismissed = false;
                }
            }
            _ => {
                ui.label(RichText::new(text).color(color).size(11.0));
            }
        }
    }

    /// Drive the in-app GitHub-release update flow. Pulls the latest status
    /// from `self.update` once per frame and renders whichever modal matches:
    ///
    /// * `Available` — "Update available" prompt with [Update now] / [Later]
    ///   buttons, plus a "View on GitHub" link if the asset URL is missing.
    /// * `Downloading` — progress bar driven by the download thread's
    ///   per-256 KB `Downloading { progress, .. }` updates. Requests a fast
    ///   repaint so the bar animates smoothly.
    /// * `Downloaded` — spawn the installer detached, flip state to `Launching`
    ///   to avoid double-spawn, then send `ViewportCommand::Close` so Inno
    ///   Setup can replace the running `.exe` cleanly.
    /// * `LaunchFailed` — surfaces the error to the user with a single OK
    ///   button. Any other status is a no-op (silent / internal).
    ///
    /// The modal is suppressed entirely while the splash solo phase is still
    /// running (Windows) and once `update_dismissed` is set for the session.
    fn maybe_show_update_modal(&mut self, ctx: &egui::Context) {
        // Don't pop the modal while the main viewport is still parked
        // off-screen behind the splash — that would look like the app has
        // "launched straight into a dialog" before the user even sees it.
        #[cfg(windows)]
        if !self.main_window_revealed {
            return;
        }
        if self.update_dismissed {
            return;
        }

        // Snapshot the status so we hold the lock for as little time as
        // possible; we only re-acquire it (briefly) if we need to mutate.
        let status = { self.update.lock().clone() };

        match status {
            crate::updater::UpdateStatus::Available(info) => {
                let current = env!("CARGO_PKG_VERSION");
                let mut clicked_update = false;
                let mut clicked_later = false;
                let dlg = Modal::new(egui::Id::new("update_available_modal")).show(ctx, |ui| {
                    ui.set_max_width(420.0);
                    ui.heading(
                        RichText::new(format!("Update available — {}", info.version)).size(16.0),
                    );
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(format!(
                            "You're on {current}. Install {} now?",
                            info.version
                        ))
                        .size(11.5)
                        .color(Color32::from_gray(210)),
                    );

                    if !info.notes.trim().is_empty() {
                        ui.add_space(10.0);
                        ui.label(
                            RichText::new("Release notes")
                                .size(10.5)
                                .color(Color32::from_gray(150)),
                        );
                        ui.add_space(2.0);
                        ScrollArea::vertical()
                            .max_height(180.0)
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(info.notes.trim())
                                        .size(10.5)
                                        .color(Color32::from_gray(195)),
                                );
                            });
                    }

                    if !info.html_url.is_empty() {
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui
                                .link(
                                    RichText::new("View release on GitHub")
                                        .size(10.5)
                                        .color(Color32::from_rgb(0x8a, 0xb4, 0xf8)),
                                )
                                .clicked()
                            {
                                ui.ctx().open_url(OpenUrl::new_tab(&info.html_url));
                            }
                        });
                    }

                    ui.add_space(16.0);
                    ui.horizontal(|ui| {
                        if ui
                            .add(egui::Button::new("Later").min_size(egui::vec2(88.0, 28.0)))
                            .clicked()
                        {
                            clicked_later = true;
                            ui.close();
                        }
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("Update now")
                                            .strong()
                                            .color(Color32::from_rgb(0xb4, 0xe0, 0xff)),
                                    )
                                    .min_size(egui::vec2(120.0, 28.0)),
                                )
                                .clicked()
                            {
                                clicked_update = true;
                                ui.close();
                            }
                        });
                    });
                });

                if clicked_update {
                    // Spawn download; next frame will show the Downloading modal.
                    crate::updater::spawn_download_and_run(info, self.update.clone());
                } else if clicked_later || dlg.should_close() {
                    // Esc / click-outside / Later all mean "not this session".
                    self.update_dismissed = true;
                }
            }

            crate::updater::UpdateStatus::Downloading {
                progress,
                total_bytes,
            } => {
                Modal::new(egui::Id::new("update_downloading_modal")).show(ctx, |ui| {
                    ui.set_max_width(360.0);
                    ui.heading(RichText::new("Downloading update…").size(15.0));
                    ui.add_space(8.0);
                    ui.add(
                        ProgressBar::new(progress)
                            .desired_width(320.0)
                            .animate(progress < 1.0),
                    );
                    if total_bytes > 0 {
                        let mb_total = total_bytes as f32 / (1024.0 * 1024.0);
                        let mb_done = mb_total * progress;
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new(format!("{:.1} / {:.1} MB", mb_done, mb_total))
                                .size(10.5)
                                .color(Color32::from_gray(160)),
                        );
                    }
                });
                // The download thread updates progress only every ~256 KB, but
                // we still want the bar to pulse between those ticks.
                ctx.request_repaint_after(std::time::Duration::from_millis(80));
            }

            crate::updater::UpdateStatus::Downloaded(path) => {
                // Flip to Launching FIRST so we don't re-enter this arm if
                // viewport close is coalesced or if the installer spawn errors
                // asynchronously on some other frame.
                *self.update.lock() = crate::updater::UpdateStatus::Launching;
                match crate::updater::launch_installer(&path) {
                    Ok(()) => {
                        crate::startup_log::log(
                            "updater: installer spawned; requesting viewport close",
                        );
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    Err(e) => {
                        *self.update.lock() =
                            crate::updater::UpdateStatus::LaunchFailed(e.to_string());
                    }
                }
            }

            crate::updater::UpdateStatus::LaunchFailed(msg) => {
                let mut clicked_ok = false;
                let dlg = Modal::new(egui::Id::new("update_failed_modal")).show(ctx, |ui| {
                    ui.set_max_width(380.0);
                    ui.heading(
                        RichText::new("Update failed")
                            .size(15.0)
                            .color(Color32::from_rgb(0xff, 0xb4, 0xb4)),
                    );
                    ui.add_space(6.0);
                    ui.label(RichText::new(&msg).size(11.0).color(Color32::from_gray(210)));
                    ui.add_space(14.0);
                    ui.horizontal(|ui| {
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui
                                .add(egui::Button::new("OK").min_size(egui::vec2(88.0, 28.0)))
                                .clicked()
                            {
                                clicked_ok = true;
                                ui.close();
                            }
                        });
                    });
                });
                if clicked_ok || dlg.should_close() {
                    self.update_dismissed = true;
                }
            }

            // All other states (Idle, Checking, UpToDate, NetworkError,
            // Launching) are UI-silent: no modal, no repaint nudge.
            _ => {}
        }
    }

    fn persist_user_settings(&self) {
        if let Some(ref p) = self.user_settings_path {
            let data = self.user_settings.lock().clone();
            if let Err(e) = save_settings_file(p, &data) {
                log::warn!("failed to save settings: {e}");
            }
        }
    }

    /// Status chip sibling of [`render_update_status_chip`], placed just after
    /// it in the title row. Shows:
    ///
    /// * `Trial: N days left` (or `last day` at N=1) — dim white, clickable;
    ///   opens the license modal preloaded in "enter key / buy" mode.
    /// * `Trial expired — enter license` — amber, clickable; forced modal.
    /// * `Licensed` — muted green, clickable; opens Manage License.
    /// * `Licensed — re-check in N d` — amber, clickable, while inside the
    ///   offline grace slope past REVALIDATE_EVERY_DAYS.
    /// * `License invalid — fix` — soft red, clickable.
    /// * `Updating license…` — dim grey, non-clickable (Activating /
    ///   Deactivating).
    /// * Hidden for `Unknown` / `NetworkError` (no noise at startup).
    fn render_license_status_chip(&mut self, ui: &mut egui::Ui) {
        use crate::license::LicenseStatus;
        let status = { self.license.lock().clone() };

        // Muted green for "you're fine", matches the ACCENT used elsewhere
        // for positive / active states but at lower intensity.
        let licensed_green = Color32::from_rgb(0x7d, 0xc6, 0x7e);
        let warn_amber = Color32::from_rgb(0xff, 0xc4, 0x7a);
        let err_red = Color32::from_rgb(0xff, 0xb4, 0xb4);

        // (text, color, clickable) tuple so the common rendering path is
        // shared. Clickable chips sense input and flip `license_modal_forced`.
        let (text, color, clickable): (String, Color32, bool) = match &status {
            LicenseStatus::Unknown => return,
            LicenseStatus::NetworkError(_) => return,
            LicenseStatus::TrialActive { days_left } => {
                let label = if *days_left <= 1 {
                    "Trial: last day".to_string()
                } else {
                    format!("Trial: {} days left", days_left)
                };
                (label, Color32::from_gray(180), true)
            }
            LicenseStatus::TrialExpired => (
                "Trial expired — enter license".to_string(),
                warn_amber,
                true,
            ),
            LicenseStatus::Licensed {
                offline_days_left, ..
            } => match offline_days_left {
                None => ("Licensed".to_string(), licensed_green, true),
                Some(n) => (
                    format!("Licensed — re-check in {} d", n),
                    warn_amber,
                    true,
                ),
            },
            LicenseStatus::Invalid(_) => {
                ("License invalid — fix".to_string(), err_red, true)
            }
            LicenseStatus::Activating => (
                "Activating license…".to_string(),
                Color32::from_gray(170),
                false,
            ),
            LicenseStatus::Deactivating => (
                "Deactivating…".to_string(),
                Color32::from_gray(170),
                false,
            ),
        };

        if clickable {
            let resp = ui.add(
                Label::new(RichText::new(text).color(color).size(11.0).strong())
                    .sense(Sense::click()),
            );
            if resp.hovered() {
                ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
            }
            if resp.clicked() {
                self.license_modal_forced = true;
                self.license_activation_error = None;
            }
        } else {
            ui.label(RichText::new(text).color(color).size(11.0));
        }
    }

    /// The license-gate equivalent of [`maybe_show_update_modal`]. Auto-shows
    /// an unskippable modal when the user is TrialExpired / Invalid (so the
    /// app is functionally behind a pay/activate wall); otherwise only shows
    /// the modal when `license_modal_forced` is set (user clicked the chip or
    /// a "Manage License" button).
    fn maybe_show_license_gate(&mut self, ctx: &egui::Context) {
        // Defer until the splash has handed off to the main window — no modal
        // should pop while the viewport is still parked off-screen.
        #[cfg(windows)]
        if !self.main_window_revealed {
            return;
        }

        let status = { self.license.lock().clone() };

        // Drain the background activation-error slot once per frame so a
        // failed `spawn_activate` surfaces as an inline error in the modal
        // without mutating the global license state (which would otherwise
        // trap a trial user who mistyped a key).
        if let Some(err) = self.license_activation_error_shared.lock().take() {
            self.license_activation_error = Some(err);
        }

        // First-run "welcome to your trial" info modal. Runs once per Windows
        // user (persisted in `user_settings.license_welcome_shown`) and only
        // when the app legitimately believes we're inside a fresh trial —
        // i.e. ≥ 12 of the 14 days are still on the clock. This guards
        // against surfacing the welcome after a support-driven license.json
        // reset on an old install.
        if let crate::license::LicenseStatus::TrialActive { days_left } = &status {
            let shown = self.user_settings.lock().license_welcome_shown;
            if !shown && *days_left as i64 >= crate::license::TRIAL_DAYS - 1 {
                self.render_trial_welcome_modal(ctx, *days_left);
                // While the welcome modal is up, suppress the license gate
                // modal so the user isn't hit with a key-entry form on their
                // very first launch.
                return;
            }
        }

        let locked = !status.grants_full_access();
        if !locked && !self.license_modal_forced {
            return;
        }
        self.render_license_modal(ctx, &status, locked);
    }

    fn render_trial_welcome_modal(&mut self, ctx: &egui::Context, days_left: u32) {
        let mut click_close = false;
        let mut click_buy = false;
        let dlg = Modal::new(egui::Id::new("license_trial_welcome_modal"))
            .show(ctx, |ui| {
                ui.set_max_width(440.0);
                ui.heading(
                    RichText::new("Welcome to AYR Audio Meter")
                        .color(TITLE_COL)
                        .size(17.0),
                );
                ui.add_space(6.0);
                ui.label(
                    RichText::new(format!(
                        "You have a free {}-day trial — every feature is unlocked. \
                         You're on day 1; {} day{} remaining.",
                        crate::license::TRIAL_DAYS,
                        days_left,
                        if days_left == 1 { "" } else { "s" }
                    ))
                    .size(11.5)
                    .color(Color32::from_gray(215)),
                );
                ui.add_space(8.0);
                ui.label(
                    RichText::new(
                        "When the trial ends, enter a license key to keep going. \
                         You can buy one any time from the License chip in the top bar.",
                    )
                    .size(11.0)
                    .color(Color32::from_gray(175)),
                );
                ui.add_space(14.0);
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Buy a license")
                                    .size(12.0)
                                    .color(Color32::from_gray(220)),
                            )
                            .min_size(egui::vec2(130.0, 28.0)),
                        )
                        .clicked()
                    {
                        click_buy = true;
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Start trial")
                                        .strong()
                                        .color(Color32::from_rgb(0xb4, 0xe0, 0xff)),
                                )
                                .min_size(egui::vec2(110.0, 28.0)),
                            )
                            .clicked()
                        {
                            click_close = true;
                            ui.close();
                        }
                    });
                });
            });
        if click_close || dlg.should_close() {
            self.user_settings.lock().license_welcome_shown = true;
            self.persist_user_settings();
        }
        if click_buy {
            ctx.open_url(OpenUrl::new_tab(crate::license::CHECKOUT_URL));
        }
    }

    fn render_license_modal(
        &mut self,
        ctx: &egui::Context,
        status: &crate::license::LicenseStatus,
        locked: bool,
    ) {
        use crate::license::{self, LicenseStatus};

        // Snapshot values we need inside the closure so we don't need a
        // double-borrow of `self`. User-driven actions set booleans we act on
        // after the closure runs.
        let mut submit_key = false;
        let mut click_buy = false;
        let mut click_deactivate = false;
        let mut click_portal = false;
        let mut click_close = false;
        let mut click_revalidate = false;

        let dlg = Modal::new(egui::Id::new("license_gate_modal")).show(ctx, |ui| {
            ui.set_max_width(460.0);

            // --- Title + subtitle based on current status ---------------
            let (title, subtitle, title_color) = match status {
                LicenseStatus::TrialActive { days_left } => (
                    "Trial active".to_string(),
                    format!(
                        "You have {} day{} of the free trial remaining.",
                        days_left,
                        if *days_left == 1 { "" } else { "s" }
                    ),
                    Color32::from_rgb(0x7d, 0xc6, 0x7e),
                ),
                LicenseStatus::TrialExpired => (
                    "Trial expired".to_string(),
                    "Enter a license key to keep using AYR Audio Meter.".into(),
                    Color32::from_rgb(0xff, 0xc4, 0x7a),
                ),
                LicenseStatus::Invalid(msg) => (
                    "License issue".to_string(),
                    msg.clone(),
                    Color32::from_rgb(0xff, 0xb4, 0xb4),
                ),
                LicenseStatus::Licensed {
                    offline_days_left, ..
                } => {
                    let sub = match offline_days_left {
                        None => "Thanks for supporting AYR Audio.".to_string(),
                        Some(n) => format!(
                            "Thanks for supporting AYR Audio. Offline grace: \
                             {} day{} until a re-check is required.",
                            n,
                            if *n == 1 { "" } else { "s" }
                        ),
                    };
                    ("Manage license".to_string(), sub, Color32::WHITE)
                }
                LicenseStatus::Activating => (
                    "Activating…".to_string(),
                    "Contacting the license server — this normally takes a second or two.".into(),
                    Color32::from_gray(210),
                ),
                LicenseStatus::Deactivating => (
                    "Deactivating…".to_string(),
                    "Releasing this PC from your license.".into(),
                    Color32::from_gray(210),
                ),
                // Unknown / NetworkError never open the modal, but render a
                // safe placeholder if somehow reached.
                _ => ("License".into(), String::new(), Color32::WHITE),
            };

            ui.heading(RichText::new(title).color(title_color).size(17.0));
            if !subtitle.is_empty() {
                ui.add_space(6.0);
                ui.label(
                    RichText::new(subtitle)
                        .size(11.5)
                        .color(Color32::from_gray(210)),
                );
            }

            ui.add_space(12.0);

            // --- Licensed: show summary + Manage actions ----------------
            if let LicenseStatus::Licensed {
                email,
                activations_used,
                activations_limit,
                ..
            } = status
            {
                ui.add_space(2.0);
                if !email.is_empty() {
                    kv_row(ui, "Email", email);
                }
                kv_row(
                    ui,
                    "Activations",
                    &format!("{} of {}", activations_used, activations_limit),
                );
                ui.add_space(14.0);
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Re-check now")
                                    .size(12.0)
                                    .color(Color32::from_gray(220)),
                            )
                            .min_size(egui::vec2(110.0, 28.0)),
                        )
                        .clicked()
                    {
                        click_revalidate = true;
                    }
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Customer portal")
                                    .size(12.0)
                                    .color(Color32::from_gray(220)),
                            )
                            .min_size(egui::vec2(140.0, 28.0)),
                        )
                        .clicked()
                    {
                        click_portal = true;
                    }
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Deactivate this PC")
                                    .size(12.0)
                                    .color(Color32::from_rgb(0xff, 0xc4, 0x7a)),
                            )
                            .min_size(egui::vec2(170.0, 28.0)),
                        )
                        .clicked()
                    {
                        click_deactivate = true;
                    }
                });
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Close")
                                        .strong()
                                        .color(Color32::from_rgb(0xb4, 0xe0, 0xff)),
                                )
                                .min_size(egui::vec2(100.0, 28.0)),
                            )
                            .clicked()
                        {
                            click_close = true;
                            ui.close();
                        }
                    });
                });
                return;
            }

            // --- Unlicensed (trial / expired / invalid) ------------------
            // Key entry row.
            ui.label(
                RichText::new("License key")
                    .size(10.5)
                    .color(Color32::from_gray(160)),
            );
            ui.add_space(3.0);
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.license_key_input)
                    .hint_text("XXXX-XXXX-XXXX-XXXX-XXXX")
                    .desired_width(420.0)
                    .font(egui::TextStyle::Monospace),
            );
            if resp.changed() {
                self.license_activation_error = None;
            }
            if resp.lost_focus()
                && ui.input(|i| i.key_pressed(egui::Key::Enter))
                && !matches!(status, LicenseStatus::Activating)
            {
                submit_key = true;
            }

            if let Some(ref err) = self.license_activation_error {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(err)
                        .size(10.5)
                        .color(Color32::from_rgb(0xff, 0xb4, 0xb4)),
                );
            }

            ui.add_space(12.0);

            let activating = matches!(status, LicenseStatus::Activating);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        !activating,
                        egui::Button::new(
                            RichText::new("Activate")
                                .strong()
                                .color(Color32::from_rgb(0xb4, 0xe0, 0xff))
                                .size(12.0),
                        )
                        .min_size(egui::vec2(110.0, 30.0)),
                    )
                    .clicked()
                {
                    submit_key = true;
                }
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("Buy license — €24.95")
                                .size(12.0)
                                .color(Color32::from_gray(220)),
                        )
                        .min_size(egui::vec2(180.0, 30.0)),
                    )
                    .clicked()
                {
                    click_buy = true;
                }
            });

            // --- Footer: Later button only when NOT locked --------------
            if !locked {
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new("Later").min_size(egui::vec2(92.0, 28.0)),
                            )
                            .clicked()
                        {
                            click_close = true;
                            ui.close();
                        }
                    });
                });
            }
        });

        // --- Handle actions ---------------------------------------------
        if submit_key {
            let key = self.license_key_input.trim().to_string();
            if key.len() < 8 {
                self.license_activation_error =
                    Some("That key doesn't look right. Please check and try again.".into());
            } else {
                self.license_activation_error = None;
                // Clear any stale error from a previous attempt so the UI
                // doesn't flash an out-of-date message while the new activate
                // call is in flight.
                *self.license_activation_error_shared.lock() = None;
                license::spawn_activate(
                    key,
                    self.license.clone(),
                    self.license_activation_error_shared.clone(),
                );
            }
        }
        if click_buy {
            ctx.open_url(OpenUrl::new_tab(license::CHECKOUT_URL));
        }
        if click_portal {
            ctx.open_url(OpenUrl::new_tab(license::CUSTOMER_PORTAL_URL));
        }
        if click_revalidate {
            license::spawn_validate(self.license.clone());
        }
        if click_deactivate {
            license::spawn_deactivate(self.license.clone());
        }

        // Auto-dismiss when a click-outside / Esc was honored by the modal —
        // but only when the user isn't locked (otherwise the modal is
        // effectively required and we re-open on the next frame).
        if (click_close || dlg.should_close()) && !locked {
            self.license_modal_forced = false;
        }
    }

    /// Full clips library: folder chooser, “How capture works”, waveform list.
    fn clips_library_panel(&mut self, ui: &mut egui::Ui, width: f32) {
        let clip_detail = format!(
            "When enabled, saves {:.0} s of audio before and {:.0} s after a block's peak first reaches the threshold. Files are 32-bit float WAV.\n\n\
             Trigger uses the same per-callback peak as the meters: frequency weighting, true-peak vs sample-peak mode, and the hottest channel in that block. \
             The moving meter can still read a little lower between blocks or because of ballistics.\n\n\
             Set the trigger threshold by dragging the horizontal line on the Level History graph (open the History view).",
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
                    SourceKind::Network => (
                        "NETWORK",
                        "AYR Audio Link (LAN)",
                        Color32::from_rgb(0x48, 0x36, 0x78),
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

        left_panel_collapsing(
            ui,
            "left_tab_ctrl_monitor",
            "AUDIO MONITORING",
            width,
            LEFT_SECTION_CARD_BODY_PAD_X,
            |ui| {
                ui.add_space(4.0);
                self.monitor_device_dropdown(ui, "monitor_device_combo", cw);
                ui.add_space(4.0);
                self.monitor_status_line(ui);
            },
        );

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
                    .selected_text(truncate(self.mode.display_name(), 36))
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

    }

    fn left_controls_network_tab(&mut self, ui: &mut egui::Ui, width: f32) {
        let cw = left_panel_section_content_w(width);
        let link_sender_on_lan = self
            .devices
            .iter()
            .any(|d| d.kind == SourceKind::Network);
        let metering_active = self.engine.is_some();
        let network_buffer_editable = link_sender_on_lan && !metering_active;

        left_panel_collapsing(
            ui,
            "left_tab_ctrl_network",
            "NETWORK BUFFER",
            width,
            LEFT_SECTION_CARD_BODY_PAD_X,
            |ui| {
                ui.add_space(4.0);
                ui.add_enabled_ui(network_buffer_editable, |ui| {
                    self.network_latency_preset_dropdown(ui, "network_preset_combo", cw);
                });
                ui.add_space(4.0);
                if !link_sender_on_lan {
                    ui.label(
                        RichText::new(
                            "No AYR Audio Link sender on the network. \
                             Start the Link app and begin broadcasting, then use \
                             Refresh devices above if this section stays dim.",
                        )
                        .color(LEFT_LABEL_SECONDARY)
                        .size(10.5),
                    );
                } else if metering_active {
                    ui.label(
                        RichText::new(
                            "Jitter-buffer target for incoming AYR Audio Link streams. \
                             Applies on the next reconnect.",
                        )
                        .color(LEFT_LABEL_SECONDARY)
                        .size(10.5),
                    );
                    ui.add_space(2.0);
                    ui.label(
                        RichText::new("Stop metering to change the buffer preset.")
                            .color(LEFT_LABEL_SECONDARY)
                            .size(10.5),
                    );
                } else {
                    ui.label(
                        RichText::new(
                            "Jitter-buffer target for incoming AYR Audio Link streams. \
                             Applies on the next reconnect.",
                        )
                        .color(LEFT_LABEL_SECONDARY)
                        .size(10.5),
                    );
                }
            },
        );
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
                            (LeftPanelTab::Network, "NETWORK"),
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
                                    LeftPanelTab::Network => {
                                        self.left_controls_network_tab(ui, width);
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
                    SourceKind::Network => "NET",
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
                let has_network = self
                    .devices
                    .iter()
                    .any(|d| d.kind == SourceKind::Network);

                if has_network {
                    ui.label(
                        RichText::new("NETWORK  (AYR Audio Link)")
                            .color(LEFT_LABEL_SECONDARY)
                            .size(10.0)
                            .strong(),
                    );
                    for (i, d) in self.devices.iter().enumerate() {
                        if d.kind != SourceKind::Network {
                            continue;
                        }
                        let selected = self.selected_device == Some(i);
                        let label = format!("[NET] {}", d.name);
                        if ui.selectable_label(selected, label).clicked() {
                            self.selected_device = Some(i);
                        }
                    }
                    if has_outputs || has_inputs {
                        ui.add_space(4.0);
                        ui.separator();
                    }
                }

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

                if !has_inputs && !has_outputs && !has_network {
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
                let meter_col_w =
                    crate::meter_widget::meter_strip_content_width(self.meter_states.len());
                let meter_strip_ir = ui.allocate_ui_with_layout(
                    egui::Vec2::new(meter_col_w, ui.available_height()),
                    Layout::left_to_right(Align::TOP),
                    |ui| {
                        let prev_spacing = ui.spacing().item_spacing;
                        ui.spacing_mut().item_spacing = egui::vec2(0.0, prev_spacing.y);

                        ui.add_space(crate::meter_widget::METER_STRIP_MARGIN_START);
                        let n = self.meter_states.len();
                        let mut reset_all_meters = false;
                        for (i, state) in self.meter_states.iter_mut().enumerate() {
                            let label = channel_label(self.channels, i).to_string();
                            let meter_resp = show_meter(
                                ui,
                                MeterPaintArgs {
                                    label: &label,
                                    mode: self.mode,
                                    state,
                                },
                                now,
                            );
                            if meter_resp.clicked() {
                                reset_all_meters = true;
                            }
                            if i + 1 < n {
                                ui.add_space(crate::meter_widget::METER_STRIP_GAP_X);
                            }
                        }
                        if reset_all_meters {
                            for s in &mut self.meter_states {
                                s.clear();
                            }
                        }
                        ui.add_space(crate::meter_widget::METER_STRIP_GAP_TO_REF);
                        show_reference_label(ui, self.mode, self.channels);
                        ui.add_space(crate::meter_widget::METER_STRIP_MARGIN_END);

                        ui.spacing_mut().item_spacing = prev_spacing;
                    },
                );

                // Overlay the headphone listen toggle in the reference-label column,
                // vertically aligned with the -40 dBFS ruler mark on the Sample Peak
                // meter and horizontally centered in the ref column (same column as
                // the "0 dB ref" text).
                {
                    let strip_rect = meter_strip_ir.response.rect;
                    let bar_top = strip_rect.top() + crate::meter_widget::BAR_TOP_PAD;
                    let bar_bottom = strip_rect.bottom()
                        - crate::meter_widget::meter_bar_bottom_pad_for_strip(self.channels);
                    // Anchor to the Sample Peak (dBFS) scale regardless of the
                    // currently selected meter mode, so the button position is
                    // stable across VU/K/SMPTE views.
                    let sample_peak_scale =
                        crate::meter::MeterScale::for_mode(crate::meter::MeterMode::SamplePeak);
                    let y_at_minus_40 = sample_peak_scale.dbfs_to_y(-40.0, bar_top, bar_bottom);

                    // Horizontally center in the full visible dark gutter between
                    // the right edge of the last meter bar and the left edge of
                    // the graph container. This gutter spans, from left to right:
                    //   METER_STRIP_GAP_TO_REF
                    // + METER_STRIP_REF_COLUMN_W (where the "0 dB ref" text sits)
                    // + METER_STRIP_MARGIN_END
                    // + outer add_space(3) + divider (2 px) + outer add_space(3)
                    // The divider and outer spaces live *after* strip_rect.right().
                    let last_meter_right = strip_rect.right()
                        - crate::meter_widget::METER_STRIP_MARGIN_END
                        - crate::meter_widget::METER_STRIP_REF_COLUMN_W
                        - crate::meter_widget::METER_STRIP_GAP_TO_REF;
                    let graph_left = strip_rect.right() + 3.0 + 2.0 + 3.0;
                    let gutter_center_x = (last_meter_right + graph_left) * 0.5 - 10.0;

                    let btn_size = egui::vec2(32.0, 26.0);
                    let btn_rect = egui::Rect::from_center_size(
                        egui::pos2(gutter_center_x, y_at_minus_40),
                        btn_size,
                    );
                    ui.scope_builder(
                        UiBuilder::new()
                            .max_rect(btn_rect)
                            .layout(Layout::left_to_right(Align::Center)),
                        |ui| self.meter_listen_toggle(ui),
                    );
                }

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
                                let graph_w = ui.available_width().max(1.0);
                                // Pin graph height to `rest_h` — same constraint the old
                                // `ScrollArea::horizontal().max_height(rest_h)` enforced.
                                // Without this, `allocate_exact_size(ui.available_size())` in the
                                // graph widgets can still see the full column height and overlaps
                                // the transport toolbar.
                                if self.panel != GraphView::Off {
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(graph_w, rest_h),
                                        Layout::top_down(Align::Min),
                                        |ui| {
                                            match self.panel {
                                                GraphView::History => {
                                                    show_history_graph(
                                                        ui,
                                                        &self.history,
                                                        self.mode,
                                                        now,
                                                        &self.clip_settings,
                                                        &self.clip_status,
                                                    );
                                                }
                                                GraphView::Spectrum => {
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
                                                }
                                                GraphView::Comp => {
                                                    show_comp(ui, &self.comp, now);
                                                }
                                                GraphView::Off => {}
                                            }
                                        },
                                    );
                                }
                            });
                    },
                );
            },
        );
    }

    /// Headphone-shaped toggle: routes the metered PCM to the user-chosen
    /// monitor render endpoint (see the AUDIO MONITORING panel). Matches
    /// OBS's explicit-monitor-device model — the button refuses to engage
    /// when no device is picked, or when the chosen device is the same
    /// endpoint we're loopback-capturing (feedback-loop protection).
    fn meter_listen_toggle(&mut self, ui: &mut egui::Ui) {
        let engine_on = self.engine.is_some();
        let listen_on = self.meter_monitor.is_enabled();
        let target = self.meter_monitor.target_device();
        let conflict = self.meter_monitor.self_monitor_conflict();
        let has_target = target.is_some();
        let can_listen = engine_on && has_target && !conflict;

        let tip = if !engine_on {
            "Start metering first to hear the signal.".to_string()
        } else if !has_target {
            "Pick a monitor output in Controls > AUDIO MONITORING first.".to_string()
        } else if conflict {
            format!(
                "Monitor device matches the capture source ({}). Pick a different output to avoid a feedback loop.",
                target.clone().unwrap_or_default()
            )
        } else if listen_on {
            format!(
                "Listen on -> {} — same audio the meters use. Click to turn off.",
                target.clone().unwrap_or_default()
            )
        } else {
            format!(
                "Listen off — click to monitor on {}.  Watch your level, loud sources can clip.",
                target.clone().unwrap_or_default()
            )
        };

        let size = egui::vec2(32.0, 26.0);
        let ir = ui.add_enabled_ui(can_listen || listen_on, |ui| {
            ui.add_sized(size, HeadphoneListenButton { on: listen_on })
        });
        ir.response.on_hover_text(tip);
        if ir.inner.clicked() {
            if listen_on {
                self.meter_monitor.set_enabled(false);
            } else if can_listen {
                self.meter_monitor.set_enabled(true);
            }
        }
    }

    /// Dropdown that matches OBS's "Settings → Audio → Monitoring Device".
    /// Choice is persisted to `user_settings.monitor_device_name` so the
    /// next launch remembers it, and mirrored live into `MeterMonitorBus`
    /// so the output thread can reopen on the new endpoint.
    fn monitor_device_dropdown(&mut self, ui: &mut egui::Ui, id: &str, width: f32) {
        let current = self.meter_monitor.target_device();
        let current_label = current
            .clone()
            .unwrap_or_else(|| "Disabled (no monitoring)".into());

        with_left_panel_combo_style(ui, |ui| {
            egui::ComboBox::from_id_salt(id)
                .width(width)
                .selected_text(truncate(&current_label, 36))
                .popup_style(left_panel_combo_popup_style())
                .show_ui(ui, |ui| {
                    let disabled = current.is_none();
                    if ui
                        .selectable_label(disabled, "Disabled (no monitoring)")
                        .clicked()
                        && !disabled
                    {
                        self.set_monitor_device(None);
                    }
                    ui.separator();
                    let names = crate::audio::list_monitor_output_device_names();
                    if names.is_empty() {
                        ui.label(
                            RichText::new("no output devices found")
                                .color(Color32::from_gray(130)),
                        );
                    } else {
                        ui.label(
                            RichText::new("OUTPUT  (speakers / headphones)")
                                .color(LEFT_LABEL_SECONDARY)
                                .size(10.0)
                                .strong(),
                        );
                        for name in names {
                            let selected = current.as_deref() == Some(name.as_str());
                            if ui.selectable_label(selected, &name).clicked() && !selected {
                                self.set_monitor_device(Some(name));
                            }
                        }
                    }
                });
        });
    }

    fn set_monitor_device(&mut self, name: Option<String>) {
        self.meter_monitor.set_target_device(name.clone());
        self.user_settings.lock().monitor_device_name = name;
    }

    /// Dropdown choosing the receiver-side jitter-buffer preset for AYR
    /// Audio Link streams. Persisted to `user_settings.network_latency_preset`
    /// and read by `run_network_source` on every (re-)subscribe, so a
    /// change here takes effect the next time the sender bounces or the
    /// user re-selects the network source.
    fn network_latency_preset_dropdown(&mut self, ui: &mut egui::Ui, id: &str, width: f32) {
        let current_byte = self.user_settings.lock().network_latency_preset;
        let current = ayr_audio_link_core::wire::LatencyPreset::from_u8(current_byte)
            .unwrap_or(ayr_audio_link_core::wire::LatencyPreset::Home);
        let current_label = current.label();

        with_left_panel_combo_style(ui, |ui| {
            egui::ComboBox::from_id_salt(id)
                .width(width)
                .selected_text(truncate(current_label, 36))
                .popup_style(left_panel_combo_popup_style())
                .show_ui(ui, |ui| {
                    for preset in ayr_audio_link_core::wire::LatencyPreset::ALL {
                        let selected = preset == current;
                        if ui.selectable_label(selected, preset.label()).clicked() && !selected {
                            self.user_settings.lock().network_latency_preset = preset as u8;
                        }
                    }
                });
        });
    }

    /// One-line status under the monitor picker: currently-routed device,
    /// any self-monitor conflict, or an error the output thread surfaced.
    fn monitor_status_line(&mut self, ui: &mut egui::Ui) {
        let target = self.meter_monitor.target_device();
        let source = self.meter_monitor.source_device();
        let conflict = self.meter_monitor.self_monitor_conflict();
        let err = self.meter_monitor.last_error();
        let on = self.meter_monitor.is_enabled();

        let (text, color) = if conflict {
            (
                format!(
                    "Can't monitor: output matches capture source ({}). Pick another device.",
                    source.unwrap_or_default()
                ),
                Color32::from_rgb(0xE0, 0x8B, 0x60),
            )
        } else if let Some(msg) = err {
            (msg, Color32::from_rgb(0xE0, 0x8B, 0x60))
        } else if on {
            match target {
                Some(name) => (
                    format!("Routing meter audio -> {name}"),
                    Color32::from_rgb(0x5D, 0xC4, 0x90),
                ),
                None => ("Listen off".to_string(), LEFT_LABEL_SECONDARY),
            }
        } else if target.is_some() {
            (
                "Pick the headphone button to start monitoring.".to_string(),
                LEFT_LABEL_SECONDARY,
            )
        } else {
            (
                "Pick an output to enable the headphone button.".to_string(),
                LEFT_LABEL_SECONDARY,
            )
        };

        ui.label(RichText::new(text).color(color).size(10.5));
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
                            if ui
                                .selectable_label(self.panel == GraphView::Comp, "Comp")
                                .clicked()
                            {
                                self.panel = GraphView::Comp;
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
                            self.clip_settings.set_enabled(!rec_on);
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

/// Headphone-shaped listen toggle next to the meters. Painted from primitives
/// so it renders regardless of whether the default egui font has 🎧 (U+1F3A7).
struct HeadphoneListenButton {
    on: bool,
}

impl Widget for HeadphoneListenButton {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, Sense::click());
        if !ui.is_rect_visible(rect) {
            return response;
        }
        let painter = ui.painter_at(rect);
        let enabled = ui.is_enabled();
        let hovered = response.hovered() && enabled;

        let (fill_bg, border_col, glyph_col) = if !enabled {
            (
                PANEL_BG,
                Color32::from_rgb(0x38, 0x38, 0x42),
                Color32::from_rgb(0x5a, 0x5a, 0x66),
            )
        } else if self.on {
            (
                Color32::from_rgb(0x14, 0x2a, 0x18),
                ACCENT,
                Color32::from_rgb(0xdc, 0xf3, 0xde),
            )
        } else if hovered {
            (
                Color32::from_rgb(0x2e, 0x2e, 0x36),
                Color32::from_rgb(0x62, 0x62, 0x70),
                Color32::from_rgb(0xe1, 0xe1, 0xea),
            )
        } else {
            (
                PANEL_BG,
                DIVIDER,
                Color32::from_rgb(0xc0, 0xc0, 0xc9),
            )
        };

        painter.rect(
            rect,
            CornerRadius::same(4),
            fill_bg,
            Stroke::new(1.0, border_col),
            StrokeKind::Inside,
        );

        let side = rect.width().min(rect.height());
        let center = rect.center();
        let half = side * 0.5;
        let top = center.y - half * 0.62;
        let band_bottom = center.y - half * 0.14;
        let cup_y = center.y + half * 0.18;
        let ear_r = half * 0.30;
        let ear_offset_x = half * 0.52;
        let band_stroke = (half * 0.20).max(1.6);
        let arc_pts = {
            const N: usize = 18;
            let mut pts = Vec::with_capacity(N);
            for i in 0..N {
                let t = i as f32 / (N - 1) as f32;
                let x = center.x + (t * 2.0 - 1.0) * (ear_offset_x + ear_r * 0.15);
                let dy = {
                    let s = (t * std::f32::consts::PI).sin();
                    -s * (band_bottom - top)
                };
                pts.push(egui::pos2(x, band_bottom + dy));
            }
            pts
        };
        painter.add(egui::Shape::line(arc_pts, Stroke::new(band_stroke, glyph_col)));

        let left_cup = egui::pos2(center.x - ear_offset_x, cup_y);
        let right_cup = egui::pos2(center.x + ear_offset_x, cup_y);
        painter.circle_filled(left_cup, ear_r, glyph_col);
        painter.circle_filled(right_cup, ear_r, glyph_col);

        if self.on {
            painter.circle_filled(left_cup, ear_r * 0.38, ACCENT);
            painter.circle_filled(right_cup, ear_r * 0.38, ACCENT);
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
/// On non-Windows, probes the Glow (OpenGL) capabilities we actually
/// received and logs them. We ask for 8× MSAA but the driver is free
/// to give us fewer samples — NVIDIA Optimus and some Intel Arc
/// profiles force "application-controlled MSAA" off regardless of
/// what the app requests. Logging `GL_SAMPLES` makes the difference
/// between "trace is stair-stepped because AA plan is wrong" and
/// "trace is stair-stepped because MSAA silently isn't running"
/// trivially debuggable.
///
/// On Windows the function is a thin one-liner — the binary doesn't
/// compile the `glow` feature on purpose (see the comment in
/// `Cargo.toml` about `nvoglv64.dll`), so there's no GL context to
/// probe and no `eframe::glow` module to reach for.
#[cfg(not(windows))]
fn log_gl_caps(cc: &eframe::CreationContext<'_>) {
    let Some(gl) = cc.gl.as_ref() else {
        log::info!("renderer: wgpu (no glow context)");
        crate::startup_log::log("renderer: wgpu");
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
        crate::startup_log::log(format!(
            "renderer: glow (GL renderer='{renderer}', version='{version}', MSAA samples={samples})"
        ));
        if samples < 2 {
            log::warn!(
                "MSAA not active (GL_SAMPLES={samples}). Requested 8× in main.rs but the driver declined. \
                 Check NVIDIA/Intel control-panel 'Antialiasing - Mode' and set it to 'Application-controlled'."
            );
        }
    }
}

#[cfg(windows)]
fn log_gl_caps(_cc: &eframe::CreationContext<'_>) {
    log::info!("renderer: wgpu (glow feature intentionally disabled on Windows)");
    crate::startup_log::log("renderer: wgpu (glow feature disabled)");
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
///
/// Three octaves of integer hash noise drive the per-pixel luminance so the
/// pattern reads as uneven matte metal rather than a regular dither. Range is
/// intentionally wider than the old build (0x2c..0x78 instead of 0x2c..0x58)
/// so the texture stays visible when the compositor draws it at a moderate
/// alpha on top of the vignette in `paint_metallic_background`.
fn build_metallic_noise_image() -> egui::ColorImage {
    const W: usize = 128;
    const H: usize = 128;
    let mut pixels = Vec::with_capacity(W * H);
    for y in 0..H {
        for x in 0..W {
            let h = hash_bg_noise_u8(x as u32, y as u32);
            let h2 = hash_bg_noise_u8(x as u32 / 3, y as u32 / 3);
            let h3 = hash_bg_noise_u8(x as u32 / 11, y as u32 / 11);
            // Wider per-pixel range than the previous build: the three
            // octaves together contribute up to ~64 grey levels so the
            // brightest grain pixels reach ~0x78 (was ~0x58). Keeps the
            // average luminance close to the old value so the overall
            // window colour doesn't shift visibly.
            let mut v = 0x2cu8
                .saturating_add(h % 30)
                .saturating_add(h2 % 20)
                .saturating_add(h3 % 14);
            v = v.min(0x78);
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
    // Vignette: lighter center `#434250` → darker edge `#232328`. The
    // center stop used to sit at `#36353b`, only ~5 luma above the edge,
    // which made the "light pool" almost invisible on most displays. Bumping
    // the centre another ~12 luma and nudging the blue channel slightly
    // gives the gradient a readable highlight without lifting the edges.
    let c_col = Color32::from_rgb(0x43, 0x42, 0x50);
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
    let sheen_tl = Color32::from_rgba_unmultiplied(0x43, 0x42, 0x50, 0x1a);
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
    // Tint + alpha of the tiled grain. Tint stays neutral slate so the noise
    // texels aren't colour-shifted; alpha controls how boldly the texture
    // sits on top of the vignette. 0x58 (~35%) is roughly 2.5× the previous
    // 0x24 and makes the grain clearly visible without turning the window
    // into visible noise.
    painter.image(
        noise.id(),
        rect,
        uv,
        Color32::from_rgba_unmultiplied(0x2c, 0x2c, 0x32, 0x58),
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

/// Always-visible thin rail that sits to the LEFT of the left column and toggles
/// the column's collapse state when clicked. Returns `true` if the rail was clicked
/// this frame so the caller can flip `AudioMeterApp::left_collapsed` + persist.
///
/// Styling: no rounded-rect border; blends with the window background by default
/// and brightens subtly on hover. A two-tone inset divider (matching
/// [`inset_vertical_divider`] — 1 px dark shadow + 1 px light highlight) runs down
/// the rail's inner edge so it reads as a separator flush with the rest of the
/// panel chrome, not a button island. A sideways chevron (`◀` when expanded,
/// `▶` when collapsed) sits vertically centered on the rail.
fn paint_left_collapse_rail(ui: &mut egui::Ui, collapsed: bool) -> bool {
    let h = ui.available_height().max(1.0);
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(LEFT_COLLAPSE_RAIL_W, h), Sense::click());
    let hovered = resp.hovered();
    let p = ui.painter();

    // Hover-only subtle white fill at ~10% opacity. Must be a *valid* premultiplied
    // color (RGB ≤ A), otherwise egui renders it far brighter than intended.
    if hovered {
        const HOVER_A: u8 = 0x1a; // ≈ 10% of 255
        p.rect_filled(
            rect,
            egui::CornerRadius::ZERO,
            Color32::from_rgba_premultiplied(HOVER_A, HOVER_A, HOVER_A, HOVER_A),
        );
    }

    // Two-tone inset divider on the rail's inner edge — matches `inset_vertical_divider`
    // (1 px dark shadow + 1 px light highlight). Only drawn when EXPANDED so the rail
    // visually caps the left column. When collapsed, the meters area already draws its
    // own `inset_vertical_divider` 2 px to the right, and painting ours too yields
    // two parallel lines.
    if !collapsed {
        let div_x = rect.right() - 2.0;
        let dark = egui::Rect::from_min_size(
            egui::pos2(div_x, rect.top()),
            egui::vec2(1.0, rect.height()),
        );
        let light = egui::Rect::from_min_size(
            egui::pos2(div_x + 1.0, rect.top()),
            egui::vec2(1.0, rect.height()),
        );
        p.rect_filled(dark, CornerRadius::ZERO, LEFT_SECTION_DIVIDER_DARK);
        p.rect_filled(light, CornerRadius::ZERO, LEFT_SECTION_INNER_HIGHLIGHT);
    }

    // Chevron: true-centered on the rail (both axes). The rail is wide enough
    // (26 px) that no horizontal nudge is needed to avoid crowding the right-edge
    // divider — roughly 11 px of clearance sits on each side.
    let cx = rect.center().x;
    let cy = rect.center().y;
    let s = 3.2;
    let arm = 3.0;
    let stroke_color = if hovered {
        Color32::WHITE
    } else {
        Color32::from_rgb(0xc0, 0xc0, 0xc8)
    };
    let stroke = Stroke::new(1.5, stroke_color);
    if collapsed {
        // Right-pointing chevron (click to expand).
        let tip = egui::pos2(cx + s * 0.6, cy);
        let top = egui::pos2(cx - s * 0.4, cy - arm);
        let bot = egui::pos2(cx - s * 0.4, cy + arm);
        p.line_segment([top, tip], stroke);
        p.line_segment([tip, bot], stroke);
    } else {
        // Left-pointing chevron (click to collapse).
        let tip = egui::pos2(cx - s * 0.6, cy);
        let top = egui::pos2(cx + s * 0.4, cy - arm);
        let bot = egui::pos2(cx + s * 0.4, cy + arm);
        p.line_segment([top, tip], stroke);
        p.line_segment([tip, bot], stroke);
    }

    if hovered {
        ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
    }
    let tooltip = if collapsed {
        "Show controls panel"
    } else {
        "Hide controls panel"
    };
    resp.on_hover_text(tooltip).clicked()
}

/// Paints a full-width 1 px horizontal band (used for inset highlights and dividers).
fn left_panel_1px_band(ui: &mut egui::Ui, color: Color32) {
    let w = ui.available_width().max(1.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 1.0), Sense::hover());
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, color);
}

/// Simple two-column key/value row used inside the license modal. Left label
/// is dim, right value is full-width so long emails don't wrap oddly.
fn kv_row(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [92.0, 16.0],
            egui::Label::new(
                RichText::new(key)
                    .size(10.5)
                    .color(Color32::from_gray(150)),
            ),
        );
        ui.label(
            RichText::new(value)
                .size(11.0)
                .color(Color32::from_gray(225)),
        );
    });
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

