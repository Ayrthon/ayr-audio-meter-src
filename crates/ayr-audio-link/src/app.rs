//! AYR Audio Link sender UI.
//!
//! Tiny single-window layout built directly on `egui` (no splash, no
//! license gate, no multi-tab — this is a one-purpose utility):
//!
//! ```text
//! ┌───────────────────────────────────────────────┐
//! │  AYR Audio Link                               │
//! │  Broadcasting as: MikeStudio-PC (Input 1/2)   │
//! │                                               │
//! │  Source: [ dropdown ]           [ refresh ]   │
//! │  ████████████████████░░░░░░░░░  -12.4 dBFS   │
//! │                                               │
//! │  Status: Listening on :45451                  │
//! │  Connected meters:                            │
//! │    * 192.168.1.21:54122    12 345 frames      │
//! │                                               │
//! │  [ Start broadcasting ] [ Stop ]              │
//! └───────────────────────────────────────────────┘
//! ```
//!
//! Keeping the layout this linear makes the "is it working?" answer one
//! glance — if the level bar moves and the peer list has at least one
//! entry, everything is fine.

use std::sync::Arc;

use eframe::egui::{self, Color32, RichText};
use parking_lot::Mutex;

use ayr_audio_link_core::{DEFAULT_PORT, discovery, wire::Handshake};

use crate::audio::{self, Capture, DeviceInfo, SourceKind};
use crate::identity::IdentityStore;
use crate::net::{Server, ServerStatus};
use crate::updater;

pub struct LinkApp {
    identity: Arc<IdentityStore>,
    devices: Vec<DeviceInfo>,
    selected_device: Option<usize>,
    capture: Option<Capture>,
    server: Server,
    advertiser: Option<discovery::Advertiser>,

    /// Cheap running peak (dBFS) for the UI level bar, updated from the
    /// audio callback. Not meter-grade — this is just a heartbeat for the
    /// user so they know audio is flowing.
    level_shared: Arc<Mutex<LiveLevel>>,

    /// Set once the capture produces its first block; controls whether
    /// we've registered the mDNS advertisement yet.
    announced_format: Option<Handshake>,

    /// Transient error surfaced at the bottom of the window. Cleared when
    /// the user picks a different device or clicks "Retry".
    last_error: Option<String>,

    port: u16,
    display_name_editor: String,
    show_advanced: bool,
    /// When `true`, pressing the window close button minimizes the app
    /// instead of quitting. This is a poor-man's tray-minimize — a real
    /// `tray-icon` integration is a Phase 2 item per the product plan.
    minimize_on_close: bool,

    /// Auto-updater status, populated by `updater::spawn_check` on boot
    /// and read each frame to show a small "update available" banner.
    /// Kept intentionally unobtrusive — Link is a background utility, so
    /// the update flow is a one-click banner rather than a modal.
    update: updater::SharedUpdateStatus,
    update_dismissed: bool,
}

#[derive(Clone, Copy, Default)]
struct LiveLevel {
    peak_db: f32,
    active: bool,
}

impl LinkApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let identity = Arc::new(IdentityStore::load_or_init());
        let devices = audio::list_devices().unwrap_or_default();
        let selected_device = pick_default_device(&devices);
        let server = Server::new(identity.clone());
        let display_name_editor = identity.display_name();
        Self {
            identity,
            devices,
            selected_device,
            capture: None,
            server,
            advertiser: None,
            level_shared: Arc::new(Mutex::new(LiveLevel::default())),
            announced_format: None,
            last_error: None,
            port: DEFAULT_PORT,
            display_name_editor,
            show_advanced: false,
            minimize_on_close: true,
            update: {
                let shared = updater::make_shared_status();
                updater::spawn_check(
                    updater::UPDATE_REPO,
                    env!("CARGO_PKG_VERSION"),
                    shared.clone(),
                );
                shared
            },
            update_dismissed: false,
        }
    }

    fn is_running(&self) -> bool {
        self.capture.is_some()
    }

    fn selected(&self) -> Option<&DeviceInfo> {
        self.selected_device.and_then(|i| self.devices.get(i))
    }

    fn start_broadcasting(&mut self) {
        self.last_error = None;
        let Some(dev) = self.selected().cloned() else {
            self.last_error = Some("Pick an audio device first.".into());
            return;
        };
        if let Err(e) = self.server.start(self.port) {
            self.last_error = Some(format!("Couldn't bind port {}: {e}", self.port));
            return;
        }

        let server_handle = self.server.handle();
        let level_shared = self.level_shared.clone();
        let capture_result = audio::start_capture(dev.clone(), move |samples: &[f32], ch: u8, sr: u32| {
            // Cheap running peak for the level bar.
            let peak = samples
                .iter()
                .fold(0.0f32, |acc, &s| acc.max(s.abs()));
            let db = if peak > 0.0 {
                20.0 * peak.log10()
            } else {
                -120.0
            };
            *level_shared.lock() = LiveLevel {
                peak_db: db.max(-120.0),
                active: true,
            };
            // Forward to the TCP server in one big little-endian blob. We
            // do the conversion here rather than in `submit_frame` so the
            // audio thread pays the cost (not the accept thread, which
            // would otherwise serialise across peers).
            let mut bytes = Vec::with_capacity(samples.len() * 4);
            for &s in samples {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            // Ensure the server has the current handshake on every block —
            // it's cheap (compare + skip) and handles devices that change
            // SR mid-stream (unusual but happens with screensavers resuming
            // a WASAPI endpoint at a different mix format).
            server_handle.update_format_if_changed(ch, sr);
            server_handle.submit(bytes);
        });

        match capture_result {
            Ok(cap) => self.capture = Some(cap),
            Err(e) => {
                self.last_error = Some(format!("Couldn't start capture: {e:#}"));
                self.server.stop();
            }
        }
    }

    fn stop_broadcasting(&mut self) {
        if let Some(mut c) = self.capture.take() {
            c.stop();
        }
        self.server.stop();
        self.advertiser = None;
        self.announced_format = None;
        *self.level_shared.lock() = LiveLevel::default();
    }

    /// Render a small "update available" banner if the updater resolved
    /// to `Available` / `Downloading` / `Downloaded` / `LaunchFailed`.
    /// Other states (including network errors) are deliberately silent —
    /// Link is a background tool and a nagging offline-state banner would
    /// be disproportionate to the cost.
    fn maybe_draw_update_banner(&mut self, ui: &mut egui::Ui) {
        use updater::UpdateStatus;

        if self.update_dismissed {
            return;
        }
        // Take a snapshot so we can release the lock before mutating
        // `self`. The updater thread is still free to overwrite the
        // slot we cloned — the UI just re-reads next frame.
        let status = self.update.lock().clone();
        match status {
            UpdateStatus::Available(info) => {
                let frame = egui::Frame::group(ui.style())
                    .fill(Color32::from_rgb(30, 50, 30))
                    .inner_margin(egui::Margin::same(8));
                frame.show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!(
                                "Update available — v{} (you have v{})",
                                info.version,
                                env!("CARGO_PKG_VERSION"),
                            ))
                            .color(Color32::from_rgb(180, 230, 180))
                            .strong(),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Later").clicked() {
                                self.update_dismissed = true;
                            }
                            if ui.button("Update now").clicked() {
                                updater::spawn_download_and_run(info.clone(), self.update.clone());
                            }
                        });
                    });
                });
                ui.add_space(6.0);
            }
            UpdateStatus::Downloading {
                progress,
                total_bytes,
            } => {
                let mb = total_bytes as f32 / (1024.0 * 1024.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Downloading update…").weak());
                    ui.add(
                        egui::ProgressBar::new(progress)
                            .desired_width(180.0)
                            .show_percentage(),
                    );
                    if total_bytes > 0 {
                        ui.label(RichText::new(format!("{mb:.1} MB")).weak());
                    }
                });
                ui.add_space(6.0);
            }
            UpdateStatus::Downloaded(path) => {
                // We don't auto-run the installer for Link — the user is
                // usually mid-session sending audio, and swapping the
                // .exe while a meter is connected would drop the peer.
                // Give them an explicit "Install now" button.
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Update downloaded.")
                            .color(Color32::from_rgb(180, 230, 180))
                            .strong(),
                    );
                    if ui.button("Install now").clicked() {
                        // Stop the broadcast before handing off to the
                        // installer so the Restart Manager doesn't have
                        // to grapple with a busy TCP port.
                        self.stop_broadcasting();
                        match updater::launch_installer(&path) {
                            Ok(()) => {
                                *self.update.lock() = UpdateStatus::Launching;
                                ui.ctx()
                                    .send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                            Err(e) => {
                                *self.update.lock() =
                                    UpdateStatus::LaunchFailed(e.to_string());
                            }
                        }
                    }
                    if ui.button("Later").clicked() {
                        self.update_dismissed = true;
                    }
                });
                ui.add_space(6.0);
            }
            UpdateStatus::LaunchFailed(msg) => {
                ui.horizontal(|ui| {
                    ui.colored_label(
                        Color32::from_rgb(230, 180, 80),
                        format!("Update failed: {msg}"),
                    );
                    if ui.button("Dismiss").clicked() {
                        self.update_dismissed = true;
                    }
                });
                ui.add_space(6.0);
            }
            _ => {}
        }
    }

    /// Refresh the mDNS advertisement. Called every frame; cheap when the
    /// format hasn't changed.
    fn reconcile_mdns(&mut self) {
        let want_format = self.server.current_handshake();
        match (&want_format, &self.announced_format) {
            (Some(new), prev) if Some(new) != prev.as_ref() => {
                // Tear down the old advert; build a fresh one.
                self.advertiser = None;
                match discovery::Advertiser::new(
                    &self.identity.instance_name(),
                    self.port,
                    &new.display_name,
                    &crate::identity::local_peer_fingerprint(),
                    new.sample_rate,
                    new.channels,
                ) {
                    Ok(a) => self.advertiser = Some(a),
                    Err(e) => {
                        log::warn!("mdns advertise: {e}");
                        self.last_error = Some(format!("mDNS announce failed: {e}"));
                    }
                }
                self.announced_format = Some(new.clone());
            }
            (None, Some(_)) => {
                // Capture stopped — drop advert.
                self.advertiser = None;
                self.announced_format = None;
            }
            _ => {}
        }
    }
}

impl eframe::App for LinkApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Poll at ~30Hz so the level bar animates and peer stats refresh.
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(33));

        // Poor-man's "tray minimize" — catch the close event and turn it
        // into a minimise instead. Real tray icon (so users can right-click
        // a notification-area glyph to quit) is Phase 2.
        if self.minimize_on_close
            && ui.ctx().input(|i| i.viewport().close_requested())
            && self.is_running()
        {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Minimized(true));
        }

        self.reconcile_mdns();

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.add_space(6.0);
            ui.heading("AYR Audio Link");
            ui.label(
                RichText::new("Sends audio from this PC to AYR Audio Meter on the same network.")
                    .italics()
                    .weak(),
            );
            ui.add_space(8.0);
            self.maybe_draw_update_banner(ui);

            ui.horizontal(|ui| {
                ui.label("Broadcasting as:");
                ui.monospace(&self.display_name_editor);
                if ui.button("Edit").clicked() {
                    self.show_advanced = !self.show_advanced;
                }
            });

            if self.show_advanced {
                ui.indent("advanced", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Display name:");
                        let r = ui.text_edit_singleline(&mut self.display_name_editor);
                        if r.lost_focus() {
                            self.identity.set_display_name(
                                Some(self.display_name_editor.clone()),
                            );
                            // Reflect immediately in the running server so
                            // the next handshake / mDNS announce uses it.
                            self.server.set_display_name(self.display_name_editor.clone());
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Port:");
                        let mut port_txt = self.port.to_string();
                        if ui.text_edit_singleline(&mut port_txt).lost_focus() {
                            if let Ok(p) = port_txt.parse::<u16>() {
                                self.port = p;
                            }
                        }
                    });
                    ui.checkbox(
                        &mut self.minimize_on_close,
                        "Minimize to taskbar on close (instead of quitting)",
                    );
                });
            }

            ui.separator();

            // Copy out just what we need so the combobox closure doesn't
            // need `&mut self`. Pending selection change is applied after
            // the closure returns.
            let current_sel = self.selected_device;
            let devices_snapshot: Vec<(SourceKind, String)> = self
                .devices
                .iter()
                .map(|d| (d.kind, d.name.clone()))
                .collect();
            let mut pending_select: Option<usize> = None;
            let mut pending_refresh = false;
            ui.horizontal(|ui| {
                ui.label("Source:");
                let selected_label = current_sel
                    .and_then(|i| self.devices.get(i))
                    .map(device_label)
                    .unwrap_or_else(|| "(none)".into());
                egui::ComboBox::from_id_salt("device_combo")
                    .selected_text(selected_label)
                    .width(280.0)
                    .show_ui(ui, |ui| {
                        let mut section = |ui: &mut egui::Ui, title: &str, kind: SourceKind| {
                            ui.label(RichText::new(title).weak());
                            for (i, (k, name)) in devices_snapshot.iter().enumerate() {
                                if *k != kind {
                                    continue;
                                }
                                let sel = Some(i) == current_sel;
                                if ui.selectable_label(sel, name).clicked() {
                                    pending_select = Some(i);
                                }
                            }
                        };
                        section(ui, "INPUTS (mic / line)", SourceKind::Input);
                        ui.separator();
                        section(ui, "LOOPBACK (what speakers play)", SourceKind::Loopback);
                    });
                if ui.button("Refresh").clicked() {
                    pending_refresh = true;
                }
            });
            if let Some(i) = pending_select {
                self.selected_device = Some(i);
            }
            if pending_refresh {
                let previous = self.selected().map(|d| (d.kind, d.name.clone()));
                self.devices = audio::list_devices().unwrap_or_default();
                self.selected_device = previous
                    .and_then(|prev| {
                        self.devices
                            .iter()
                            .position(|d| d.kind == prev.0 && d.name == prev.1)
                    })
                    .or_else(|| pick_default_device(&self.devices));
            }

            ui.add_space(6.0);
            draw_level_bar(ui, *self.level_shared.lock());
            ui.add_space(8.0);

            let status = self.server.status();
            draw_status_block(ui, &status, self.is_running(), self.advertiser.is_some());

            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if self.is_running() {
                    if ui.button("Stop broadcasting").clicked() {
                        self.stop_broadcasting();
                    }
                } else if ui
                    .add_enabled(
                        self.selected_device.is_some(),
                        egui::Button::new("Start broadcasting"),
                    )
                    .clicked()
                {
                    self.start_broadcasting();
                }
            });

            if let Some(err) = &self.last_error {
                ui.add_space(6.0);
                ui.colored_label(Color32::from_rgb(220, 80, 80), err);
            }
        });
    }
}

fn pick_default_device(devices: &[DeviceInfo]) -> Option<usize> {
    // Prefer the system default input. If we're on a headless Linux and
    // there isn't one, fall back to index 0.
    if let Some(default_in) = audio::default_input_name() {
        if let Some(i) = devices
            .iter()
            .position(|d| d.kind == SourceKind::Input && d.name == default_in)
        {
            return Some(i);
        }
    }
    if !devices.is_empty() {
        Some(0)
    } else {
        None
    }
}

fn device_label(d: &DeviceInfo) -> String {
    let tag = match d.kind {
        SourceKind::Input => "[IN]",
        SourceKind::Loopback => "[OUT]",
    };
    format!("{tag} {}", d.name)
}

fn draw_level_bar(ui: &mut egui::Ui, level: LiveLevel) {
    let full_width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(full_width, 16.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter();
    painter.rect_filled(rect, 3.0, Color32::from_rgb(20, 20, 24));
    // Map peak_db in [-60..0] to bar fraction [0..1].
    let f = ((level.peak_db + 60.0) / 60.0).clamp(0.0, 1.0);
    let filled = egui::Rect::from_min_max(
        rect.min,
        egui::pos2(rect.min.x + rect.width() * f, rect.max.y),
    );
    let color = if level.peak_db > -3.0 {
        Color32::from_rgb(220, 80, 80)
    } else if level.peak_db > -18.0 {
        Color32::from_rgb(220, 180, 60)
    } else {
        Color32::from_rgb(80, 200, 120)
    };
    painter.rect_filled(filled, 3.0, color);

    let label = if level.active {
        format!("{:.1} dBFS", level.peak_db)
    } else {
        "-- dBFS".to_string()
    };
    painter.text(
        rect.right_center() - egui::vec2(6.0, 0.0),
        egui::Align2::RIGHT_CENTER,
        label,
        egui::FontId::proportional(12.0),
        Color32::WHITE,
    );
}

fn draw_status_block(
    ui: &mut egui::Ui,
    status: &ServerStatus,
    running: bool,
    mdns_advertising: bool,
) {
    ui.label(
        RichText::new(match (running, status.listening_on) {
            (true, Some(addr)) => format!("Listening on {addr}"),
            (true, None) => "Listening (no local address resolved)".to_string(),
            (false, _) => "Not broadcasting".to_string(),
        })
        .strong(),
    );
    if running {
        let mdns_txt = if mdns_advertising {
            "mDNS announced — meters on the LAN can see me"
        } else {
            "mDNS not yet announced (waiting for first audio block)"
        };
        ui.label(RichText::new(mdns_txt).weak());
    }

    ui.add_space(4.0);
    ui.label(RichText::new(format!("Connected meters: {}", status.peers.len())).strong());
    if status.peers.is_empty() {
        ui.label(RichText::new("  (none yet)").weak());
    } else {
        for p in &status.peers {
            ui.label(format!("  • {}   — {} frames", p.addr, p.frames_sent));
        }
    }
}

