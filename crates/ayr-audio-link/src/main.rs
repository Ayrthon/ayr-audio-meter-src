//! AYR Audio Link — sender binary.
//!
//! The product: a small, single-window Windows app that captures one
//! chosen audio device (input or loopback of an output) and streams it
//! over the LAN to an AYR Audio Meter running on another PC, where it
//! appears as a normal entry in the device picker.
//!
//! Layout mirrors the meter:
//!
//! * `main.rs`       — eframe bootstrapping, logging, panic hook
//! * `app.rs`        — the single egui window (device picker, level bar,
//!                     peer list, pairing dialog, start/stop toggle)
//! * `audio.rs`      — capture: cpal input + WASAPI loopback on Windows
//! * `net.rs`        — TCP server that accepts meter connections and
//!                     pushes PCM frames
//! * `identity.rs`   — persistent sender identity + trust cache
//!                     (`~/.config/ayr-audio-link/state.json`)
//!
//! Keeping the whole thing in one binary (no daemons, no services) is a
//! deliberate UX choice: the user can see at a glance whether it's
//! running and which meter is attached.

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod audio;
mod identity;
mod net;
mod updater;

use eframe::egui::{IconData, ViewportBuilder};

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Log — don't paper over — a panic on the UI thread. Without this hook a
    // panic inside an `egui` update closure turns into a silent taskbar
    // disappearance; the `env_logger` record at least lands in a terminal
    // when the user is debugging.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        log::error!("ayr-audio-link panic: {info}");
        default_hook(info);
    }));

    let icon = load_icon();
    let viewport = ViewportBuilder::default()
        .with_inner_size([480.0, 560.0])
        .with_min_inner_size([400.0, 480.0])
        .with_title("AYR Audio Link")
        .with_icon(icon);

    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "AYR Audio Link",
        native_options,
        Box::new(|cc| Ok(Box::new(app::LinkApp::new(cc)))),
    )
}

fn load_icon() -> IconData {
    // Placeholder icon — a single-pixel transparent PNG is enough for the
    // dev build. The installer replaces this via winres at release-build
    // time. Keeping the fallback small avoids holding up the MVP on
    // asset pipelines.
    IconData {
        rgba: vec![0u8; 4],
        width: 1,
        height: 1,
    }
}
