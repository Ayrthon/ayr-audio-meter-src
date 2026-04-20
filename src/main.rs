// Always build as a Windows GUI binary (not a console app) so the user never
// sees an auxiliary terminal window pop up behind the app at launch. We used
// to keep the console subsystem for debug builds to get stderr logging, but
// that meant any dev-time launch flashed a command prompt on screen.
#![windows_subsystem = "windows"]

mod app;
mod audio;
mod clip_capture;
mod clip_playback;
mod clip_playlist_ui;
mod user_settings;
mod dsp;
mod history_graph;
mod loudness;
mod meter;
mod meter_widget;
mod spectrum;
mod spectrum_view;
#[cfg(windows)]
mod splash;
mod weighting;

use std::time::Duration;

use eframe::egui;

/// How long the splash icon is the only thing on screen before we bring up
/// the main window behind it.
const SPLASH_SOLO: Duration = Duration::from_secs(3);
/// How long the splash sits on top of the (now visible) main window at full
/// alpha before the fade-out begins.
const SPLASH_OVER_APP: Duration = Duration::from_secs(2);
/// Fade-out duration for the final handoff from splash to main window.
const SPLASH_FADE_OUT: Duration = Duration::from_millis(500);

/// Initial outer size of the main window in logical pixels, used to compute
/// a centered launch position.
const MAIN_W: f32 = 1140.0;
const MAIN_H: f32 = 649.0;

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Opt the process into *System* DPI awareness rather than winit's
    // default Per-Monitor-v2.  With per-monitor awareness, moving the
    // window across a DPI boundary fires WM_DPICHANGED, which forces
    // the renderer to recreate its surface mid-move — that's the main
    // reason the window "acts weird" or crashes when dragged between
    // screens on mixed-DPI / hybrid-GPU laptops.  System-aware is what
    // most stock Windows apps (Notepad, Notes) use and handles monitor
    // changes by simply letting Windows stretch the bitmap instead.
    //
    // Must be called before any window or winit event loop is created.
    #[cfg(windows)]
    unsafe {
        type DpiAwarenessContext = isize;
        const DPI_AWARENESS_CONTEXT_SYSTEM_AWARE: DpiAwarenessContext = -2;
        unsafe extern "system" {
            fn SetProcessDpiAwarenessContext(value: DpiAwarenessContext) -> i32;
        }
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_SYSTEM_AWARE);
    }

    // Conservative static min inner size.  We used to update this
    // dynamically when the user switched right panels, but that racey
    // interaction with multi-monitor DPI changes mid-drag was crashing
    // the platform layer.  Keeping it fixed at a size that fits the
    // left panel + stereo meters + "0 dB ref" column means the window
    // can't become genuinely useless, while still letting the user
    // shrink freely when the right-hand panel is hidden.
    let min_inner = [600.0_f32, 520.0_f32];

    // Kick off the transparent splash icon ASAP — it runs on its own thread
    // with a Win32 layered window, so it appears before any GL / audio init
    // work this thread is about to do. We keep the returned handle in scope
    // through `run_native` so the splash thread's lifetime is bounded by
    // `main` (it will also auto-close after the configured lifetime).
    #[cfg(windows)]
    let _splash = splash::spawn(SPLASH_SOLO + SPLASH_OVER_APP + SPLASH_FADE_OUT, SPLASH_FADE_OUT);

    // Give the splash the whole solo phase before we start standing up the
    // main window. Blocking this thread is fine — nothing else is running
    // here yet, and the splash lives on its own thread with its own pump.
    #[cfg(windows)]
    std::thread::sleep(SPLASH_SOLO);

    let mut viewport = egui::ViewportBuilder::default()
        // Inner size is logical (DPI-independent). Windows adds the title bar +
        // 1 px borders on top, ~31 px at default chrome, so inner 649 ≈ outer 680.
        .with_inner_size([MAIN_W, MAIN_H])
        .with_min_inner_size(min_inner)
        .with_title("AYR Audio Meter")
        .with_decorations(true)
        .with_resizable(true)
        // Open centered on the primary monitor so the app appears exactly
        // under the splash icon that's been sitting there for the last few
        // seconds, instead of at some OS-chosen cascading default.
        .with_position(centered_launch_position(MAIN_W, MAIN_H));
    if let Some(icon) = app_icon() {
        viewport = viewport.with_icon(icon);
    }

    let native_options = eframe::NativeOptions {
        viewport,
        // Use the Glow (OpenGL) renderer rather than wgpu.  Hybrid GPU
        // laptops (NVIDIA Optimus / Intel Arc) tend to have swap-chain
        // issues with wgpu when the window moves between monitors that
        // are served by different GPUs — Glow avoids that class of bug.
        renderer: eframe::Renderer::Glow,
        // Keep vsync off so the renderer doesn't freeze on a monitor
        // transition where the refresh rate changes.
        vsync: false,
        // 8× MSAA on the GL backbuffer. The spectrum trace is full of shallow
        // diagonals in log-frequency space; 4 samples was already noticeably
        // smoother than no MSAA but the crests of steep peaks still showed
        // faint stair-stepping on 1.5× DPI Windows scaling. 8× gets us close
        // to CAD-quality edges, at roughly twice the fill-rate cost of 4× -
        // comfortably absorbed by anything from Intel Iris onward at our
        // resolution. `apply_theme` also widens egui's own stroke feathering
        // to 1.5 px on top of this, and `log_gl_caps` reports the actual
        // sample count the driver handed us (some control-panel profiles
        // force application-requested MSAA off silently).
        multisampling: 8,
        ..Default::default()
    };

    eframe::run_native(
        "AYR Audio Meter",
        native_options,
        Box::new(|cc| Ok(Box::new(app::AudioMeterApp::new(cc)))),
    )
}

/// Computes a launch position (outer window top-left) that centers a window
/// of the given logical size on the primary monitor. On non-Windows platforms
/// we fall back to a reasonable default; the splash intro is Windows-only
/// anyway, so exact centering there is a nice-to-have, not a correctness
/// requirement.
fn centered_launch_position(width: f32, height: f32) -> egui::Pos2 {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetSystemMetrics(index: i32) -> i32;
        }
        const SM_CXSCREEN: i32 = 0;
        const SM_CYSCREEN: i32 = 1;
        let (sw, sh) = unsafe {
            (
                GetSystemMetrics(SM_CXSCREEN) as f32,
                GetSystemMetrics(SM_CYSCREEN) as f32,
            )
        };
        return egui::pos2(((sw - width) * 0.5).max(0.0), ((sh - height) * 0.5).max(0.0));
    }
    #[cfg(not(windows))]
    {
        let _ = (width, height);
        egui::pos2(80.0, 80.0)
    }
}

fn app_icon() -> Option<egui::IconData> {
    const PNG: &[u8] = include_bytes!("../assets/app_icon.png");
    let mut rgba = image::load_from_memory(PNG).ok()?.into_rgba8();
    const MAX_DIM: u32 = 256;
    let (w, h) = rgba.dimensions();
    if w > MAX_DIM || h > MAX_DIM {
        let scale = (MAX_DIM as f32 / w.max(h) as f32).min(1.0);
        let nw = ((w as f32) * scale).round().clamp(1.0, MAX_DIM as f32) as u32;
        let nh = ((h as f32) * scale).round().clamp(1.0, MAX_DIM as f32) as u32;
        rgba = image::imageops::resize(&rgba, nw, nh, image::imageops::FilterType::Triangle);
    }
    let (w, h) = rgba.dimensions();
    Some(egui::IconData {
        rgba: rgba.into_raw(),
        width: w,
        height: h,
    })
}
