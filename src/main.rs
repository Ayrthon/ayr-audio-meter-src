#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod audio;
mod clip_capture;
mod dsp;
mod history_graph;
mod loudness;
mod meter;
mod meter_widget;
mod spectrum;
mod spectrum_view;
mod weighting;

use eframe::egui;

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

    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1140.0, 640.0])
        .with_min_inner_size(min_inner)
        .with_title("Audio Meter")
        .with_decorations(true)
        .with_resizable(true);
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
        ..Default::default()
    };

    eframe::run_native(
        "Audio Meter",
        native_options,
        Box::new(|cc| Ok(Box::new(app::AudioMeterApp::new(cc)))),
    )
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
