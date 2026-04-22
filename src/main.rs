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
mod comp_view;
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
mod startup_log;
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
pub const MAIN_W: f32 = 1220.0;
pub const MAIN_H: f32 = 649.0;

/// Re-exported so `app.rs` can know when to un-park the main window and
/// bring it to the foreground. Keeping the single source of truth here
/// means the splash in `splash::spawn(...)` below and the reveal in the
/// UI loop agree on timing.
pub const SPLASH_SOLO_DURATION: Duration = SPLASH_SOLO;

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Set up the on-disk diagnostic log before anything else so we capture
    // failures that happen during early init (COM / WASAPI enumeration, GL
    // context creation, etc.) even though we ship as a windows-subsystem
    // binary without a stderr console.
    startup_log::install_panic_hook();
    #[cfg(windows)]
    startup_log::install_seh_handler();
    startup_log::log(format!(
        "main: start (exe = {})",
        std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<unknown>".into())
    ));

    // Pings the log every 2 s until eframe reaches our creator, and pops
    // a Win32 message box after 20 s of no progress so the user isn't
    // left staring at a finished splash. Harmless on healthy launches:
    // the thread exits on its first tick the moment the creator runs.
    startup_log::spawn_launch_watchdog(Duration::from_secs(20));

    // Diagnostic / workaround env vars. `AM_SKIP_SPLASH=1` skips the Win32
    // layered splash thread entirely — if this makes the "app won't
    // launch alongside a running game" hang go away, the splash's
    // topmost layered window is interacting badly with the game's
    // compositor / anti-cheat. `AM_SKIP_OFFSCREEN_PARK=1` additionally
    // launches the main window directly at the primary-monitor center
    // instead of parking it off-screen during the splash's solo phase,
    // in case `winit`'s window placement is the stuck call.
    let skip_splash = std::env::var_os("AM_SKIP_SPLASH").is_some();
    let skip_offscreen_park = std::env::var_os("AM_SKIP_OFFSCREEN_PARK").is_some();
    startup_log::log(format!(
        "config: skip_splash={skip_splash} skip_offscreen_park={skip_offscreen_park}"
    ));

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
    let _splash = if skip_splash {
        None
    } else {
        startup_log::log("splash: spawning");
        let h = splash::spawn(SPLASH_SOLO + SPLASH_OVER_APP + SPLASH_FADE_OUT, SPLASH_FADE_OUT);
        startup_log::log("splash: spawn() returned");
        Some(h)
    };

    let mut viewport = egui::ViewportBuilder::default()
        // Inner size is logical (DPI-independent). Windows adds the title bar +
        // 1 px borders on top, ~31 px at default chrome, so inner 649 ≈ outer 680.
        .with_inner_size([MAIN_W, MAIN_H])
        .with_min_inner_size(min_inner)
        .with_title("AYR Audio Meter")
        .with_decorations(true)
        .with_resizable(true);

    // Windows-only launch choreography:
    //
    // We used to sleep the main thread for 3 s before calling
    // `run_native` so the splash had solo screen time. That turned out to
    // be load-bearing for a launch-time bug: if the user launched us while
    // a fullscreen game (e.g. Battlefield) was running, the 3 s delay
    // burned through the freshly-launched process's brief foreground
    // privilege. When `run_native` finally created the window, Windows'
    // anti-focus-stealing logic refused to bring it to the front — the
    // app was running, but invisible behind the game.
    //
    // Instead we now start `run_native` immediately (so the window is
    // created within the foreground-privilege window) but park it far
    // off-screen at `-20_000, -20_000`. The reveal is driven from the UI
    // loop itself (see `AudioMeterApp::ui`), which teleports the window
    // to the primary-monitor center and sends `Focus` once the splash
    // solo phase is up. This keeps audio and GL init happening in
    // parallel with the splash and preserves our ability to steal focus
    // from a foreground app.
    #[cfg(windows)]
    {
        if skip_offscreen_park {
            viewport = viewport.with_position(centered_launch_position(MAIN_W, MAIN_H));
            startup_log::log("viewport: launching centered (AM_SKIP_OFFSCREEN_PARK)");
        } else {
            viewport = viewport.with_position(egui::pos2(-20_000.0, -20_000.0));
            startup_log::log("viewport: launching parked off-screen");
        }
    }
    #[cfg(not(windows))]
    {
        viewport = viewport.with_position(centered_launch_position(MAIN_W, MAIN_H));
    }

    if let Some(icon) = app_icon() {
        viewport = viewport.with_icon(icon);
    }

    // Renderer selection per OS.
    //
    // Windows → **wgpu (D3D12)**. We used to ship Glow (OpenGL) everywhere
    // because hybrid-GPU laptops had historical swap-chain issues with
    // wgpu on monitor transitions. That trade-off flipped in practice: if
    // the user launched Audio Meter while a fullscreen game (e.g.
    // Battlefield 6 in borderless fullscreen) was running, `wglCreateContext`
    // would block inside `eframe::run_native` on GPU driver locks held by
    // the game — the splash would play out but the main window never
    // appeared, not even after alt-tab. D3D12 shares the GPU cooperatively
    // through the OS scheduler so the context comes up immediately.
    //
    // Linux → **Glow**. The repo's existing visual baseline was tuned
    // against Glow output and Wayland/X11 wgpu has its own quirks we
    // haven't validated for this app yet.
    // Renderer + MSAA selection per OS. See `Cargo.toml` for the
    // feature-set rationale; in short, Windows has only `wgpu` compiled
    // in (not `glow`) on purpose — linking `opengl32.dll` was causing
    // NVIDIA's `nvoglv64.dll` to crash our process when games were
    // running. Non-Windows stays on `glow`.
    //
    // MSAA: the spectrum trace is full of shallow diagonals in
    // log-frequency space, so more samples = smoother crests on steep
    // peaks at fractional DPI scaling.
    //
    // * Glow (OpenGL): 8×. The WGL backbuffer path handles 8× MSAA on
    //   BGRA8 fine across NVIDIA/Intel/AMD drivers.
    // * wgpu (D3D12): 4×. The WebGPU spec only guarantees 1 and 4
    //   samples on `Bgra8Unorm` (egui's pipeline format). 8× would
    //   require the `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` adapter
    //   feature, which egui doesn't enable — asking for 8× causes
    //   `Device::create_render_pipeline` to fail validation and the
    //   app to panic at startup.
    //
    // `apply_theme` also widens egui's own stroke feathering to 1.5 px
    // on top of this, and `log_gl_caps` (Glow only) reports the actual
    // sample count the driver handed us — some NVIDIA / Intel control
    // panel profiles silently force application-requested MSAA off.
    #[cfg(windows)]
    let renderer = eframe::Renderer::Wgpu;
    #[cfg(windows)]
    let multisampling: u16 = 4;

    #[cfg(not(windows))]
    let renderer = eframe::Renderer::Glow;
    #[cfg(not(windows))]
    let multisampling: u16 = 8;

    // Restrict wgpu's backend probe set on Windows to DX12 only.
    //
    // egui-wgpu's default `InstanceDescriptor` is
    //     backends: Backends::PRIMARY | Backends::GL
    // — i.e. wgpu is asked to probe *every* available backend
    // (DX12, Vulkan, AND OpenGL) and pick one at runtime. Even if
    // DX12 wins the adapter selection, the GL backend's probe path
    // still gets initialized: `wgpu-hal` calls into `wglGetProcAddress`
    // which causes Windows to resolve the OpenGL ICD and load
    // NVIDIA's `nvoglv64.dll` into our process. When another process
    // on the machine (e.g. Battlefield 6) already has the GPU busy,
    // that driver's init path recurses on itself and blows the stack
    // with `EXCEPTION_STACK_OVERFLOW`, terminating the process before
    // eframe ever invokes our creator callback. We only need DX12 on
    // Windows, so restrict `backends` to that and the GL probe never
    // fires.
    #[cfg(windows)]
    let wgpu_options = {
        let mut cfg = eframe::egui_wgpu::WgpuConfiguration::default();
        if let eframe::egui_wgpu::WgpuSetup::CreateNew(create_new) = &mut cfg.wgpu_setup {
            create_new.instance_descriptor.backends = eframe::wgpu::Backends::DX12;
        }
        cfg
    };

    let native_options = eframe::NativeOptions {
        viewport,
        renderer,
        // Keep vsync off so the renderer doesn't freeze on a monitor
        // transition where the refresh rate changes.
        vsync: false,
        multisampling,
        #[cfg(windows)]
        wgpu_options,
        ..Default::default()
    };

    // Snapshot all DLLs currently loaded into our process. When something
    // later crashes inside eframe/winit/wgpu init with a `SEH:` line in
    // the log, we can grep this snapshot for the DLL whose address range
    // contains the faulting address — which tells us exactly *which*
    // injected library (overlay, anti-cheat, screen recorder, …) caused
    // the crash, not just "something did".
    #[cfg(windows)]
    {
        startup_log::snapshot_loaded_modules_to_log();
        // Plus a sidecar that keeps logging *new* DLL loads for a few
        // seconds. Needed to catch lazily-loaded GPU driver DLLs
        // (`nvoglv64.dll`, `atio6axx.dll`, `ig9icd64.dll`, …) and
        // D3D12 runtime (`d3d12.dll`, `d3d12core.dll`), which are
        // what winit/wgpu pull in while we're still inside
        // `run_native` — so they never appear in the static
        // pre-snapshot even though they're exactly what tends to
        // crash when another app (e.g. Battlefield) has the GPU busy.
        startup_log::spawn_module_load_tracer(Duration::from_secs(10));
    }

    startup_log::log("main: calling eframe::run_native");
    let result = eframe::run_native(
        "AYR Audio Meter",
        native_options,
        Box::new(|cc| {
            startup_log::log("eframe: creator invoked, building AudioMeterApp");
            startup_log::mark_run_native_reached_creator();
            let app = app::AudioMeterApp::new(cc);
            startup_log::log("eframe: AudioMeterApp constructed");
            Ok(Box::new(app))
        }),
    );
    match &result {
        Ok(_) => startup_log::log("main: run_native returned Ok"),
        Err(e) => startup_log::log(format!("main: run_native returned Err: {e}")),
    }
    result
}

/// Computes a launch position (outer window top-left) that centers a window
/// of the given logical size on the primary monitor. On non-Windows platforms
/// we fall back to a reasonable default; the splash intro is Windows-only
/// anyway, so exact centering there is a nice-to-have, not a correctness
/// requirement.
pub fn centered_launch_position(width: f32, height: f32) -> egui::Pos2 {
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
