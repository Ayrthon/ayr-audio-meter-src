//! Timestamped append-only log file for diagnosing launch-time issues.
//!
//! Because the executable ships with `#![windows_subsystem = "windows"]` there
//! is no attached console and `stderr` / `env_logger` output is silently
//! discarded. That makes a class of launch bugs (GL context creation stalls,
//! WASAPI enumeration hangs, etc.) invisible unless we write the trail to
//! disk ourselves.
//!
//! The log lives at:
//!   Windows: `%LOCALAPPDATA%\AudioMeter\launch.log`
//!   macOS:   `~/Library/Application Support/audio-meter-rs/launch.log`
//!   Linux:   `~/.local/share/audio-meter-rs/launch.log`
//!
//! Each entry is a single line of the form
//!   `[+0.127s] splash: spawned`
//! where the prefix is seconds since process start. This is intentionally
//! plain text so users can paste it into a bug report.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use once_cell::sync::Lazy;

/// Wall-clock anchor set the first time the log is touched, so all entries
/// share a consistent zero.
static START: Lazy<Instant> = Lazy::new(Instant::now);

/// Backing file, opened lazily in append mode. Held in a `Mutex` so multiple
/// threads (e.g. the splash thread and the main thread) can write without
/// interleaving lines.
static FILE: Lazy<Mutex<Option<File>>> = Lazy::new(|| Mutex::new(open_log_file()));

fn log_path() -> Option<PathBuf> {
    let dir = dirs::data_local_dir()?.join("AudioMeter");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("launch.log"))
}

fn open_log_file() -> Option<File> {
    let path = log_path()?;
    // Truncate the log at the start of each run. We're only interested in
    // *this* launch's behavior, and multi-run logs get confusing in a hurry.
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .ok()?;
    Some(file)
}

/// Append a single timestamped line to the startup log. Silently no-ops if
/// the log file could not be opened — we never want logging to be the reason
/// the app fails to start.
pub fn log(msg: impl AsRef<str>) {
    let elapsed = START.elapsed().as_secs_f64();
    let Ok(mut guard) = FILE.lock() else {
        return;
    };
    if let Some(file) = guard.as_mut() {
        let _ = writeln!(file, "[+{:>7.3}s] {}", elapsed, msg.as_ref());
        let _ = file.flush();
    }
}

/// Installs a global panic hook that writes the panic message and location
/// into the startup log before the process dies. Without this, a panic on
/// the UI thread produces no visible artifact at all in a windows-subsystem
/// build.
pub fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown location>".into());
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        log(format!("PANIC at {loc}: {payload}"));
        default_hook(info);
    }));
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Shared flag flipped to `true` once `eframe::run_native` has successfully
/// reached our creator callback. The launch watchdog uses it to stop
/// pinging the log and to suppress the user-facing stall message box as
/// soon as eframe is actually running.
static RUN_NATIVE_REACHED_CREATOR: AtomicBool = AtomicBool::new(false);

/// Called from the eframe creator closure the moment eframe hands control
/// back to us, so the watchdog knows startup is healthy.
pub fn mark_run_native_reached_creator() {
    RUN_NATIVE_REACHED_CREATOR.store(true, Ordering::Release);
}

/// Spawns a background watchdog thread that:
///
///   * Every 2 s, writes a `watchdog: still inside run_native after Xs`
///     line to the startup log while eframe hasn't reached the creator
///     yet. This converts the otherwise-silent "log stops at
///     `main: calling eframe::run_native`" into a minute-by-minute trace
///     so we can tell a slow start from a genuinely indefinite hang.
///   * After `fallback_after`, if the creator still hasn't fired, pops a
///     Win32 `MessageBox` so the user sees a user-facing hint instead of
///     staring at a disappeared splash. On non-Windows only the log
///     lines happen.
///
/// The thread exits cleanly as soon as the creator flips the flag, so
/// healthy launches leave at most one or two watchdog lines in the log.
pub fn spawn_launch_watchdog(fallback_after: Duration) {
    thread::spawn(move || {
        let start = std::time::Instant::now();
        let tick = Duration::from_millis(2000);
        let mut fallback_fired = false;
        loop {
            thread::sleep(tick);
            if RUN_NATIVE_REACHED_CREATOR.load(Ordering::Acquire) {
                log("watchdog: creator reached, exiting");
                return;
            }
            let elapsed = start.elapsed();
            log(format!(
                "watchdog: still inside run_native after {:.1}s",
                elapsed.as_secs_f64()
            ));
            if !fallback_fired && elapsed >= fallback_after {
                fallback_fired = true;
                log("watchdog: firing user-facing fallback");
                #[cfg(windows)]
                show_stall_message_box();
            }
        }
    });
}

/// Installs a Windows `SetUnhandledExceptionFilter` so that structured
/// exceptions — access violations, stack overflows, illegal
/// instructions, heap corruption from FFI / anti-cheat shims, etc. —
/// leave a trace in the startup log even though they bypass Rust's
/// panic hook entirely. The filter also resolves the faulting address
/// back to a loaded DLL name so we can tell *which* injected library
/// blew up (overlay, anti-cheat, screen recorder, etc.), then returns
/// `EXCEPTION_CONTINUE_SEARCH` so Windows still does its normal crash
/// handling afterwards.
#[cfg(windows)]
pub fn install_seh_handler() {
    use std::ffi::c_void;

    #[repr(C)]
    struct ExceptionPointers {
        exception_record: *mut ExceptionRecord,
        context_record: *mut c_void,
    }

    #[repr(C)]
    struct ExceptionRecord {
        exception_code: u32,
        exception_flags: u32,
        exception_record: *mut ExceptionRecord,
        exception_address: *mut c_void,
        number_parameters: u32,
        exception_information: [usize; 15],
    }

    type VectoredHandler =
        unsafe extern "system" fn(*mut ExceptionPointers) -> i32;

    unsafe extern "system" {
        fn SetUnhandledExceptionFilter(filter: VectoredHandler) -> VectoredHandler;
    }

    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

    unsafe extern "system" fn handler(info: *mut ExceptionPointers) -> i32 {
        // IMPORTANT: keep this handler tiny. For `STATUS_STACK_OVERFLOW`
        // (0xC00000FD) we are running on a stack whose guard page has
        // already been tripped — additional heavy work (heap allocations,
        // Vec, heap-using formatters, module enumeration) can double-fault
        // and we lose the log entry entirely. That's why the richer
        // module / module-for-address correlation happens in
        // `snapshot_loaded_modules_to_log` *before* `run_native` instead.
        if !info.is_null() {
            let info = unsafe { &*info };
            if !info.exception_record.is_null() {
                let rec = unsafe { &*info.exception_record };
                let code = rec.exception_code;
                let addr = rec.exception_address;
                let pretty_code = exception_code_name(code);
                log(format!(
                    "SEH: {pretty_code} (0x{code:08X}) at addr={addr:p} flags=0x{:08X}",
                    rec.exception_flags
                ));
            } else {
                log("SEH: unhandled exception (no record)");
            }
        } else {
            log("SEH: unhandled exception (no info pointer)");
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    unsafe {
        let _prev = SetUnhandledExceptionFilter(handler);
    }
}

#[cfg(windows)]
fn exception_code_name(code: u32) -> &'static str {
    // Only the handful we actually care about diagnosing; anything else
    // gets the raw hex value via the calling log line.
    match code {
        0xC0000005 => "EXCEPTION_ACCESS_VIOLATION",
        0xC000001D => "EXCEPTION_ILLEGAL_INSTRUCTION",
        0xC0000025 => "EXCEPTION_NONCONTINUABLE_EXCEPTION",
        0xC0000026 => "EXCEPTION_INVALID_DISPOSITION",
        0xC000008C => "EXCEPTION_ARRAY_BOUNDS_EXCEEDED",
        0xC000008D => "EXCEPTION_FLT_DENORMAL_OPERAND",
        0xC000008E => "EXCEPTION_FLT_DIVIDE_BY_ZERO",
        0xC00000FD => "EXCEPTION_STACK_OVERFLOW",
        0xC0000094 => "EXCEPTION_INT_DIVIDE_BY_ZERO",
        0xC0000096 => "EXCEPTION_PRIV_INSTRUCTION",
        0xC0000135 => "STATUS_DLL_NOT_FOUND",
        0xC0000142 => "STATUS_DLL_INIT_FAILED",
        0xC0000409 => "STATUS_STACK_BUFFER_OVERRUN",
        0xC0000374 => "STATUS_HEAP_CORRUPTION",
        _ => "EXCEPTION",
    }
}

/// Dumps every loaded module's base address, size, and name to the
/// startup log. Call this *before* anything else risky (e.g.
/// `eframe::run_native`) so that if we later see a `SEH:` line
/// pointing at address X in the log, we can look a few lines earlier
/// and match X to a specific DLL. We do this up-front rather than in
/// the SEH handler itself because for `STATUS_STACK_OVERFLOW` the
/// handler runs on a nearly-depleted stack and any extra heap /
/// allocation work there can double-fault.
#[cfg(windows)]
pub fn snapshot_loaded_modules_to_log() {
    let mods = enumerate_modules();
    log(format!("modules: {} loaded before run_native", mods.len()));
    for m in mods.iter().take(256) {
        log(format!(
            "  0x{:016X}..0x{:016X} {}",
            m.base,
            m.base.saturating_add(m.size),
            m.name
        ));
    }
}

/// Kicks off a background thread that polls the process's loaded
/// modules every 25 ms for `total_duration`, logging any newly loaded
/// DLL as it appears.
///
/// This exists to catch the module whose code is executing at the
/// moment of a fatal SEH. The one-shot
/// `snapshot_loaded_modules_to_log` before `run_native` only shows
/// static-link-time DLLs — the actual GPU vendor user-mode driver
/// (`nvoglv64.dll`, `atio6axx.dll`, `ig9icd64.dll`, …) and D3D12
/// runtime DLLs (`d3d12.dll`, `d3d12core.dll`, …) are loaded lazily
/// by winit/wgpu/glutin *while* we're hung-or-crashing inside
/// `run_native`, so they never appear in the pre-snapshot.
///
/// The sidecar runs on a completely independent thread, so even if
/// the main thread blows its stack at +0.25 s, this thread keeps
/// logging for a bit longer and gets the observation into the file
/// before Windows tears the process down.
#[cfg(windows)]
pub fn spawn_module_load_tracer(total_duration: Duration) {
    std::thread::spawn(move || {
        let tick = Duration::from_millis(25);
        let start = std::time::Instant::now();
        // Seed with currently-loaded module bases so we only log *new*
        // arrivals. Pre-run_native snapshot already printed the full
        // starting set.
        let mut known: std::collections::HashSet<usize> = enumerate_modules()
            .into_iter()
            .map(|m| m.base)
            .collect();
        while start.elapsed() < total_duration {
            std::thread::sleep(tick);
            for m in enumerate_modules() {
                if known.insert(m.base) {
                    log(format!(
                        "module-loaded: 0x{:016X}..0x{:016X} {}",
                        m.base,
                        m.base.saturating_add(m.size),
                        m.name
                    ));
                }
            }
        }
    });
}

#[cfg(windows)]
struct ModuleInfo {
    name: String,
    base: usize,
    size: usize,
}

#[cfg(windows)]
fn enumerate_modules() -> Vec<ModuleInfo> {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct ModuleInfoRaw {
        lp_base_of_dll: *mut c_void,
        size_of_image: u32,
        entry_point: *mut c_void,
    }

    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32EnumProcessModules(
            h_process: *mut c_void,
            lphmodule: *mut *mut c_void,
            cb: u32,
            lpcb_needed: *mut u32,
        ) -> i32;
        fn K32GetModuleBaseNameW(
            h_process: *mut c_void,
            h_module: *mut c_void,
            base_name: *mut u16,
            size: u32,
        ) -> u32;
        fn K32GetModuleInformation(
            h_process: *mut c_void,
            h_module: *mut c_void,
            lp_mod_info: *mut ModuleInfoRaw,
            cb: u32,
        ) -> i32;
    }

    let mut out = Vec::new();
    let process = unsafe { GetCurrentProcess() };

    // First call: figure out how many bytes of HMODULE handles we need.
    let mut needed: u32 = 0;
    let rc = unsafe { K32EnumProcessModules(process, std::ptr::null_mut(), 0, &mut needed) };
    if rc == 0 || needed == 0 {
        return out;
    }

    let count = (needed as usize) / std::mem::size_of::<*mut c_void>();
    let mut handles: Vec<*mut c_void> = vec![std::ptr::null_mut(); count];
    let rc = unsafe {
        K32EnumProcessModules(
            process,
            handles.as_mut_ptr(),
            (handles.len() * std::mem::size_of::<*mut c_void>()) as u32,
            &mut needed,
        )
    };
    if rc == 0 {
        return out;
    }

    for handle in handles {
        if handle.is_null() {
            continue;
        }
        let mut name_buf = [0u16; 260];
        let len = unsafe {
            K32GetModuleBaseNameW(process, handle, name_buf.as_mut_ptr(), name_buf.len() as u32)
        };
        let name = if len > 0 {
            String::from_utf16_lossy(&name_buf[..len as usize])
        } else {
            String::from("<unknown>")
        };
        let mut info = ModuleInfoRaw::default();
        let rc = unsafe {
            K32GetModuleInformation(
                process,
                handle,
                &mut info,
                std::mem::size_of::<ModuleInfoRaw>() as u32,
            )
        };
        if rc == 0 {
            continue;
        }
        out.push(ModuleInfo {
            name,
            base: info.lp_base_of_dll as usize,
            size: info.size_of_image as usize,
        });
    }
    out.sort_by_key(|m| m.base);
    out
}

#[cfg(windows)]
fn show_stall_message_box() {
    use std::ffi::OsStr;
    use std::iter::once;
    use std::os::windows::ffi::OsStrExt;

    unsafe extern "system" {
        fn MessageBoxW(
            hwnd: *mut core::ffi::c_void,
            text: *const u16,
            caption: *const u16,
            u_type: u32,
        ) -> i32;
    }
    const MB_OK: u32 = 0x0000_0000;
    const MB_ICONWARNING: u32 = 0x0000_0030;
    const MB_SETFOREGROUND: u32 = 0x0001_0000;
    const MB_TOPMOST: u32 = 0x0004_0000;

    fn to_wide(s: &str) -> Vec<u16> {
        OsStr::new(s).encode_wide().chain(once(0)).collect()
    }

    let text = to_wide(
        "Audio Meter is having trouble starting.\n\n\
         This usually happens when launching alongside a game whose \
         anti-cheat or graphics driver holds exclusive resources. \
         Please close the game (or switch it to Windowed mode) and \
         try launching Audio Meter again.\n\n\
         A diagnostic log has been written to:\n\
         %LOCALAPPDATA%\\AudioMeter\\launch.log",
    );
    let caption = to_wide("AYR Audio Meter");

    unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            text.as_ptr(),
            caption.as_ptr(),
            MB_OK | MB_ICONWARNING | MB_SETFOREGROUND | MB_TOPMOST,
        );
    }
}

