//! Windows splash screen — a transparent, borderless, always-on-top icon shown
//! over the desktop while the main audio-meter window is still initializing.
//!
//! We use a native Win32 **layered window** instead of an `egui` / `eframe`
//! viewport because `eframe` + `glow` transparency on Windows is famously
//! unreliable (tracked upstream as emilk/egui#4451 and friends). A layered
//! window composited by DWM with `UpdateLayeredWindow(..., ULW_ALPHA)` gives
//! rock-solid per-pixel alpha on every Windows version we care about, so the
//! icon actually floats over the desktop instead of appearing inside a black
//! box.
//!
//! The splash runs on its own thread with its own message pump, so it pops up
//! instantly before `eframe` (which takes a moment to spin up GL + audio
//! devices) has to do anything, and fades out on its own timetable regardless
//! of whatever the main UI thread is doing.

#![cfg(windows)]
#![allow(non_snake_case, non_camel_case_types, dead_code)]

use std::ffi::OsStr;
use std::mem;
use std::os::windows::ffi::OsStrExt;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Handle keeping the splash thread alive. Dropping it signals the splash
/// window to close early (e.g. on main-thread panic during init) and joins
/// the thread.
pub struct SplashHandle {
    done: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl Drop for SplashHandle {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(join) = self.join.take() {
            // We don't really care if the splash thread panicked — it's
            // purely decorative. Eat the error.
            let _ = join.join();
        }
    }
}

/// Launches the splash window on a background thread.
///
/// * `total_lifetime` — how long the splash is visible from spawn (including
///   the fade-out tail).
/// * `fade_out` — duration of the trailing fade to zero alpha. Must be less
///   than `total_lifetime`.
pub fn spawn(total_lifetime: Duration, fade_out: Duration) -> SplashHandle {
    let done = Arc::new(AtomicBool::new(false));
    let done_for_thread = done.clone();
    let join = thread::spawn(move || {
        crate::startup_log::log("splash: thread started");
        // Safety: the thread owns all resources it creates and tears them
        // down before returning, and the `done` flag is only observed, not
        // dereferenced, across threads.
        unsafe { run(done_for_thread, total_lifetime, fade_out) };
        crate::startup_log::log("splash: thread exiting");
    });
    SplashHandle {
        done,
        join: Some(join),
    }
}

// ---------------------------------------------------------------------------
// Win32 bindings
//
// Declared by hand to avoid pulling in the `windows` / `windows-sys` crates
// just for a splash screen (the project already uses raw `extern "system"` in
// main.rs and audio.rs).
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type c_void = core::ffi::c_void;
type HWND = *mut c_void;
type HINSTANCE = *mut c_void;
type HDC = *mut c_void;
type HBITMAP = *mut c_void;
type HGDIOBJ = *mut c_void;
type HICON = *mut c_void;
type HCURSOR = *mut c_void;
type HBRUSH = *mut c_void;
type HMENU = *mut c_void;
type LRESULT = isize;
type LPARAM = isize;
type WPARAM = usize;
type ATOM = u16;
type DWORD = u32;
type UINT = u32;
type BOOL = i32;
type LONG = i32;
type WORD = u16;
type BYTE = u8;

type WNDPROC = unsafe extern "system" fn(HWND, UINT, WPARAM, LPARAM) -> LRESULT;

#[repr(C)]
#[derive(Clone, Copy)]
struct POINT {
    x: LONG,
    y: LONG,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SIZE {
    cx: LONG,
    cy: LONG,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BLENDFUNCTION {
    BlendOp: BYTE,
    BlendFlags: BYTE,
    SourceConstantAlpha: BYTE,
    AlphaFormat: BYTE,
}

#[repr(C)]
struct WNDCLASSEXW {
    cbSize: UINT,
    style: UINT,
    lpfnWndProc: Option<WNDPROC>,
    cbClsExtra: i32,
    cbWndExtra: i32,
    hInstance: HINSTANCE,
    hIcon: HICON,
    hCursor: HCURSOR,
    hbrBackground: HBRUSH,
    lpszMenuName: *const u16,
    lpszClassName: *const u16,
    hIconSm: HICON,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MSG {
    hwnd: HWND,
    message: UINT,
    wParam: WPARAM,
    lParam: LPARAM,
    time: DWORD,
    pt: POINT,
    lPrivate: DWORD,
}

#[repr(C)]
struct BITMAPINFOHEADER {
    biSize: DWORD,
    biWidth: LONG,
    biHeight: LONG,
    biPlanes: WORD,
    biBitCount: WORD,
    biCompression: DWORD,
    biSizeImage: DWORD,
    biXPelsPerMeter: LONG,
    biYPelsPerMeter: LONG,
    biClrUsed: DWORD,
    biClrImportant: DWORD,
}

#[repr(C)]
struct BITMAPINFO {
    bmiHeader: BITMAPINFOHEADER,
    // Unused for 32-bit DIBs but kept for struct size.
    bmiColors: [u32; 3],
}

// Window styles & constants we actually use.
const WS_POPUP: DWORD = 0x8000_0000;
const WS_EX_LAYERED: DWORD = 0x0008_0000;
const WS_EX_TOPMOST: DWORD = 0x0000_0008;
const WS_EX_TRANSPARENT: DWORD = 0x0000_0020;
const WS_EX_TOOLWINDOW: DWORD = 0x0000_0080;
const WS_EX_NOACTIVATE: DWORD = 0x0800_0000;

const SW_SHOWNOACTIVATE: i32 = 4;

const SM_CXSCREEN: i32 = 0;
const SM_CYSCREEN: i32 = 1;

const ULW_ALPHA: DWORD = 0x0000_0002;

const AC_SRC_OVER: BYTE = 0x00;
const AC_SRC_ALPHA: BYTE = 0x01;

const BI_RGB: DWORD = 0;
const DIB_RGB_COLORS: UINT = 0;

const WM_CLOSE: UINT = 0x0010;
const WM_DESTROY: UINT = 0x0002;
const WM_QUIT: UINT = 0x0012;
const PM_REMOVE: UINT = 0x0001;

#[link(name = "user32")]
unsafe extern "system" {
    fn RegisterClassExW(unnamedParam1: *const WNDCLASSEXW) -> ATOM;
    fn UnregisterClassW(lpClassName: *const u16, hInstance: HINSTANCE) -> BOOL;
    fn CreateWindowExW(
        dwExStyle: DWORD,
        lpClassName: *const u16,
        lpWindowName: *const u16,
        dwStyle: DWORD,
        X: i32,
        Y: i32,
        nWidth: i32,
        nHeight: i32,
        hWndParent: HWND,
        hMenu: HMENU,
        hInstance: HINSTANCE,
        lpParam: *mut c_void,
    ) -> HWND;
    fn DestroyWindow(hWnd: HWND) -> BOOL;
    fn ShowWindow(hWnd: HWND, nCmdShow: i32) -> BOOL;
    fn DefWindowProcW(hWnd: HWND, Msg: UINT, wParam: WPARAM, lParam: LPARAM) -> LRESULT;
    fn GetSystemMetrics(nIndex: i32) -> i32;
    fn GetDC(hWnd: HWND) -> HDC;
    fn ReleaseDC(hWnd: HWND, hDC: HDC) -> i32;
    fn UpdateLayeredWindow(
        hWnd: HWND,
        hdcDst: HDC,
        pptDst: *const POINT,
        psize: *const SIZE,
        hdcSrc: HDC,
        pptSrc: *const POINT,
        crKey: u32,
        pblend: *const BLENDFUNCTION,
        dwFlags: DWORD,
    ) -> BOOL;
    fn PeekMessageW(
        lpMsg: *mut MSG,
        hWnd: HWND,
        wMsgFilterMin: UINT,
        wMsgFilterMax: UINT,
        wRemoveMsg: UINT,
    ) -> BOOL;
    fn TranslateMessage(lpMsg: *const MSG) -> BOOL;
    fn DispatchMessageW(lpMsg: *const MSG) -> LRESULT;
    fn PostQuitMessage(nExitCode: i32);
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleW(lpModuleName: *const u16) -> HINSTANCE;
}

#[link(name = "gdi32")]
unsafe extern "system" {
    fn CreateCompatibleDC(hDC: HDC) -> HDC;
    fn DeleteDC(hDC: HDC) -> BOOL;
    fn CreateDIBSection(
        hdc: HDC,
        pbmi: *const BITMAPINFO,
        usage: UINT,
        ppvBits: *mut *mut c_void,
        hSection: *mut c_void,
        offset: DWORD,
    ) -> HBITMAP;
    fn SelectObject(hdc: HDC, h: HGDIOBJ) -> HGDIOBJ;
    fn DeleteObject(ho: HGDIOBJ) -> BOOL;
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/// Square size of the splash window on screen, in physical pixels. Since the
/// process is opted into system-DPI awareness by `main.rs` before the splash
/// spawns, physical pixels line up with what users perceive as a comfortable
/// icon size (≈ 2.5 cm on typical displays).
const SPLASH_SIDE: u32 = 256;

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: UINT,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_CLOSE => {
            unsafe { DestroyWindow(hwnd) };
            0
        }
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            0
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

/// Encode a UTF-8 string as a NUL-terminated UTF-16 buffer suitable for
/// Win32 wide-string APIs.
fn wide(s: &str) -> Vec<u16> {
    OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
}

unsafe fn run(done: Arc<AtomicBool>, total_lifetime: Duration, fade_out: Duration) {
    // -----------------------------------------------------------------
    // Decode the icon up-front. If this fails for any reason we just skip
    // the splash — it's cosmetic, not critical.
    // -----------------------------------------------------------------
    const PNG: &[u8] = include_bytes!("../assets/app_icon.png");
    let Ok(src) = image::load_from_memory(PNG) else {
        return;
    };
    let resized = image::imageops::resize(
        &src.to_rgba8(),
        SPLASH_SIDE,
        SPLASH_SIDE,
        image::imageops::FilterType::Lanczos3,
    );

    // -----------------------------------------------------------------
    // Register a window class unique to this splash.
    // -----------------------------------------------------------------
    let instance = unsafe { GetModuleHandleW(ptr::null()) };
    let class_name = wide("AYRAudioMeterSplash");
    let window_name = wide("AYR Audio Meter");

    let class = WNDCLASSEXW {
        cbSize: mem::size_of::<WNDCLASSEXW>() as UINT,
        style: 0,
        lpfnWndProc: Some(wnd_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: instance,
        hIcon: ptr::null_mut(),
        hCursor: ptr::null_mut(),
        hbrBackground: ptr::null_mut(),
        lpszMenuName: ptr::null(),
        lpszClassName: class_name.as_ptr(),
        hIconSm: ptr::null_mut(),
    };
    if unsafe { RegisterClassExW(&class) } == 0 {
        return;
    }

    // -----------------------------------------------------------------
    // Create a borderless, topmost, layered, non-activating, click-through
    // window centered on the primary monitor.
    // -----------------------------------------------------------------
    let screen_w = unsafe { GetSystemMetrics(SM_CXSCREEN) };
    let screen_h = unsafe { GetSystemMetrics(SM_CYSCREEN) };
    let x = (screen_w - SPLASH_SIDE as i32) / 2;
    let y = (screen_h - SPLASH_SIDE as i32) / 2;

    let hwnd = unsafe {
        CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT
                | WS_EX_NOACTIVATE,
            class_name.as_ptr(),
            window_name.as_ptr(),
            WS_POPUP,
            x,
            y,
            SPLASH_SIDE as i32,
            SPLASH_SIDE as i32,
            ptr::null_mut(),
            ptr::null_mut(),
            instance,
            ptr::null_mut(),
        )
    };
    if hwnd.is_null() {
        unsafe { UnregisterClassW(class_name.as_ptr(), instance) };
        return;
    }

    // -----------------------------------------------------------------
    // Allocate a top-down 32-bit DIB section, blit the icon into it with
    // premultiplied alpha (required by `UpdateLayeredWindow` when using
    // `AC_SRC_ALPHA`), and bind it to an in-memory DC.
    // -----------------------------------------------------------------
    let screen_dc = unsafe { GetDC(ptr::null_mut()) };
    let mem_dc = unsafe { CreateCompatibleDC(screen_dc) };

    let bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: mem::size_of::<BITMAPINFOHEADER>() as DWORD,
            biWidth: SPLASH_SIDE as LONG,
            // Negative height = top-down DIB, which matches the row order
            // produced by the `image` crate after `.to_rgba8()`.
            biHeight: -(SPLASH_SIDE as LONG),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB,
            biSizeImage: 0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        },
        bmiColors: [0; 3],
    };
    let mut bits_ptr: *mut c_void = ptr::null_mut();
    let dib = unsafe {
        CreateDIBSection(
            mem_dc,
            &bmi,
            DIB_RGB_COLORS,
            &mut bits_ptr,
            ptr::null_mut(),
            0,
        )
    };
    if dib.is_null() || bits_ptr.is_null() {
        unsafe {
            DeleteDC(mem_dc);
            ReleaseDC(ptr::null_mut(), screen_dc);
            DestroyWindow(hwnd);
            UnregisterClassW(class_name.as_ptr(), instance);
        }
        return;
    }
    let old_bitmap = unsafe { SelectObject(mem_dc, dib as HGDIOBJ) };

    // Copy RGBA → BGRA with premultiplied alpha. The DIB is the same
    // pixel count as `resized`, so this is a straight one-to-one map.
    let byte_count = (SPLASH_SIDE as usize) * (SPLASH_SIDE as usize) * 4;
    let dib_bytes = unsafe { std::slice::from_raw_parts_mut(bits_ptr as *mut u8, byte_count) };
    for (dst, src) in dib_bytes.chunks_exact_mut(4).zip(resized.chunks_exact(4)) {
        let r = src[0] as u32;
        let g = src[1] as u32;
        let b = src[2] as u32;
        let a = src[3] as u32;
        dst[0] = ((b * a + 127) / 255) as u8;
        dst[1] = ((g * a + 127) / 255) as u8;
        dst[2] = ((r * a + 127) / 255) as u8;
        dst[3] = a as u8;
    }

    // -----------------------------------------------------------------
    // Show the window at full alpha, then enter a small animation loop
    // that handles fade-out and timing.
    // -----------------------------------------------------------------
    let pt_dst = POINT { x, y };
    let pt_src = POINT { x: 0, y: 0 };
    let size = SIZE {
        cx: SPLASH_SIDE as LONG,
        cy: SPLASH_SIDE as LONG,
    };

    let push = |alpha: BYTE| {
        let blend = BLENDFUNCTION {
            BlendOp: AC_SRC_OVER,
            BlendFlags: 0,
            SourceConstantAlpha: alpha,
            AlphaFormat: AC_SRC_ALPHA,
        };
        unsafe {
            UpdateLayeredWindow(
                hwnd,
                screen_dc,
                &pt_dst,
                &size,
                mem_dc,
                &pt_src,
                0,
                &blend,
                ULW_ALPHA,
            );
        }
    };

    push(255);
    unsafe { ShowWindow(hwnd, SW_SHOWNOACTIVATE) };

    let start = Instant::now();
    let fade_start = total_lifetime.saturating_sub(fade_out);
    let fade_len_s = fade_out.as_secs_f32().max(1e-6);

    loop {
        // Drain any pending messages so the window stays responsive to
        // system events (destroy, WM_QUIT, etc.) without blocking on them.
        let mut msg = unsafe { mem::zeroed::<MSG>() };
        while unsafe { PeekMessageW(&mut msg, ptr::null_mut(), 0, 0, PM_REMOVE) } != 0 {
            if msg.message == WM_QUIT {
                // Something asked us to stop — bail out of the animation
                // loop; cleanup follows below.
                break;
            }
            unsafe {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        if done.load(Ordering::Acquire) {
            break;
        }

        let elapsed = start.elapsed();
        if elapsed >= total_lifetime {
            break;
        }

        let alpha = if elapsed < fade_start {
            255
        } else {
            let k = (elapsed - fade_start).as_secs_f32() / fade_len_s;
            ((1.0 - k.clamp(0.0, 1.0)) * 255.0) as BYTE
        };
        push(alpha);

        // ~60 Hz animation tick. Short enough that the fade is smooth,
        // long enough that we don't peg a CPU core on an idle splash.
        thread::sleep(Duration::from_millis(16));
    }

    // -----------------------------------------------------------------
    // Teardown — restore GDI ownership then destroy the window + class.
    // -----------------------------------------------------------------
    unsafe {
        SelectObject(mem_dc, old_bitmap);
        DeleteObject(dib as HGDIOBJ);
        DeleteDC(mem_dc);
        ReleaseDC(ptr::null_mut(), screen_dc);
        DestroyWindow(hwnd);
        UnregisterClassW(class_name.as_ptr(), instance);
    }
}
