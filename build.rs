//! Icon pipeline for the AYR Audio Meter build.
//!
//! The source of truth is `assets/app_icon.ico` — a multi-frame Windows ICO
//! with a proper alpha channel. We take that file as-is and fan it out to the
//! three consumers that need the icon in different shapes:
//!
//! * **Windows executable resource** — the `.ico` is embedded into the PE via
//!   `winres` so it shows up as the taskbar / Alt-Tab icon.
//! * **Inno Setup installer** — the same `.ico` is copied to
//!   `installer/AudioMeter.ico` so installed Start-Menu / Desktop shortcuts
//!   use it directly.
//! * **Runtime image (splash + egui window icon)** — we extract the largest
//!   RGBA frame and write it as `assets/app_icon.png` so the runtime code in
//!   `src/splash.rs` and `src/main.rs` can `include_bytes!` a PNG (which they
//!   need because `image`'s PNG decoder is what they're wired to).
//!
//! All three outputs are regenerated whenever `assets/app_icon.ico` changes.

use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// `winres`'s registry discovery (`reg query`) can fail for MSVC build scripts when the project
/// lives on `\\wsl...\`, leaving an empty toolkit path. Older winres logic also joins
/// `<Kits\10>\bin\x64\rc.exe`, but modern SDKs ship `bin\<version>\x64\rc.exe` only.
/// Resolve the directory that actually contains `rc.exe` and pass it via [`set_toolkit_path`]
/// (that directory is `bin\<version>\x64`, per winres docs).
#[cfg(windows)]
fn windows_kits_10_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("WindowsSdkDir") {
        let p = PathBuf::from(dir.trim_end_matches(['\\', '/']));
        if p.is_dir() {
            return Some(p);
        }
    }
    for candidate in [
        PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10"),
        PathBuf::from(r"C:\Program Files\Windows Kits\10"),
    ] {
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(windows)]
fn winres_msvc_toolkit_dir() -> Option<PathBuf> {
    let kit_root = windows_kits_10_root()?;
    let bin = kit_root.join("bin");
    let mut with_rc: Vec<PathBuf> = std::fs::read_dir(&bin).ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("x64").join("rc.exe").is_file())
        .collect();
    with_rc.sort_by(|a, b| {
        let sa = a.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let sb = b.file_name().and_then(|s| s.to_str()).unwrap_or("");
        sb.cmp(sa)
    });
    with_rc.into_iter().next().map(|v| v.join("x64"))
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let ico_path = manifest_dir.join("assets/app_icon.ico");
    println!("cargo:rerun-if-changed={}", ico_path.display());

    let ico_bytes = std::fs::read(&ico_path).expect("read assets/app_icon.ico");
    let icon_dir = ico::IconDir::read(Cursor::new(&ico_bytes)).expect("parse app_icon.ico");

    // Runtime PNG (splash + egui window icon) — largest frame, alpha preserved.
    let png_path = manifest_dir.join("assets/app_icon.png");
    write_largest_frame_as_png(&icon_dir, &png_path);

    // Installer — copy the ICO verbatim. The designer shipped a proper
    // multi-frame ICO with anti-aliased transparency, so re-encoding is
    // pointless and only risks degrading it.
    let installer_ico = manifest_dir.join("installer/AudioMeter.ico");
    if let Some(parent) = installer_ico.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::copy(&ico_path, &installer_ico).expect("copy installer/AudioMeter.ico");

    // Windows PE resource — embed the original ICO as icon #1 so Explorer and
    // the taskbar show it on the .exe itself.
    #[cfg(windows)]
    {
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let staged_ico = out_dir.join("app.ico");
        std::fs::copy(&ico_path, &staged_ico).expect("stage app.ico for winres");

        // `llvm-rc` / the MSVC resource toolchain often fails on UNC paths such
        // as `\\wsl.localhost\...` when the project lives on a WSL mount. Point
        // winres at a normal path under `%TEMP%` in that case.
        let icon_for_rc = {
            let s = staged_ico.to_string_lossy();
            if s.starts_with("\\\\") {
                let tmp = std::env::temp_dir().join(format!(
                    "ayr-audio-meter-winres-{}.ico",
                    std::process::id()
                ));
                std::fs::copy(&staged_ico, &tmp).expect("copy ico to temp for winres");
                tmp
            } else {
                staged_ico
            }
        };

        let mut res = winres::WindowsResource::new();
        if let Some(toolkit) = winres_msvc_toolkit_dir() {
            if let Some(s) = toolkit.to_str() {
                res.set_toolkit_path(s);
            }
        }
        res.set_icon(icon_for_rc.to_str().expect("ico path utf-8"));
        res.set("FileDescription", "AYR Audio Meter");
        res.set("ProductName", "AYR Audio Meter");
        res.compile().expect("WindowsResource::compile");
    }
}

/// Extracts the largest RGBA frame from `icon_dir` and writes it as a PNG at
/// `out_path`. Decoding is strict — a malformed ICO fails the build rather
/// than silently producing a broken runtime image.
fn write_largest_frame_as_png(icon_dir: &ico::IconDir, out_path: &Path) {
    let best = icon_dir
        .entries()
        .iter()
        .max_by_key(|e| (e.width() as u32) * (e.height() as u32))
        .expect("app_icon.ico must contain at least one frame");

    let image = best.decode().expect("decode largest ico frame");
    let (w, h) = (image.width(), image.height());
    let rgba = image.rgba_data().to_vec();

    let buf = image::RgbaImage::from_vec(w, h, rgba)
        .expect("largest frame rgba dimensions match width * height * 4");
    let mut file = File::create(out_path).expect("create assets/app_icon.png");
    buf.write_to(&mut file, image::ImageFormat::Png)
        .expect("write assets/app_icon.png");
}
