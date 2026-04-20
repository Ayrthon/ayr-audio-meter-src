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

        let mut res = winres::WindowsResource::new();
        res.set_icon(staged_ico.to_str().expect("ico path utf-8"));
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
