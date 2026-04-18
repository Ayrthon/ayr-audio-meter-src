//! Embeds a Windows `.ico` into the PE, refreshes `installer/AudioMeter.ico` for Inno Setup,
//! and rebuilds when `assets/app_icon.png` changes.

use std::fs::File;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let png_path = manifest_dir.join("assets/app_icon.png");
    println!("cargo:rerun-if-changed={}", png_path.display());

    let png_bytes = std::fs::read(&png_path).expect("read assets/app_icon.png");
    let master =
        ico::IconImage::read_png(std::io::Cursor::new(&png_bytes)).expect("decode app_icon.png");

    let mut icon_dir = ico::IconDir::new(ico::ResourceType::Icon);
    for size in [16_u32, 32, 48, 64, 256] {
        let image = if master.width() == size && master.height() == size {
            master.clone()
        } else {
            resize_icon(&master, size, size)
        };
        icon_dir.add_entry(ico::IconDirEntry::encode(&image).expect("encode ico entry"));
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ico_path = out_dir.join("app.ico");
    let mut file = File::create(&ico_path).expect("create app.ico");
    icon_dir.write(&mut file).expect("write app.ico");

    let installer_ico = manifest_dir.join("installer/AudioMeter.ico");
    if let Some(parent) = installer_ico.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::copy(&ico_path, &installer_ico).expect("copy installer/AudioMeter.ico");

    #[cfg(windows)]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon(ico_path.to_str().expect("ico path utf-8"));
        res.set("FileDescription", "Audio Meter");
        res.set("ProductName", "Audio Meter");
        res.compile().expect("WindowsResource::compile");
    }
}

fn resize_icon(src: &ico::IconImage, w: u32, h: u32) -> ico::IconImage {
    let rgba = src.rgba_data().to_vec();
    let img = image::RgbaImage::from_raw(src.width(), src.height(), rgba)
        .expect("icon rgba dimensions");
    let resized = image::imageops::resize(&img, w, h, image::imageops::FilterType::Lanczos3);
    ico::IconImage::from_rgba_data(w, h, resized.into_raw())
}
