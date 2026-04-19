//! When peak level crosses a user threshold, save 5 s pre-roll + 5 s post-roll
//! to a float WAV under the app data directory (timestamp filename).
//!
//! Triggering uses a **latch**: the level must drop **below** the threshold before a fire can
//! occur, and changing the threshold resets that requirement. That way lowering the red line
//! does not instantly trip on an old peak-hold — only a **new** meter excursion after a clear.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use chrono::Local;
use parking_lot::Mutex;

/// Seconds of audio kept before the trigger (ring buffer).
pub const PRE_ROLL_SECS: f64 = 5.0;
/// Seconds recorded after the trigger before writing the file.
pub const POST_ROLL_SECS: f64 = 5.0;
/// Minimum time between saved clips (avoids flooding the disk).
pub const COOLDOWN_SECS: f64 = 5.0;

/// Clip threshold slider / graph-drag range (dBFS), kept in sync with the UI slider.
pub const CLIP_THRESHOLD_SLIDER_MIN_DB: f32 = -48.0;
pub const CLIP_THRESHOLD_SLIDER_MAX_DB: f32 = 0.0;

/// User-toggled capture + threshold (read from the audio thread).
pub struct ClipSettings {
    enabled: AtomicBool,
    threshold_db_bits: AtomicU32,
}

impl ClipSettings {
    pub fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            // Default −1 dBFS — “almost clipping”.
            threshold_db_bits: AtomicU32::new((-1.0f32).to_bits()),
        }
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, v: bool) {
        self.enabled.store(v, Ordering::Relaxed);
    }

    #[inline]
    pub fn threshold_db(&self) -> f32 {
        f32::from_bits(self.threshold_db_bits.load(Ordering::Relaxed))
    }

    pub fn set_threshold_db(&self, db: f32) {
        self.threshold_db_bits.store(db.to_bits(), Ordering::Relaxed);
    }
}

pub type SharedClipSettings = Arc<ClipSettings>;

pub fn make_shared_clip_settings() -> SharedClipSettings {
    Arc::new(ClipSettings::new())
}

/// Last write result for the UI.
#[derive(Default)]
pub struct ClipStatus {
    pub last_path: Option<String>,
    pub last_error: Option<String>,
    /// Successful clip writes since app start (UI derives “this arm” via a baseline).
    pub total_clips_saved: u64,
    /// When a clip **trigger** just fired (threshold crossed). History graph fades red briefly.
    pub graph_trigger_flash_at: Option<Instant>,
}

pub type SharedClipStatus = Arc<Mutex<ClipStatus>>;

pub fn make_shared_clip_status() -> SharedClipStatus {
    Arc::new(Mutex::new(ClipStatus::default()))
}

pub type SharedClipCapture = Arc<Mutex<ClipCapture>>;

pub fn make_shared_clip_capture() -> SharedClipCapture {
    Arc::new(Mutex::new(ClipCapture::new()))
}

struct InterleavedRing {
    buf: Vec<f32>,
    w: usize,
    total: u64,
}

impl InterleavedRing {
    fn new(cap_samples: usize) -> Self {
        Self {
            buf: vec![0.0; cap_samples.max(1)],
            w: 0,
            total: 0,
        }
    }

    fn push_block(&mut self, data: &[f32]) {
        let cap = self.buf.len();
        if cap == 0 {
            return;
        }
        for &s in data {
            self.buf[self.w] = s;
            self.w = (self.w + 1) % cap;
            self.total = self.total.saturating_add(1);
        }
    }

    /// Oldest → newest (length == capacity once `total` ≥ capacity).
    fn ordered_snapshot(&self) -> Vec<f32> {
        let cap = self.buf.len();
        if cap == 0 {
            return Vec::new();
        }
        if self.total < cap as u64 {
            return self.buf[..self.w].to_vec();
        }
        let mut out = Vec::with_capacity(cap);
        out.extend_from_slice(&self.buf[self.w..]);
        out.extend_from_slice(&self.buf[..self.w]);
        out
    }
}

enum Phase {
    Idle,
    CollectingPost {
        pre: Vec<f32>,
        post: Vec<f32>,
        needed: usize,
    },
}

pub struct ClipCapture {
    ring: InterleavedRing,
    phase: Phase,
    sample_rate: u32,
    channels: usize,
    cooldown_until: Option<Instant>,
    /// Threshold bits last applied to the latch (`u32::MAX` = never initialized).
    last_threshold_bits: u32,
    /// After a trigger, a threshold edit, or first arm: false until `trigger_peak_db` has fallen
    /// **below** the current threshold. Prevents firing from an old peak-hold when the user
    /// lowers the line — only a **new** excursion after a clear counts.
    armed: bool,
}

impl ClipCapture {
    pub fn new() -> Self {
        Self {
            ring: InterleavedRing::new(1),
            phase: Phase::Idle,
            sample_rate: 0,
            channels: 0,
            cooldown_until: None,
            last_threshold_bits: u32::MAX,
            armed: false,
        }
    }

    fn ensure_capacity(&mut self, sample_rate: u32, channels: usize) {
        if sample_rate == 0 || channels == 0 {
            return;
        }
        if self.sample_rate == sample_rate && self.channels == channels {
            return;
        }
        self.sample_rate = sample_rate;
        self.channels = channels;
        let cap = (PRE_ROLL_SECS * sample_rate as f64 * channels as f64).ceil() as usize;
        self.ring = InterleavedRing::new(cap.max(1));
        self.phase = Phase::Idle;
        self.cooldown_until = None;
        self.armed = false;
    }

    /// Call from the audio callback on **raw** float samples (same samples written to WAV).
    ///
    /// `trigger_peak_db` must match the **peak-hold** white line on the meters: same **frequency
    /// weighting**, **true peak** setting, and **meter mode** ballistics after
    /// [`crate::audio::process_weighted_levels`] (not the raw block peak). The history graph
    /// traces the coloured bar fill, not this hold value.
    pub fn process_block(
        &mut self,
        data: &[f32],
        channels: usize,
        sample_rate: u32,
        settings: &ClipSettings,
        status: &SharedClipStatus,
        trigger_peak_db: f32,
    ) {
        if channels == 0 || data.is_empty() {
            return;
        }
        self.ensure_capacity(sample_rate, channels);

        let phase = std::mem::replace(&mut self.phase, Phase::Idle);

        match phase {
            Phase::CollectingPost {
                mut pre,
                mut post,
                needed,
            } => {
                self.ring.push_block(data);
                let space = needed.saturating_sub(post.len());
                let take = space.min(data.len());
                post.extend_from_slice(&data[..take]);
                if post.len() >= needed {
                    self.cooldown_until =
                        Some(Instant::now() + Duration::from_secs_f64(COOLDOWN_SECS));
                    pre.extend(post);
                    spawn_wav_write(pre, channels as u16, sample_rate, status.clone());
                    self.phase = Phase::Idle;
                } else {
                    self.phase = Phase::CollectingPost {
                        pre,
                        post,
                        needed,
                    };
                }
                return;
            }
            Phase::Idle => {}
        }

        // Idle path
        self.ring.push_block(data);

        if !settings.enabled() {
            self.phase = Phase::Idle;
            self.armed = false;
            return;
        }

        let thr = settings.threshold_db();
        let thr_bits = thr.to_bits();
        if thr_bits != self.last_threshold_bits {
            self.last_threshold_bits = thr_bits;
            self.armed = false;
        }

        // Schmitt: arm after we see level **below** threshold (matches “new peak” on the meters).
        // Run even during cooldown so a dip while waiting still counts toward re-arming.
        if trigger_peak_db < thr {
            self.armed = true;
        }

        if let Some(until) = self.cooldown_until {
            if Instant::now() < until {
                self.phase = Phase::Idle;
                return;
            }
            self.cooldown_until = None;
        }

        if !self.armed || trigger_peak_db < thr {
            self.phase = Phase::Idle;
            return;
        }

        self.armed = false;

        let pre = self.ring.ordered_snapshot();
        let needed = (POST_ROLL_SECS * sample_rate as f64 * channels as f64).ceil() as usize;
        status.lock().graph_trigger_flash_at = Some(Instant::now());
        self.phase = Phase::CollectingPost {
            pre,
            post: Vec::with_capacity(needed),
            needed,
        };
    }
}

fn clips_dir() -> Option<PathBuf> {
    let base = dirs::data_local_dir()?;
    let dir = base.join("audio-meter-rs").join("clips");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Display path for the UI (folder is created on first successful save).
pub fn clips_directory_display() -> String {
    clips_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(could not resolve app data folder)".into())
}

/// Resolved clips folder (creates it if possible). For "open in explorer" etc.
pub fn clips_dir_path() -> Option<PathBuf> {
    clips_dir()
}

/// One `.wav` file in the clips folder (newest files first).
#[derive(Clone, Debug)]
pub struct ClipFileInfo {
    pub path: PathBuf,
    pub file_name: String,
}

/// Lists `*.wav` / `*.WAV` in the clips directory, **newest modification time first**.
pub fn list_clip_wavs() -> Vec<ClipFileInfo> {
    let Some(dir) = clips_dir() else {
        return Vec::new();
    };
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut v: Vec<ClipFileInfo> = rd
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            let name = path.file_name()?.to_string_lossy().into_owned();
            let lower = name.to_ascii_lowercase();
            if !lower.ends_with(".wav") {
                return None;
            }
            Some(ClipFileInfo {
                path,
                file_name: name,
            })
        })
        .collect();
    v.sort_by(|a, b| {
        let ta = std::fs::metadata(&a.path)
            .and_then(|m| m.modified())
            .ok();
        let tb = std::fs::metadata(&b.path)
            .and_then(|m| m.modified())
            .ok();
        tb.cmp(&ta)
    });
    v
}

fn spawn_wav_write(
    samples: Vec<f32>,
    channels: u16,
    sample_rate: u32,
    status: SharedClipStatus,
) {
    if samples.is_empty() {
        return;
    }
    let Some(dir) = clips_dir() else {
        status.lock().last_error = Some("Could not create clips folder.".into());
        return;
    };
    let ts = Local::now();
    let name = format!(
        "{}_{:03}.wav",
        ts.format("%Y%m%d_%H%M%S"),
        ts.timestamp_subsec_millis()
    );
    let path = dir.join(name);

    std::thread::Builder::new()
        .name("clip-wav".into())
        .spawn(move || {
            let result = write_wav_f32(&path, &samples, channels, sample_rate);
            let mut s = status.lock();
            match result {
                Ok(()) => {
                    s.last_error = None;
                    s.last_path = Some(path.display().to_string());
                    s.total_clips_saved = s.total_clips_saved.saturating_add(1);
                }
                Err(e) => {
                    s.last_error = Some(e.to_string());
                }
            }
        })
        .ok();
}

fn write_wav_f32(
    path: &std::path::Path,
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}
