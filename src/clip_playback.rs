//! In-app clip preview: load float WAV (same format as clip capture) and play
//! through the default output device via cpal, with seekable playback position.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use parking_lot::Mutex;

#[derive(Debug, Clone)]
pub struct LoadedClip {
    pub sample_rate: u32,
    pub channels: usize,
    pub samples: Vec<f32>,
}

impl LoadedClip {
    pub fn duration_sec(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        let frames = self.samples.len() / self.channels;
        frames as f64 / self.sample_rate as f64
    }

    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels
    }
}

/// Load a 32-bit float WAV (interleaved). Other formats return an error string.
pub fn load_wav_f32(path: &std::path::Path) -> Result<LoadedClip, String> {
    let mut r = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = r.spec();
    if spec.sample_format != hound::SampleFormat::Float || spec.bits_per_sample != 32 {
        return Err("Only 32-bit float WAV clips are supported.".into());
    }
    let ch = spec.channels as usize;
    if ch == 0 {
        return Err("WAV has zero channels.".into());
    }
    let mut samples = Vec::new();
    for s in r.samples::<f32>() {
        samples.push(s.map_err(|e| e.to_string())?);
    }
    Ok(LoadedClip {
        sample_rate: spec.sample_rate,
        channels: ch,
        samples,
    })
}

/// Peak envelope for drawing (one peak per horizontal bucket, max abs sample).
pub fn waveform_peaks(samples: &[f32], channels: usize, max_buckets: usize) -> Vec<f32> {
    if channels == 0 || samples.is_empty() || max_buckets == 0 {
        return Vec::new();
    }
    let frames = samples.len() / channels;
    if frames == 0 {
        return Vec::new();
    }
    let buckets = max_buckets.min(frames).max(1);
    let mut out = vec![0.0_f32; buckets];
    let frames_per = frames as f64 / buckets as f64;
    for b in 0..buckets {
        let t0 = ((b as f64) * frames_per).floor() as usize;
        let t1 = (((b + 1) as f64) * frames_per).ceil() as usize;
        let t1 = t1.min(frames);
        let mut pk = 0.0_f32;
        for f in t0..t1 {
            let base = f * channels;
            for c in 0..channels {
                pk = pk.max(samples[base + c].abs());
            }
        }
        out[b] = pk;
    }
    out
}

enum PlayerCommand {
    ReplacePcm {
        pcm: Vec<f32>,
        channels: usize,
        sample_rate: u32,
    },
    SetPlaying(bool),
    SetPositionFrames(f64),
    /// Seek + play state in one step (avoids EOF race: play=true before pos resets).
    SeekAndSetPlaying {
        position_frames: f64,
        playing: bool,
    },
    Shutdown,
}

struct PlaybackInner {
    pcm: Vec<f32>,
    channels: usize,
    src_rate: u32,
    /// Fractional frame index (0 .. frame_count).
    pos_frames: f64,
    playing: bool,
    /// Smoothed scrub rate (Mixxx-style: UI target is low-passed in the audio thread).
    scrub_rate_smooth: f64,
    /// After a seek / new clip, snap [`scrub_rate_smooth`] to the next target.
    scrub_rate_snap: bool,
    /// Gentle low-pass on output while scratching (reduces “digital” edge).
    scrub_lp: [f32; 8],
}

impl Default for PlaybackInner {
    fn default() -> Self {
        Self {
            pcm: Vec::new(),
            channels: 2,
            src_rate: 48_000,
            pos_frames: 0.0,
            playing: false,
            scrub_rate_smooth: 1.0,
            scrub_rate_snap: true,
            scrub_lp: [0.0; 8],
        }
    }
}

/// Max absolute playback-rate multiplier (file-seconds per wall-second) to avoid huge jumps.
pub const MAX_PLAYBACK_RATE_MUL: f64 = 64.0;

const MAX_RATE_MUL: f64 = MAX_PLAYBACK_RATE_MUL;

/// Catmull–Rom between the middle two samples (`y1`→`y2`, `t` ∈ [0,1]).
#[inline]
fn catmull_rom(y0: f32, y1: f32, y2: f32, y3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * y1)
        + (-y0 + y2) * t
        + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2
        + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3)
}

#[inline]
fn sample_cubic(pcm: &[f32], ch: usize, nf: usize, frame: f64, c: usize) -> f32 {
    if nf == 0 || pcm.is_empty() {
        return 0.0;
    }
    let nf_i = nf as isize;
    let f = frame.clamp(0.0, (nf - 1) as f64 + 1e-9);
    let t = (f - f.floor()) as f32;
    let i = f.floor() as isize;
    let i0 = (i - 1).clamp(0, nf_i - 1) as usize;
    let i1 = i.clamp(0, nf_i - 1) as usize;
    let i2 = (i + 1).clamp(0, nf_i - 1) as usize;
    let i3 = (i + 2).clamp(0, nf_i - 1) as usize;
    let y0 = pcm[i0 * ch + c];
    let y1 = pcm[i1 * ch + c];
    let y2 = pcm[i2 * ch + c];
    let y3 = pcm[i3 * ch + c];
    catmull_rom(y0, y1, y2, y3, t)
}

fn fill_f32_output(
    inner: &Mutex<PlaybackInner>,
    playhead_sec: &Mutex<f64>,
    playing_flag: &AtomicBool,
    scrub_active: &AtomicBool,
    scrub_rate_bits: &AtomicU64,
    data: &mut [f32],
    out_ch: usize,
    out_rate: u32,
) {
    let mut g = inner.lock();
    let nf = if g.channels > 0 {
        g.pcm.len() / g.channels
    } else {
        0
    };
    let out_rate = out_rate.max(1);
    let src_rate = g.src_rate.max(1) as f64;
    let n = data.len() / out_ch.max(1);
    let ch = g.channels.max(1);

    for i in 0..n {
        let scrub_on = scrub_active.load(Ordering::Relaxed);
        let rate_mul = f64::from_bits(scrub_rate_bits.load(Ordering::Relaxed))
            .clamp(-MAX_RATE_MUL, MAX_RATE_MUL);
        let target_rate = if scrub_on {
            rate_mul
        } else if playing_flag.load(Ordering::Relaxed) {
            1.0
        } else {
            0.0
        };

        if scrub_on {
            if g.scrub_rate_snap {
                g.scrub_rate_smooth = target_rate;
                g.scrub_rate_snap = false;
            } else if target_rate.abs() < 1e-9 {
                // Instant silence when UI sends 0 (no exponential “slow‑down”).
                g.scrub_rate_smooth = 0.0;
            } else {
                // ~3–6 ms time constant at 48 kHz — smooths non‑zero control steps only.
                g.scrub_rate_smooth +=
                    (target_rate - g.scrub_rate_smooth) * 0.0065;
            }
        } else {
            g.scrub_rate_smooth = target_rate;
        }

        let effective_rate = if scrub_on {
            g.scrub_rate_smooth
        } else {
            target_rate
        };

        if nf == 0 || effective_rate.abs() < 1e-9 {
            for c in 0..out_ch {
                data[i * out_ch + c] = 0.0;
            }
            continue;
        }

        // Normal 1× playback: stop once the playhead passes the end of the file.
        if !scrub_on && g.pos_frames >= nf as f64 {
            g.playing = false;
            playing_flag.store(false, Ordering::Relaxed);
            g.pos_frames = nf as f64;
            for c in 0..out_ch {
                data[i * out_ch + c] = 0.0;
            }
            *playhead_sec.lock() = nf as f64 / src_rate;
            continue;
        }

        // Scratch forward into the last sample: hold silence until the user pulls back.
        if scrub_on && effective_rate > 0.0 && g.pos_frames >= nf as f64 - 1e-9 {
            for c in 0..out_ch {
                data[i * out_ch + c] = 0.0;
            }
            continue;
        }

        // Scratch reverse past the start.
        if scrub_on && effective_rate < 0.0 && g.pos_frames <= 0.0 {
            for c in 0..out_ch {
                data[i * out_ch + c] = 0.0;
            }
            continue;
        }

        let pos_read = g
            .pos_frames
            .clamp(0.0, nf.saturating_sub(1) as f64 + 1.0 - 1e-9);

        for oc in 0..out_ch {
            let ic = oc % ch;
            let mut s = sample_cubic(&g.pcm, ch, nf, pos_read, ic);
            if scrub_on {
                let z = &mut g.scrub_lp[oc.min(7)];
                // Softens zipper / aliasing edge when scrubbing (DJ apps band-limit heavily).
                *z += 0.11 * (s - *z);
                s = *z;
            } else {
                g.scrub_lp[oc.min(7)] = s;
            }
            data[i * out_ch + oc] = s;
        }

        g.pos_frames += effective_rate * (src_rate / out_rate as f64);
        *playhead_sec.lock() = (g.pos_frames / src_rate).clamp(0.0, nf as f64 / src_rate);
    }
}

fn fill_i16_output(
    inner: &Mutex<PlaybackInner>,
    playhead_sec: &Mutex<f64>,
    playing_flag: &AtomicBool,
    scrub_active: &AtomicBool,
    scrub_rate_bits: &AtomicU64,
    data: &mut [i16],
    out_ch: usize,
    out_rate: u32,
) {
    let mut tmp = vec![0.0_f32; data.len()];
    fill_f32_output(
        inner,
        playhead_sec,
        playing_flag,
        scrub_active,
        scrub_rate_bits,
        &mut tmp,
        out_ch,
        out_rate,
    );
    for (d, &s) in data.iter_mut().zip(tmp.iter()) {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i32;
        *d = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

fn fill_u16_output(
    inner: &Mutex<PlaybackInner>,
    playhead_sec: &Mutex<f64>,
    playing_flag: &AtomicBool,
    scrub_active: &AtomicBool,
    scrub_rate_bits: &AtomicU64,
    data: &mut [u16],
    out_ch: usize,
    out_rate: u32,
) {
    let mut tmp = vec![0.0_f32; data.len()];
    fill_f32_output(
        inner,
        playhead_sec,
        playing_flag,
        scrub_active,
        scrub_rate_bits,
        &mut tmp,
        out_ch,
        out_rate,
    );
    for (d, &s) in data.iter_mut().zip(tmp.iter()) {
        let v = ((s.clamp(-1.0, 1.0) * 32767.0) + 32768.0).round() as i32;
        *d = v.clamp(0, 65535) as u16;
    }
}

/// Owns the cpal output stream on a background thread; cheap to drop (joins thread).
pub struct ClipPlaybackHandle {
    tx: Sender<PlayerCommand>,
    join: Option<JoinHandle<()>>,
    playhead_sec: Arc<Mutex<f64>>,
    duration_sec: Arc<Mutex<f64>>,
    playing: Arc<AtomicBool>,
    /// True while the user drags on the waveform — advance by [`Self::set_playback_rate_mul`].
    scrub_active: Arc<AtomicBool>,
    /// `f64::to_bits` of playback multiplier (file-seconds per wall-second).
    playback_rate_bits: Arc<AtomicU64>,
    /// `(frame_count, sample_rate)` for the last [`Self::load_clip`].
    clip_meta: Arc<Mutex<(usize, u32)>>,
}

impl ClipPlaybackHandle {
    pub fn spawn() -> Result<Self, String> {
        let (tx, rx) = mpsc::channel();
        let inner = Arc::new(Mutex::new(PlaybackInner::default()));
        let playhead_sec = Arc::new(Mutex::new(0.0));
        let duration_sec = Arc::new(Mutex::new(0.0));
        let playing = Arc::new(AtomicBool::new(false));
        let scrub_active = Arc::new(AtomicBool::new(false));
        let playback_rate_bits = Arc::new(AtomicU64::new(1.0f64.to_bits()));
        let clip_meta = Arc::new(Mutex::new((0usize, 48_000u32)));

        let inner_th = inner.clone();
        let playhead_th = playhead_sec.clone();
        let playing_th = playing.clone();
        let scrub_th = scrub_active.clone();
        let rate_th = playback_rate_bits.clone();

        let join = std::thread::Builder::new()
            .name("clip-playback".into())
            .spawn(move || {
                run_player_thread(
                    rx,
                    inner_th,
                    playhead_th,
                    playing_th,
                    scrub_th,
                    rate_th,
                );
            })
            .map_err(|e| e.to_string())?;

        Ok(Self {
            tx,
            join: Some(join),
            playhead_sec,
            duration_sec,
            playing,
            scrub_active,
            playback_rate_bits,
            clip_meta,
        })
    }

    pub fn load_clip(&self, clip: &LoadedClip) -> Result<(), String> {
        let dur = clip.duration_sec();
        *self.duration_sec.lock() = dur;
        *self.playhead_sec.lock() = 0.0;
        self.playing.store(false, Ordering::Relaxed);
        self.scrub_active.store(false, Ordering::Relaxed);
        self.playback_rate_bits
            .store(1.0f64.to_bits(), Ordering::Relaxed);
        *self.clip_meta.lock() = (clip.frame_count(), clip.sample_rate.max(1));
        self
            .tx
            .send(PlayerCommand::ReplacePcm {
                pcm: clip.samples.clone(),
                channels: clip.channels,
                sample_rate: clip.sample_rate,
            })
            .map_err(|_| "Clip player thread stopped.".to_string())?;
        Ok(())
    }

    pub fn set_playing(&self, v: bool) {
        // Do not update `playing` here: the audio callback can run before the player thread
        // applies the command. The player thread sets the atomic when it processes `SetPlaying`.
        let _ = self.tx.send(PlayerCommand::SetPlaying(v));
    }

    /// Seek (in seconds) and set play state in **one** player-thread step so the audio callback
    /// never sees `playing == true` with `pos_frames` still parked at EOF.
    pub fn seek_and_set_playing(&self, sec: f64, playing: bool) {
        let (nf, sr) = *self.clip_meta.lock();
        let sr_f = sr.max(1) as f64;
        let mut frames = (sec * sr_f).clamp(0.0, nf as f64);
        // `seek_sec` clamps to `nf` frames — but the render loop treats `pos_frames >= nf` as
        // "already past the file" and clears playing on the first sample. Starting playback from
        // that parked EOF position never produces audio; restart from 0 instead.
        if playing && nf > 0 && frames >= nf as f64 - 0.5 {
            frames = 0.0;
        }
        *self.playhead_sec.lock() = frames / sr_f;
        let _ = self.tx.send(PlayerCommand::SeekAndSetPlaying {
            position_frames: frames,
            playing,
        });
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    pub fn seek_sec(&self, sec: f64) {
        let (nf, sr) = *self.clip_meta.lock();
        let sr_f = sr.max(1) as f64;
        let frames = (sec * sr_f).clamp(0.0, nf as f64);
        *self.playhead_sec.lock() = frames / sr_f;
        let _ = self.tx.send(PlayerCommand::SetPositionFrames(frames));
    }

    pub fn playhead_sec(&self) -> f64 {
        *self.playhead_sec.lock()
    }

    pub fn duration_sec(&self) -> f64 {
        *self.duration_sec.lock()
    }

    /// Waveform scrub drag is active (DJ-style variable-speed preview).
    pub fn set_scrub_active(&self, v: bool) {
        self.scrub_active.store(v, Ordering::Relaxed);
    }

    pub fn is_scrub_active(&self) -> bool {
        self.scrub_active.load(Ordering::Relaxed)
    }

    /// Multiplier: file-time advanced per wall-second (`1.0` = normal; negative = reverse).
    pub fn set_playback_rate_mul(&self, r: f64) {
        let b = r
            .clamp(-MAX_PLAYBACK_RATE_MUL, MAX_PLAYBACK_RATE_MUL)
            .to_bits();
        self.playback_rate_bits.store(b, Ordering::Relaxed);
    }
}

impl Drop for ClipPlaybackHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(PlayerCommand::Shutdown);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

fn run_player_thread(
    rx: Receiver<PlayerCommand>,
    inner: Arc<Mutex<PlaybackInner>>,
    playhead_sec: Arc<Mutex<f64>>,
    playing_flag: Arc<AtomicBool>,
    scrub_active: Arc<AtomicBool>,
    playback_rate_bits: Arc<AtomicU64>,
) {
    let host = cpal::default_host();
    let Some(device) = host.default_output_device() else {
        log::error!("clip preview: no default output device");
        drain_player_commands(rx);
        return;
    };
    let Ok(default_conf) = device.default_output_config() else {
        log::error!("clip preview: no default output config");
        drain_player_commands(rx);
        return;
    };
    let sample_format = default_conf.sample_format();
    let config: StreamConfig = default_conf.config().into();
    let out_ch = config.channels as usize;
    let out_rate = config.sample_rate;

    let err_fn = |e| log::error!("clip preview output stream error: {e}");

    let stream_result = match sample_format {
        SampleFormat::F32 => device.build_output_stream(
            &config,
            {
                let inner = inner.clone();
                let ph = playhead_sec.clone();
                let pf = playing_flag.clone();
                let sa = scrub_active.clone();
                let rm = playback_rate_bits.clone();
                move |data: &mut [f32], _| {
                    fill_f32_output(&inner, &ph, &pf, &sa, &rm, data, out_ch, out_rate)
                }
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_output_stream(
            &config,
            {
                let inner = inner.clone();
                let ph = playhead_sec.clone();
                let pf = playing_flag.clone();
                let sa = scrub_active.clone();
                let rm = playback_rate_bits.clone();
                move |data: &mut [i16], _| {
                    fill_i16_output(&inner, &ph, &pf, &sa, &rm, data, out_ch, out_rate)
                }
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_output_stream(
            &config,
            {
                let inner = inner.clone();
                let ph = playhead_sec.clone();
                let pf = playing_flag.clone();
                let sa = scrub_active.clone();
                let rm = playback_rate_bits.clone();
                move |data: &mut [u16], _| {
                    fill_u16_output(&inner, &ph, &pf, &sa, &rm, data, out_ch, out_rate)
                }
            },
            err_fn,
            None,
        ),
        _ => {
            log::error!("clip preview: unsupported output sample format {sample_format:?}");
            drain_player_commands(rx);
            return;
        }
    };

    let stream = match stream_result {
        Ok(s) => s,
        Err(e) => {
            log::error!("clip preview: build_output_stream failed: {e}");
            drain_player_commands(rx);
            return;
        }
    };

    if let Err(e) = stream.play() {
        log::error!("clip preview: stream.play failed: {e}");
        drop(stream);
        drain_player_commands(rx);
        return;
    }

    loop {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(PlayerCommand::Shutdown) => break,
            Ok(PlayerCommand::ReplacePcm {
                pcm,
                channels,
                sample_rate,
            }) => {
                let mut g = inner.lock();
                g.pcm = pcm;
                g.channels = channels.max(1);
                g.src_rate = sample_rate.max(1);
                g.pos_frames = 0.0;
                g.playing = playing_flag.load(Ordering::Relaxed);
                g.scrub_rate_smooth = 1.0;
                g.scrub_rate_snap = true;
                g.scrub_lp = [0.0; 8];
                *playhead_sec.lock() = 0.0;
            }
            Ok(PlayerCommand::SetPlaying(v)) => {
                playing_flag.store(v, Ordering::Relaxed);
                inner.lock().playing = v;
            }
            Ok(PlayerCommand::SetPositionFrames(p)) => {
                let mut g = inner.lock();
                g.pos_frames = p.max(0.0);
                g.scrub_rate_snap = true;
            }
            Ok(PlayerCommand::SeekAndSetPlaying {
                position_frames,
                playing,
            }) => {
                let mut g = inner.lock();
                let nf = if g.channels > 0 {
                    g.pcm.len() / g.channels
                } else {
                    0
                };
                let pf = position_frames.max(0.0).min(nf as f64);
                g.pos_frames = pf;
                g.scrub_rate_snap = true;
                g.playing = playing;
                playing_flag.store(playing, Ordering::Relaxed);
                let sr = g.src_rate.max(1) as f64;
                *playhead_sec.lock() = pf / sr;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    drop(stream);
}

/// Keep receiving until shutdown so the UI `Sender` is not disconnected while the app still holds
/// a [`ClipPlaybackHandle`].
fn drain_player_commands(rx: Receiver<PlayerCommand>) {
    loop {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(PlayerCommand::Shutdown) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
            Ok(_) => {}
        }
    }
}
