//! In-app clip preview: load float WAV (same format as clip capture) and play
//! through the default output device via cpal, with seekable playback position.

use std::sync::atomic::{AtomicBool, Ordering};
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
    Shutdown,
}

struct PlaybackInner {
    pcm: Vec<f32>,
    channels: usize,
    src_rate: u32,
    /// Fractional frame index (0 .. frame_count).
    pos_frames: f64,
    playing: bool,
}

impl Default for PlaybackInner {
    fn default() -> Self {
        Self {
            pcm: Vec::new(),
            channels: 2,
            src_rate: 48_000,
            pos_frames: 0.0,
            playing: false,
        }
    }
}

fn fill_f32_output(
    inner: &Mutex<PlaybackInner>,
    playhead_sec: &Mutex<f64>,
    playing_flag: &AtomicBool,
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
        if !g.playing || nf == 0 {
            for c in 0..out_ch {
                data[i * out_ch + c] = 0.0;
            }
            continue;
        }

        if g.pos_frames >= nf as f64 {
            g.playing = false;
            playing_flag.store(false, Ordering::Relaxed);
            g.pos_frames = nf as f64;
            for c in 0..out_ch {
                data[i * out_ch + c] = 0.0;
            }
            *playhead_sec.lock() = nf as f64 / src_rate;
            continue;
        }

        let p0 = g.pos_frames.floor() as usize;
        let frac = (g.pos_frames - p0 as f64) as f32;
        let p1 = (p0 + 1).min(nf.saturating_sub(1));
        let base0 = p0 * ch;
        let base1 = p1 * ch;

        for oc in 0..out_ch {
            let ic = oc % ch;
            let s0 = g.pcm[base0 + ic];
            let s1 = g.pcm[base1 + ic];
            data[i * out_ch + oc] = s0 + (s1 - s0) * frac;
        }

        g.pos_frames += src_rate / out_rate as f64;
        *playhead_sec.lock() = (g.pos_frames / src_rate).min(nf as f64 / src_rate);
    }
}

fn fill_i16_output(
    inner: &Mutex<PlaybackInner>,
    playhead_sec: &Mutex<f64>,
    playing_flag: &AtomicBool,
    data: &mut [i16],
    out_ch: usize,
    out_rate: u32,
) {
    let mut tmp = vec![0.0_f32; data.len()];
    fill_f32_output(inner, playhead_sec, playing_flag, &mut tmp, out_ch, out_rate);
    for (d, &s) in data.iter_mut().zip(tmp.iter()) {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i32;
        *d = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

fn fill_u16_output(
    inner: &Mutex<PlaybackInner>,
    playhead_sec: &Mutex<f64>,
    playing_flag: &AtomicBool,
    data: &mut [u16],
    out_ch: usize,
    out_rate: u32,
) {
    let mut tmp = vec![0.0_f32; data.len()];
    fill_f32_output(inner, playhead_sec, playing_flag, &mut tmp, out_ch, out_rate);
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
        let clip_meta = Arc::new(Mutex::new((0usize, 48_000u32)));

        let inner_th = inner.clone();
        let playhead_th = playhead_sec.clone();
        let playing_th = playing.clone();

        let join = std::thread::Builder::new()
            .name("clip-playback".into())
            .spawn(move || {
                run_player_thread(
                    rx,
                    inner_th,
                    playhead_th,
                    playing_th,
                );
            })
            .map_err(|e| e.to_string())?;

        Ok(Self {
            tx,
            join: Some(join),
            playhead_sec,
            duration_sec,
            playing,
            clip_meta,
        })
    }

    pub fn load_clip(&self, clip: &LoadedClip) -> Result<(), String> {
        let dur = clip.duration_sec();
        *self.duration_sec.lock() = dur;
        *self.playhead_sec.lock() = 0.0;
        self.playing.store(false, Ordering::Relaxed);
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
        self.playing.store(v, Ordering::Relaxed);
        let _ = self.tx.send(PlayerCommand::SetPlaying(v));
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
                move |data: &mut [f32], _| {
                    fill_f32_output(&inner, &ph, &pf, data, out_ch, out_rate)
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
                move |data: &mut [i16], _| {
                    fill_i16_output(&inner, &ph, &pf, data, out_ch, out_rate)
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
                move |data: &mut [u16], _| {
                    fill_u16_output(&inner, &ph, &pf, data, out_ch, out_rate)
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
                *playhead_sec.lock() = 0.0;
            }
            Ok(PlayerCommand::SetPlaying(v)) => {
                playing_flag.store(v, Ordering::Relaxed);
                inner.lock().playing = v;
            }
            Ok(PlayerCommand::SetPositionFrames(p)) => {
                inner.lock().pos_frames = p.max(0.0);
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
