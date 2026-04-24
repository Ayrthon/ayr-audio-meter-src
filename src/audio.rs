//! Audio capture engine.
//!
//! * Input devices are captured via `cpal` (cross-platform).
//! * Output devices are captured via the `wasapi` crate in loopback
//!   mode (Windows only).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread;

use anyhow::{Context, Result, anyhow};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::{Mutex, RwLock};

use crate::clip_capture::{ClipSettings, SharedClipCapture, SharedClipSettings, SharedClipStatus};
use crate::comp_view::SharedComp;
use crate::user_settings::SharedUserSettings;
use crate::dsp::{INF_FLOOR_DB, block_peak_rms};
use crate::history_graph::SharedHistory;
use crate::loudness::{LoudnessAnalyzer, SharedLoudness};
use crate::meter::{MeterMode, config_for, wall_clock_secs};
use crate::meter_widget::MeterState;
use crate::spectrum::{self, SharedSpectrum};
use crate::meter_monitor::MeterMonitorBus;
use crate::weighting::{FreqWeighting, MeterScratch};

pub const MAX_CHANNELS: usize = 8;

pub struct Levels {
    pub channels: usize,
    pub peak_db: [f32; MAX_CHANNELS],
    pub rms_db: [f32; MAX_CHANNELS],
    pub active: bool,
    pub sample_rate: u32,
    pub device_name: String,
    pub source_kind: SourceKind,
}

impl Default for Levels {
    fn default() -> Self {
        Self {
            channels: 0,
            peak_db: [INF_FLOOR_DB; MAX_CHANNELS],
            rms_db: [INF_FLOOR_DB; MAX_CHANNELS],
            active: false,
            sample_rate: 0,
            device_name: String::new(),
            source_kind: SourceKind::Input,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Input,
    Loopback,
    /// Remote sender discovered over mDNS and connected to over TCP. The
    /// endpoint metadata lives on [`DeviceInfo::network`] — keeping
    /// `SourceKind` as a fieldless enum preserves its `Copy` semantics
    /// that the rest of the meter relies on.
    Network,
}

#[derive(Debug, Clone)]
pub struct NetworkEndpoint {
    /// Stable id the meter uses to match this sender across refreshes —
    /// derived from the mDNS instance name + fingerprint so two senders
    /// with the same friendly name don't collide.
    pub service_id: String,
    pub host: std::net::IpAddr,
    pub port: u16,
    /// Protocol version the sender advertised in its TXT record. Stored
    /// so the UI can warn about version mismatches pre-connect.
    pub protocol_version: u16,
    /// M3: IPv4 multicast group for L24 (same `port` as the sender). When
    /// set, the receiver binds to `0.0.0.0:port` and `join_multicast_v4`
    /// so it can follow the session without unicast-audio (still uses
    /// unicast to the sender for `SUBSCRIBE` / `STREAM_INFO` / `KEEPALIVE`).
    pub mcast: Option<std::net::Ipv4Addr>,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub kind: SourceKind,
    pub max_channels: usize,
    /// Only populated for `SourceKind::Network` entries. `None` for
    /// cpal inputs / WASAPI loopbacks.
    pub network: Option<NetworkEndpoint>,
}

pub type SharedLevels = Arc<RwLock<Levels>>;

pub fn make_shared_levels() -> SharedLevels {
    Arc::new(RwLock::new(Levels::default()))
}

/// Per-channel dBFS values for the history graph: **ballistic** levels matching the bar fill.
pub type SharedHistoryTrace = Arc<Mutex<[f32; MAX_CHANNELS]>>;

pub fn make_shared_history_trace() -> SharedHistoryTrace {
    Arc::new(Mutex::new([INF_FLOOR_DB; MAX_CHANNELS]))
}

/// Ballistic meter state driven on the audio thread so clip threshold compares to the same
/// **peak-hold** (`hold_db`) as the white line on each meter, not the raw block peak.
pub type SharedClipTriggerStates = Arc<Mutex<[MeterState; MAX_CHANNELS]>>;

pub fn make_shared_clip_trigger_states() -> SharedClipTriggerStates {
    Arc::new(Mutex::new(std::array::from_fn(|_| MeterState::default())))
}

#[derive(Clone, Copy)]
pub struct BlockLevelsSnapshot {
    pub channels: usize,
    pub peak_db: [f32; MAX_CHANNELS],
    pub rms_db: [f32; MAX_CHANNELS],
}

pub fn list_devices() -> Result<Vec<DeviceInfo>> {
    let mut devices = Vec::new();
    let host = cpal::default_host();
    for dev in host.input_devices().context("cpal input_devices")? {
        #[allow(deprecated)]
        let name = dev.name().unwrap_or_else(|_| "<unnamed input>".into());
        let max = dev
            .default_input_config()
            .map(|c| c.channels() as usize)
            .unwrap_or(2);
        devices.push(DeviceInfo {
            name,
            kind: SourceKind::Input,
            max_channels: max,
            network: None,
        });
    }
    devices.sort_by(|a, b| a.name.cmp(&b.name));

    #[cfg(windows)]
    {
        match windows_loopback::list_output_devices() {
            Ok(mut outs) => {
                outs.sort_by(|a, b| a.name.cmp(&b.name));
                devices.extend(outs);
            }
            Err(err) => log::warn!("loopback device enumeration failed: {err:?}"),
        }
    }

    Ok(devices)
}

pub fn default_input_name() -> Option<String> {
    let host = cpal::default_host();
    #[allow(deprecated)]
    host.default_input_device().and_then(|d| d.name().ok())
}

pub fn default_output_name() -> Option<String> {
    #[cfg(windows)]
    {
        windows_loopback::default_output_name().ok().flatten()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Friendly names of all render endpoints eligible as headphone-monitor
/// targets. On Windows this reuses `windows_loopback::list_output_devices()`
/// so the names match exactly what the user sees in Windows Sound Settings
/// (and therefore match the friendly name stored in
/// [`crate::user_settings::UserSettingsData::monitor_device_name`]). Other
/// platforms fall back to `cpal::Host::output_devices()`.
pub fn list_monitor_output_device_names() -> Vec<String> {
    #[cfg(windows)]
    {
        match windows_loopback::list_output_devices() {
            Ok(devs) => {
                let mut names: Vec<String> = devs.into_iter().map(|d| d.name).collect();
                names.sort();
                names.dedup();
                names
            }
            Err(err) => {
                log::warn!("list_monitor_output_device_names: {err:?}");
                Vec::new()
            }
        }
    }
    #[cfg(not(windows))]
    {
        let host = cpal::default_host();
        let Ok(devs) = host.output_devices() else {
            return Vec::new();
        };
        let mut names: Vec<String> = devs
            .filter_map(|d| d.name().ok())
            .collect();
        names.sort();
        names.dedup();
        names
    }
}

pub struct AudioEngine {
    stop_flag: Arc<AtomicBool>,
    levels: SharedLevels,
    meter_monitor: Arc<MeterMonitorBus>,
    thread: Option<thread::JoinHandle<()>>,
}

impl AudioEngine {
    pub fn start(
        device: &DeviceInfo,
        meter_true_peak: Arc<AtomicBool>,
        levels: SharedLevels,
        loudness: SharedLoudness,
        spectrum: SharedSpectrum,
        comp: SharedComp,
        history: SharedHistory,
        history_trace: SharedHistoryTrace,
        weighting: Arc<AtomicU8>,
        meter_scratch: Arc<Mutex<MeterScratch>>,
        meter_mode: Arc<AtomicU8>,
        clip_trigger_states: SharedClipTriggerStates,
        clip_settings: SharedClipSettings,
        clip_capture: SharedClipCapture,
        clip_status: SharedClipStatus,
        clip_dirs: SharedUserSettings,
        meter_monitor: Arc<MeterMonitorBus>,
    ) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = stop.clone();
        let meter_monitor_for_thread = meter_monitor.clone();
        let levels_for_thread = levels.clone();
        let loudness_for_thread = loudness.clone();
        let spectrum_for_thread = spectrum.clone();
        let comp_for_thread = comp.clone();
        let history_for_thread = history.clone();
        let history_trace_for_thread = history_trace.clone();
        let weighting_for_thread = weighting.clone();
        let scratch_for_thread = meter_scratch.clone();
        let meter_mode_th = meter_mode.clone();
        let clip_states_th = clip_trigger_states.clone();
        let clip_settings_th = clip_settings.clone();
        let clip_capture_th = clip_capture.clone();
        let clip_status_th = clip_status.clone();
        let clip_dirs_th = clip_dirs.clone();
        let device_name = device.name.clone();
        let kind = device.kind;
        let network_endpoint = device.network.clone();

        let handle = thread::Builder::new()
            .name(format!("audio-{:?}", kind))
            .spawn(move || {
                let result = match kind {
                    SourceKind::Input => run_cpal_input(
                        &device_name,
                        meter_true_peak.clone(),
                        &levels_for_thread,
                        &loudness_for_thread,
                        &spectrum_for_thread,
                        &comp_for_thread,
                        &history_for_thread,
                        &history_trace_for_thread,
                        &weighting_for_thread,
                        &scratch_for_thread,
                        &meter_mode_th,
                        &clip_states_th,
                        &clip_settings_th,
                        &clip_capture_th,
                        &clip_status_th,
                        &clip_dirs_th,
                        &stop_for_thread,
                        &meter_monitor_for_thread,
                    ),
                    #[cfg(windows)]
                    SourceKind::Loopback => windows_loopback::run(
                        &device_name,
                        meter_true_peak.clone(),
                        &levels_for_thread,
                        &loudness_for_thread,
                        &spectrum_for_thread,
                        &comp_for_thread,
                        &history_for_thread,
                        &history_trace_for_thread,
                        &weighting_for_thread,
                        &scratch_for_thread,
                        &meter_mode_th,
                        &clip_states_th,
                        &clip_settings_th,
                        &clip_capture_th,
                        &clip_status_th,
                        &clip_dirs_th,
                        &stop_for_thread,
                        &meter_monitor_for_thread,
                    ),
                    #[cfg(not(windows))]
                    SourceKind::Loopback => Err(anyhow!(
                        "loopback capture is not supported on this platform"
                    )),
                    SourceKind::Network => match network_endpoint {
                        Some(ep) => run_network_source(
                            &device_name,
                            ep,
                            meter_true_peak.clone(),
                            &levels_for_thread,
                            &loudness_for_thread,
                            &spectrum_for_thread,
                            &comp_for_thread,
                            &history_for_thread,
                            &history_trace_for_thread,
                            &weighting_for_thread,
                            &scratch_for_thread,
                            &meter_mode_th,
                            &clip_states_th,
                            &clip_settings_th,
                            &clip_capture_th,
                            &clip_status_th,
                            &clip_dirs_th,
                            &stop_for_thread,
                            &meter_monitor_for_thread,
                        ),
                        None => Err(anyhow!(
                            "network source `{device_name}` is missing endpoint metadata"
                        )),
                    },
                };
                if let Err(err) = result {
                    log::error!("audio thread exited with error: {err:?}");
                    let mut lv = levels_for_thread.write();
                    lv.active = false;
                    lv.channels = 0;
                    spectrum::configure(&spectrum_for_thread, 48_000, false);
                }
            })
            .context("spawn audio thread")?;

        Ok(Self {
            stop_flag: stop,
            levels,
            meter_monitor,
            thread: Some(handle),
        })
    }

    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
        self.meter_monitor.end_capture();
        let mut lv = self.levels.write();
        lv.active = false;
        for v in &mut lv.peak_db[..] {
            *v = INF_FLOOR_DB;
        }
        for v in &mut lv.rms_db[..] {
            *v = INF_FLOOR_DB;
        }
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run_cpal_input(
    device_name: &str,
    meter_true_peak: Arc<AtomicBool>,
    levels: &SharedLevels,
    loudness: &SharedLoudness,
    spectrum: &SharedSpectrum,
    comp: &SharedComp,
    history: &SharedHistory,
    history_trace: &SharedHistoryTrace,
    weighting: &Arc<AtomicU8>,
    meter_scratch: &Arc<Mutex<MeterScratch>>,
    meter_mode: &Arc<AtomicU8>,
    clip_trigger_states: &SharedClipTriggerStates,
    clip_settings: &SharedClipSettings,
    clip_capture: &SharedClipCapture,
    clip_status: &SharedClipStatus,
    clip_dirs: &SharedUserSettings,
    stop: &Arc<AtomicBool>,
    meter_monitor: &Arc<MeterMonitorBus>,
) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .input_devices()
        .context("cpal input_devices")?
        .find(|d| {
            #[allow(deprecated)]
            d.name().map(|n| n == device_name).unwrap_or(false)
        })
        .ok_or_else(|| anyhow!("input device `{device_name}` not found"))?;

    let config = device
        .default_input_config()
        .with_context(|| format!("default_input_config for `{device_name}`"))?;
    let sample_format = config.sample_format();
    let channels = config.channels() as usize;
    let sample_rate: u32 = config.sample_rate().into();

    {
        let mut lv = levels.write();
        lv.active = true;
        lv.channels = channels.min(MAX_CHANNELS);
        lv.sample_rate = sample_rate;
        lv.device_name = device_name.to_owned();
        lv.source_kind = SourceKind::Input;
    }
    // Fresh loudness analyzer per stream.
    {
        let mut l = loudness.write();
        *l = Default::default();
    }
    let mut analyzer = match LoudnessAnalyzer::new(channels as u32, sample_rate) {
        Ok(a) => Some(a),
        Err(err) => {
            log::warn!("loudness analyzer init failed: {err:?}");
            None
        }
    };
    spectrum::configure(spectrum, sample_rate, true);
    {
        let mut c = comp.write();
        c.clear();
        c.configure(sample_rate);
    }

    let stream_config: cpal::StreamConfig = config.clone().into();
    let levels_clone = levels.clone();
    let loudness_clone = loudness.clone();
    let spectrum_clone = spectrum.clone();
    let comp_clone = comp.clone();
    let weight_clone = weighting.clone();
    let scratch_clone = meter_scratch.clone();
    let clip_settings_cb = clip_settings.clone();
    let clip_capture_cb = clip_capture.clone();
    let clip_status_cb = clip_status.clone();
    let clip_dirs_cb = clip_dirs.clone();
    let meter_mode_cb = meter_mode.clone();
    let clip_states_cb = clip_trigger_states.clone();
    let meter_tp_f32 = meter_true_peak.clone();
    let meter_tp_i16 = meter_true_peak.clone();
    let meter_tp_u16 = meter_true_peak.clone();
    let mm_f32 = meter_monitor.clone();
    let mm_i16 = meter_monitor.clone();
    let mm_u16 = meter_monitor.clone();
    let err_fn = |e| log::error!("cpal stream error: {e}");

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                let now = wall_clock_secs();
                let tp = meter_tp_f32.load(Ordering::Relaxed);
                let snap = process_weighted_levels(
                    &scratch_clone,
                    &weight_clone,
                    data,
                    channels,
                    sample_rate,
                    tp,
                    &levels_clone,
                );
                let trig = clip_trigger_hold_max_db(
                    &mut clip_states_cb.lock(),
                    meter_mode_cb.load(Ordering::Relaxed),
                    now,
                    &snap,
                );
                process_clip(
                    data,
                    channels,
                    sample_rate,
                    clip_settings_cb.as_ref(),
                    &clip_capture_cb,
                    &clip_status_cb,
                    &clip_dirs_cb,
                    trig,
                );
                if let Some(a) = analyzer.as_mut() {
                    a.process(data, &loudness_clone);
                }
                spectrum::push_samples(&spectrum_clone, data, channels);
                comp_clone
                    .write()
                    .push_samples_mono_sum(data, channels, sample_rate);
                mm_f32.push_interleaved_f32(data, channels);
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let now = wall_clock_secs();
                let tp = meter_tp_i16.load(Ordering::Relaxed);
                let mut buf = Vec::with_capacity(data.len());
                for &s in data {
                    buf.push(s as f32 / i16::MAX as f32);
                }
                let snap = process_weighted_levels(
                    &scratch_clone,
                    &weight_clone,
                    &buf,
                    channels,
                    sample_rate,
                    tp,
                    &levels_clone,
                );
                let trig = clip_trigger_hold_max_db(
                    &mut clip_states_cb.lock(),
                    meter_mode_cb.load(Ordering::Relaxed),
                    now,
                    &snap,
                );
                process_clip(
                    &buf,
                    channels,
                    sample_rate,
                    clip_settings_cb.as_ref(),
                    &clip_capture_cb,
                    &clip_status_cb,
                    &clip_dirs_cb,
                    trig,
                );
                if let Some(a) = analyzer.as_mut() {
                    a.process(&buf, &loudness_clone);
                }
                spectrum::push_samples(&spectrum_clone, &buf, channels);
                comp_clone
                    .write()
                    .push_samples_mono_sum(&buf, channels, sample_rate);
                mm_i16.push_interleaved_f32(&buf, channels);
            },
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                let now = wall_clock_secs();
                let tp = meter_tp_u16.load(Ordering::Relaxed);
                let mut buf = Vec::with_capacity(data.len());
                for &s in data {
                    buf.push((s as f32 - 32768.0) / 32768.0);
                }
                let snap = process_weighted_levels(
                    &scratch_clone,
                    &weight_clone,
                    &buf,
                    channels,
                    sample_rate,
                    tp,
                    &levels_clone,
                );
                let trig = clip_trigger_hold_max_db(
                    &mut clip_states_cb.lock(),
                    meter_mode_cb.load(Ordering::Relaxed),
                    now,
                    &snap,
                );
                process_clip(
                    &buf,
                    channels,
                    sample_rate,
                    clip_settings_cb.as_ref(),
                    &clip_capture_cb,
                    &clip_status_cb,
                    &clip_dirs_cb,
                    trig,
                );
                if let Some(a) = analyzer.as_mut() {
                    a.process(&buf, &loudness_clone);
                }
                spectrum::push_samples(&spectrum_clone, &buf, channels);
                comp_clone
                    .write()
                    .push_samples_mono_sum(&buf, channels, sample_rate);
                mm_u16.push_interleaved_f32(&buf, channels);
            },
            err_fn,
            None,
        )?,
        other => return Err(anyhow!("unsupported sample format {other:?}")),
    };

    stream.play()?;
    meter_monitor.begin_capture(
        sample_rate,
        channels,
        Some(device_name.to_string()),
        stop.clone(),
    );

    history.write().reset_tick_clock();
    let start_instant = std::time::Instant::now();
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(std::time::Duration::from_millis(25));
        tick_shared_history(levels, history, history_trace, start_instant);
    }
    drop(stream);
    spectrum::configure(spectrum, sample_rate, false);
    comp.write().clear();
    Ok(())
}

/// Snapshot per-channel **meter bar** levels (written from the UI thread — same
/// ballistic peak or RMS as the coloured fill) into the history ring and advance it.
/// Runs from the audio thread so it keeps ticking when the window is minimized
/// (values may plateau until the next UI update).
fn tick_shared_history(
    levels: &SharedLevels,
    history: &SharedHistory,
    history_trace: &SharedHistoryTrace,
    start: std::time::Instant,
) {
    let peaks = {
        let lv = levels.read();
        let ch = lv.channels.min(MAX_CHANNELS);
        let trace = history_trace.lock();
        let mut p = [INF_FLOOR_DB; MAX_CHANNELS];
        for i in 0..ch {
            p[i] = trace[i];
        }
        (p, ch)
    };
    let (p, ch) = peaks;
    let now = start.elapsed().as_secs_f64();
    let mut hist = history.write();
    hist.push_latest(&p[..ch]);
    hist.tick(now);
}

fn process_clip(
    data: &[f32],
    channels: usize,
    sample_rate: u32,
    settings: &ClipSettings,
    capture: &SharedClipCapture,
    status: &SharedClipStatus,
    clip_dirs: &SharedUserSettings,
    trigger_peak_db: f32,
) {
    capture.lock().process_block(
        data,
        channels,
        sample_rate,
        settings,
        status,
        trigger_peak_db,
        clip_dirs,
    );
}

/// Updates shared levels and returns per-channel peaks / RMS using the same weighting +
/// true-peak path as the meters.
fn process_weighted_levels(
    meter_scratch: &Arc<Mutex<MeterScratch>>,
    weighting: &Arc<AtomicU8>,
    data: &[f32],
    channels: usize,
    sample_rate: u32,
    true_peak: bool,
    levels: &SharedLevels,
) -> BlockLevelsSnapshot {
    let w = FreqWeighting::from_u8(weighting.load(Ordering::Relaxed));
    let mut g = meter_scratch.lock();
    g.prepare_meter_block(data, channels, sample_rate, w);
    let ch = channels.min(MAX_CHANNELS);
    let mut peaks = [INF_FLOOR_DB; MAX_CHANNELS];
    let mut rms = [INF_FLOOR_DB; MAX_CHANNELS];
    block_peak_rms(g.buf(), ch, true_peak, &mut peaks[..ch], &mut rms[..ch]);
    let mut lv = levels.write();
    lv.channels = ch;
    lv.peak_db[..ch].copy_from_slice(&peaks[..ch]);
    lv.rms_db[..ch].copy_from_slice(&rms[..ch]);
    BlockLevelsSnapshot {
        channels: ch,
        peak_db: peaks,
        rms_db: rms,
    }
}

/// Advances per-channel [`MeterState`] on the audio thread and returns the worst-channel
/// **peak-hold** level — the same quantity as the white line on each meter.
fn clip_trigger_hold_max_db(
    states: &mut [MeterState; MAX_CHANNELS],
    mode_discriminant: u8,
    now: f64,
    snap: &BlockLevelsSnapshot,
) -> f32 {
    let mode = MeterMode::from_discriminant(mode_discriminant);
    let cfg = config_for(mode);
    let ch = snap.channels.min(MAX_CHANNELS);
    let mut max_hold = INF_FLOOR_DB;
    for i in 0..ch {
        states[i].set_target(snap.peak_db[i], snap.rms_db[i]);
        states[i].tick(&cfg, now);
        max_hold = max_hold.max(states[i].hold_db);
    }
    for i in ch..MAX_CHANNELS {
        states[i].set_target(INF_FLOOR_DB, INF_FLOOR_DB);
        states[i].tick(&cfg, now);
    }
    max_hold
}

// -------------------------------------------------------------------
// Windows WASAPI loopback path
// -------------------------------------------------------------------

#[cfg(windows)]
mod windows_loopback {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::Ordering;
    use wasapi::{
        Direction, DeviceEnumerator, SampleType, StreamMode, WaveFormat, initialize_mta,
    };

    pub fn list_output_devices() -> Result<Vec<DeviceInfo>> {
        initialize_mta().ok();
        let enumerator = DeviceEnumerator::new().context("DeviceEnumerator::new")?;
        let collection = enumerator
            .get_device_collection(&Direction::Render)
            .context("get_device_collection(Render)")?;
        let count = collection.get_nbr_devices().context("get_nbr_devices")?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let dev = match collection.get_device_at_index(i) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let name = dev
                .get_friendlyname()
                .unwrap_or_else(|_| format!("Render {i}"));
            let mut channels = 2usize;
            if let Ok(client) = dev.get_iaudioclient() {
                if let Ok(mix) = client.get_mixformat() {
                    channels = mix.get_nchannels() as usize;
                }
            }
            out.push(DeviceInfo {
                name,
                kind: SourceKind::Loopback,
                max_channels: channels,
                network: None,
            });
        }
        Ok(out)
    }

    pub fn default_output_name() -> Result<Option<String>> {
        initialize_mta().ok();
        let enumerator = DeviceEnumerator::new().context("DeviceEnumerator::new")?;
        let dev = enumerator
            .get_default_device(&Direction::Render)
            .context("default render device")?;
        Ok(Some(dev.get_friendlyname().unwrap_or_default()))
    }

    pub fn run(
        device_name: &str,
        meter_true_peak: Arc<AtomicBool>,
        levels: &SharedLevels,
        loudness: &SharedLoudness,
        spectrum: &SharedSpectrum,
        comp: &SharedComp,
        history: &SharedHistory,
        history_trace: &SharedHistoryTrace,
        weighting: &Arc<AtomicU8>,
        meter_scratch: &Arc<Mutex<MeterScratch>>,
        meter_mode: &Arc<AtomicU8>,
        clip_trigger_states: &SharedClipTriggerStates,
        clip_settings: &SharedClipSettings,
        clip_capture: &SharedClipCapture,
        clip_status: &SharedClipStatus,
        clip_dirs: &SharedUserSettings,
        stop: &Arc<AtomicBool>,
        meter_monitor: &Arc<MeterMonitorBus>,
    ) -> Result<()> {
        initialize_mta().ok();
        let enumerator = DeviceEnumerator::new().context("DeviceEnumerator::new")?;
        let collection = enumerator
            .get_device_collection(&Direction::Render)
            .context("get_device_collection")?;
        let count = collection.get_nbr_devices().context("get_nbr_devices")?;
        let mut chosen = None;
        for i in 0..count {
            let dev = match collection.get_device_at_index(i) {
                Ok(d) => d,
                Err(_) => continue,
            };
            if dev
                .get_friendlyname()
                .map(|n| n == device_name)
                .unwrap_or(false)
            {
                chosen = Some(dev);
                break;
            }
        }
        let device = chosen.ok_or_else(|| anyhow!("render device `{device_name}` not found"))?;

        let mut audio_client = device.get_iaudioclient().context("get_iaudioclient")?;
        let mix_format = audio_client.get_mixformat().context("get_mixformat")?;
        let sample_rate = mix_format.get_samplespersec();
        let channels = mix_format.get_nchannels() as usize;
        let bits = mix_format.get_bitspersample() as usize;
        let valid_bits = mix_format.get_validbitspersample() as usize;
        let block_align = mix_format.get_blockalign() as usize;

        log::info!(
            "loopback mix format: {device_name} @ {sample_rate} Hz, {channels} ch, bits={bits} valid={valid_bits}, block_align={block_align}"
        );

        // Use the mix format directly (it is guaranteed supported in shared mode).
        let desired_format = WaveFormat::new(
            bits,
            valid_bits,
            &if bits == 32 && valid_bits == 32 {
                SampleType::Float
            } else {
                SampleType::Int
            },
            sample_rate as usize,
            channels,
            None,
        );

        let (_def_period, min_period) = audio_client
            .get_device_period()
            .context("get_device_period")?;
        let buffer_duration_hns = min_period.max(100_000);

        let mode = StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns,
        };
        audio_client
            .initialize_client(&desired_format, &Direction::Capture, &mode)
            .context("initialize_client loopback")?;

        let h_event = audio_client
            .set_get_eventhandle()
            .context("set_get_eventhandle")?;
        let capture_client = audio_client
            .get_audiocaptureclient()
            .context("get_audiocaptureclient")?;

        audio_client.start_stream().context("start_stream")?;
        log::info!("loopback started");

        meter_monitor.begin_capture(
            sample_rate,
            channels,
            Some(device_name.to_string()),
            stop.clone(),
        );

        history.write().reset_tick_clock();
        let start_instant = std::time::Instant::now();

        {
            let mut lv = levels.write();
            lv.active = true;
            lv.channels = channels.min(MAX_CHANNELS);
            lv.sample_rate = sample_rate;
            lv.device_name = device_name.to_owned();
            lv.source_kind = SourceKind::Loopback;
        }
        {
            let mut l = loudness.write();
            *l = Default::default();
        }
        let mut analyzer = match LoudnessAnalyzer::new(channels as u32, sample_rate) {
            Ok(a) => Some(a),
            Err(err) => {
                log::warn!("loudness analyzer init failed: {err:?}");
                None
            }
        };
        spectrum::configure(spectrum, sample_rate, true);
        {
            let mut c = comp.write();
            c.clear();
            c.configure(sample_rate);
        }

        let bytes_per_sample = bits / 8;
        let is_float = bits == 32 && valid_bits == 32;

        let mut sample_queue: VecDeque<u8> =
            VecDeque::with_capacity(block_align * sample_rate as usize / 10);
        let mut f32_scratch: Vec<f32> = Vec::with_capacity(channels * sample_rate as usize / 50);
        let mut contig: Vec<u8> = Vec::with_capacity(block_align * sample_rate as usize / 50);

        while !stop.load(Ordering::Relaxed) {
            // Drain any already-queued data into level updates.
            while sample_queue.len() >= block_align {
                let frames = sample_queue.len() / block_align;
                contig.clear();
                contig.extend(sample_queue.drain(..frames * block_align));
                f32_scratch.clear();
                f32_scratch.reserve(frames * channels);
                decode(&contig, &mut f32_scratch, bytes_per_sample, is_float);
                let now = crate::meter::wall_clock_secs();
                let tp = meter_true_peak.load(Ordering::Relaxed);
                let snap = super::process_weighted_levels(
                    meter_scratch,
                    weighting,
                    &f32_scratch,
                    channels,
                    sample_rate,
                    tp,
                    levels,
                );
                let trig = super::clip_trigger_hold_max_db(
                    &mut clip_trigger_states.lock(),
                    meter_mode.load(Ordering::Relaxed),
                    now,
                    &snap,
                );
                super::process_clip(
                    &f32_scratch,
                    channels,
                    sample_rate,
                    clip_settings.as_ref(),
                    clip_capture,
                    clip_status,
                    clip_dirs,
                    trig,
                );
                if let Some(a) = analyzer.as_mut() {
                    a.process(&f32_scratch, loudness);
                }
                spectrum::push_samples(spectrum, &f32_scratch, channels);
                comp.write()
                    .push_samples_mono_sum(&f32_scratch, channels, sample_rate);
                meter_monitor.push_interleaved_f32(&f32_scratch, channels);
            }

            match capture_client.get_next_packet_size() {
                Ok(Some(n)) if n > 0 => {
                    let _ = capture_client
                        .read_from_device_to_deque(&mut sample_queue);
                }
                Ok(_) => {
                    if h_event.wait_for_event(200).is_err() {
                        continue;
                    }
                }
                Err(err) => {
                    log::warn!("get_next_packet_size error: {err:?}");
                    let _ = h_event.wait_for_event(200);
                }
            }
            tick_shared_history(levels, history, history_trace, start_instant);
        }

        let _ = audio_client.stop_stream();
        spectrum::configure(spectrum, sample_rate, false);
        comp.write().clear();
        Ok(())
    }

    fn decode(bytes: &[u8], out: &mut Vec<f32>, bytes_per_sample: usize, is_float: bool) {
        debug_assert!(bytes.len() % bytes_per_sample == 0);
        match (bytes_per_sample, is_float) {
            (4, true) => {
                for chunk in bytes.chunks_exact(4) {
                    let s = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    out.push(s);
                }
            }
            (2, false) => {
                for chunk in bytes.chunks_exact(2) {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(s as f32 / i16::MAX as f32);
                }
            }
            (3, false) => {
                for chunk in bytes.chunks_exact(3) {
                    // 24-bit signed LE -> f32
                    let mut v = (chunk[0] as i32) | ((chunk[1] as i32) << 8) | ((chunk[2] as i32) << 16);
                    if v & 0x0080_0000 != 0 {
                        v |= -0x0100_0000i32;
                    }
                    out.push(v as f32 / 8_388_607.0);
                }
            }
            (4, false) => {
                for chunk in bytes.chunks_exact(4) {
                    let s = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    out.push(s as f32 / 2_147_483_647.0);
                }
            }
            _ => {
                // Unknown format — emit silence to keep the UI responsive.
                out.extend(std::iter::repeat(0.0).take(bytes.len() / bytes_per_sample.max(1)));
            }
        }
    }
}

// -------------------------------------------------------------------
// Network source — AYR Audio Link (UDP, protocol v2)
// -------------------------------------------------------------------

/// Runs a network capture session to an AYR Audio Link sender over UDP.
///
/// Protocol shape (see `ayr_audio_link_core::wire` for full details):
///
/// 1. Bind an ephemeral UDP socket and connect it to the sender's
///    advertised endpoint.
/// 2. Send `SUBSCRIBE { client_id, preset }` and wait for a
///    `STREAM_INFO { sample_rate, channels, ssrc, … }` reply before we
///    initialise DSP — the reply is the only source of truth that the
///    sender is currently capturing (the mDNS TXT record may be stale).
/// 3. Receive loop: classify each datagram as either RTP (audio) or
///    control (`STREAM_INFO` updates). RTP packets feed a per-channel
///    jitter buffer; out-of-order packets whose sequence is behind the
///    current play-out are dropped, forward gaps are zero-filled so the
///    DSP clock stays aligned with the sender's sample counter.
/// 4. A small background thread fires `KEEPALIVE` every 500 ms so the
///    sender's reaper doesn't evict us.
/// 5. On stop we send `UNSUBSCRIBE` best-effort.
///
/// Reconnect loop: if the sender disappears (format change, restart,
/// sender process bounce) the receive path drains, we resend SUBSCRIBE
/// and continue. So a Link "Stop broadcasting"/"Start broadcasting"
/// cycle resumes automatically without the user touching the meter.
#[allow(clippy::too_many_arguments)]
fn run_network_source(
    device_name: &str,
    endpoint: NetworkEndpoint,
    meter_true_peak: Arc<AtomicBool>,
    levels: &SharedLevels,
    loudness: &SharedLoudness,
    spectrum: &SharedSpectrum,
    comp: &SharedComp,
    history: &SharedHistory,
    history_trace: &SharedHistoryTrace,
    weighting: &Arc<AtomicU8>,
    meter_scratch: &Arc<Mutex<MeterScratch>>,
    meter_mode: &Arc<AtomicU8>,
    clip_trigger_states: &SharedClipTriggerStates,
    clip_settings: &SharedClipSettings,
    clip_capture: &SharedClipCapture,
    clip_status: &SharedClipStatus,
    clip_dirs: &SharedUserSettings,
    stop: &Arc<AtomicBool>,
    meter_monitor: &Arc<MeterMonitorBus>,
) -> Result<()> {
    use ayr_audio_link_core::wire::{
        self, classify_datagram, ControlMsg, DatagramKind, LatencyPreset, L24_BYTES_PER_SAMPLE,
        PT_F32LE, PT_L24,
    };
    use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
    use std::time::{Duration, Instant};

    let addr = SocketAddr::new(endpoint.host, endpoint.port);
    // Pick a stable-per-process client id so repeated subscribes from the
    // same meter instance look like re-subscriptions to the sender rather
    // than fresh clients.
    static CLIENT_ID_SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let client_id: u64 = *CLIENT_ID_SEED.get_or_init(|| {
        let mut h = std::process::id() as u64;
        if let Ok(d) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            h ^= (d.as_nanos() as u64).rotate_left(17);
        }
        h ^ 0xA1EC_CAFE_DEAD_BEEF
    });
    // Multicast + IGMP: IPv4 mDNS + `mc` TXT; IPv6-only senders keep unicast.
    let wants_mcast_igmp = endpoint
        .mcast
        .is_some()
        && matches!(endpoint.host, IpAddr::V4(_));

    log::info!(
        "network source: AYR Audio Link at {addr} (`{device_name}`) client_id={client_id:#018x} mcast_igmp={wants_mcast_igmp}"
    );

    while !stop.load(Ordering::Relaxed) {
        {
            let mut lv = levels.write();
            lv.active = false;
        }
        spectrum::configure(spectrum, 48_000, false);
        comp.write().clear();

        // Read the user's current latency preset every reconnect, so a
        // dropdown change applies on the next subscribe without tearing
        // down the audio thread.
        let preset = {
            let byte = clip_dirs.lock().network_latency_preset;
            LatencyPreset::from_u8(byte).unwrap_or(LatencyPreset::Home)
        };
        log::debug!("network source: preset={preset:?} ({}ms target)", preset.target_ms());

        // Fresh socket per session. M3: for AES67 L24 the sender also fans
        // RTP to a `239.69.x.x:<same-port>` group — we bind that local port
        // and `join_multicast_v4` so we receive those packets (and still use
        // the same socket for unicast `SUBSCRIBE` / `STREAM_INFO` /
        // `KEEPALIVE` to the sender, without `connect`, because a connected
        // socket would not deliver multicast copies).
        let (sock, mcast_mode): (UdpSocket, bool) =
            if let (true, Some(g)) = (wants_mcast_igmp, endpoint.mcast) {
                    let bind_addr =
                        SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), endpoint.port);
                    match UdpSocket::bind(bind_addr) {
                        Ok(s) => {
                            if let Err(e) =
                                s.join_multicast_v4(&g, &Ipv4Addr::UNSPECIFIED)
                            {
                                log::warn!("network source: join_multicast {g}: {e}");
                            }
                            log::info!(
                                "network source: bound {} + joined mcast {g} for L24",
                                s.local_addr().unwrap_or(bind_addr)
                            );
                            (s, true)
                        }
                        Err(e) => {
                            log::warn!(
                                "network source: could not bind port {} (mcast): {e} — try closing other AYR/meter or pick another input",
                                endpoint.port
                            );
                            let s = match UdpSocket::bind("0.0.0.0:0") {
                                Ok(s) => s,
                                Err(e2) => {
                                    log::warn!("network source: bind fallback: {e2}");
                                    thread::sleep(Duration::from_millis(400));
                                    continue;
                                }
                            };
                            if let Err(e) = s.connect(addr) {
                                log::warn!("network source: connect {addr}: {e}");
                                thread::sleep(Duration::from_millis(400));
                                continue;
                            }
                            (s, false)
                        }
                    }
            } else {
                let s = match UdpSocket::bind("0.0.0.0:0") {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!("network source: bind ephemeral udp: {e}");
                        thread::sleep(Duration::from_millis(400));
                        continue;
                    }
                };
                if let Err(e) = s.connect(addr) {
                    log::warn!("network source: connect {addr}: {e}");
                    thread::sleep(Duration::from_millis(400));
                    continue;
                }
                (s, false)
            };
        sock.set_read_timeout(Some(Duration::from_millis(200))).ok();

        // Subscribe + wait for STREAM_INFO.
        let mut ctrl_buf: Vec<u8> = Vec::with_capacity(128);
        let subscribe = ControlMsg::Subscribe { client_id, preset };
        if wire::encode_control(&mut ctrl_buf, &subscribe).is_ok() {
            let _ = if mcast_mode {
                sock.send_to(&ctrl_buf, addr)
            } else {
                sock.send(&ctrl_buf)
            };
        }
        let mut recv_buf = vec![0u8; 2048];
        let stream_info_deadline = Instant::now() + Duration::from_secs(2);
        let mut stream_info: Option<(u32, u8, u8, u32, String)> = None;
        while stream_info.is_none() && Instant::now() < stream_info_deadline {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let recv_res = if mcast_mode {
                sock.recv_from(&mut recv_buf).map(|(n, _)| n)
            } else {
                sock.recv(&mut recv_buf)
            };
            match recv_res {
                Ok(n) if n > 0 => {
                    if classify_datagram(recv_buf[0]) == DatagramKind::Control {
                        match wire::decode_control(&recv_buf[..n]) {
                            Ok(ControlMsg::StreamInfo {
                                sample_rate,
                                channels,
                                payload_type,
                                ssrc,
                                display_name,
                            }) => {
                                stream_info =
                                    Some((sample_rate, channels, payload_type, ssrc, display_name));
                            }
                            Ok(_) => {}
                            Err(e) => log::debug!("bad control from {addr}: {e}"),
                        }
                    }
                    // Any RTP arriving before STREAM_INFO is dropped — we
                    // can't decode interleaved samples without knowing
                    // channels yet.
                }
                Ok(_) => {}
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    // Resend SUBSCRIBE occasionally in case the first
                    // datagram was dropped (UDP offers no guarantees).
                    let _ = if mcast_mode {
                        sock.send_to(&ctrl_buf, addr)
                    } else {
                        sock.send(&ctrl_buf)
                    };
                }
                Err(e) => {
                    log::warn!("network source: recv waiting for STREAM_INFO: {e}");
                    break;
                }
            }
        }
        let Some((sample_rate, channels_u8, payload_type, ssrc, display_name)) = stream_info else {
            log::warn!(
                "network source: no STREAM_INFO from {addr} within 2s — sender offline or firewalled"
            );
            thread::sleep(Duration::from_millis(400));
            continue;
        };
        if payload_type != PT_F32LE && payload_type != PT_L24 {
            log::warn!(
                "network source: sender advertised payload_type={payload_type}, \
                 only f32le ({PT_F32LE}) and AES67 L24 ({PT_L24}) are supported"
            );
            thread::sleep(Duration::from_millis(1000));
            continue;
        }
        let channels = (channels_u8 as usize).min(MAX_CHANNELS);
        log::info!(
            "network source connected: {display_name} @ {sample_rate}Hz, {channels}ch, ssrc={ssrc:#010x}"
        );

        {
            let mut lv = levels.write();
            lv.active = true;
            lv.channels = channels;
            lv.sample_rate = sample_rate;
            lv.device_name = device_name.to_owned();
            lv.source_kind = SourceKind::Network;
        }
        {
            let mut l = loudness.write();
            *l = Default::default();
        }
        let mut analyzer = match LoudnessAnalyzer::new(channels as u32, sample_rate) {
            Ok(a) => Some(a),
            Err(err) => {
                log::warn!("loudness analyzer init failed (network): {err:?}");
                None
            }
        };
        spectrum::configure(spectrum, sample_rate, true);
        {
            let mut c = comp.write();
            c.clear();
            c.configure(sample_rate);
        }
        meter_monitor.begin_capture(sample_rate, channels, None, stop.clone());
        history.write().reset_tick_clock();
        let start_instant = std::time::Instant::now();

        // Jitter buffer sized by the preset. `target_fill` primes the buffer
        // before we start emitting so underruns only happen on a genuine
        // network stall. `drop_threshold` caps growth when the sender
        // bursts (e.g. after a routing hiccup releases a backlog).
        let target_ms = preset.target_ms() as usize;
        let target_fill = (sample_rate as usize * channels * target_ms / 1000).max(1);
        let drop_threshold =
            (sample_rate as usize * channels * (target_ms * 6).max(50) / 1000).max(target_fill);
        let emit_chunk = (sample_rate as usize * channels * 5 / 1000).max(1);

        let mut jitter: std::collections::VecDeque<f32> =
            std::collections::VecDeque::with_capacity(drop_threshold);
        let mut primed = false;
        let mut emit_scratch: Vec<f32> = Vec::with_capacity(emit_chunk);
        let mut decoded: Vec<f32> = Vec::with_capacity(MAX_CHANNELS * sample_rate as usize / 100);

        // Track sequence + SSRC so we detect packet loss / source changes.
        let mut current_ssrc = ssrc;
        let mut last_seq: Option<u16> = None;
        let mut consecutive_loss = 0u32;

        // Keepalive thread: clone the socket, fire a keepalive every 500 ms
        // until the stop flag is set or the receive loop exits.
        let ka_sock = match sock.try_clone() {
            Ok(s) => s,
            Err(e) => {
                log::warn!("network source: try_clone for keepalive: {e}");
                thread::sleep(Duration::from_millis(400));
                continue;
            }
        };
        let ka_stop = Arc::new(AtomicBool::new(false));
        let ka_stop_bg = ka_stop.clone();
        let ka_global = stop.clone();
        let ka_mcast = mcast_mode;
        let keepalive_thread = std::thread::Builder::new()
            .name("meter-link-keepalive".into())
            .spawn(move || {
                let mut ka_buf: Vec<u8> = Vec::with_capacity(32);
                while !ka_stop_bg.load(Ordering::Relaxed) && !ka_global.load(Ordering::Relaxed) {
                    if wire::encode_control(&mut ka_buf, &ControlMsg::Keepalive { client_id })
                        .is_ok()
                    {
                        let _ = if ka_mcast {
                            ka_sock.send_to(&ka_buf, addr)
                        } else {
                            ka_sock.send(&ka_buf)
                        };
                    }
                    thread::sleep(Duration::from_millis(500));
                }
            })
            .ok();

        let mut last_activity = Instant::now();

        'read: while !stop.load(Ordering::Relaxed) {
            let main_recv = if mcast_mode {
                sock.recv_from(&mut recv_buf).map(|(n, _)| n)
            } else {
                sock.recv(&mut recv_buf)
            };
            match main_recv {
                Ok(n) if n > 0 => {
                    last_activity = Instant::now();
                    match classify_datagram(recv_buf[0]) {
                        DatagramKind::Rtp => {
                            match wire::decode_audio_packet(&recv_buf[..n]) {
                                Ok((hdr, payload)) => {
                                    if hdr.ssrc != current_ssrc {
                                        log::info!(
                                            "network source: ssrc changed {current_ssrc:#010x} -> {:#010x}, resetting jitter buffer",
                                            hdr.ssrc
                                        );
                                        current_ssrc = hdr.ssrc;
                                        jitter.clear();
                                        last_seq = None;
                                        primed = false;
                                    }
                                    // Per-PT bytes-per-sample. Used below
                                    // for validation, zero-fill sizing and
                                    // to reject packets whose PT doesn't
                                    // match the active stream format.
                                    let bps_sample = match hdr.payload_type {
                                        PT_F32LE => 4usize,
                                        PT_L24 => L24_BYTES_PER_SAMPLE,
                                        _ => {
                                            // Unknown payload — skip.
                                            continue 'read;
                                        }
                                    };
                                    if hdr.payload_type != payload_type {
                                        // Sender changed wire format mid-
                                        // stream; the companion STREAM_INFO
                                        // arrives on a separate datagram so
                                        // we may hit one or two RTP packets
                                        // with the new PT before the
                                        // control path catches up. Drop
                                        // them rather than decode at the
                                        // wrong stride — STREAM_INFO will
                                        // trigger a reconnect.
                                        continue 'read;
                                    }
                                    let valid = match hdr.payload_type {
                                        PT_F32LE => {
                                            wire::validate_f32_payload(payload, channels_u8).is_ok()
                                        }
                                        PT_L24 => {
                                            wire::validate_l24_payload(payload, channels_u8).is_ok()
                                        }
                                        _ => false,
                                    };
                                    if !valid {
                                        continue 'read;
                                    }
                                    // Sequence gap detection. `wrapping_sub`
                                    // on `u16` does the right thing across
                                    // the 16-bit rollover for typical drift.
                                    let gap_samples = match last_seq {
                                        Some(prev) => {
                                            let delta = hdr.sequence.wrapping_sub(prev);
                                            if delta == 0 {
                                                // Duplicate — ignore.
                                                continue 'read;
                                            }
                                            if delta > 0x8000 {
                                                // Packet is "in the past" —
                                                // arrived late; drop.
                                                continue 'read;
                                            }
                                            // Zero-fill for every missing
                                            // packet so the DSP sample clock
                                            // stays aligned with the sender.
                                            (delta as u32).saturating_sub(1)
                                        }
                                        None => 0,
                                    };
                                    last_seq = Some(hdr.sequence);
                                    if gap_samples > 0 {
                                        consecutive_loss =
                                            consecutive_loss.saturating_add(gap_samples);
                                        // Cap the zero-fill so a huge gap
                                        // (e.g. routing blip) doesn't dump a
                                        // megabyte of silence into the
                                        // buffer at once.
                                        let per_pkt_samples =
                                            (payload.len() / bps_sample).max(1) as u32;
                                        let fill_samples =
                                            gap_samples.min(4) * per_pkt_samples;
                                        for _ in 0..fill_samples {
                                            jitter.push_back(0.0);
                                        }
                                    } else {
                                        consecutive_loss = 0;
                                    }
                                    match hdr.payload_type {
                                        PT_F32LE => {
                                            wire::decode_f32_payload(payload, &mut decoded);
                                        }
                                        PT_L24 => {
                                            wire::decode_l24_payload(payload, &mut decoded);
                                        }
                                        _ => continue 'read,
                                    }
                                    for &s in &decoded {
                                        jitter.push_back(s);
                                    }
                                    while jitter.len() > drop_threshold {
                                        jitter.pop_front();
                                    }
                                }
                                Err(e) => log::debug!("bad RTP from {addr}: {e}"),
                            }
                        }
                        DatagramKind::Control => {
                            if let Ok(ControlMsg::StreamInfo {
                                sample_rate: sr,
                                channels: ch,
                                payload_type: pt,
                                ssrc: new_ssrc,
                                ..
                            }) = wire::decode_control(&recv_buf[..n])
                            {
                                if sr != sample_rate
                                    || ch as usize != channels
                                    || pt != payload_type
                                {
                                    log::info!(
                                        "network source: format changed ({sample_rate}/{channels}/{payload_type}) -> ({sr}/{ch}/{pt}), reconnecting"
                                    );
                                    break 'read;
                                }
                                if new_ssrc != current_ssrc {
                                    current_ssrc = new_ssrc;
                                    jitter.clear();
                                    last_seq = None;
                                    primed = false;
                                }
                            }
                        }
                        DatagramKind::Unknown => {}
                    }
                }
                Ok(_) => {}
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    if last_activity.elapsed() > Duration::from_secs(3) {
                        log::info!(
                            "network source: no audio from {addr} for 3s, resubscribing"
                        );
                        break 'read;
                    }
                    // Loop back to check stop + process any queued audio.
                }
                Err(e) => {
                    log::warn!("network source: recv: {e}");
                    break 'read;
                }
            }

            if !primed {
                if jitter.len() >= target_fill {
                    primed = true;
                } else {
                    continue;
                }
            }

            while jitter.len() >= emit_chunk {
                emit_scratch.clear();
                for _ in 0..emit_chunk {
                    if let Some(v) = jitter.pop_front() {
                        emit_scratch.push(v);
                    }
                }
                let now = wall_clock_secs();
                let tp = meter_true_peak.load(Ordering::Relaxed);
                let snap = process_weighted_levels(
                    meter_scratch,
                    weighting,
                    &emit_scratch,
                    channels,
                    sample_rate,
                    tp,
                    levels,
                );
                let trig = clip_trigger_hold_max_db(
                    &mut clip_trigger_states.lock(),
                    meter_mode.load(Ordering::Relaxed),
                    now,
                    &snap,
                );
                process_clip(
                    &emit_scratch,
                    channels,
                    sample_rate,
                    clip_settings.as_ref(),
                    clip_capture,
                    clip_status,
                    clip_dirs,
                    trig,
                );
                if let Some(a) = analyzer.as_mut() {
                    a.process(&emit_scratch, loudness);
                }
                spectrum::push_samples(spectrum, &emit_scratch, channels);
                comp.write()
                    .push_samples_mono_sum(&emit_scratch, channels, sample_rate);
                meter_monitor.push_interleaved_f32(&emit_scratch, channels);
            }
            tick_shared_history(levels, history, history_trace, start_instant);
        }

        // Session ended: stop keepalive thread, best-effort UNSUBSCRIBE.
        ka_stop.store(true, Ordering::Relaxed);
        if let Some(h) = keepalive_thread {
            let _ = h.join();
        }
        if wire::encode_control(&mut ctrl_buf, &ControlMsg::Unsubscribe { client_id }).is_ok() {
            let _ = if mcast_mode {
                sock.send_to(&ctrl_buf, addr)
            } else {
                sock.send(&ctrl_buf)
            };
        }

        spectrum::configure(spectrum, sample_rate, false);
        comp.write().clear();
        if stop.load(Ordering::Relaxed) {
            break;
        }
        thread::sleep(Duration::from_millis(150));
    }

    spectrum::configure(spectrum, 48_000, false);
    comp.write().clear();
    {
        let mut lv = levels.write();
        lv.active = false;
        lv.channels = 0;
    }
    Ok(())
}
