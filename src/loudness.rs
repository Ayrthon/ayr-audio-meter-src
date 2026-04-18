//! EBU R128 / ITU-R BS.1770 loudness metering.
//!
//! Wraps the [`ebur128`] crate (EBU TECH 3341/3342 compliant) into a
//! small analyzer that is owned by the audio thread and pushes the
//! latest momentary / short-term / integrated / true-peak figures into
//! a [`SharedLoudness`] for the UI thread to read.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use ebur128::{EbuR128, Mode};
use parking_lot::RwLock;

/// Sentinel value representing "no useful reading yet".
pub const NEG_INF_LUFS: f32 = -120.0;

#[derive(Clone, Copy)]
pub struct LoudnessState {
    pub momentary_lufs: f32,
    pub short_lufs: f32,
    pub integrated_lufs: f32,
    pub true_peak_dbtp: f32,
    pub active: bool,
    /// UI → audio one-shot: set to request an integrated / peak reset.
    pub reset_request: bool,
}

impl Default for LoudnessState {
    fn default() -> Self {
        Self {
            momentary_lufs: NEG_INF_LUFS,
            short_lufs: NEG_INF_LUFS,
            integrated_lufs: NEG_INF_LUFS,
            true_peak_dbtp: NEG_INF_LUFS,
            active: false,
            reset_request: false,
        }
    }
}

pub type SharedLoudness = Arc<RwLock<LoudnessState>>;

pub fn make_shared_loudness() -> SharedLoudness {
    Arc::new(RwLock::new(LoudnessState::default()))
}

pub struct LoudnessAnalyzer {
    ebu: EbuR128,
    channels: u32,
    sample_rate: u32,
    /// Rate-limit integrated/true-peak readouts — they don't change
    /// meaningfully faster than ~100 ms anyway.
    last_pub: Option<Instant>,
}

impl LoudnessAnalyzer {
    pub fn new(channels: u32, sample_rate: u32) -> Result<Self> {
        let mode = Mode::M | Mode::S | Mode::I | Mode::TRUE_PEAK;
        let ebu = EbuR128::new(channels.max(1), sample_rate, mode).context("EbuR128::new")?;
        Ok(Self {
            ebu,
            channels: channels.max(1),
            sample_rate,
            last_pub: None,
        })
    }

    /// Push a block of interleaved f32 samples (range ±1.0).
    pub fn process(&mut self, interleaved: &[f32], shared: &SharedLoudness) {
        if interleaved.is_empty() {
            return;
        }

        // Handle UI reset request.
        {
            let r = shared.read();
            if r.reset_request {
                drop(r);
                if let Ok(fresh) = EbuR128::new(
                    self.channels,
                    self.sample_rate,
                    Mode::M | Mode::S | Mode::I | Mode::TRUE_PEAK,
                ) {
                    self.ebu = fresh;
                }
                let mut w = shared.write();
                w.reset_request = false;
                w.integrated_lufs = NEG_INF_LUFS;
                w.true_peak_dbtp = NEG_INF_LUFS;
                w.momentary_lufs = NEG_INF_LUFS;
                w.short_lufs = NEG_INF_LUFS;
            }
        }

        if let Err(err) = self.ebu.add_frames_f32(interleaved) {
            log::debug!("ebur128 add_frames_f32: {err:?}");
            return;
        }

        // Rate-limit publish to ~every 80 ms — the underlying block
        // cadence is 100 ms so this is plenty.
        let now = Instant::now();
        let should_pub = match self.last_pub {
            None => true,
            Some(t) => now.duration_since(t).as_millis() >= 80,
        };
        if !should_pub {
            return;
        }
        self.last_pub = Some(now);

        let momentary = self.ebu.loudness_momentary().unwrap_or(f64::NEG_INFINITY);
        let short = self.ebu.loudness_shortterm().unwrap_or(f64::NEG_INFINITY);
        let integrated = self.ebu.loudness_global().unwrap_or(f64::NEG_INFINITY);

        // Maximum true peak across channels (linear amplitude → dBTP).
        let mut tp_linear = 0.0_f64;
        for ch in 0..self.channels {
            if let Ok(v) = self.ebu.true_peak(ch) {
                if v > tp_linear {
                    tp_linear = v;
                }
            }
        }
        let true_peak = if tp_linear > 0.0 {
            20.0 * tp_linear.log10()
        } else {
            f64::NEG_INFINITY
        };

        let mut s = shared.write();
        s.momentary_lufs = sanitize(momentary);
        s.short_lufs = sanitize(short);
        s.integrated_lufs = sanitize(integrated);
        s.true_peak_dbtp = sanitize(true_peak);
        s.active = true;
    }
}

fn sanitize(x: f64) -> f32 {
    if !x.is_finite() || x <= NEG_INF_LUFS as f64 {
        NEG_INF_LUFS
    } else {
        x as f32
    }
}
