//! Meter modes, per-mode ballistics and color-zone thresholds.

/// Lower dBFS bound used across the whole UI — mirrored in
/// `meter_widget::MIN_DB` so palette zones can reference it.
pub const MIN_DB: f32 = -60.0;
/// Digital full scale (top of the dBFS ruler).
pub const MAX_DB: f32 = 0.0;

/// OBS **Settings → Audio → Meter decay rate → Fast** (`VOLUME_METER_DECAY_FAST`
/// in `OBSBasicSettings.hpp`, default in `OBSBasic.cpp`). Linear dB/s fall of the
/// painted peak; ~40 dB / 1.7 s.
pub const OBS_METER_DECAY_FAST_DB_PER_SEC: f32 = 23.53;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeterMode {
    SamplePeak = 0,
    Obs = 1,
    DigitalPpm = 2,
    TruePeak = 3,
    Vu = 4,
    K20 = 5,
    K14 = 6,
    K12 = 7,
}

impl MeterMode {
    pub const ALL: [MeterMode; 8] = [
        MeterMode::SamplePeak,
        MeterMode::Obs,
        MeterMode::DigitalPpm,
        MeterMode::TruePeak,
        MeterMode::Vu,
        MeterMode::K20,
        MeterMode::K14,
        MeterMode::K12,
    ];

    pub fn display_name(self) -> &'static str {
        match self {
            MeterMode::SamplePeak => "Sample Peak  (Pro Tools)",
            MeterMode::Obs => "OBS  (Fast decay, Sample Peak, AL -20 / PML -9)",
            MeterMode::DigitalPpm => "Digital PPM  (ATEM, EBU/BBC)",
            MeterMode::TruePeak => "True Peak  (EBU R128, Loudness)",
            MeterMode::Vu => "VU  (Analog / SMPTE)",
            MeterMode::K20 => "K-20  (Mastering, film)",
            MeterMode::K14 => "K-14  (Mastering, pop)",
            MeterMode::K12 => "K-12  (Broadcast)",
        }
    }

    /// The part of [`display_name`](Self::display_name) before `(`,
    /// end-trimmed. Kept around as a short stable-width label option even
    /// though the meter combo currently shows the full display name.
    #[allow(dead_code)]
    pub fn name_outside_paren(self) -> &'static str {
        let s = self.display_name();
        match s.split_once('(') {
            Some((head, _)) => head.trim_end(),
            None => s.trim_end(),
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            MeterMode::SamplePeak => "PEAK",
            MeterMode::Obs => "OBS",
            MeterMode::DigitalPpm => "PPM",
            MeterMode::TruePeak => "TP",
            MeterMode::Vu => "VU",
            MeterMode::K20 => "K-20",
            MeterMode::K14 => "K-14",
            MeterMode::K12 => "K-12",
        }
    }

    #[inline]
    pub fn to_discriminant(self) -> u8 {
        self as u8
    }

    #[inline]
    pub fn from_discriminant(d: u8) -> Self {
        match d {
            0 => Self::SamplePeak,
            1 => Self::Obs,
            2 => Self::DigitalPpm,
            3 => Self::TruePeak,
            4 => Self::Vu,
            5 => Self::K20,
            6 => Self::K14,
            7 => Self::K12,
            _ => Self::SamplePeak,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillSource {
    Peak,
    Rms,
}

/// Color treatment for the colored bands inside the bar.
///
/// Pro Tools uses three bar bands (dark olive → bright green → orange)
/// and treats red exclusively as the clip LED above the bar.  Other
/// DAWs / broadcast meters use the classic bright-green → yellow → red
/// band layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPalette {
    /// Bright green → yellow → red.  Matches OBS, PPM, K-System, VU.
    Standard,
    /// OBS-style face: green (safe) → mustard/amber (caution) → red (hot).
    Obs,
    /// Dark olive → bright green → orange (+ red LED).  Matches Avid
    /// Pro Tools' Sample Peak / Pro Tools Classic meter.
    ProTools,
}

#[derive(Debug, Clone, Copy)]
pub struct MeterConfig {
    pub peak_attack_tau: f32,
    pub peak_release_db_per_sec: f32,
    pub rms_attack_tau: f32,
    pub rms_release_tau: f32,
    pub hold_time: f32,
    pub hold_release_db_per_sec: f32,
    /// When true, after `hold_time` without a new maximum the hold **jumps** to
    /// the current level (OBS `VolumeMeter` peak-hold). When false, the hold
    /// decays at `hold_release_db_per_sec`.
    pub hold_snap_after_timeout: bool,
    /// When set, RMS / magnitude smoothing uses OBS's per-frame linear step in
    /// dB (`Δ * (dt/τ) * 0.99`, clamped) instead of exponential `approach`.
    pub rms_linear_blend_tau: Option<f32>,
    pub fill_source: FillSource,
    pub show_rms_line: bool,
    pub true_peak: bool,
    pub reference_db: f32,

    /// Top of the "low" band (dark olive / dim green).  Set to
    /// `MIN_DB` to suppress the low band (3-band classic layout).
    pub zone_low_top_db: f32,
    /// Top of the "mid" band (bright green).
    pub zone_mid_top_db: f32,
    /// Top of the "high" band (orange in Pro Tools, yellow in the
    /// standard palette).  Anything above this up to 0 dBFS renders
    /// in the "over" color (red).  Set to `0.0` to suppress a red
    /// band inside the bar (Pro Tools style — clip LED only).
    pub zone_high_top_db: f32,

    pub palette: ColorPalette,
}

pub fn config_for(mode: MeterMode) -> MeterConfig {
    match mode {
        // ── Avid Pro Tools 2024/2025 Sample Peak defaults ──────────
        //   Dim green     : -∞   … -20 dBFS   (lower ~2/3 of the bar)
        //   Bright green  : -20  …  -5 dBFS
        //   Orange        :  -5  …   0 dBFS   (thin top band)
        //   Red (clip LED):   0 dBFS+         (separate from the bar)
        MeterMode::SamplePeak => MeterConfig {
            peak_attack_tau: 0.020,
            peak_release_db_per_sec: 20.0 / 0.650,
            rms_attack_tau: 0.200,
            rms_release_tau: 0.700,
            // Pro Tools default: Peak Hold = 3 seconds.
            hold_time: 3.0,
            hold_release_db_per_sec: 10.0,
            hold_snap_after_timeout: false,
            rms_linear_blend_tau: None,
            fill_source: FillSource::Peak,
            show_rms_line: true,
            true_peak: false,
            reference_db: -18.0,
            zone_low_top_db: -20.0,
            zone_mid_top_db: -5.0,
            zone_high_top_db: 0.0,
            palette: ColorPalette::ProTools,
        },
        MeterMode::Obs => MeterConfig {
            // Peak attack/decay/hold/RMS: `VolumeMeter.cpp`. Decay rate is taken
            // from the user's `MeterDecayRate` setting; we lock to OBS default
            // **Fast** — `OBS_METER_DECAY_FAST_DB_PER_SEC` (23.53 dB/s).
            peak_attack_tau: 0.0,
            peak_release_db_per_sec: OBS_METER_DECAY_FAST_DB_PER_SEC,
            rms_attack_tau: 0.300,
            rms_release_tau: 0.300,
            hold_time: 20.0,
            hold_release_db_per_sec: 10.0,
            hold_snap_after_timeout: true,
            rms_linear_blend_tau: Some(0.3),
            fill_source: FillSource::Peak,
            show_rms_line: true,
            true_peak: false,
            // OBS sample-peak meter definitions from upstream:
            //   Alignment Level (AL)       = -20 dBFS
            //   Permitted Maximum Level    =  -9 dBFS
            // with ruler labels every 6 dB. This places transitions visually
            // between -24/-18 and -12/-6 ticks, matching OBS.
            reference_db: -20.0,
            zone_low_top_db: MIN_DB,
            zone_mid_top_db: -20.0,
            zone_high_top_db: -9.0,
            palette: ColorPalette::Obs,
        },
        MeterMode::DigitalPpm => MeterConfig {
            peak_attack_tau: 0.010,
            peak_release_db_per_sec: 20.0 / 1.700,
            rms_attack_tau: 0.200,
            rms_release_tau: 1.700,
            hold_time: 2.2,
            hold_release_db_per_sec: 6.0,
            hold_snap_after_timeout: false,
            rms_linear_blend_tau: None,
            fill_source: FillSource::Peak,
            show_rms_line: true,
            true_peak: false,
            reference_db: -18.0,
            zone_low_top_db: MIN_DB,
            zone_mid_top_db: -20.0,
            zone_high_top_db: -9.0,
            palette: ColorPalette::Standard,
        },
        MeterMode::TruePeak => MeterConfig {
            peak_attack_tau: 0.020,
            peak_release_db_per_sec: 20.0 / 0.650,
            rms_attack_tau: 0.200,
            rms_release_tau: 0.700,
            hold_time: 2.2,
            hold_release_db_per_sec: 10.0,
            hold_snap_after_timeout: false,
            rms_linear_blend_tau: None,
            fill_source: FillSource::Peak,
            show_rms_line: true,
            true_peak: true,
            reference_db: -18.0,
            zone_low_top_db: MIN_DB,
            zone_mid_top_db: -3.0,
            zone_high_top_db: -1.0,
            palette: ColorPalette::Standard,
        },
        MeterMode::Vu => MeterConfig {
            peak_attack_tau: 0.020,
            peak_release_db_per_sec: 20.0 / 0.650,
            rms_attack_tau: 0.300,
            rms_release_tau: 0.300,
            hold_time: 2.2,
            hold_release_db_per_sec: 10.0,
            hold_snap_after_timeout: false,
            rms_linear_blend_tau: None,
            fill_source: FillSource::Peak,
            show_rms_line: true,
            true_peak: false,
            reference_db: -18.0,
            zone_low_top_db: MIN_DB,
            zone_mid_top_db: -18.0,
            zone_high_top_db: -15.0,
            palette: ColorPalette::Standard,
        },
        MeterMode::K20 => MeterConfig {
            peak_attack_tau: 0.010,
            peak_release_db_per_sec: 20.0 / 1.700,
            rms_attack_tau: 0.300,
            rms_release_tau: 0.300,
            hold_time: 2.2,
            hold_release_db_per_sec: 6.0,
            hold_snap_after_timeout: false,
            rms_linear_blend_tau: None,
            fill_source: FillSource::Peak,
            show_rms_line: true,
            true_peak: false,
            reference_db: -20.0,
            zone_low_top_db: MIN_DB,
            zone_mid_top_db: -20.0,
            zone_high_top_db: -16.0,
            palette: ColorPalette::Standard,
        },
        MeterMode::K14 => MeterConfig {
            peak_attack_tau: 0.010,
            peak_release_db_per_sec: 20.0 / 1.700,
            rms_attack_tau: 0.300,
            rms_release_tau: 0.300,
            hold_time: 2.2,
            hold_release_db_per_sec: 6.0,
            hold_snap_after_timeout: false,
            rms_linear_blend_tau: None,
            fill_source: FillSource::Peak,
            show_rms_line: true,
            true_peak: false,
            reference_db: -14.0,
            zone_low_top_db: MIN_DB,
            zone_mid_top_db: -14.0,
            zone_high_top_db: -10.0,
            palette: ColorPalette::Standard,
        },
        MeterMode::K12 => MeterConfig {
            peak_attack_tau: 0.010,
            peak_release_db_per_sec: 20.0 / 1.700,
            rms_attack_tau: 0.300,
            rms_release_tau: 0.300,
            hold_time: 2.2,
            hold_release_db_per_sec: 6.0,
            hold_snap_after_timeout: false,
            rms_linear_blend_tau: None,
            fill_source: FillSource::Peak,
            show_rms_line: true,
            true_peak: false,
            reference_db: -12.0,
            zone_low_top_db: MIN_DB,
            zone_mid_top_db: -12.0,
            zone_high_top_db: -8.0,
            palette: ColorPalette::Standard,
        },
    }
}

// ── Per-mode vertical scale (dBFS vs VU vs K-relative) ───────────────

/// Which numbering the meter **ruler** uses (ballistics stay in dBFS internally).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeterScaleKind {
    /// −60 … 0 dBFS.
    Dbf,
    /// SMPTE-style VU face: −20 … +3 VU, 0 VU at [`MeterScale::reference_dbfs`].
    Vu,
    /// K-system style: relative dB with 0 at headroom reference (Bob Katz layout).
    K,
}

#[derive(Debug, Clone, Copy)]
pub struct MeterScale {
    pub kind: MeterScaleKind,
    /// dBFS level that reads as display **0** (0 VU or 0 K).
    pub reference_dbfs: f32,
    /// Display value at the **bottom** of the bar (quiet end).
    pub disp_min: f32,
    /// Display value at the **top** of the bar (hot end, just below clip LED).
    pub disp_max: f32,
    /// Major ruler ticks in display units (dBFS/VU/K). Populated per-mode so K-20/K-14/K-12
    /// each get their own Katz-correct layout instead of sharing one "+9 K max" fallback.
    pub ticks: &'static [f32],
}

/// Major ticks in **display units** (dBFS, VU, or K).
pub const TICK_DBF: [f32; 11] = [
    0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -50.0, -60.0,
];
/// OBS-style dBFS ruler: 6 dB steps from 0 to -60.
pub const TICK_OBS: [f32; 11] = [
    0.0, -6.0, -12.0, -18.0, -24.0, -30.0, -36.0, -42.0, -48.0, -54.0, -60.0,
];
pub const TICK_VU: [f32; 12] = [
    -20.0, -15.0, -10.0, -7.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0,
];
// Bob Katz K-system meters: 0 K = the mode's reference (e.g. -20 dBFS for K-20), and the top
// of the ruler is the mode's full-scale headroom (+20 K = 0 dBFS for K-20, +14 K for K-14,
// +12 K for K-12). Each mode therefore has its own tick layout.
pub const TICK_K20: [f32; 11] = [-40.0, -30.0, -20.0, -10.0, -5.0, 0.0, 4.0, 8.0, 12.0, 16.0, 20.0];
pub const TICK_K14: [f32; 10] = [-30.0, -20.0, -14.0, -10.0, -5.0, 0.0, 3.0, 6.0, 10.0, 14.0];
pub const TICK_K12: [f32; 10] = [-30.0, -20.0, -12.0, -10.0, -5.0, 0.0, 3.0, 6.0, 9.0, 12.0];

impl MeterScale {
    pub fn for_mode(mode: MeterMode) -> Self {
        let cfg = config_for(mode);
        match mode {
            MeterMode::Vu => Self {
                kind: MeterScaleKind::Vu,
                reference_dbfs: cfg.reference_db,
                disp_min: -20.0,
                disp_max: 3.0,
                ticks: &TICK_VU,
            },
            MeterMode::K20 => Self {
                kind: MeterScaleKind::K,
                reference_dbfs: cfg.reference_db,
                disp_min: -42.0,
                disp_max: 20.0,
                ticks: &TICK_K20,
            },
            MeterMode::K14 => Self {
                kind: MeterScaleKind::K,
                reference_dbfs: cfg.reference_db,
                disp_min: -32.0,
                disp_max: 14.0,
                ticks: &TICK_K14,
            },
            MeterMode::K12 => Self {
                kind: MeterScaleKind::K,
                reference_dbfs: cfg.reference_db,
                disp_min: -32.0,
                disp_max: 12.0,
                ticks: &TICK_K12,
            },
            MeterMode::Obs => Self {
                kind: MeterScaleKind::Dbf,
                reference_dbfs: 0.0,
                disp_min: MIN_DB,
                disp_max: MAX_DB,
                ticks: &TICK_OBS,
            },
            _ => Self {
                kind: MeterScaleKind::Dbf,
                reference_dbfs: 0.0,
                disp_min: MIN_DB,
                disp_max: MAX_DB,
                ticks: &TICK_DBF,
            },
        }
    }

    #[inline]
    pub fn dbfs_to_display(&self, dbfs: f32) -> f32 {
        match self.kind {
            MeterScaleKind::Dbf => dbfs,
            MeterScaleKind::Vu | MeterScaleKind::K => dbfs - self.reference_dbfs,
        }
    }

    /// Map dBFS sample level to vertical **y** (same convention as legacy `db_to_y`: top = hot).
    pub fn dbfs_to_y(&self, dbfs: f32, bar_top: f32, bar_bottom: f32) -> f32 {
        match self.kind {
            MeterScaleKind::Dbf => {
                let d = dbfs.clamp(MIN_DB, MAX_DB);
                bar_top + (MAX_DB - d) / (MAX_DB - MIN_DB) * (bar_bottom - bar_top)
            }
            MeterScaleKind::Vu | MeterScaleKind::K => {
                let disp = dbfs - self.reference_dbfs;
                let span = self.disp_max - self.disp_min;
                if span <= 0.0 {
                    return bar_bottom;
                }
                let frac = ((disp - self.disp_min) / span).clamp(0.0, 1.0);
                bar_bottom - frac * (bar_bottom - bar_top)
            }
        }
    }

    /// Inverse of [`Self::dbfs_to_y`]: map a vertical pixel **inside** the bar to dBFS.
    pub fn y_to_dbfs(&self, y: f32, bar_top: f32, bar_bottom: f32) -> f32 {
        let h = bar_bottom - bar_top;
        if h.abs() < 1e-6 {
            return MIN_DB;
        }
        match self.kind {
            MeterScaleKind::Dbf => MAX_DB - (y - bar_top) / h * (MAX_DB - MIN_DB),
            MeterScaleKind::Vu | MeterScaleKind::K => {
                let span = self.disp_max - self.disp_min;
                if span <= 0.0 {
                    return self.reference_dbfs + self.disp_min;
                }
                let frac = ((bar_bottom - y) / h).clamp(0.0, 1.0);
                self.reference_dbfs + self.disp_min + frac * span
            }
        }
    }

    pub fn tick_display_values(&self) -> &'static [f32] {
        self.ticks
    }

    pub fn tick_label(&self, display_val: f32) -> String {
        match self.kind {
            MeterScaleKind::Dbf => {
                if display_val >= -0.05 {
                    "0".into()
                } else {
                    format!("{}", (-display_val) as i32)
                }
            }
            MeterScaleKind::Vu => {
                if display_val.abs() < 0.05 {
                    "0".into()
                } else if display_val > 0.0 {
                    format!("+{:.0}", display_val)
                } else {
                    format!("{:.0}", display_val)
                }
            }
            MeterScaleKind::K => {
                if display_val.abs() < 0.05 {
                    "0".into()
                } else if display_val > 0.0 {
                    format!("+{:.0}", display_val)
                } else {
                    format!("{:.0}", display_val)
                }
            }
        }
    }

    /// Numeric readout under the meter (still driven by internal dBFS ballistics).
    pub fn format_readout_dbfs(&self, dbfs: f32) -> String {
        use crate::dsp::INF_FLOOR_DB;
        if dbfs <= INF_FLOOR_DB + 1.0 {
            return "\u{2212}inf".to_string();
        }
        match self.kind {
            MeterScaleKind::Dbf => {
                if dbfs.abs() < 0.05 {
                    "0.0".into()
                } else {
                    format!("{:+.1}", dbfs)
                }
            }
            MeterScaleKind::Vu => format!("{:+.1} VU", self.dbfs_to_display(dbfs)),
            MeterScaleKind::K => format!("{:+.1} K", self.dbfs_to_display(dbfs)),
        }
    }

    /// Short unit for the reference column (e.g. next to the dashed line).
    pub fn reference_label_text(&self) -> &'static str {
        match self.kind {
            MeterScaleKind::Dbf => "0 dB ref",
            MeterScaleKind::Vu => "0 VU",
            MeterScaleKind::K => "0 K",
        }
    }
}

/// Wall-clock seconds since the Unix epoch. Used for meter ballistics and the
/// clip-trigger path so UI `MeterState` and audio-thread `clip_trigger_states`
/// stay on the same time base (egui’s `input.time` is not comparable to stream
/// elapsed time).
#[inline]
pub fn wall_clock_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Exponential approach (tau-based).
pub fn approach(current: f32, target: f32, dt: f32, attack_tau: f32, release_tau: f32) -> f32 {
    let tau = if target > current { attack_tau } else { release_tau };
    if tau <= 0.0 {
        return target;
    }
    let alpha = 1.0 - (-dt / tau).exp();
    current + (target - current) * alpha
}
