//! IEC 61672–style A- and C-weighting for peak/RMS meters (ANSI S1.4 / IEC
//! analogue prototype + matched- z transform, SOS from SciPy reference).
//! LUFS / spectrum paths stay broadband — only classic meters use this.

/// Frequency weighting for **meter ballistics** (not LUFS, which uses K-weighting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FreqWeighting {
    /// Unweighted / broadband (Z-like within the audio band).
    #[default]
    Z = 0,
    A = 1,
    C = 2,
}

impl FreqWeighting {
    pub const ALL: [FreqWeighting; 3] = [FreqWeighting::Z, FreqWeighting::A, FreqWeighting::C];

    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => FreqWeighting::A,
            2 => FreqWeighting::C,
            _ => FreqWeighting::Z,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            FreqWeighting::Z => "Z  (broadband)",
            FreqWeighting::A => "A-weighted",
            FreqWeighting::C => "C-weighted",
        }
    }

    /// One-line label + hint for the dropdown list (selected row uses [`label`](Self::label)).
    pub fn list_entry(self) -> &'static str {
        match self {
            FreqWeighting::Z => "Z (broadband) — full bandwidth, no IEC curve",
            FreqWeighting::A => "A-weighted — more like human hearing (typical noise / SPL style)",
            FreqWeighting::C => "C-weighted — flatter than A; more bass shows in the reading",
        }
    }

    pub fn short_suffix(self) -> &'static str {
        match self {
            FreqWeighting::Z => "",
            FreqWeighting::A => " · A",
            FreqWeighting::C => " · C",
        }
    }

    /// Analytical IEC 61672–1 frequency response in dB at `f_hz`.
    ///
    /// Used for frequency-domain weighting of the spectrum analyzer (the
    /// meter path uses the matched biquad cascade further down this file).
    /// Returns 0 dB for `Z` (broadband / unweighted) and clamps to the
    /// spectrum floor at DC / negative inputs to avoid `log10(0)`.
    #[inline]
    pub fn response_db(self, f_hz: f32) -> f32 {
        if !f_hz.is_finite() || f_hz <= 0.0 {
            return match self {
                FreqWeighting::Z => 0.0,
                // Both A and C roll off toward -infinity at DC; clamp hard.
                _ => -120.0,
            };
        }
        match self {
            FreqWeighting::Z => 0.0,
            FreqWeighting::A => a_weight_db(f_hz),
            FreqWeighting::C => c_weight_db(f_hz),
        }
    }
}

/// IEC 61672-1 A-weighting analytical response (dB).
/// Pole frequencies: 20.598997, 107.65265, 737.86223, 12194.217 Hz.
/// Normalized so `R_A(1 kHz) ≈ 0 dB` via the standard +2.00 dB offset.
#[inline]
fn a_weight_db(f: f32) -> f32 {
    let f2 = f * f;
    let f4 = f2 * f2;
    const FL2: f32 = 20.598_997 * 20.598_997;
    const FM2: f32 = 107.652_65 * 107.652_65;
    const FN2: f32 = 737.862_24 * 737.862_24;
    const FH2: f32 = 12_194.217 * 12_194.217;
    let num = FH2 * f4;
    let den = (f2 + FL2) * ((f2 + FM2) * (f2 + FN2)).sqrt() * (f2 + FH2);
    let ra = num / den.max(1e-30);
    20.0 * ra.max(1e-30).log10() + 2.00
}

/// IEC 61672-1 C-weighting analytical response (dB).
/// Same low/high pole pair as A but without the middle band; +0.0619 dB 1-kHz normalisation.
#[inline]
fn c_weight_db(f: f32) -> f32 {
    let f2 = f * f;
    const FL2: f32 = 20.598_997 * 20.598_997;
    const FH2: f32 = 12_194.217 * 12_194.217;
    let num = FH2 * f2;
    let den = (f2 + FL2) * (f2 + FH2);
    let rc = num / den.max(1e-30);
    20.0 * rc.max(1e-30).log10() + 0.0619
}

/// One second-order section in scipy `sos` row order: `[b0, b1, b2, a0, a1, a2]` with `a0 = 1`.
#[derive(Clone, Copy)]
struct SosCoeff {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl SosCoeff {
    #[inline]
    fn from_row(row: [f32; 6]) -> Self {
        let a0 = row[3];
        debug_assert!((a0 - 1.0).abs() < 1e-3);
        Self {
            b0: row[0] / a0,
            b1: row[1] / a0,
            b2: row[2] / a0,
            a1: row[4] / a0,
            a2: row[5] / a0,
        }
    }
}

/// Direct-form-II transposed biquad (matches SciPy `sosfilt` streaming step).
struct Biquad {
    c: SosCoeff,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn new(c: SosCoeff) -> Self {
        Self {
            c,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let c = &self.c;
        let y = c.b0 * x + self.z1;
        self.z1 = c.b1 * x - c.a1 * y + self.z2;
        self.z2 = c.b2 * x - c.a2 * y;
        y
    }
}

/// Preset sample rates matching tabulated SOS (nearest neighbour at runtime).
const PRESET_RATES: [u32; 7] = [
    16_000, 32_000, 44_100, 48_000, 88_200, 96_000, 192_000,
];

fn nearest_preset_index(sample_rate: u32) -> usize {
    PRESET_RATES
        .iter()
        .enumerate()
        .min_by_key(|&(_, f)| (sample_rate as i64 - *f as i64).abs())
        .map(|(i, _)| i)
        .unwrap_or(3)
}

/// A-weighting SOS (3 sections), C-weighting (2 sections) per preset index.
fn sos_tables(
    preset_idx: usize,
    w: FreqWeighting,
) -> (&'static [[f32; 6]], usize) {
    const A: [[[f32; 6]; 3]; 7] = [
        // 16 kHz
        [
            [
                1.038_300_2,
                0.0,
                0.0,
                1.0,
                -0.016_647_243,
                0.000_069_282_67,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.707_050_8,
                0.717_463_7,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.983_886_8,
                0.983_951_75,
            ],
        ],
        // 32 kHz
        [
            [
                0.951_477_77,
                0.0,
                0.0,
                1.0,
                -0.182_467_76,
                0.008_323_622,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.844_211_3,
                0.847_032_3,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.991_927_1,
                0.991_943_4,
            ],
        ],
        // 44.1 kHz
        [
            [
                0.801_665_54,
                0.0,
                0.0,
                1.0,
                -0.351_961_2,
                0.030_969_17,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.884_988_8,
                0.886_507_7,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.994_138_9,
                0.994_147_5,
            ],
        ],
        // 48 kHz
        [
            [
                0.754_550_7,
                0.0,
                0.0,
                1.0,
                -0.405_322_55,
                0.041_071_59,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.893_939,
                0.895_227_3,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.994_614_5,
                0.994_621_7,
            ],
        ],
        // 88.2 kHz
        [
            [
                0.410_907_58,
                0.0,
                0.0,
                1.0,
                -0.839_0008,
                0.175_980_6,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.941_154_2,
                0.941_545_36,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.997_067_3,
                0.997_069_4,
            ],
        ],
        // 96 kHz
        [
            [
                0.369_586_36,
                0.0,
                0.0,
                1.0,
                -0.900_358_3,
                0.202_661_28,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.945_833_5,
                0.946_164_5,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.997_305_4,
                0.997_307_24,
            ],
        ],
        // 192 kHz
        [
            [
                0.134_328_29,
                0.0,
                0.0,
                1.0,
                -1.341_907_8,
                0.450_179_16,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.972_626,
                0.972_709_9,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.998_652_3,
                0.998_652_7,
            ],
        ],
    ];

    const C: [[[f32; 6]; 2]; 7] = [
        // 16 kHz
        [
            [
                0.977_17583,
                0.0,
                0.0,
                1.0,
                -0.016_647_243,
                0.000_069_282_67,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.983_886_8,
                0.983_951_75,
            ],
        ],
        // 32 kHz
        [
            [
                0.826_3659,
                0.0,
                0.0,
                1.0,
                -0.182_467_76,
                0.008_323_622,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.991_927_1,
                0.991_943_4,
            ],
        ],
        // 44.1 kHz
        [
            [
                0.680_863_2,
                0.0,
                0.0,
                1.0,
                -0.351_961_2,
                0.030_969_17,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.994_138_9,
                0.994_147_5,
            ],
        ],
        // 48 kHz
        [
            [
                0.637_76624,
                0.0,
                0.0,
                1.0,
                -0.405_322_55,
                0.041_071_59,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.994_614_5,
                0.994_621_7,
            ],
        ],
        // 88.2 kHz
        [
            [
                0.338_754_25,
                0.0,
                0.0,
                1.0,
                -0.839_0008,
                0.175_980_6,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.997_067_3,
                0.997_069_4,
            ],
        ],
        // 96 kHz
        [
            [
                0.303_94977,
                0.0,
                0.0,
                1.0,
                -0.900_358_3,
                0.202_661_28,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.997_305_4,
                0.997_307_24,
            ],
        ],
        // 192 kHz
        [
            [
                0.108_962_57,
                0.0,
                0.0,
                1.0,
                -1.341_907_8,
                0.450_179_16,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.998_652_3,
                0.998_652_7,
            ],
        ],
    ];

    let idx = preset_idx.min(A.len() - 1);
    match w {
        FreqWeighting::Z => (&[][..], 0),
        FreqWeighting::A => (&A[idx][..], 3),
        FreqWeighting::C => (&C[idx][..], 2),
    }
}

pub struct WeightingChain {
    biquads: [Biquad; 3],
    active: usize,
}

impl WeightingChain {
    pub fn new() -> Self {
        Self {
            biquads: std::array::from_fn(|_| Biquad::new(SosCoeff::from_row([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))),
            active: 0,
        }
    }

    pub fn configure(&mut self, w: FreqWeighting, sample_rate: u32) {
        let preset = nearest_preset_index(sample_rate);
        let (rows, n) = sos_tables(preset, w);
        self.active = n;
        if n == 0 {
            return;
        }
        for i in 0..n {
            let c = SosCoeff::from_row(rows[i]);
            self.biquads[i] = Biquad::new(c);
        }
        for b in &mut self.biquads[n..] {
            b.reset();
        }
    }

    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let mut y = x;
        for i in 0..self.active {
            y = self.biquads[i].tick(y);
        }
        y
    }
}

/// Per-channel scratch: interleaved buffer + one [`WeightingChain`] per channel.
pub struct MeterScratch {
    buf: Vec<f32>,
    chains: Vec<WeightingChain>,
    last_sr: u32,
    last_w: FreqWeighting,
    last_ch: usize,
}

impl MeterScratch {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            chains: Vec::new(),
            last_sr: 0,
            last_w: FreqWeighting::Z,
            last_ch: 0,
        }
    }

    /// Copies `data` into an internal buffer, applies weighting when `w` is not [`FreqWeighting::Z`],
    /// and leaves the result in [`MeterScratch::buf`].
    pub fn prepare_meter_block(
        &mut self,
        data: &[f32],
        channels: usize,
        sample_rate: u32,
        w: FreqWeighting,
    ) {
        self.buf.clear();
        self.buf.extend_from_slice(data);
        if w == FreqWeighting::Z || channels == 0 || self.buf.is_empty() {
            // Mark broadband so the next A/C block re-initialises IIR state.
            self.last_w = FreqWeighting::Z;
            return;
        }

        if self.chains.len() != channels {
            self.chains = (0..channels).map(|_| WeightingChain::new()).collect();
            self.last_sr = 0;
            self.last_w = FreqWeighting::Z;
        }

        let reconfig = self.last_sr != sample_rate
            || self.last_w != w
            || self.last_ch != channels;
        if reconfig {
            for c in &mut self.chains {
                c.configure(w, sample_rate);
            }
            self.last_sr = sample_rate;
            self.last_w = w;
            self.last_ch = channels;
        }

        let frames = self.buf.len() / channels;
        for i in 0..frames {
            for ch in 0..channels {
                let idx = i * channels + ch;
                let s = self.buf[idx];
                self.buf[idx] = self.chains[ch].tick(s);
            }
        }
    }

    #[inline]
    pub fn buf(&self) -> &[f32] {
        &self.buf
    }
}
