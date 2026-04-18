//! Block-level audio analysis: per-channel peak and RMS in dBFS,
//! with an optional 4x linear-interpolated inter-sample (true-peak)
//! approximation.

use std::f32::NEG_INFINITY;

pub const INF_FLOOR_DB: f32 = -90.0;

/// Converts a linear amplitude (0..1) to dBFS.  Values below 1e-9 are
/// clamped to the [`INF_FLOOR_DB`] so the UI can treat them as "no
/// signal".
#[inline]
pub fn amp_to_db(a: f32) -> f32 {
    if a <= 1e-9 {
        INF_FLOOR_DB
    } else {
        20.0 * a.log10()
    }
}

/// Analyses an interleaved block.  `samples.len() == frames * channels`.
///
/// Writes per-channel peak (dBFS) into `peaks[..channels]` and
/// per-channel RMS into `rms[..channels]`.
///
/// When `true_peak` is true the function also checks three
/// linearly-interpolated samples between each pair of consecutive
/// samples and takes the maximum.  This is the same cheap 4x
/// oversampled true-peak approximation used on the Python side.
pub fn block_peak_rms(
    samples: &[f32],
    channels: usize,
    true_peak: bool,
    peaks: &mut [f32],
    rms: &mut [f32],
) {
    debug_assert!(channels > 0);
    debug_assert!(peaks.len() >= channels);
    debug_assert!(rms.len() >= channels);
    if samples.is_empty() {
        for v in &mut peaks[..channels] {
            *v = INF_FLOOR_DB;
        }
        for v in &mut rms[..channels] {
            *v = INF_FLOOR_DB;
        }
        return;
    }
    let frames = samples.len() / channels;
    if frames == 0 {
        for v in &mut peaks[..channels] {
            *v = INF_FLOOR_DB;
        }
        for v in &mut rms[..channels] {
            *v = INF_FLOOR_DB;
        }
        return;
    }

    for ch in 0..channels {
        let mut peak = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut prev: Option<f32> = None;
        for i in 0..frames {
            let s = samples[i * channels + ch];
            let a = s.abs();
            if a > peak {
                peak = a;
            }
            sum_sq += (s as f64) * (s as f64);

            if true_peak {
                if let Some(p) = prev {
                    // 4x linear interpolation between ``p`` and ``s``.
                    let d = s - p;
                    let t1 = p + d * 0.25;
                    let t2 = p + d * 0.5;
                    let t3 = p + d * 0.75;
                    let m = t1.abs().max(t2.abs()).max(t3.abs());
                    if m > peak {
                        peak = m;
                    }
                }
                prev = Some(s);
            }
        }
        let rms_lin = (sum_sq / frames as f64).sqrt() as f32;
        peaks[ch] = if peak > 0.0 { amp_to_db(peak) } else { NEG_INFINITY.max(INF_FLOOR_DB) };
        rms[ch] = if rms_lin > 0.0 { amp_to_db(rms_lin) } else { INF_FLOOR_DB };

        if peaks[ch] < INF_FLOOR_DB {
            peaks[ch] = INF_FLOOR_DB;
        }
        if rms[ch] < INF_FLOOR_DB {
            rms[ch] = INF_FLOOR_DB;
        }
    }
}
