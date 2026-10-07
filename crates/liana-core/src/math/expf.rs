//! numpy 2.5.3's `f32` `exp`: `simd_exp_FLOAT`, the `SIMD_AVX2_FMA3` body of
//! `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src` (tag v2.5.3),
//! scalar-transcribed. Constants: `npy_simd_data.h` (`NPY_COEFF_*_EXPF`,
//! `NPY_CODY_WAITE_LOGE_2_*f`, `NPY_RINT_CVT_MAGICf`) and `npy_math.h`
//! (`NPY_LOG2Ef`, `NPY_INFINITYF`, `NPY_NANF`), all as raw `f32` bit patterns.

/// `NPY_CODY_WAITE_LOGE_2_HIGHf` — ln(2) high part (Cody-Waite split).
const CODY_C1: f32 = f32::from_bits(0xBF31_7200);
/// `NPY_CODY_WAITE_LOGE_2_LOWf` — ln(2) low part.
const CODY_C2: f32 = f32::from_bits(0xB5BF_BE8E);
/// `NPY_RINT_CVT_MAGICf` — 0x1.8p23: adding it rounds to nearest integer.
const RINT_MAGIC: f32 = f32::from_bits(0x4B40_0000);
/// `NPY_LOG2Ef`.
const LOG2E: f32 = f32::from_bits(0x3FB8_AA3B);
const P: [f32; 6] = [
    f32::from_bits(0x3F80_0000), // NPY_COEFF_P0_EXPf
    f32::from_bits(0x3F39_CBD5), // NPY_COEFF_P1_EXPf
    f32::from_bits(0x3E7D_4C58), // NPY_COEFF_P2_EXPf
    f32::from_bits(0x3D51_7D8C), // NPY_COEFF_P3_EXPf
    f32::from_bits(0x3BDD_7159), // NPY_COEFF_P4_EXPf
    f32::from_bits(0x3A05_3DD8), // NPY_COEFF_P5_EXPf
];
const Q: [f32; 3] = [
    f32::from_bits(0x3F80_0000), // NPY_COEFF_Q0_EXPf
    f32::from_bits(0xBE8C_6857), // NPY_COEFF_Q1_EXPf
    f32::from_bits(0x3CB0_E832), // NPY_COEFF_Q2_EXPf
];
/// The saturation guards, `xmax` / `xmin` (both literals in the kernel body).
const XMAX: f32 = f32::from_bits(0x42B1_7218);
const XMIN: f32 = f32::from_bits(0xC2CF_F1B5);
/// `NPY_NANF` / `NPY_INFINITYF`.
const NAN: f32 = f32::from_bits(0x7FC0_0000);
const INF: f32 = f32::from_bits(0x7F80_0000);

/// `exp(x)` in `f32`, bit-identical to `np.exp` on an `f32` array of this
/// (X86_V3) numpy build — including the saturation masks: `x >= xmax` gives
/// `inf`, `x <= xmin` gives `0.0`, `NaN` (and the `x` it is replaced by) never
/// reaches the polynomial.
pub fn expf(x_in: f32) -> f32 {
    let nan_mask = x_in.is_nan();
    let mut x = if nan_mask { 0.0 } else { x_in };
    // evaluated on the NaN-clobbered x, as the vector body does
    let xmax_mask = x >= XMAX;
    let xmin_mask = x <= XMIN;
    // `inf_mask` / `ninf_mask` only drive the FP status flags in the kernel
    // (`xmax ^ inf`, `xmin ^ ninf`); the values come from the masks above.
    x = if nan_mask || xmin_mask || xmax_mask {
        0.0
    } else {
        x
    };

    // y = rint(x / ln2) via the magic-number trick; the reduced x is
    // x - y*ln(2) in three Cody-Waite steps (the third against +0.0).
    let mut quadrant = x * LOG2E;
    quadrant = (quadrant + RINT_MAGIC) - RINT_MAGIC;
    x = quadrant.mul_add(CODY_C1, x);
    x = quadrant.mul_add(CODY_C2, x);
    x = quadrant.mul_add(0.0, x);

    let mut num = P[5].mul_add(x, P[4]);
    num = num.mul_add(x, P[3]);
    num = num.mul_add(x, P[2]);
    num = num.mul_add(x, P[1]);
    num = num.mul_add(x, P[0]);
    let mut den = Q[2].mul_add(x, Q[1]);
    den = den.mul_add(x, Q[0]);
    let poly = num / den;

    let mut poly = scalef(poly, quadrant);

    if nan_mask {
        poly = NAN;
    }
    if xmax_mask {
        poly = INF;
    }
    if xmin_mask {
        poly = 0.0;
    }
    poly
}

/// `fma_scalef_ps`: `poly * 2^quadrant` by adding the integer to the exponent
/// field — with the `quadrant <= -125` denormal branch, where the 2^-125
/// remainder is divided back in (`poly / 2^(quadrant + 125)`). The vector
/// version takes the branch when *any* lane needs it and blends per lane,
/// which is element-for-element the same as this per-element branch.
fn scalef(poly: f32, quadrant: f32) -> f32 {
    const MINQUADRANT: f32 = -125.0;
    if quadrant <= MINQUADRANT {
        // `_mm256_cvtps_epi32` (round to nearest even; quadrant is integral
        // here, so the cast is exact)
        let quad_diff = quad_diff(quadrant);
        let clamped = quadrant.max(MINQUADRANT);
        let scaled = f32::from_bits(poly.to_bits().wrapping_add((cvt_i32(clamped) << 23) as u32));
        scaled / (1u32 << quad_diff) as f32
    } else {
        f32::from_bits(
            poly.to_bits()
                .wrapping_add((cvt_i32(quadrant) << 23) as u32),
        )
    }
}

/// `quad_diff = 0.0 - (quadrant - (-125.0))`, i.e. how far into the denormal
/// range the quadrant sits; `cvtps_epi32` of it, exactly as the kernel.
fn quad_diff(quadrant: f32) -> i32 {
    let diff = 0.0 - (quadrant - (-125.0));
    cvt_i32(diff)
}

/// `_mm256_cvtps_epi32`: round to nearest even; `as i32` saturates like the
/// instruction, which the call sites' magnitudes never reach.
fn cvt_i32(x: f32) -> i32 {
    x.round_ties_even() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specials_take_the_mask_values() {
        assert_eq!(expf(0.0).to_bits(), 0x3F80_0000, "exp(0) == 1");
        assert_eq!(expf(-0.0).to_bits(), 0x3F80_0000, "exp(-0) == 1");
        assert_eq!(expf(f32::INFINITY).to_bits(), 0x7F80_0000);
        assert_eq!(expf(f32::NEG_INFINITY).to_bits(), 0x0000_0000, "-inf -> 0");
        assert_eq!(expf(f32::NAN).to_bits(), 0x7FC0_0000, "NaN -> NPY_NANF");
        assert_eq!(expf(XMAX).to_bits(), 0x7F80_0000, "xmax saturates");
        assert_eq!(expf(XMIN).to_bits(), 0x0000_0000, "xmin saturates");
    }
}
