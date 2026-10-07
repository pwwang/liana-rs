//! numpy 2.5.3's `f32` `log`: `simd_log_FLOAT`, the `SIMD_AVX2_FMA3` body of
//! `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src` (tag v2.5.3),
//! scalar-transcribed — mantissa/exponent bit-trick range reduction (with the
//! `FLT_MIN` denormal pre-scale by 2^100), the sqrt(1/2) normalisation, then
//! `exponent*ln2 + P/Q`. Constants: `npy_simd_data.h` (`NPY_COEFF_*_LOGf`)
//! and `npy_math.h` (`NPY_LOGE2f`, `NPY_SQRT1_2f`, `NPY_NANF`,
//! `NPY_INFINITYF`), all as raw `f32` bit patterns.

/// `NPY_SQRT1_2f` — the normalisation switch.
const SQRT1_2: f32 = f32::from_bits(0x3F35_04F3);
/// `NPY_LOGE2f`.
const LOGE2: f32 = f32::from_bits(0x3F31_7218);
/// `FLT_MIN` — below it the kernel pre-scales by 2^100.
const FLT_MIN: f32 = f32::from_bits(0x0080_0000);
/// 2^100, the denormal pre-scale (bit pattern `0x71800000` in the kernel).
const TWO_POWER_100: f32 = f32::from_bits(0x7180_0000);
const P: [f32; 6] = [
    f32::from_bits(0x0000_0000), // NPY_COEFF_P0_LOGf
    f32::from_bits(0x3F80_0000), // NPY_COEFF_P1_LOGf
    f32::from_bits(0x4007_361C), // NPY_COEFF_P2_LOGf
    f32::from_bits(0x3FBD_70A9), // NPY_COEFF_P3_LOGf
    f32::from_bits(0x3EC3_0333), // NPY_COEFF_P4_LOGf
    f32::from_bits(0x3CD4_2BCD), // NPY_COEFF_P5_LOGf
];
const Q: [f32; 6] = [
    f32::from_bits(0x3F80_0000), // NPY_COEFF_Q0_LOGf
    f32::from_bits(0x4027_361C), // NPY_COEFF_Q1_LOGf
    f32::from_bits(0x401C_FE0D), // NPY_COEFF_Q2_LOGf
    f32::from_bits(0x3F7C_8AE4), // NPY_COEFF_Q3_LOGf
    f32::from_bits(0x3E1E_5BF3), // NPY_COEFF_Q4_LOGf
    f32::from_bits(0x3BC0_83DF), // NPY_COEFF_Q5_LOGf
];
/// `NPY_NANF`, `-NPY_NANF`, `NPY_INFINITYF`, `-NPY_INFINITYF`.
const NAN: f32 = f32::from_bits(0x7FC0_0000);
const NEG_NAN: f32 = f32::from_bits(0xFFC0_0000);
const INF: f32 = f32::from_bits(0x7F80_0000);
const NEG_INF: f32 = f32::from_bits(0xFF80_0000);

/// `ln(x)` in `f32`, bit-identical to `np.log` on an `f32` array of this
/// (X86_V3) numpy build — including its fallbacks: `0.0` (either sign) gives
/// `-inf`, any negative gives `-NPY_NANF`, any NaN gives canonical
/// `NPY_NANF` (the payload is discarded), `+inf` gives `+inf`.
pub fn logf(x_in: f32) -> f32 {
    let negx_mask = x_in < 0.0;
    let zero_mask = x_in == 0.0;
    let inf_mask = x_in == INF;
    let nan_mask = x_in.is_nan();

    let mut x = if negx_mask { 0.0 } else { x_in };

    // range reduction: x -> normalised mantissa, `exponent` -> the power of 2
    let mut exponent = get_exponent(x);
    x = get_mantissa(x);

    // if x < sqrt(2) { exponent -= 1; x *= 2 } — mantissa into [sqrt(2)/2, sqrt(2))
    let sqrt2_mask = x <= SQRT1_2;
    x = if sqrt2_mask { x + x } else { x };
    exponent = if sqrt2_mask { exponent - 1.0 } else { exponent };

    x -= 1.0;

    let mut num = P[5].mul_add(x, P[4]);
    num = num.mul_add(x, P[3]);
    num = num.mul_add(x, P[2]);
    num = num.mul_add(x, P[1]);
    num = num.mul_add(x, P[0]);
    let mut den = Q[5].mul_add(x, Q[4]);
    den = den.mul_add(x, Q[3]);
    den = den.mul_add(x, Q[2]);
    den = den.mul_add(x, Q[1]);
    den = den.mul_add(x, Q[0]);
    let mut poly = num / den;
    poly = exponent.mul_add(LOGE2, poly);

    if nan_mask {
        poly = NAN;
    }
    if negx_mask {
        poly = NEG_NAN;
    }
    if zero_mask {
        poly = NEG_INF;
    }
    if inf_mask {
        poly = INF;
    }
    poly
}

/// `fma_get_exponent`: the unbiased exponent, for denormals via the 2^100
/// pre-scale and an extra -100. The two `FLT_MIN` comparisons are deliberately
/// not negations of each other — both are false for NaN, which the blend then
/// passes through unchanged, exactly as the vector body does.
fn get_exponent(x: f32) -> f32 {
    let denormal_mask = x < FLT_MIN;
    let normal_mask = x >= FLT_MIN;
    let temp1 = if normal_mask { 0.0 } else { x };
    let temp = temp1 * TWO_POWER_100;
    let x = if denormal_mask { temp } else { x };

    // (bits >> 23) - 0x7E, then to f32 — `srli_epi32`, `sub_epi32`,
    // `cvtepi32_ps` (exact for these magnitudes)
    let exponent = ((x.to_bits() >> 23) as i32 - 0x7E) as f32;
    let denorm_exponent = exponent - 100.0;
    if denormal_mask {
        denorm_exponent
    } else {
        exponent
    }
}

/// `fma_get_mantissa`: the 23 mantissa bits re-based to `[0.5, 1)` — for
/// denormals the 2^100 pre-scale leaves them as-is.
fn get_mantissa(x: f32) -> f32 {
    let denormal_mask = x < FLT_MIN;
    let normal_mask = x >= FLT_MIN;
    let temp1 = if normal_mask { 0.0 } else { x };
    let temp = temp1 * TWO_POWER_100;
    let x = if denormal_mask { temp } else { x };

    f32::from_bits((x.to_bits() & 0x7F_FFFF) | (126 << 23))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specials_take_the_mask_values() {
        assert_eq!(logf(1.0).to_bits(), 0x0000_0000, "ln(1) == 0");
        assert_eq!(logf(2.0).to_bits(), 0x3F31_7218, "ln(2) == NPY_LOGE2f");
        assert_eq!(logf(0.0).to_bits(), 0xFF80_0000, "ln(0) == -inf");
        assert_eq!(logf(-0.0).to_bits(), 0xFF80_0000, "ln(-0) == -inf");
        assert_eq!(logf(INF).to_bits(), 0x7F80_0000);
        assert_eq!(logf(NEG_INF).to_bits(), 0xFFC0_0000, "-inf -> -NPY_NANF");
        assert_eq!(logf(-1.0).to_bits(), 0xFFC0_0000, "negative -> -NPY_NANF");
        assert_eq!(logf(NAN).to_bits(), 0x7FC0_0000, "NaN -> NPY_NANF");
    }
}
