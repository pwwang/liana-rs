//! A bit-exact scalar port of scipy 1.18.1's `ndtr` — the standard normal CDF.
//!
//! `scipy.stats.norm.cdf` (which `liana`'s `_gene_cdf` calls,
//! `method/sc/_liana_pipe.py:755-767`) is `scipy.special.ndtr`, scipy's cephes
//! transcription (`scipy/special/xsf/cephes/ndtr.h`): `ndtr` splits at
//! `|a·√½| = 1` into `0.5 + 0.5·erf` and `0.5·erfc`, and `erf`/`erfc` are the
//! cephes rational polynomials — `polevl`/`p1evl` over the `T`/`U` (`erf`) and
//! `P`/`Q` vs `R`/`S` (`erfc`, switching at `|a| = 8`) coefficient sets.
//!
//! The only libm call inside is `erfc`'s `exp(-a·a)`; every other operation is
//! transcribed 1:1 — same constants (as raw bit patterns), same order, plain
//! `f64` arithmetic (Rust does not contract `a * x + c` into an FMA, matching
//! the scipy build the reference was dumped from). Bit-exactness is gated by
//! `tests/ndtr_parity.rs` against `testdata/math_ref/scipy_ndtr_ref.json` —
//! 100% match required, no tolerance.

// All coefficients are the cephes source literals (`ndtr.h`), carried as raw
// `f64` bit patterns; the decimal spelling from the header is above each.

/// cephes `erf`'s polynomial numerator coefficients —
/// `9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3,
/// 7.00332514112805075473E3, 5.55923013010394962768E4`.
const T: [f64; 5] = [
    f64::from_bits(0x402335BF_1E375D88),
    f64::from_bits(0x405681AA_4E9E067F),
    f64::from_bits(0x40A17002_BCB435B7),
    f64::from_bits(0x40BB5B53_3C72EF90),
    f64::from_bits(0x40EB2509_A44213DC),
];

/// cephes `erf`'s polynomial denominator coefficients (`p1evl`) —
/// `3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3,
/// 2.26290000613890934246E4, 4.92673942608635921086E4`.
const U: [f64; 5] = [
    f64::from_bits(0x4040C7E6_3FEFA6BA),
    f64::from_bits(0x40804ADD_14C63AEE),
    f64::from_bits(0x40B1F252_E680FD12),
    f64::from_bits(0x40D61940_01017C0A),
    f64::from_bits(0x40E80E6C_9DC8F567),
];

/// cephes `erfc`'s small-`|a|` numerator coefficients —
/// `2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0,
/// 4.86371970985681366614E1, 1.96520832956077098242E2, 5.26445194995477358631E2,
/// 9.34528527171957607540E2, 1.02755188689515710272E3, 5.57535335369399327526E2`.
const P: [f64; 9] = [
    f64::from_bits(0x3DF0EB24_A24F6479),
    f64::from_bits(0x3FE20DD7_46363488),
    f64::from_bits(0x401DDA53_DEC56DC4),
    f64::from_bits(0x4048518F_ACADBA66),
    f64::from_bits(0x406890AA_A9E020F7),
    f64::from_bits(0x4080738F_C264CF58),
    f64::from_bits(0x408D343A_6C7434D8),
    f64::from_bits(0x40900E35_21D6972A),
    f64::from_bits(0x40816C48_5DE8FFB3),
];

/// cephes `erfc`'s small-`|a|` denominator coefficients (`p1evl`) —
/// `1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2,
/// 9.75708501743205489753E2, 1.82390916687909736289E3, 2.24633760818710981792E3,
/// 1.65666309194161350182E3, 5.57535340817727675546E2`.
const Q: [f64; 8] = [
    f64::from_bits(0x402A74D5_FD7C23CC),
    f64::from_bits(0x4055AD42_FEE17365),
    f64::from_bits(0x40762F01_246F610D),
    f64::from_bits(0x408E7DAB_02F641D0),
    f64::from_bits(0x409C7FA2_FCA47151),
    f64::from_bits(0x40A18CAC_DAFAF4FF),
    f64::from_bits(0x4099E2A7_0192EDE2),
    f64::from_bits(0x40816C48_60C442D6),
];

/// cephes `erfc`'s large-`|a|` numerator coefficients —
/// `5.64189583547755073984E-1, 1.27536670759978104416E0, 5.01905042251180477414E0,
/// 6.16021097993053585195E0, 7.40974269950448939160E0, 2.97886665372100240670E0`.
const R: [f64; 6] = [
    f64::from_bits(0x3FE20DD7_50429B62),
    f64::from_bits(0x3FF467E6_EBB8C5A6),
    f64::from_bits(0x40141381_F436A71A),
    f64::from_bits(0x4018A40E_58DD0C0C),
    f64::from_bits(0x401DA393_9718960E),
    f64::from_bits(0x4007D4B8_0A470367),
];

/// cephes `erfc`'s large-`|a|` denominator coefficients (`p1evl`) —
/// `2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1,
/// 1.70814450747565897222E1, 9.60896809063285878198E0, 3.36907645100081516050E0`.
const S: [f64; 6] = [
    f64::from_bits(0x40021590_0917CE21),
    f64::from_bits(0x4022CAC5_21D84CFD),
    f64::from_bits(0x40281910_7F052C4D),
    f64::from_bits(0x403114D9_959C7FF5),
    f64::from_bits(0x402337CA_AA6326C1),
    f64::from_bits(0x400AF3DE_5AB62D90),
];

/// `MAXLOG` (`7.097827128933839730962063185871E2`): above `-a·a`, `exp`
/// underflows and `erfc` saturates.
const MAXLOG: f64 = f64::from_bits(0x40862E42_FEFA39EF);

/// `SQRTH` (`0.707106781186547524401`), the `ndtr` argument normalisation.
const SQRTH: f64 = f64::from_bits(0x3FE6A09E_667F3BCD);

/// `polevl`: `c[0]·x^(n-1) + … + c[n-1]`, Horner from the front.
fn polevl(x: f64, c: &[f64]) -> f64 {
    let mut ans = c[0];
    for &coefficient in &c[1..] {
        ans = ans * x + coefficient;
    }
    ans
}

/// `p1evl`: as [`polevl`] but with the implicit leading `x`.
fn p1evl(x: f64, c: &[f64]) -> f64 {
    let mut ans = x + c[0];
    for &coefficient in &c[1..] {
        ans = ans * x + coefficient;
    }
    ans
}

/// cephes `erf`.
fn erf(x: f64) -> f64 {
    if x < 0.0 {
        return -erf(-x);
    }
    if x > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &T) / p1evl(z, &U)
}

/// cephes `erfc`.
fn erfc(a: f64) -> f64 {
    let x = if a < 0.0 { -a } else { a };
    if x < 1.0 {
        return 1.0 - erf(a);
    }
    let z = -a * a;
    if z < -MAXLOG {
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    let z = z.exp();
    let (p, q) = if x < 8.0 {
        (polevl(x, &P), p1evl(x, &Q))
    } else {
        (polevl(x, &R), p1evl(x, &S))
    };
    let y = (z * p) / q;
    let y = if a < 0.0 { 2.0 - y } else { y };
    if y != 0.0 {
        return y;
    }
    if a < 0.0 { 2.0 } else { 0.0 }
}

/// The standard normal CDF, `P(Z <= x)` — scipy's `ndtr`/`norm.cdf`.
///
/// NaN inputs come back as the canonical quiet NaN (`0x7FF8000000000000`).
/// scipy's ufunc does not propagate the input's sign or payload — checked
/// over ±quiet/signalling NaNs with assorted payloads — and the sentinel
/// `f64::NAN` is that same bit pattern.
pub fn ndtr(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a * SQRTH;
    let z = x.abs();
    if z < 1.0 {
        0.5 + 0.5 * erf(x)
    } else {
        let y = 0.5 * erfc(z);
        if x > 0.0 { 1.0 - y } else { y }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The known values every port must reproduce, and the two saturation
    /// tails (`erfc`'s `2 - y` and the `-MAXLOG` cut).
    #[test]
    fn ndtr_hits_the_textbook_values() {
        assert_eq!(ndtr(0.0), 0.5);
        assert_eq!(ndtr(-0.0), 0.5);
        assert_eq!(ndtr(f64::INFINITY), 1.0);
        assert_eq!(ndtr(f64::NEG_INFINITY), 0.0);
        assert!(ndtr(f64::NAN).is_nan());
        assert!((ndtr(1.0) - 0.8413447460685429).abs() < 1e-15);
        assert!((ndtr(-1.0) - 0.15865525393145707).abs() < 1e-16);
        assert_eq!(ndtr(-40.0), 0.0, "the far negative tail saturates");
        assert_eq!(ndtr(40.0), 1.0);
    }
}
