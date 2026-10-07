//! A bit-exact scalar port of scipy 1.18.1's `betainc` — the incomplete beta
//! function behind `rank_aggregate`'s robust rank aggregation.
//!
//! `scipy.stats.beta.cdf(x, a, b)` (what liana's `_rho_scores` calls,
//! `liana/_core/_pipe_utils/_aggregate.py:212`) is `scipy.special.betainc`,
//! and in scipy 1.18.1 that ufunc is **Boost**, not cephes:
//! `scipy/special/functions.json` maps `betainc -> boost_special_functions.h++:
//! ibeta_double`, i.e. scipy's `ibeta_wrap` (`boost_special_functions.h:72`)
//! around `boost::math::detail::ibeta_imp`
//! (`subprojects/boost_math/.../beta.hpp:1198`).
//!
//! **Domain.** The RRA only ever asks for `a = j + 1`, `b = k - j + 1` over a
//! rank matrix `k` columns wide, so `a`, `b` are positive integers with
//! `a + b <= 8` and `x` is a normalised rank in `(1/440, 1)`. On that domain
//! `ibeta_imp` reaches exactly two branches: the `b == 1` closed form (with
//! `expm1`/`log1p`/`pow`, or `powm1` when `y >= 0.5`) and the integer binomial
//! branch `binomial_ccdf(b + a - 1, a - 1, x, 1 - x)` (`beta.hpp:1493`). The
//! series expansions (`min(a, b) <= 1`, the `0.5, 0.5` arcsine case, the
//! `a > max_factorial` paths) and `binomial_ccdf`'s own underflow leg cannot
//! be reached from RRA and are not ported; `debug_assert!` documents that.
//!
//! Every operation is transcribed 1:1 — same order, plain `f64` arithmetic
//! (Rust does not contract `a * x + c` into an FMA, matching the scipy build
//! the reference was dumped from). The libm calls inside (`pow`, `exp`,
//! `expm1`, `log1p`) go to the same glibc symbols boost and Rust both use:
//! boost's `expm1`/`log1p` for `double` defer to `std::expm1`/`std::log1p`,
//! which is Rust's `f64::exp_m1`/`f64::ln_1p`. Bit-exactness is gated by
//! `tests/betainc_parity.rs` against `testdata/math_ref/scipy_betainc_ref.json`
//! — 100% match required, no tolerance.

/// boost `powm1` (`powm1.hpp:27`): `x^y - 1` for the arguments `beta.hpp`
/// passes, which is the `pow` path — the `expm1` shortcut it guards is
/// unreachable once `b == 1` has sent `y >= 0.5` (there `|a·(x-1)| = a·y >= 1`).
fn powm1(x: f64, y: f64) -> f64 {
    if (y * (x - 1.0)).abs() < 0.5 || y.abs() < 0.2 {
        let l = y * x.ln();
        if l < 0.5 {
            // boost raises an overflow policy error for `l > log_max_value`;
            // unreachable from RRA, where `x <= 0.5` keeps `l < 0`.
            return l.exp_m1();
        }
    }
    x.powf(y) - 1.0
}

/// boost `binomial_ccdf` (`beta.hpp:1078`): `sum_{i>k} C(n, i)·x^i·y^(n-i)`,
/// the finite sum behind `ibeta_imp`'s integer branch.
///
/// `n` and `k` are integral `f64`s (`n = b + a - 1`, `k = a - 1`). The
/// underflow leg boost takes when `x^n <= min_value` is not ported: RRA's `x`
/// is a normalised rank, so `x^n >= (1/440)^7` here.
fn binomial_ccdf(n: f64, k: f64, x: f64, y: f64) -> f64 {
    let mut result = x.powf(n);
    debug_assert!(
        result > f64::MIN_POSITIVE,
        "binomial_ccdf underflow leg (x^n subnormal) is outside the RRA domain"
    );
    let mut term = result;
    let mut i = n - 1.0;
    while i > k {
        term *= ((i + 1.0) * y) / ((n - i) * x);
        result += term;
        i -= 1.0;
    }
    result
}

/// `boost::math::ibeta(a, b, x)` — the normalised incomplete beta function
/// `I_x(a, b)`, `ibeta_imp` with `inv = false, normalised = true`.
fn ibeta(a: f64, b: f64, x: f64) -> f64 {
    debug_assert!(
        a >= 1.0 && b >= 1.0 && a.fract() == 0.0 && b.fract() == 0.0,
        "outside the positive-integer RRA domain"
    );
    let mut a = a;
    let mut b = b;
    let mut x = x;
    let mut y = 1.0 - x;
    let mut invert = false;

    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    // (`a == 0.5 && b == 0.5`, the arcsine case, cannot be an integer pair.)
    if a == 1.0 {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut x, &mut y);
        invert = !invert;
    }
    if b == 1.0 {
        if a == 1.0 {
            return if invert { y } else { x };
        }
        return if y < 0.5 {
            let l = a * (-y).ln_1p();
            if invert { -l.exp_m1() } else { l.exp() }
        } else if invert {
            -powm1(x, a)
        } else {
            x.powf(a)
        };
    }

    // Both a, b > 1: break the symmetry toward the smaller tail, then use the
    // binomial relation (b < 40 and both shapes integral, always true here).
    let lambda = if a < b {
        a - (a + b) * x
    } else {
        (a + b) * y - b
    };
    if lambda < 0.0 {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut x, &mut y);
        invert = !invert;
    }
    let k = a - 1.0;
    let n = b + k;
    let fract = binomial_ccdf(n, k, x, y);
    if invert { 1.0 - fract } else { fract }
}

/// The regularised incomplete beta function `I_x(a, b)` — scipy's `betainc`
/// / `scipy.stats.beta.cdf`.
///
/// The guards are scipy's `ibeta_wrap`: NaN first, then the domain check, then
/// the `(a, b) -> (0, 0)` / `(inf, inf)` limits that stay indeterminate, and
/// the degenerate-beta limits that collapse to a point mass at an endpoint.
/// NaN inputs and indeterminate limits come back as the canonical quiet NaN
/// (`0x7FF8000000000000`), which `f64::NAN` already is — scipy's ufunc does
/// not carry the input's sign or payload (see `ndtr`'s note).
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a < 0.0 || b < 0.0 || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if (a == 0.0 && b == 0.0) || (a.is_infinite() && b.is_infinite()) {
        return f64::NAN;
    }
    if a == 0.0 || b.is_infinite() {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    if b == 0.0 || a.is_infinite() {
        return if x < 1.0 { 0.0 } else { 1.0 };
    }
    ibeta(a, b, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed forms for the shapes RRA uses, plus the wrapper's limits.
    #[test]
    fn betainc_hits_the_closed_forms() {
        assert_eq!(betainc(1.0, 1.0, 0.25), 0.25, "I_x(1, 1) = x");
        assert_eq!(betainc(2.0, 1.0, 0.25), 0.0625, "I_x(2, 1) = x^2");
        assert_eq!(betainc(1.0, 2.0, 0.25), 0.4375, "I_x(1, 2) = 1 - (1 - x)^2");
        assert_eq!(betainc(2.0, 3.0, 1.0), 1.0);
        assert_eq!(betainc(2.0, 3.0, 0.0), 0.0);
        assert_eq!(betainc(0.0, 1.0, 0.5), 1.0);
        assert_eq!(betainc(1.0, 0.0, 0.5), 0.0);
        assert!(betainc(0.0, 0.0, 0.5).is_nan());
        assert!(betainc(1.0, 1.0, 1.5).is_nan());
        assert!(betainc(f64::NAN, 1.0, 0.5).is_nan());
        // I_0.5(2, 2) = 0.5 by symmetry of Beta(2, 2) about x = 0.5
        assert_eq!(betainc(2.0, 2.0, 0.5), 0.5);
    }
}
