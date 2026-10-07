//! A bit-exact scalar port of numpy 2.5.3's `f32` pairwise reductions —
//! `np.sum` and `np.std` on a contiguous `f32` array.
//!
//! numpy does not fold such an array sequentially: the add loop is
//! `FLOAT_pairwise_sum` (`numpy/_core/src/umath/loops_utils.h.src`), eight
//! lane accumulators over blocks of 128, recursing on halves above that, and
//! its SIMD dispatch (`loops_arithm_fp.dispatch.c.src`) keeps the same shape
//! with zero-initialised lanes — so the lane accumulators start at `+0.0`,
//! the `n < 8` tail is a plain sequential sum from `+0.0`, and an empty
//! reduction is the `+0.0` identity. `np.std` (`_methods.py::_var`, `ddof=0`)
//! is that sum twice — the `f32` mean, then the `f32` sum of squared
//! deviations — divided by `n` in `f32` and rooted.
//!
//! Both are gated by `tests/pairwise_parity.rs` against
//! `testdata/math_ref/numpy_pairwise_ref.json` — 100% match required.

/// `np.add.reduce` over a contiguous `f32` array.
pub fn sum_f32(values: &[f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0; // the reduce's identity, before the loop is entered
    }
    if n < 8 {
        let mut result = 0.0f32;
        for &value in values {
            result += value;
        }
        return result;
    }
    if n <= 128 {
        let mut lanes = [0.0f32; 8];
        let mut index = 0;
        while index < n - (n % 8) {
            for (lane, &value) in lanes.iter_mut().zip(&values[index..index + 8]) {
                *lane += value;
            }
            index += 8;
        }
        let mut result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        while index < n {
            result += values[index];
            index += 1;
        }
        return result;
    }
    let mut half = n / 2;
    half -= half % 8;
    sum_f32(&values[..half]) + sum_f32(&values[half..])
}

/// `np.std` of a contiguous `f32` array, `ddof=0` — the `f32` mean, the
/// squared deviations' `f32` sum, `/ n` and `sqrt` in `f32`.
///
/// A zero-length array is undefined here (numpy's `0/0` NaN), and the callers
/// never pass one: every cluster block holds at least one cell.
pub fn std_f32(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    let mean = sum_f32(values) / n;
    let squares: Vec<f32> = values
        .iter()
        .map(|&value| (value - mean) * (value - mean))
        .collect();
    (sum_f32(&squares) / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The branch structure, on hand-checkable values: the sequential tail,
    /// the six-of-eight block tail, and a `-0.0` block's lane init.
    #[test]
    fn sum_f32_walks_the_pairwise_branches() {
        assert_eq!(sum_f32(&[]), 0.0);
        assert_eq!(sum_f32(&[1.5]), 1.5);
        assert_eq!(sum_f32(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(sum_f32(&[2.0; 8]), 16.0);
        // 8 lanes then a tail of 3: (2+2+2) per lane is exact
        assert_eq!(sum_f32(&[2.0; 11]), 22.0);
        assert_eq!(sum_f32(&[2.0; 200]), 400.0);
        // every lane starts at +0.0, so an all-(-0.0) block sums to +0.0
        assert_eq!(sum_f32(&[-0.0; 16]).to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn std_f32_uses_the_pairwise_mean_and_squares() {
        assert_eq!(std_f32(&[3.5]), 0.0);
        assert_eq!(std_f32(&[1.0, 3.0]), 1.0);
        // population standard deviation, ddof=0: sqrt(2/3) for {0, 1, 2}
        assert_eq!(std_f32(&[0.0, 1.0, 2.0]), (2.0f32 / 3.0).sqrt());
    }
}
