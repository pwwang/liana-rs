//! Bit-exact port of the permutation stream liana draws from `numpy.random`.
//!
//! The chain is: `SeedSequence(seed)` pool mixing → `PCG64` seeding → numpy's
//! `random_interval` masked rejection over `PCG64.next_uint32` → Fisher–Yates
//! as `_shuffle_raw` runs it. Faithful to numpy 2.5.3, which is where the
//! reference dumps in `testdata/rng_ref/` come from; the corresponding numpy
//! sources are `numpy/random/bit_generator.pyx` (SeedSequence),
//! `numpy/random/_pcg64.pyx` + `src/pcg64/pcg64.h` (PCG64), and
//! `numpy/random/_generator.pyx` + `src/distributions/distributions.c`
//! (`_shuffle_raw` / `random_interval`).

/// Mixing constants from `numpy/random/bit_generator.pyx`.
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

/// `PCG_DEFAULT_MULTIPLIER_128` from `numpy/random/src/pcg64/pcg64.h`.
const PCG64_MULTIPLIER: u128 = ((2_549_297_995_355_413_924_u128) << 64) | 4_865_540_595_714_422_341;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let mut result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

/// `SeedSequence(seed).pool` — the 4-word mixed entropy pool.
pub fn seed_sequence_pool(seed: u64) -> [u32; 4] {
    // `_coerce_to_uint32_array`: little-endian words, and seed 0 is one zero word.
    let mut entropy = Vec::new();
    let mut remaining = seed;
    while remaining > 0 {
        entropy.push(remaining as u32);
        remaining >>= 32;
    }
    if entropy.is_empty() {
        entropy.push(0);
    }

    let mut mixer = [0_u32; 4];
    let mut hash_const = INIT_A;
    for (i, slot) in mixer.iter_mut().enumerate() {
        let word = entropy.get(i).copied().unwrap_or(0);
        *slot = hashmix(word, &mut hash_const);
    }
    // Mix all bits together so late bits can affect earlier bits.
    for i_src in 0..mixer.len() {
        for i_dst in 0..mixer.len() {
            if i_src != i_dst {
                let src = mixer[i_src];
                mixer[i_dst] = mix(mixer[i_dst], hashmix(src, &mut hash_const));
            }
        }
    }
    // Add any remaining entropy, mixing each new word with each pool word.
    for &word in entropy.iter().skip(mixer.len()) {
        for slot in mixer.iter_mut() {
            *slot = mix(*slot, hashmix(word, &mut hash_const));
        }
    }
    mixer
}

/// `SeedSequence(seed).generate_state(n_words, uint64)` as raw little-endian words.
pub fn seed_sequence_state_u64(seed: u64, n_words: usize) -> Vec<u64> {
    let pool = seed_sequence_pool(seed);
    let mut hash_const = INIT_B;
    let words32: Vec<u32> = (0..n_words * 2)
        .map(|i| {
            let mut data = pool[i % pool.len()] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data = data.wrapping_mul(hash_const);
            data ^ (data >> XSHIFT)
        })
        .collect();
    words32
        .chunks_exact(2)
        .map(|w| (w[0] as u64) | (w[1] as u64) << 32)
        .collect()
}

/// `PCG64` seeded the way `numpy/random/_pcg64.pyx` seeds it from a `SeedSequence`.
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    pub fn from_seed(seed: u64) -> Self {
        // `_pcg64.pyx`: val = seed_seq.generate_state(4, uint64);
        // pcg64_set_seed(&seed[0], &inc[2]).
        let words = seed_sequence_state_u64(seed, 4);
        let initstate = ((words[0] as u128) << 64) | words[1] as u128;
        let initseq = ((words[2] as u128) << 64) | words[3] as u128;
        let mut rng = Self {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    /// `pcg_setseq_128_step_r`: the LCG step.
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG64_MULTIPLIER)
            .wrapping_add(self.inc);
    }

    /// `pcg_output_xsl_rr_128_64`: xor the halves, then rotate by the top 6 bits.
    fn output(state: u128) -> u64 {
        let xsl = ((state >> 64) as u64) ^ (state as u64);
        xsl.rotate_right((state >> 122) as u32)
    }

    /// `pcg64_next64`: step, then `xsl_rr` output.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        Self::output(self.state)
    }

    /// `pcg64_next32`: low 32 bits, caching the high 32 bits for the next call.
    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        next as u32
    }

    /// `random_interval(max)`: uniform in `[0, max]` by masked rejection.
    pub fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        if max <= u32::MAX as u64 {
            loop {
                let value = (self.next_u32() as u64) & mask;
                if value <= max {
                    return value;
                }
            }
        } else {
            loop {
                let value = self.next_u64() & mask;
                if value <= max {
                    return value;
                }
            }
        }
    }

    /// `_shuffle_raw`: Fisher–Yates, `i` from `n-1` down to 1.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.random_interval(i as u64) as usize;
            x.swap(i, j);
        }
    }
}

/// The permutation stream `_chunk_permutations` draws, yielded a block at a
/// time so a consumer that takes the permutations in order never holds more
/// than one block.
///
/// The indices are `u32` (`numpy.min_scalar_type(n_obs - 1)`'s `uint32` for
/// `n_obs > 65536`; the values are the same dtype-independent stream below it —
/// `tests/rng_parity.rs` pins the byte-level agreement against the `uint16`
/// reference dumps).
pub struct PermsStream {
    rng: Pcg64,
    template: Vec<u32>,
    idx: Vec<u32>,
}

impl PermsStream {
    pub fn new(seed: u64, n_obs: usize) -> Self {
        let template: Vec<u32> = (0..n_obs as u32).collect();
        Self {
            rng: Pcg64::from_seed(seed),
            idx: template.clone(),
            template,
        }
    }

    /// The next `block` permutations, `(block, n_obs)` row-major `u32`.
    pub fn next_block(&mut self, block: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.template.len() * block);
        for _ in 0..block {
            self.idx.copy_from_slice(&self.template);
            self.rng.shuffle(&mut self.idx);
            out.extend_from_slice(&self.idx);
        }
        out
    }
}

/// The `(n_perms, n_obs)` permutation matrix `_chunk_permutations` yields.
pub fn permutation_matrix(seed: u64, n_obs: usize, n_perms: usize) -> Vec<u32> {
    PermsStream::new(seed, n_obs).next_block(n_perms)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vectors dumped from the pinned oracle (numpy 2.5.3), e.g.
    /// `np.random.SeedSequence(0).pool`.
    #[test]
    fn seed_sequence_pool_matches_numpy() {
        assert_eq!(
            seed_sequence_pool(0),
            [0xfe40_eb07, 0x4f36_3a36, 0x4eb2_009d, 0xc89a_7aa7]
        );
        assert_eq!(
            seed_sequence_pool(1),
            [0xa137_e185, 0x0de2_fab1, 0x950e_ae78, 0x54dd_0a81]
        );
        assert_eq!(
            seed_sequence_pool(1337),
            [0x31db_188b, 0x7500_096e, 0xdad4_311a, 0x69ec_0325]
        );
    }

    /// `np.random.SeedSequence(s).generate_state(4, np.uint64)`.
    #[test]
    fn generate_state_matches_numpy() {
        assert_eq!(
            seed_sequence_state_u64(0, 4),
            [
                0xdb2c_d7e7_b0f4_78be,
                0xabf4_641a_2c71_ba49,
                0x20c6_ed6d_9d7b_8d41,
                0x2c40_99de_223c_39d4
            ]
        );
        assert_eq!(
            seed_sequence_state_u64(1, 4),
            [
                0x672d_8ee5_6d67_91ff,
                0x8ae1_9ca1_4eb1_072c,
                0x4915_796d_1322_fc4a,
                0xd0cc_2bdc_aba0_49bd
            ]
        );
        assert_eq!(
            seed_sequence_state_u64(1337, 4),
            [
                0xbc7e_664d_90aa_98cc,
                0x83e7_a3f3_04ca_3335,
                0x959b_a968_f316_3f00,
                0x08b7_0893_38cc_d3d3
            ]
        );
    }

    /// `np.random.PCG64(s).state` and `.random_raw(4)`.
    #[test]
    fn pcg64_stream_matches_numpy() {
        let mut rng = Pcg64::from_seed(0);
        assert_eq!(rng.state, 0x1aa1_b534_5996_452d_0958_5eb7_a695_61e3);
        assert_eq!(rng.inc, 0x418d_dadb_3af7_1a82_5881_33bc_4478_73a9);
        assert_eq!(
            [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64()
            ],
            [
                0xa30f_ebcf_d9c2_825f,
                0x4510_bdf8_82d9_d721,
                0x0a7d_3da9_4ecd_e8b8,
                0x043b_27b6_1342_f01d
            ]
        );

        let rng = Pcg64::from_seed(1337);
        assert_eq!(rng.state, 0x5803_c055_a0ea_2797_5153_d738_5a23_0cf3);
        assert_eq!(rng.inc, 0x2b37_52d1_e62c_7e00_116e_1126_7199_a7a7);
    }

    /// `np.random.default_rng(s).permutation(np.arange(n))`.
    #[test]
    fn shuffle_matches_numpy() {
        assert_eq!(
            permutation_matrix(0, 10, 1),
            vec![4_u32, 6, 2, 7, 3, 5, 9, 0, 8, 1]
        );
        assert_eq!(
            permutation_matrix(1337, 10, 1),
            vec![7_u32, 6, 3, 5, 2, 8, 4, 9, 0, 1]
        );
        assert_eq!(permutation_matrix(1, 5, 1), vec![4_u32, 0, 1, 2, 3]);
        // a block-wise draw is the same stream, whatever the block size
        let mut stream = PermsStream::new(0, 10);
        let mut blocked = stream.next_block(1);
        blocked.extend(stream.next_block(2));
        assert_eq!(blocked, permutation_matrix(0, 10, 3));
        assert_eq!(
            permutation_matrix(0, 128, 1)[..24].to_vec(),
            vec![
                125_u32, 53, 102, 34, 121, 117, 66, 74, 64, 112, 88, 37, 71, 1, 13, 80, 11, 43, 16,
                5, 93, 124, 107, 105
            ]
        );
    }
}
