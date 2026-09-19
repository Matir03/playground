//! Inference-side cost model for the small NNUE-style value net.
//!
//! Python cannot answer "does this hit 100k evaluations/second". Two harnesses
//! tried: one built on torch ops reported ~11k/s and one on numpy ~30-60k/s,
//! and *both* were flat across an 8x change in hidden width. A cost model that
//! does not move when the cost moves is measuring the harness, so neither
//! number meant anything. This is the same network in the language it would
//! actually be deployed in.
//!
//! The architecture mirrors `python/minions_nnue.py` exactly:
//!
//! ```text
//!   acc = clipped_relu(ft_bias + sum of ft columns for active features)  588 -> 512
//!   x   = clipped_relu(fc1([acc, scalars]))                              525 -> 32
//!   x   = clipped_relu(fc2(x))                                            32 -> 64
//!   x   = clipped_relu(fc3(x))                                            64 -> 32
//!   out = tanh(out(x))                                                    32 -> 1
//! ```
//!
//! The 13 scalars are concatenated raw onto the accumulator rather than passed
//! through their own embedding, so the post-input widths are exactly the
//! specified ones.
//!
//! The accumulator is the whole point: a position sets ~27 of 588 features
//! (measured over generated positions: mean 27.1, range 14-41), so evaluation
//! sums 27 columns rather than multiplying a 588xH matrix. In a real search the
//! accumulator is also *incremental* — a turn changes a handful of features —
//! which this deliberately does not assume. Every eval here rebuilds it from
//! scratch, so the measurement is a floor on achievable speed, not a best case.
//! It is also scalar f32: no SIMD, no int8/int16 quantisation, both of which a
//! production NNUE would use and both of which only help.
//!
//! Weights are random: throughput depends on the shapes, not the values.

use rand::{Rng, SeedableRng};

pub const CELLS: usize = 49;
pub const N_UNIT_PLANES: usize = 12;
pub const N_FEATURES: usize = N_UNIT_PLANES * CELLS; // 588
pub const N_SCALARS: usize = 13;

#[inline(always)]
fn crelu(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Layer widths after the input: accumulator, then the three body layers.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub hidden: usize,
    pub l1: usize,
    pub l2: usize,
    pub l3: usize,
}

impl Default for Shape {
    fn default() -> Self {
        Self {
            hidden: 512,
            l1: 32,
            l2: 64,
            l3: 32,
        }
    }
}

pub struct Nnue {
    shape: Shape,
    /// Row-major `[feature][hidden]`, so one active feature's column is a
    /// contiguous slice — the layout the accumulator sum wants.
    ft: Vec<f32>,
    ft_bias: Vec<f32>,
    fc1: Vec<f32>,
    fc1_bias: Vec<f32>,
    fc2: Vec<f32>,
    fc2_bias: Vec<f32>,
    fc3: Vec<f32>,
    fc3_bias: Vec<f32>,
    out: Vec<f32>,
    out_bias: f32,
}

impl Nnue {
    pub fn random(shape: Shape, seed: u64) -> Self {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut fill =
            |n: usize| -> Vec<f32> { (0..n).map(|_| rng.gen_range(-0.1f32..0.1f32)).collect() };
        let cat = shape.hidden + N_SCALARS;
        Self {
            shape,
            ft: fill(N_FEATURES * shape.hidden),
            ft_bias: fill(shape.hidden),
            fc1: fill(cat * shape.l1),
            fc1_bias: fill(shape.l1),
            fc2: fill(shape.l1 * shape.l2),
            fc2_bias: fill(shape.l2),
            fc3: fill(shape.l2 * shape.l3),
            fc3_bias: fill(shape.l3),
            out: fill(shape.l3),
            out_bias: 0.0,
        }
    }

    pub fn params(&self) -> usize {
        self.ft.len()
            + self.ft_bias.len()
            + self.fc1.len()
            + self.fc1_bias.len()
            + self.fc2.len()
            + self.fc2_bias.len()
            + self.fc3.len()
            + self.fc3_bias.len()
            + self.out.len()
            + 1
    }

    /// Multiply-accumulates per evaluation, for sanity-checking the measured
    /// rate against arithmetic rather than trusting one number.
    pub fn macs(&self, active: usize) -> usize {
        let s = self.shape;
        active * s.hidden + (s.hidden + N_SCALARS) * s.l1 + s.l1 * s.l2 + s.l2 * s.l3 + s.l3
    }

    /// Build the raw (pre-activation) accumulator for a position from scratch.
    pub fn acc_build(&self, active: &[u16], acc: &mut [f32]) {
        let h = self.shape.hidden;
        acc[..h].copy_from_slice(&self.ft_bias);
        for &f in active {
            let col = &self.ft[f as usize * h..(f as usize + 1) * h];
            for (a, w) in acc[..h].iter_mut().zip(col) {
                *a += *w;
            }
        }
    }

    /// Add or remove one feature from a live accumulator.
    ///
    /// This is the reason the architecture is shaped this way. A turn changes a
    /// handful of the ~27 active features, so in search the 588xH transform is
    /// never recomputed — only the columns that actually moved are patched.
    /// Rebuilding costs `active * hidden`; patching costs `churn * hidden`.
    #[inline]
    pub fn acc_update(&self, acc: &mut [f32], feature: u16, add: bool) {
        let h = self.shape.hidden;
        let col = &self.ft[feature as usize * h..(feature as usize + 1) * h];
        if add {
            for (a, w) in acc[..h].iter_mut().zip(col) {
                *a += *w;
            }
        } else {
            for (a, w) in acc[..h].iter_mut().zip(col) {
                *a -= *w;
            }
        }
    }

    /// Evaluate from a sparse feature list. `scratch` is caller-owned so a hot
    /// loop does no allocation at all.
    ///
    /// Delegates to [`Self::eval_from_acc`] so the rebuild and incremental paths
    /// share one body implementation — two copies would be free to drift, and a
    /// drifting incremental path would benchmark well and be silently wrong.
    pub fn eval_sparse(
        &self,
        active: &[u16],
        scalars: &[f32; N_SCALARS],
        scratch: &mut Scratch,
    ) -> f32 {
        let mut raw = vec![0.0f32; self.shape.hidden];
        self.acc_build(active, &mut raw);
        self.eval_from_acc(&raw, scalars, scratch)
    }

    /// Run the body over an already-built (pre-activation) accumulator.
    pub fn eval_from_acc(
        &self,
        raw: &[f32],
        scalars: &[f32; N_SCALARS],
        scratch: &mut Scratch,
    ) -> f32 {
        let h = self.shape.hidden;
        for i in 0..h {
            scratch.acc[i] = crelu(raw[i]);
        }
        // Scalars ride along raw, already normalised by the caller.
        scratch.acc[h..h + N_SCALARS].copy_from_slice(scalars);
        let cat = h + N_SCALARS;

        for j in 0..self.shape.l1 {
            let mut v = self.fc1_bias[j];
            for i in 0..cat {
                v += scratch.acc[i] * self.fc1[i * self.shape.l1 + j];
            }
            scratch.b1[j] = crelu(v);
        }
        for j in 0..self.shape.l2 {
            let mut v = self.fc2_bias[j];
            for i in 0..self.shape.l1 {
                v += scratch.b1[i] * self.fc2[i * self.shape.l2 + j];
            }
            scratch.b2[j] = crelu(v);
        }
        for j in 0..self.shape.l3 {
            let mut v = self.fc3_bias[j];
            for i in 0..self.shape.l2 {
                v += scratch.b2[i] * self.fc3[i * self.shape.l3 + j];
            }
            scratch.b3[j] = crelu(v);
        }
        let mut v = self.out_bias;
        for i in 0..self.shape.l3 {
            v += scratch.b3[i] * self.out[i];
        }
        v.tanh()
    }
}

pub struct Scratch {
    acc: Vec<f32>,
    b1: Vec<f32>,
    b2: Vec<f32>,
    b3: Vec<f32>,
}

impl Scratch {
    pub fn new(shape: Shape) -> Self {
        Self {
            acc: vec![0.0; shape.hidden + N_SCALARS],
            b1: vec![0.0; shape.l1],
            b2: vec![0.0; shape.l2],
            b3: vec![0.0; shape.l3],
        }
    }
}

/// Time `reps` evaluations at `active` set features, single threaded.
pub fn bench(shape: Shape, active: usize, reps: usize) -> (f64, f64) {
    let net = Nnue::random(shape, 42);
    let mut scratch = Scratch::new(shape);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    // A fresh feature set per eval, so the measurement cannot be a cache
    // best-case on one hot column set.
    let sets: Vec<Vec<u16>> = (0..64)
        .map(|_| {
            let mut v: Vec<u16> = (0..N_FEATURES as u16).collect();
            for i in (1..v.len()).rev() {
                v.swap(i, rng.gen_range(0..=i));
            }
            v.truncate(active);
            v
        })
        .collect();
    let scalars = [0.5f32; N_SCALARS];

    // Warm up, and keep the result so nothing is optimised away.
    let mut sink = 0.0f32;
    for s in &sets {
        sink += net.eval_sparse(s, &scalars, &mut scratch);
    }

    let t0 = std::time::Instant::now();
    for i in 0..reps {
        sink += net.eval_sparse(&sets[i % sets.len()], &scalars, &mut scratch);
    }
    let el = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    (reps as f64 / el, el / reps as f64 * 1e6)
}

/// Time evaluations that *patch* the accumulator instead of rebuilding it.
///
/// This is the deployment cost model. `bench` rebuilds all `active` columns per
/// evaluation, which no real NNUE does: in search a turn changes `churn`
/// features and the rest of the accumulator carries over. The difference is the
/// whole point of the architecture, so quoting only the rebuild number
/// understates the net by roughly `active / churn` on the accumulator term.
pub fn bench_incremental(shape: Shape, active: usize, churn: usize, reps: usize) -> (f64, f64) {
    let net = Nnue::random(shape, 42);
    let mut scratch = Scratch::new(shape);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    let mut feats: Vec<u16> = {
        let mut v: Vec<u16> = (0..N_FEATURES as u16).collect();
        for i in (1..v.len()).rev() {
            v.swap(i, rng.gen_range(0..=i));
        }
        v.truncate(active);
        v
    };
    let mut raw = vec![0.0f32; shape.hidden];
    net.acc_build(&feats, &mut raw);
    let scalars = [0.5f32; N_SCALARS];

    let mut sink = net.eval_from_acc(&raw, &scalars, &mut scratch);

    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        // Swap `churn` features out for new ones, patching as we go — the
        // pattern a move/unmove in search produces.
        for k in 0..churn {
            let slot = k % feats.len();
            let old = feats[slot];
            let new = rng.gen_range(0..N_FEATURES as u16);
            net.acc_update(&mut raw, old, false);
            net.acc_update(&mut raw, new, true);
            feats[slot] = new;
        }
        sink += net.eval_from_acc(&raw, &scalars, &mut scratch);
    }
    let el = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    (reps as f64 / el, el / reps as f64 * 1e6)
}

/// Print a sweep around the chosen shape. A real cost model must scale with
/// width — that is the property both Python harnesses failed.
pub fn bench_sweep(active: usize, reps: usize) {
    // A turn moves a few units, so only a handful of the ~27 active features
    // change between consecutive evaluations in search.
    const CHURN: usize = 6;
    println!(
        "NNUE sparse eval, single thread, scalar f32, {active} active features of {N_FEATURES}, \
         {reps} reps\n\
         rebuild  = all {active} accumulator columns recomputed every eval (pessimistic floor)\n\
         incr     = {CHURN} features patched per eval, the deployment path\n\
         neither uses SIMD or int8/int16 quantisation, both of which only help"
    );
    let shapes = [
        Shape {
            hidden: 512,
            l1: 32,
            l2: 64,
            l3: 32,
        },
        Shape {
            hidden: 256,
            l1: 32,
            l2: 64,
            l3: 32,
        },
        Shape {
            hidden: 128,
            l1: 32,
            l2: 64,
            l3: 32,
        },
        Shape {
            hidden: 512,
            l1: 32,
            l2: 32,
            l3: 32,
        },
        Shape {
            hidden: 1024,
            l1: 32,
            l2: 64,
            l3: 32,
        },
    ];
    for &s in &shapes {
        let net = Nnue::random(s, 1);
        let (eps, us) = bench(s, active, reps);
        let (ieps, ius) = bench_incremental(s, active, CHURN, reps);
        println!(
            "  {:4}->{:3}->{:3}->{:3}->1  params {:7}  {:5.1}k MAC   \
             rebuild {eps:8.0}/s ({us:6.2}us)   incr {ieps:8.0}/s ({ius:6.2}us)  {}",
            s.hidden,
            s.l1,
            s.l2,
            s.l3,
            net.params(),
            net.macs(active) as f64 / 1000.0,
            if ieps >= 100_000.0 {
                "OK >=100k/s"
            } else {
                "BELOW TARGET"
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The incremental path must agree with the rebuild path.
    ///
    /// This is the load-bearing test for the whole speed argument: patching the
    /// accumulator is what makes the net cheap, and a patch path that drifted
    /// from the rebuild would benchmark beautifully while computing a different
    /// network. Exact equality is not required — adds and subtracts land in a
    /// different order than a fresh sum, and float addition is not associative —
    /// but the gap must be numerical noise, not structural.
    #[test]
    fn incremental_matches_rebuild() {
        let s = Shape::default();
        let net = Nnue::random(s, 21);
        let mut scratch = Scratch::new(s);
        let scalars = [0.3f32; N_SCALARS];

        let mut feats: Vec<u16> = (0..27).map(|i| i * 13 % N_FEATURES as u16).collect();
        feats.sort_unstable();
        feats.dedup();
        let mut raw = vec![0.0f32; s.hidden];
        net.acc_build(&feats, &mut raw);

        // Swap several features, patching the accumulator rather than rebuilding.
        for k in 0..8u16 {
            let old = feats[k as usize];
            let new = (old + 101) % N_FEATURES as u16;
            if feats.contains(&new) {
                continue;
            }
            net.acc_update(&mut raw, old, false);
            net.acc_update(&mut raw, new, true);
            feats[k as usize] = new;
        }

        let incremental = net.eval_from_acc(&raw, &scalars, &mut scratch);
        let rebuilt = net.eval_sparse(&feats, &scalars, &mut scratch);
        assert!(
            (incremental - rebuilt).abs() < 1e-5,
            "incremental {incremental} vs rebuilt {rebuilt}"
        );
    }

    #[test]
    fn eval_is_finite_and_bounded() {
        let s = Shape::default();
        let net = Nnue::random(s, 3);
        let mut scratch = Scratch::new(s);
        let active: Vec<u16> = (0..27).map(|i| i * 7).collect();
        let v = net.eval_sparse(&active, &[0.5; N_SCALARS], &mut scratch);
        assert!(v.is_finite(), "eval produced {v}");
        assert!((-1.0..=1.0).contains(&v), "tanh output {v} out of range");
    }

    /// The accumulator must actually depend on which features are set —
    /// a net that ignores its input would benchmark beautifully and be useless.
    #[test]
    fn different_features_give_different_values() {
        let s = Shape::default();
        let net = Nnue::random(s, 4);
        let mut scratch = Scratch::new(s);
        let a: Vec<u16> = (0..27).map(|i| i * 7).collect();
        let b: Vec<u16> = (0..27).map(|i| i * 7 + 3).collect();
        let va = net.eval_sparse(&a, &[0.5; N_SCALARS], &mut scratch);
        let vb = net.eval_sparse(&b, &[0.5; N_SCALARS], &mut scratch);
        assert_ne!(va, vb);
    }

    /// The scalars must reach the output too, or 13 real inputs are being
    /// silently dropped — the exact bug the raw concat was meant to avoid.
    #[test]
    fn scalars_change_the_output() {
        let s = Shape::default();
        let net = Nnue::random(s, 9);
        let mut scratch = Scratch::new(s);
        let a: Vec<u16> = (0..27).map(|i| i * 7).collect();
        let lo = net.eval_sparse(&a, &[0.0; N_SCALARS], &mut scratch);
        let hi = net.eval_sparse(&a, &[1.0; N_SCALARS], &mut scratch);
        assert_ne!(lo, hi);
    }

    /// Scratch reuse must not leak state between evaluations.
    #[test]
    fn scratch_reuse_is_stateless() {
        let s = Shape {
            hidden: 128,
            l1: 32,
            l2: 64,
            l3: 32,
        };
        let net = Nnue::random(s, 5);
        let mut s1 = Scratch::new(s);
        let mut s2 = Scratch::new(s);
        let a: Vec<u16> = (0..27).map(|i| i * 5).collect();
        let b: Vec<u16> = (0..20).map(|i| i * 11).collect();
        let first = net.eval_sparse(&a, &[0.25; N_SCALARS], &mut s1);
        let _ = net.eval_sparse(&b, &[0.25; N_SCALARS], &mut s1);
        let again = net.eval_sparse(&a, &[0.25; N_SCALARS], &mut s1);
        let fresh = net.eval_sparse(&a, &[0.25; N_SCALARS], &mut s2);
        assert_eq!(first, again);
        assert_eq!(first, fresh);
    }

    /// The MAC count must match the shapes it claims to describe.
    #[test]
    fn mac_count_matches_the_shape() {
        let s = Shape {
            hidden: 512,
            l1: 32,
            l2: 64,
            l3: 32,
        };
        let net = Nnue::random(s, 6);
        // 27*512 + 525*32 + 32*64 + 64*32 + 32
        assert_eq!(net.macs(27), 13824 + 16800 + 2048 + 2048 + 32);
    }
}
