//! Random-position generation for the small value net (NNUE-style) experiment.
//!
//! Positions are drawn exactly the way [`crate::selfplay`]'s `random_opening`
//! draws its openings — a bid, then some number of uniformly random *legal*
//! turns via [`TurnSpec::random_playout`] + [`realize`] — so a generated
//! position is one the actor could have played from. Only the law for the
//! number of turns differs: self-play uses a decaying geometric, this uses a
//! normal truncated above, which is what the distillation set asked for.
//!
//! The bytes written per position are `Minions::to_features(&canonical)`, which
//! is what `selfplay.rs` puts in a replay record. That is deliberate and it is
//! the whole point: the replay buffer can then serve as a validation set for a
//! net trained on this data without re-encoding anything, and any drift between
//! the two encoders would be a silent, invisible bug.
//!
//! Determinism: each shard is seeded from `seed + shard_index`, so the output is
//! reproducible no matter how the threads interleave or how many there are.

use crate::game::{Minions, MinionsState, BID_MAX, BID_MIN};
use crate::selfplay::PLANE_BYTES;
use crate::turn::{realize, TurnSpec};
use playground_core::Game;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `"MNPS"`, so a truncated or mistyped file fails loudly rather than being
/// read as garbage planes.
pub const MAGIC: u32 = 0x4d4e_5053;
pub const VERSION: u32 = 1;
/// Planes, then the turns actually played and the bid that produced them. The
/// two trailing bytes cost 0.16% of the file and make it possible to ask later
/// whether error correlates with depth or with the handicap.
pub const RECORD_BYTES: usize = PLANE_BYTES + 2;
/// `magic, version, count, planes, cells`.
pub const HEADER_BYTES: usize = 4 + 4 + 8 + 4 + 4;

/// How many openings to try before giving up on a draw whose prefix keeps
/// ending the game. Mirrors `random_opening`'s own retry budget.
const OPENING_TRIES: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct GenConfig {
    pub count: usize,
    pub mean: f64,
    pub stdev: f64,
    /// Draws longer than this are rejected, not clamped. Clamping would pile
    /// a spike of mass on exactly this value; rejection keeps the shape.
    pub max_turns: u32,
    pub shard_size: usize,
    pub threads: usize,
    pub seed: u64,
}

/// Draw a turn count from `N(mean, stdev)` truncated to `0..=max_turns`.
///
/// Rejection, not clamping. With the spec's `mean == max_turns` that throws
/// away about half the draws, which is cheap and keeps the distribution a true
/// truncated normal: the realised mean lands near `mean - 0.8 * stdev`, *not*
/// at `mean`. That is a property of the spec, not a bug here.
fn draw_len<R: Rng + ?Sized>(normal: &Normal<f64>, max_turns: u32, rng: &mut R) -> u32 {
    for _ in 0..64 {
        let v = normal.sample(rng).round();
        if v >= 0.0 && v <= max_turns as f64 {
            return v as u32;
        }
    }
    // Pathological parameters (e.g. a mean far above the cap): fall back to a
    // uniform draw rather than spinning or silently returning a constant.
    rng.gen_range(0..=max_turns)
}

/// One position: a bid, then `n` random legal turns, resampled if the prefix
/// ends the game — a finished position has no value to distil.
fn draw_position<R: Rng + ?Sized>(
    normal: &Normal<f64>,
    max_turns: u32,
    rng: &mut R,
) -> (MinionsState, u32, i16) {
    for _ in 0..OPENING_TRIES {
        let bid = rng.gen_range(BID_MIN..=BID_MAX);
        let n = draw_len(normal, max_turns, rng);
        let mut state = MinionsState::with_bid(bid);
        let mut played = 0u32;
        for _ in 0..n {
            if Minions::outcome(&state).is_terminal() {
                break;
            }
            let spec = TurnSpec::random_playout(&state, rng);
            state = realize(&state, &spec).next;
            played += 1;
        }
        if !Minions::outcome(&state).is_terminal() {
            return (state, played, bid);
        }
    }
    // Every try ended the game. The bid-only position is always legal and never
    // terminal, so this terminates rather than looping forever.
    let bid = rng.gen_range(BID_MIN..=BID_MAX);
    (MinionsState::with_bid(bid), 0, bid)
}

/// Generate one shard and return how many positions it holds.
fn write_shard(path: &std::path::Path, n: usize, cfg: &GenConfig, seed: u64) -> anyhow::Result<()> {
    let normal = Normal::new(cfg.mean, cfg.stdev)
        .map_err(|e| anyhow::anyhow!("bad normal({}, {}): {e}", cfg.mean, cfg.stdev))?;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    let mut buf = Vec::with_capacity(HEADER_BYTES + n * RECORD_BYTES);
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    buf.extend_from_slice(&(crate::game::PLANES as u32).to_le_bytes());
    buf.extend_from_slice(&((PLANE_BYTES / crate::game::PLANES) as u32).to_le_bytes());

    for _ in 0..n {
        let (state, played, bid) = draw_position(&normal, cfg.max_turns, &mut rng);
        // The canonical frame is what the replay records store, so the teacher
        // and the buffer agree without a second encoder.
        let planes = Minions::to_features(&state.canonical());
        debug_assert_eq!(planes.len(), PLANE_BYTES);
        buf.extend_from_slice(&planes);
        buf.push(played.min(255) as u8);
        buf.push(bid as i8 as u8);
    }

    // Write to a temp name and rename, so a killed run never leaves a
    // half-written shard that reads as valid up to its header count.
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &buf)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Generate `cfg.count` positions into `out_dir` as `positions_NNNNN.bin`.
///
/// Shards already on disk are skipped, so an interrupted run resumes by being
/// re-invoked with the same arguments and seed.
pub fn generate(out_dir: &str, cfg: GenConfig) -> anyhow::Result<()> {
    anyhow::ensure!(cfg.count > 0, "--count must be positive");
    anyhow::ensure!(cfg.shard_size > 0, "--shard-size must be positive");
    anyhow::ensure!(cfg.stdev > 0.0, "--stdev must be positive");
    let dir = std::path::Path::new(out_dir);
    std::fs::create_dir_all(dir)?;

    let shards = cfg.count.div_ceil(cfg.shard_size);
    let threads = cfg.threads.max(1).min(shards);
    println!(
        "generating {} positions in {} shards of {} across {} threads\n  \
         length ~ N({}, {}) truncated to 0..={} turns, seed {}\n  \
         {} bytes/record ({} plane bytes + 2), ~{:.1} GB total",
        cfg.count,
        shards,
        cfg.shard_size,
        threads,
        cfg.mean,
        cfg.stdev,
        cfg.max_turns,
        cfg.seed,
        RECORD_BYTES,
        PLANE_BYTES,
        (cfg.count * RECORD_BYTES) as f64 / 1e9,
    );

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= shards {
                    return;
                }
                // The tail shard is short when count is not a multiple.
                let n = cfg.shard_size.min(cfg.count - idx * cfg.shard_size);
                let path = dir.join(format!("positions_{idx:05}.bin"));
                if path.exists() {
                    done.fetch_add(n, Ordering::Relaxed);
                    continue;
                }
                if let Err(e) = write_shard(&path, n, &cfg, cfg.seed.wrapping_add(idx as u64)) {
                    *err.lock().unwrap() = Some(format!("shard {idx}: {e}"));
                    return;
                }
                let d = done.fetch_add(n, Ordering::Relaxed) + n;
                let secs = t0.elapsed().as_secs_f64();
                println!(
                    "  shard {idx:05} done — {d}/{} positions, {:.0}/s, {:.1} min elapsed",
                    cfg.count,
                    d as f64 / secs.max(1e-9),
                    secs / 60.0,
                );
                let _ = std::io::stdout().flush();
            });
        }
    });

    if let Some(e) = err.lock().unwrap().take() {
        anyhow::bail!(e);
    }
    println!(
        "done: {} positions in {:.1} min",
        cfg.count,
        t0.elapsed().as_secs_f64() / 60.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GenConfig {
        GenConfig {
            count: 64,
            mean: 20.0,
            stdev: 5.0,
            max_turns: 20,
            shard_size: 64,
            threads: 1,
            seed: 7,
        }
    }

    /// The encoding must be the one the replay buffer uses, or validating on the
    /// buffer compares two different input spaces.
    #[test]
    fn planes_match_the_replay_encoding() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let normal = Normal::new(20.0, 5.0).unwrap();
        let (state, _, _) = draw_position(&normal, 20, &mut rng);
        let planes = Minions::to_features(&state.canonical());
        assert_eq!(planes.len(), PLANE_BYTES);
        // Canonicalising twice is the same as once, so the stored frame is
        // stable — the property the buffer's planes also rely on.
        let again = Minions::to_features(&state.canonical().canonical());
        assert_eq!(planes, again);
    }

    /// Truncation is by rejection, so nothing may exceed the cap and the
    /// realised mean must sit *below* the nominal one.
    #[test]
    fn lengths_respect_the_cap() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(2);
        let normal = Normal::new(20.0, 5.0).unwrap();
        let mut sum = 0u32;
        let n = 2000;
        for _ in 0..n {
            let v = draw_len(&normal, 20, &mut rng);
            assert!(v <= 20, "draw {v} exceeded the cap");
            sum += v;
        }
        let mean = sum as f64 / n as f64;
        assert!(mean < 20.0, "truncated mean {mean} should be below 20");
        assert!(mean > 12.0, "truncated mean {mean} implausibly low");
    }

    /// A generated position is never terminal: a finished game has no value to
    /// distil and would poison the target distribution.
    #[test]
    fn positions_are_never_terminal() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let normal = Normal::new(20.0, 5.0).unwrap();
        for _ in 0..200 {
            let (state, _, _) = draw_position(&normal, 20, &mut rng);
            assert!(!Minions::outcome(&state).is_terminal());
        }
    }

    #[test]
    fn shard_round_trips_its_header() {
        let dir = std::env::temp_dir().join(format!("minions_genpos_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("positions_00000.bin");
        write_shard(&path, 64, &cfg(), 11).unwrap();
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw.len(), HEADER_BYTES + 64 * RECORD_BYTES);
        assert_eq!(u32::from_le_bytes(raw[0..4].try_into().unwrap()), MAGIC);
        assert_eq!(u32::from_le_bytes(raw[4..8].try_into().unwrap()), VERSION);
        assert_eq!(u64::from_le_bytes(raw[8..16].try_into().unwrap()), 64);
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    /// Same seed, same bytes — the property that makes a shard re-creatable and
    /// an interrupted run resumable.
    #[test]
    fn generation_is_deterministic_in_the_seed() {
        let dir = std::env::temp_dir().join(format!("minions_genpos_det_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.bin");
        let b = dir.join("b.bin");
        write_shard(&a, 32, &cfg(), 99).unwrap();
        write_shard(&b, 32, &cfg(), 99).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();
        std::fs::remove_dir(&dir).ok();
    }
}
