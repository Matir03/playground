//! The search self-play runs, recorded once and shared.
//!
//! The self-play actor writes the settings it actually runs with to
//! `search.json` in the live dir, and the UI's engine (`serve`) starts from
//! that file, so the UI shows the search that produced the training data. The
//! UI can then change any of it (`apply_json`) and restore it. Eval keeps
//! settings of its own (`eval::match_search`). Every search, whatever its
//! settings, is built through `ActorConfig::search_params`.

use crate::sampler::SamplerConfig;
use crate::selfplay::ActorConfig;
use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// File name inside the live dir.
pub const FILE: &str = "search.json";

/// Every setting that changes what a search does, and nothing else: seats,
/// batching, the backend and random openings are not part of the search.
///
/// Strict on load -- no field has a default and unknown fields are refused --
/// so a file from an older binary fails loudly instead of quietly filling the
/// gap with whatever this binary's defaults happen to be.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
struct Recorded {
    sims: u32,
    fast_sims: u32,
    full_search_prob: f64,
    c_puct: f32,
    k_forced: f32,
    min_explore_visits: f32,
    forced_visits: u32,
    forced_extra: u32,
    temp_moves: u32,
    pw_alpha: f32,
    pw_beta: f32,
    pw_chunk: usize,
    draw_retries: u32,
    kernel_sigma: f32,
    resample_every: f32,
    resample_max_depth: u32,
    resample_t: usize,
    resample_frac: f32,
    candidates: usize,
    candidates_fast: usize,
    draws: usize,
    draws_fast: usize,
    t_steps: usize,
    t_steps_fast: usize,
    random_fraction: f32,
    temperature: f32,
    use_energy: bool,
}

fn record(a: &ActorConfig, s: &SamplerConfig) -> Recorded {
    Recorded {
        sims: a.sims,
        fast_sims: a.fast_sims,
        full_search_prob: a.full_search_prob,
        c_puct: a.c_puct,
        k_forced: a.k_forced,
        min_explore_visits: a.min_explore_visits,
        forced_visits: a.forced_visits,
        forced_extra: a.forced_extra,
        temp_moves: a.temp_moves,
        pw_alpha: a.pw_alpha,
        pw_beta: a.pw_beta,
        pw_chunk: a.pw_chunk,
        draw_retries: a.draw_retries,
        kernel_sigma: a.kernel_sigma,
        resample_every: a.resample_every,
        resample_max_depth: a.resample_max_depth,
        resample_t: s.resample_t,
        resample_frac: s.resample_frac,
        candidates: s.k_candidates,
        candidates_fast: s.k_candidates_fast,
        draws: s.k_draws,
        draws_fast: s.k_draws_fast,
        t_steps: s.t_steps,
        t_steps_fast: s.t_steps_fast,
        random_fraction: s.random_fraction,
        temperature: s.temperature,
        use_energy: s.use_energy,
    }
}

/// Write the search `actor` and `sampler` describe. Atomic, so a reader never
/// sees half a file.
pub fn save(path: &Path, actor: &ActorConfig, sampler: &SamplerConfig) -> anyhow::Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(
        &tmp,
        serde_json::to_string_pretty(&record(actor, sampler))? + "\n",
    )
    .with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming onto {}", path.display()))?;
    Ok(())
}

/// The actor and sampler settings self-play recorded. Everything outside the
/// search keeps its default; the caller sets seeds, padding and the backend.
pub fn load(path: &Path) -> anyhow::Result<(ActorConfig, SamplerConfig)> {
    let text = std::fs::read_to_string(path).with_context(|| {
        format!(
            "reading {} -- self-play writes it on start; `minions-engine selfplay-loop \
             --write-search-config` writes it without playing",
            path.display()
        )
    })?;
    let (mut actor, mut sampler) = (ActorConfig::default(), SamplerConfig::default());
    apply_json(&text, &mut actor, &mut sampler)
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok((actor, sampler))
}

/// The search `actor` and `sampler` describe, in the file's format.
pub fn to_json(actor: &ActorConfig, sampler: &SamplerConfig) -> String {
    serde_json::to_string_pretty(&record(actor, sampler)).expect("numbers and bools serialise")
}

/// Overwrite the search settings in `actor` and `sampler` with those in
/// `text`, leaving everything else -- seeds, padding, the backend -- alone.
///
/// As strict as the file: every setting must be given and nothing else, and a
/// value no search can run with is refused by name rather than run.
pub fn apply_json(
    text: &str,
    actor: &mut ActorConfig,
    sampler: &mut SamplerConfig,
) -> anyhow::Result<()> {
    let r: Recorded = serde_json::from_str(text)?;
    validate(&r)?;
    actor.sims = r.sims;
    actor.fast_sims = r.fast_sims;
    actor.full_search_prob = r.full_search_prob;
    actor.c_puct = r.c_puct;
    actor.k_forced = r.k_forced;
    actor.min_explore_visits = r.min_explore_visits;
    actor.forced_visits = r.forced_visits;
    actor.forced_extra = r.forced_extra;
    actor.temp_moves = r.temp_moves;
    actor.pw_alpha = r.pw_alpha;
    actor.pw_beta = r.pw_beta;
    actor.pw_chunk = r.pw_chunk;
    actor.draw_retries = r.draw_retries;
    actor.kernel_sigma = r.kernel_sigma;
    actor.resample_every = r.resample_every;
    actor.resample_max_depth = r.resample_max_depth;
    sampler.resample_t = r.resample_t;
    sampler.resample_frac = r.resample_frac;
    sampler.k_candidates = r.candidates;
    sampler.k_candidates_fast = r.candidates_fast;
    sampler.k_draws = r.draws;
    sampler.k_draws_fast = r.draws_fast;
    sampler.t_steps = r.t_steps;
    sampler.t_steps_fast = r.t_steps_fast;
    sampler.random_fraction = r.random_fraction;
    sampler.temperature = r.temperature;
    sampler.use_energy = r.use_energy;
    Ok(())
}

fn validate(r: &Recorded) -> anyhow::Result<()> {
    for (name, v) in [
        ("sims", r.sims as usize),
        ("fast_sims", r.fast_sims as usize),
        ("candidates", r.candidates),
        ("candidates_fast", r.candidates_fast),
        ("t_steps", r.t_steps),
        ("t_steps_fast", r.t_steps_fast),
        ("resample_t", r.resample_t),
    ] {
        anyhow::ensure!(v > 0, "{name} must be at least 1");
    }
    for (name, v) in [
        ("full_search_prob", r.full_search_prob as f32),
        ("random_fraction", r.random_fraction),
        ("resample_frac", r.resample_frac),
    ] {
        anyhow::ensure!(
            (0.0..=1.0).contains(&v),
            "{name} must be within [0, 1], got {v}"
        );
    }
    for (name, v) in [
        ("c_puct", r.c_puct),
        ("k_forced", r.k_forced),
        ("min_explore_visits", r.min_explore_visits),
        ("pw_alpha", r.pw_alpha),
        ("kernel_sigma", r.kernel_sigma),
        ("resample_every", r.resample_every),
    ] {
        anyhow::ensure!(
            v.is_finite() && v >= 0.0,
            "{name} must be a number >= 0, got {v}"
        );
    }
    anyhow::ensure!(r.pw_beta.is_finite(), "pw_beta must be a number");
    anyhow::ensure!(
        r.temperature.is_finite() && r.temperature > 0.0,
        "temperature must be above 0, got {}",
        r.temperature
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("minions_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(FILE)
    }

    /// What eval and the UI load must be the search self-play wrote -- every
    /// field, not just the ones someone remembered to copy.
    #[test]
    fn a_recorded_search_loads_back_unchanged() {
        let actor = ActorConfig {
            sims: 400,
            fast_sims: 100,
            forced_visits: 30,
            forced_extra: 4,
            ..Default::default()
        };
        let sampler = SamplerConfig {
            k_candidates: 16,
            k_candidates_fast: 8,
            t_steps: 4,
            t_steps_fast: 2,
            random_fraction: 0.25,
            ..Default::default()
        };
        let path = scratch("roundtrip");
        save(&path, &actor, &sampler).unwrap();
        let (a2, s2) = load(&path).unwrap();
        assert_eq!(record(&actor, &sampler), record(&a2, &s2));
        std::fs::remove_file(&path).ok();
    }

    /// A file missing a setting is an error, not a default.
    #[test]
    fn a_file_missing_a_setting_is_refused() {
        let path = scratch("missing");
        let mut v =
            serde_json::to_value(record(&ActorConfig::default(), &SamplerConfig::default()))
                .unwrap();
        v.as_object_mut().unwrap().remove("random_fraction");
        std::fs::write(&path, v.to_string()).unwrap();
        assert!(
            load(&path).is_err(),
            "a search.json without random_fraction loaded"
        );
        std::fs::remove_file(&path).ok();
    }

    /// The UI edits the search in place: applying settings must leave what is
    /// not a setting alone, and refuse a value no search can run with.
    #[test]
    fn applying_settings_keeps_the_rest_and_refuses_bad_values() {
        let mut actor = ActorConfig {
            workers: 7,
            ..Default::default()
        };
        let mut sampler = SamplerConfig {
            seed: 99,
            pad_to: 12,
            ..Default::default()
        };
        let mut v = serde_json::to_value(record(&actor, &sampler)).unwrap();
        v["sims"] = 123.into();
        v["random_fraction"] = 0.5.into();
        apply_json(&v.to_string(), &mut actor, &mut sampler).unwrap();
        assert_eq!((actor.sims, sampler.random_fraction), (123, 0.5));
        assert_eq!((actor.workers, sampler.seed, sampler.pad_to), (7, 99, 12));

        v["random_fraction"] = 1.5.into();
        let err = apply_json(&v.to_string(), &mut actor, &mut sampler).unwrap_err();
        assert!(err.to_string().contains("random_fraction"), "{err}");
        assert_eq!(
            sampler.random_fraction, 0.5,
            "a refused apply must change nothing"
        );
    }
}
