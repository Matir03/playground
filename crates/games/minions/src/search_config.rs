//! The search self-play runs, recorded once and shared.
//!
//! Self-play, eval and the UI used to build their searches separately -- five
//! copies of the parameter list, each with its own defaults -- and they
//! drifted: eval drew no random candidates, and the UI seat took
//! `SampledParams::default()` wholesale. Now the self-play actor writes the
//! settings it actually runs with to `search.json` in the live dir, and eval
//! and the UI load that file. Every one of them then builds its search through
//! `ActorConfig::search_params`, so what the UI shows and what a match measures
//! is the search that produced the training data.

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
    pw_resume: bool,
    kernel_sigma: f32,
    candidates: usize,
    candidates_fast: usize,
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
        pw_resume: a.pw_resume,
        kernel_sigma: a.kernel_sigma,
        candidates: s.k_candidates,
        candidates_fast: s.k_candidates_fast,
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
    std::fs::write(&tmp, serde_json::to_string_pretty(&record(actor, sampler))? + "\n")
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
    let r: Recorded =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let actor = ActorConfig {
        sims: r.sims,
        fast_sims: r.fast_sims,
        full_search_prob: r.full_search_prob,
        c_puct: r.c_puct,
        k_forced: r.k_forced,
        min_explore_visits: r.min_explore_visits,
        forced_visits: r.forced_visits,
        forced_extra: r.forced_extra,
        temp_moves: r.temp_moves,
        pw_alpha: r.pw_alpha,
        pw_beta: r.pw_beta,
        pw_chunk: r.pw_chunk,
        pw_resume: r.pw_resume,
        kernel_sigma: r.kernel_sigma,
        ..Default::default()
    };
    let sampler = SamplerConfig {
        k_candidates: r.candidates,
        k_candidates_fast: r.candidates_fast,
        t_steps: r.t_steps,
        t_steps_fast: r.t_steps_fast,
        random_fraction: r.random_fraction,
        temperature: r.temperature,
        use_energy: r.use_energy,
        ..Default::default()
    };
    Ok((actor, sampler))
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
            pw_resume: false,
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
        let mut v = serde_json::to_value(record(&ActorConfig::default(), &SamplerConfig::default()))
            .unwrap();
        v.as_object_mut().unwrap().remove("random_fraction");
        std::fs::write(&path, v.to_string()).unwrap();
        assert!(load(&path).is_err(), "a search.json without random_fraction loaded");
        std::fs::remove_file(&path).ok();
    }
}
