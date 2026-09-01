//! Attacker→target assignment: the one genuinely combinatorial part of
//! realization.
//!
//! Given fixed positions and a set of enemies declared dead, choose disjoint
//! groups of attackers so each declared target takes at least its remaining
//! health. That is bin covering with assignment restrictions, so there is no
//! known polynomial algorithm — but the instances here are tiny and structured,
//! so a complete search is affordable *as a fallback*.
//!
//! Three tiers, cheapest first:
//!
//! 1. **Greedy**, most-constrained target first. Handles essentially everything:
//!    measured within `2.4e-6` of optimal over 133M kills.
//! 2. **Deficiency certificate** ([`hall_violation`]). If some set of targets
//!    demands more health than the attackers reaching it can possibly deal, the
//!    declaration is *provably* unrealizable and dropping it is correct rather
//!    than a guess.
//! 3. **Complete search** ([`solve`]). Exhaustive over minimal sufficient
//!    subsets, so it returns either an assignment or a proof that none exists.
//!    Bounded by a node budget; exceeding it yields [`Assignment::Unknown`],
//!    which is reported rather than silently treated as infeasible.
//!
//! Deliberately independent of hex geometry and of one-attack-per-unit:
//! adjacency is an arbitrary relation and attackers carry an attack count, so
//! range > 1 and multi-attack units need no change here.

/// One enemy declared dead.
#[derive(Clone, Debug)]
pub struct Target {
    /// Board cell, for reporting back.
    pub cell: u8,
    /// Damage still needed to kill it.
    pub need: u32,
    /// Indices into the attacker list that can reach it, with the damage each
    /// would deal *to this target* (unsummon is target-dependent).
    pub reach: Vec<(usize, u32)>,
}

/// One of our units that still has an attack.
#[derive(Clone, Copy, Debug)]
pub struct Attacker {
    pub cell: u8,
    /// Attacks remaining. 1 for every unit in the simplified game; the search
    /// handles more so multi-attack units need no special case.
    pub attacks: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Assignment {
    /// Attacker indices per target, parallel to the target list.
    Found(Vec<Vec<usize>>),
    /// Proven: no assignment kills every declared target.
    Infeasible,
    /// Search budget exhausted — neither found nor disproven.
    Unknown,
}

/// A subset of targets that provably cannot all be killed, if one exists among
/// the cheap witnesses we check.
///
/// The condition is a deficiency (Hall-type) bound: if the targets in `S` demand
/// more total health than *every* attacker reaching any of them could deal, then
/// `S` is unsatisfiable and so is any superset. Checking all subsets is
/// exponential, so we check the two that matter in practice — singletons, and
/// whole connected components — which together catch the overwhelming majority
/// of genuinely impossible declarations.
pub fn hall_violation(targets: &[Target], attackers: &[Attacker]) -> Option<Vec<usize>> {
    // Singletons: a target no reachable set of attackers can finish.
    for (i, t) in targets.iter().enumerate() {
        let supply: u32 = t
            .reach
            .iter()
            .map(|&(a, d)| d * attackers[a].attacks as u32)
            .sum();
        if supply < t.need {
            return Some(vec![i]);
        }
    }
    // Components: the shared attacker pool cannot cover the shared demand.
    for comp in components(targets, attackers) {
        let demand: u32 = comp.targets.iter().map(|&i| targets[i].need).sum();
        // Best case for the defence: every attacker deals its maximum over the
        // targets it can reach, and does so `attacks` times.
        let mut supply = 0u32;
        for &a in &comp.attackers {
            let best = comp
                .targets
                .iter()
                .flat_map(|&t| targets[t].reach.iter())
                .filter(|&&(ai, _)| ai == a)
                .map(|&(_, d)| d)
                .max()
                .unwrap_or(0);
            supply += best * attackers[a].attacks as u32;
        }
        if supply < demand {
            return Some(comp.targets);
        }
    }
    None
}

struct Component {
    targets: Vec<usize>,
    attackers: Vec<usize>,
}

/// Split the instance into independent pieces. Targets interact only through
/// shared attackers, so components can be solved separately — which is what
/// keeps the complete search affordable, since components are almost always
/// two to five units wide.
fn components(targets: &[Target], attackers: &[Attacker]) -> Vec<Component> {
    let n_a = attackers.len();
    let mut parent: Vec<usize> = (0..n_a + targets.len()).collect();
    fn find(p: &mut Vec<usize>, x: usize) -> usize {
        if p[x] != x {
            let r = find(p, p[x]);
            p[x] = r;
        }
        p[x]
    }
    for (ti, t) in targets.iter().enumerate() {
        for &(a, _) in &t.reach {
            let (ra, rt) = (find(&mut parent, a), find(&mut parent, n_a + ti));
            if ra != rt {
                parent[ra] = rt;
            }
        }
    }
    let mut by_root: std::collections::HashMap<usize, Component> = Default::default();
    for ti in 0..targets.len() {
        let r = find(&mut parent, n_a + ti);
        by_root
            .entry(r)
            .or_insert_with(|| Component { targets: vec![], attackers: vec![] })
            .targets
            .push(ti);
    }
    for ai in 0..n_a {
        let r = find(&mut parent, ai);
        if let Some(c) = by_root.get_mut(&r) {
            c.attackers.push(ai);
        }
    }
    by_root.into_values().collect()
}

/// Widest reachable set we will enumerate exhaustively. Beyond this the search
/// cannot claim completeness, so [`solve`] refuses to run rather than risk a
/// false proof of infeasibility. Range 1 caps `reach` at 6, so this only bites
/// for the range > 1 generalisation.
pub const MAX_ENUMERABLE_REACH: usize = 20;

/// Minimal subsets of `t.reach` whose damage meets `need`. Minimal only: a
/// superset never helps and only widens the search.
///
/// Callers must have checked `reach.len() <= MAX_ENUMERABLE_REACH`; otherwise
/// this would silently drop candidates and a failed search would look like a
/// proof of impossibility.
fn sufficient_subsets(t: &Target, capacity: &[u8]) -> Vec<Vec<usize>> {
    let reach: Vec<(usize, u32)> =
        t.reach.iter().copied().filter(|&(a, d)| capacity[a] > 0 && d > 0).collect();
    debug_assert!(reach.len() <= MAX_ENUMERABLE_REACH);
    let n = reach.len();
    let mut out: Vec<Vec<usize>> = Vec::new();
    for mask in 1u32..(1 << n) {
        let dmg: u32 = (0..n)
            .filter(|b| mask & (1 << b) != 0)
            .map(|b| reach[b].1)
            .sum();
        if dmg < t.need {
            continue;
        }
        // Minimality: dropping any member must break it.
        let minimal = (0..n).filter(|b| mask & (1 << b) != 0).all(|b| dmg - reach[b].1 < t.need);
        if minimal {
            out.push((0..n).filter(|b| mask & (1 << b) != 0).map(|b| reach[b].0).collect());
        }
    }
    // Fewest units spent first: keeps attackers free for later targets and finds
    // a solution earlier when one exists.
    out.sort_by_key(|s| s.len());
    out
}

/// Greedy plan, most-constrained target first: the fast tier.
///
/// `None` for a target means greedy could not finish it — which is a cue to
/// escalate, *not* evidence that it is impossible. Candidates are taken in the
/// order `reach` was built, so callers control preference (we put units with no
/// pending move first, since spending their attack costs nothing).
pub fn greedy(targets: &[Target], attackers: &[Attacker]) -> Vec<Option<Vec<usize>>> {
    let mut capacity: Vec<u8> = attackers.iter().map(|a| a.attacks).collect();
    let mut plan: Vec<Option<Vec<usize>>> = vec![None; targets.len()];
    loop {
        // Fewest available finishers first: a target with one way to die is
        // served before one with five, so a shared attacker is not squandered.
        let mut best: Option<(usize, usize, Vec<usize>)> = None;
        for (i, t) in targets.iter().enumerate() {
            if plan[i].is_some() {
                continue;
            }
            let avail: Vec<(usize, u32)> =
                t.reach.iter().copied().filter(|&(a, d)| capacity[a] > 0 && d > 0).collect();
            if avail.iter().map(|&(_, d)| d).sum::<u32>() < t.need {
                continue;
            }
            let n = avail.len();
            if best.as_ref().is_none_or(|(bn, _, _)| n < *bn) {
                best = Some((n, i, avail.iter().map(|&(a, _)| a).collect()));
            }
        }
        let Some((_, i, order)) = best else { break };
        let t = &targets[i];
        let dmg = |a: usize| t.reach.iter().find(|&&(x, _)| x == a).map(|&(_, d)| d).unwrap_or(0);
        // Biggest hitters first to reach the threshold, then drop any member the
        // remainder no longer needs — fewest units spent, least overkill.
        let mut order = order;
        order.sort_by_key(|&a| (std::cmp::Reverse(dmg(a)), a));
        let mut chosen: Vec<usize> = Vec::new();
        let mut acc = 0u32;
        for a in order {
            if acc >= t.need {
                break;
            }
            acc += dmg(a);
            chosen.push(a);
        }
        let mut k = 0;
        while k < chosen.len() {
            if acc - dmg(chosen[k]) >= t.need {
                acc -= dmg(chosen[k]);
                chosen.remove(k);
            } else {
                k += 1;
            }
        }
        for &a in &chosen {
            capacity[a] -= 1;
        }
        plan[i] = Some(chosen);
    }
    plan
}

/// Complete search: an assignment, or a proof there is none.
pub fn solve(targets: &[Target], attackers: &[Attacker], budget: u64) -> Assignment {
    if targets.is_empty() {
        return Assignment::Found(Vec::new());
    }
    if hall_violation(targets, attackers).is_some() {
        return Assignment::Infeasible;
    }
    // Completeness precondition: every target's option set must be enumerable,
    // or an exhausted search would be a *false* proof of infeasibility.
    if targets.iter().any(|t| t.reach.len() > MAX_ENUMERABLE_REACH) {
        return Assignment::Unknown;
    }
    let mut capacity: Vec<u8> = attackers.iter().map(|a| a.attacks).collect();
    let mut chosen: Vec<Vec<usize>> = vec![Vec::new(); targets.len()];
    let mut done = vec![false; targets.len()];
    let mut nodes = 0u64;
    match search(targets, &mut capacity, &mut chosen, &mut done, &mut nodes, budget) {
        Some(true) => Assignment::Found(chosen),
        Some(false) => Assignment::Infeasible,
        None => Assignment::Unknown,
    }
}

/// `Some(true)` solved, `Some(false)` proven impossible, `None` out of budget.
fn search(
    targets: &[Target],
    capacity: &mut Vec<u8>,
    chosen: &mut Vec<Vec<usize>>,
    done: &mut Vec<bool>,
    nodes: &mut u64,
    budget: u64,
) -> Option<bool> {
    *nodes += 1;
    if *nodes > budget {
        return None;
    }
    // Most-constrained target first: the one with fewest ways left to die. If any
    // has none, this branch is dead immediately.
    let mut best: Option<(usize, Vec<Vec<usize>>)> = None;
    for (i, t) in targets.iter().enumerate() {
        if done[i] {
            continue;
        }
        let opts = sufficient_subsets(t, capacity);
        if opts.is_empty() {
            return Some(false);
        }
        if best.as_ref().is_none_or(|(_, b)| opts.len() < b.len()) {
            best = Some((i, opts));
        }
    }
    let Some((i, options)) = best else { return Some(true) };

    let mut exhausted = true;
    for opt in options {
        for &a in &opt {
            capacity[a] -= 1;
        }
        chosen[i] = opt.clone();
        done[i] = true;
        match search(targets, capacity, chosen, done, nodes, budget) {
            Some(true) => return Some(true),
            // A budget stop means this branch is unresolved, so the whole call
            // can no longer claim to have proven anything.
            None => exhausted = false,
            Some(false) => {}
        }
        done[i] = false;
        chosen[i].clear();
        for &a in &opt {
            capacity[a] += 1;
        }
    }
    if exhausted {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(cell: u8, need: u32, reach: &[(usize, u32)]) -> Target {
        Target { cell, need, reach: reach.to_vec() }
    }
    fn a(n: usize) -> Vec<Attacker> {
        (0..n).map(|i| Attacker { cell: i as u8, attacks: 1 }).collect()
    }

    #[test]
    fn a_single_sufficient_attacker_is_found() {
        let r = solve(&[t(0, 4, &[(0, 5)])], &a(1), 1000);
        assert_eq!(r, Assignment::Found(vec![vec![0]]));
    }

    #[test]
    fn insufficient_damage_is_proven_impossible() {
        // 3 damage against 4 health, and nothing else in range.
        let r = solve(&[t(0, 4, &[(0, 3)])], &a(1), 1000);
        assert_eq!(r, Assignment::Infeasible);
        assert_eq!(hall_violation(&[t(0, 4, &[(0, 3)])], &a(1)), Some(vec![0]));
    }

    #[test]
    fn damage_is_pooled_across_attackers() {
        // 3 + 1 kills a ghost; neither alone does.
        let r = solve(&[t(0, 4, &[(0, 3), (1, 1)])], &a(2), 1000);
        match r {
            Assignment::Found(v) => {
                let mut s = v[0].clone();
                s.sort();
                assert_eq!(s, vec![0, 1]);
            }
            other => panic!("expected a pooled kill, got {other:?}"),
        }
    }

    #[test]
    fn contention_is_resolved_where_greedy_could_fail() {
        // Two ghosts (health 4). Attacker 0 deals 5 and reaches both; 1 and 2
        // deal 3 and 1 and reach only the second. Spending the 5 on the second
        // target strands the first — the search must give the 5 to target 0.
        let targets = vec![
            t(10, 4, &[(0, 5)]),
            t(11, 4, &[(0, 5), (1, 3), (2, 1)]),
        ];
        match solve(&targets, &a(3), 10_000) {
            Assignment::Found(v) => {
                assert_eq!(v[0], vec![0], "the only attacker for target 0 must go there");
                let mut s = v[1].clone();
                s.sort();
                assert_eq!(s, vec![1, 2]);
            }
            other => panic!("expected a solution, got {other:?}"),
        }
    }

    #[test]
    fn shared_scarcity_is_proven_impossible() {
        // Two targets needing 4 each, one attacker dealing 5 reaching both:
        // total supply 5 < demand 8.
        let targets = vec![t(10, 4, &[(0, 5)]), t(11, 4, &[(0, 5)])];
        assert_eq!(solve(&targets, &a(1), 10_000), Assignment::Infeasible);
        assert!(hall_violation(&targets, &a(1)).is_some());
    }

    #[test]
    fn multi_attack_units_split_across_targets() {
        // One unit with three 1-damage attacks covers three 1-health targets —
        // the generalisation the simplified game does not yet exercise.
        let targets = vec![
            t(10, 1, &[(0, 1)]),
            t(11, 1, &[(0, 1)]),
            t(12, 1, &[(0, 1)]),
        ];
        let atk = vec![Attacker { cell: 0, attacks: 3 }];
        match solve(&targets, &atk, 10_000) {
            Assignment::Found(v) => assert_eq!(v, vec![vec![0], vec![0], vec![0]]),
            other => panic!("expected the multi-attacker to cover all three, got {other:?}"),
        }
        // With only two attacks it is provably impossible.
        let atk = vec![Attacker { cell: 0, attacks: 2 }];
        assert_eq!(solve(&targets, &atk, 10_000), Assignment::Infeasible);
    }

    #[test]
    fn independent_components_do_not_interact() {
        let targets = vec![t(10, 1, &[(0, 1)]), t(50, 1, &[(1, 1)])];
        let comps = components(&targets, &a(2));
        assert_eq!(comps.len(), 2, "disjoint targets should not share a component");
        assert!(matches!(solve(&targets, &a(2), 1000), Assignment::Found(_)));
    }

    #[test]
    fn a_tiny_budget_reports_unknown_rather_than_guessing() {
        // Wide contention with a budget of one node: the search must admit it
        // does not know, never claim infeasibility.
        // Feasible (demand 6, supply 8) so the deficiency check passes, but the
        // search space is wide enough that one node cannot settle it.
        let reach: Vec<(usize, u32)> = (0..8).map(|k| (k, 1u32)).collect();
        let targets: Vec<Target> = (0..3).map(|i| t(i, 2, &reach)).collect();
        assert_eq!(solve(&targets, &a(8), 1), Assignment::Unknown);
        // With room to search it is solvable, so "unknown" really was ignorance.
        assert!(matches!(solve(&targets, &a(8), 100_000), Assignment::Found(_)));
    }

    #[test]
    fn an_unenumerable_reach_refuses_to_claim_a_proof() {
        // More reachable attackers than we can enumerate: the honest answer is
        // Unknown, never Infeasible.
        let reach: Vec<(usize, u32)> =
            (0..MAX_ENUMERABLE_REACH + 1).map(|k| (k, 1u32)).collect();
        let targets = vec![t(0, 99, &reach)];
        assert_eq!(
            solve(&targets, &a(MAX_ENUMERABLE_REACH + 1), 100_000),
            Assignment::Infeasible,
            "the deficiency bound still applies and is sound"
        );
        let targets = vec![t(0, 2, &reach)];
        assert_eq!(solve(&targets, &a(MAX_ENUMERABLE_REACH + 1), 100_000), Assignment::Unknown);
    }

    #[test]
    fn minimal_subsets_exclude_wasteful_supersets() {
        let target = t(0, 2, &[(0, 1), (1, 1), (2, 3)]);
        let subs = sufficient_subsets(&target, &[1, 1, 1]);
        assert!(subs.contains(&vec![2]), "the single 3 suffices");
        assert!(subs.contains(&vec![0, 1]), "so does 1+1");
        assert!(
            !subs.iter().any(|s| s.len() == 3),
            "using all three is never minimal: {subs:?}"
        );
    }
}
