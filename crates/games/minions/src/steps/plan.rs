//! Plan-space witness search: the exact question "does some completion of
//! this partial declaration realize?" answered by enumerating *plans* rather
//! than atomic action orders.
//!
//! A plan is a complete declaration: which enemies die (the declared kills
//! plus some of the unfated ones, killed to free a hex or a path), where every
//! attacker ends its move, and where the other undecided friends go (they stay
//! unless their hex is needed). Once all of that is fixed, ordering the
//! actions is the realizer's job, which is cheap. So the search is:
//!
//! 1. the kill set, smallest extra set first, with a lethal-damage and
//!    hex-capacity bound on every target before anything is enumerated;
//! 2. attacker placements covering every target's remaining health on
//!    distinct hexes (each attacker set counted once, not once per order);
//! 3. relocation of friends standing on a needed hex;
//! 4. verification of the complete declaration.
//!
//! A kill of the enemy necromancer ends the game, so it is its own branch in
//! which the other declared fates need not happen.
//!
//! [`Plan::Impossible`] is claimed only when every branch was enumerated to
//! the end; any cap reached makes the answer [`Plan::Unknown`] and the caller
//! falls back to the atomic search.

use super::*;

pub(super) enum Plan {
    Found(TurnSpec),
    Impossible,
    Unknown,
}

/// Caps per query: placement-search nodes, and complete declarations
/// verified. Kept small on purpose: self-play waits for the slowest leaf of
/// each wave, so a long plan query stalls every build thread. At 128
/// verifies x 20k nodes x 2048 sets, self-play lost 10% for 190 unknowns
/// per 25k queries instead of 428.
const NODE_CAP: u32 = 20_000;
const VERIFY_CAP: u32 = 256;
/// Exact-search nodes for verifying one complete declaration: the first
/// `VERIFY_DEEP` get `VERIFY_BUDGET`, the rest `VERIFY_SHALLOW`. Big late-game
/// positions have hundreds of placements that each verify in a few nodes, so
/// 48 deep verifies spent up to 240k nodes on one query and a best-first eval
/// game took 15 minutes; this schedule finds 9% more witnesses on those
/// queries in a sixth of the time, and as many on self-play's.
const VERIFY_BUDGET: u32 = 5_000;
const VERIFY_DEEP: u32 = 2;
const VERIFY_SHALLOW: u32 = 25;
/// Kill sets (with their fate variants) tried per branch.
const SETS_CAP: u32 = 512;

/// Why plan searches ended incomplete: too many extra kill candidates,
/// placement nodes, verifications, an exact verification out of budget.
pub(crate) static INCOMPLETE: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// A cap, overridable from the environment for tuning.
fn plan_env(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn why(i: usize) {
    INCOMPLETE[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

struct Budget {
    nodes: u32,
    verifies: u32,
    /// Kill sets (with fate variants) still to try, per branch.
    sets: u32,
    /// Some branch was cut short: "nothing found" proves nothing.
    incomplete: bool,
    /// Complete declarations verified so far.
    verified: u32,
}

fn dest(spec: &TurnSpec, o: u8) -> u8 {
    let k = spec.moves[o as usize];
    if k == NO_MOVE || k == 0 {
        o
    } else {
        MOVE_DEST[o as usize][k as usize]
    }
}

pub(super) fn plan_realize(base: &MinionsState, spec: &TurnSpec, decided: Bb, fated: Bb) -> Plan {
    plan_realize_scaled(base, spec, decided, fated, 1)
}

/// [`plan_realize`] with every cap multiplied by `scale`.
pub(super) fn plan_realize_scaled(
    base: &MinionsState,
    spec: &TurnSpec,
    decided: Bb,
    fated: Bb,
    scale: u32,
) -> Plan {
    let friends: Vec<u8> = (0..NUM_CELLS as u8)
        .filter(|&c| base.piece_at(c).is_some_and(|(sd, _)| sd == 0))
        .collect();
    let enemies: Vec<u8> = (0..NUM_CELLS as u8)
        .filter(|&c| base.piece_at(c).is_some_and(|(sd, _)| sd != 0))
        .collect();
    let decided_list: Vec<u8> = friends
        .iter()
        .copied()
        .filter(|&f| has(decided, f))
        .collect();
    // Hexes any friend could pass through or end on: only enemies there can
    // matter as extra kills.
    let mut reach_area: Bb = 0;
    for &f in &friends {
        let (_, u) = base.piece_at(f).expect("friend");
        if base.can_move_piece(f) {
            reach_area |= bitboard::reachable(bit(f), u.stats().speed, FULL);
        }
    }
    let unfated: Vec<u8> = enemies
        .iter()
        .copied()
        .filter(|&e| !has(fated, e) && has(reach_area, e))
        .collect();
    let mut b = Budget {
        nodes: NODE_CAP * scale,
        verifies: plan_env("PLAN_VERIFY_CAP", VERIFY_CAP) * scale,
        sets: plan_env("PLAN_SETS_CAP", SETS_CAP) * scale,
        incomplete: false,
        verified: 0,
    };

    // Branch A: every declared fate happens, plus extra kills.
    let mut base_spec = *spec;
    for &e in &enemies {
        if !has(fated, e) {
            base_spec.targets[e as usize] = TARGET_STAY;
        }
    }
    if let Some(w) = kill_sets(base, &base_spec, &decided_list, &unfated, &friends, &mut b) {
        return Plan::Found(w);
    }
    // Branch B: kill the enemy necromancer (the game ends; other declared
    // fates are moot), unless it is declared to stay.
    let necro = enemies.iter().copied().find(|&e| {
        base.piece_at(e).is_some_and(|(_, u)| u.stats().necromancer)
            && !(has(fated, e) && spec.targets[e as usize] == TARGET_STAY)
    });
    if let Some(n) = necro {
        {
            // Other enemies may die first to open the way, fated or not.
            let mut win = *spec;
            for &e in &enemies {
                if !has(fated, e) {
                    win.targets[e as usize] = TARGET_STAY;
                }
            }
            win.targets[n as usize] = TARGET_KILL;
            let others: Vec<u8> = enemies
                .iter()
                .copied()
                .filter(|&e| e != n && has(reach_area, e))
                .collect();
            // Only the necromancer and the chosen extras are targets here.
            let mut only = win;
            for &e in &enemies {
                if e != n {
                    only.targets[e as usize] = TARGET_STAY;
                }
            }
            if let Some(mut w) = kill_sets(base, &only, &decided_list, &others, &friends, &mut b) {
                // Keep the declared fates in the witness: the game ends
                // before they matter.
                for &e in &enemies {
                    if has(fated, e) {
                        w.targets[e as usize] = spec.targets[e as usize];
                    }
                }
                return Plan::Found(w);
            }
        }
    }
    if b.incomplete {
        Plan::Unknown
    } else {
        Plan::Impossible
    }
}

/// Try `spec` (whose targets are final except for `extras`) with every
/// subset of `extras` killed or unsummoned, smallest first.
fn kill_sets(
    base: &MinionsState,
    spec: &TurnSpec,
    decided: &[u8],
    extras: &[u8],
    friends: &[u8],
    b: &mut Budget,
) -> Option<TurnSpec> {
    // Only enemies some attacker could hit at all can be extra kills, and
    // unsummoning is only a separate option if we have an unsummoner.
    // (Optimistic: every candidate removable, since one kill can open the
    // way to another.)
    let open = extras.iter().fold(0, |m, &e| m | bit(e));
    let atk_all = attackers(base, spec, decided, 0, open);
    let unsummoner = atk_all
        .iter()
        .any(|&(_, u, _)| matches!(u.stats().attack, Attack::Unsummon));
    let extras: Vec<u8> = extras
        .iter()
        .copied()
        .filter(|&e| {
            atk_all.iter().any(|&(_, u, ends)| {
                ends & bitboard::reachable(bit(e), u.stats().range, FULL) & !bit(e) != 0
            })
        })
        .collect();
    let n = extras.len();
    let max = n;
    let mut sets = b.sets;
    for size in 0..=max {
        let mut idx: Vec<usize> = (0..size).collect();
        loop {
            // Every fate combination for the chosen extras.
            let fates: Vec<Vec<u8>> = idx
                .iter()
                .map(|&i| {
                    let e = extras[i];
                    let (_, u) = base.piece_at(e).expect("enemy");
                    if u.stats().persistent || !unsummoner {
                        vec![TARGET_KILL]
                    } else {
                        vec![TARGET_KILL, TARGET_UNSUMMON]
                    }
                })
                .collect();
            let combos: usize = fates.iter().map(Vec::len).product();
            for mut k in 0..combos {
                if sets == 0 {
                    b.incomplete = true;
                    why(0);
                    return None;
                }
                sets -= 1;
                let mut sp = *spec;
                for (j, &i) in idx.iter().enumerate() {
                    let f = &fates[j];
                    sp.targets[extras[i] as usize] = f[k % f.len()];
                    k /= f.len();
                }
                if let Some(w) = placements(base, &sp, decided, friends, b) {
                    return Some(w);
                }
                if b.verifies == 0 || b.nodes == 0 {
                    b.incomplete = true;
                    return None;
                }
            }
            if !next_combination(&mut idx, n) {
                break;
            }
        }
    }
    None
}

/// Advance `idx` (strictly increasing, below `n`) to the next combination;
/// `false` after the last.
fn next_combination(idx: &mut [usize], n: usize) -> bool {
    let k = idx.len();
    for i in (0..k).rev() {
        if idx[i] < n - k + i {
            idx[i] += 1;
            for j in i + 1..k {
                idx[j] = idx[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// All targets and fates fixed: place attackers, relocate, verify.
fn placements(
    base: &MinionsState,
    spec: &TurnSpec,
    decided: &[u8],
    friends: &[u8],
    b: &mut Budget,
) -> Option<TurnSpec> {
    let atk = attackers(base, spec, decided, 0, 0);
    let targets = kill_targets(base, spec, &atk)?;
    if !lethal_capacity(base, spec, &targets, &atk) {
        return None;
    }
    let attackers_: Vec<assign::Attacker> = atk
        .iter()
        .map(|&(c, _, _)| assign::Attacker {
            cell: c,
            attacks: 1,
        })
        .collect();
    if assign::hall_violation(&targets, &attackers_).is_some() {
        return None;
    }
    let mut claimed: Bb = 0;
    for &f in decided {
        claimed |= bit(dest(spec, f));
    }
    let mut blockers: Bb = 0;
    let mut target_hexes: Bb = 0;
    for e in 0..NUM_CELLS as u8 {
        if base.piece_at(e).is_some_and(|(sd, _)| sd != 0) {
            if spec.targets[e as usize] == TARGET_STAY {
                blockers |= bit(e);
            } else {
                target_hexes |= bit(e);
            }
        }
    }
    let mut order: Vec<usize> = (0..targets.len()).collect();
    order.sort_by_key(|&i| targets[i].reach.len());
    let rings: Vec<Vec<Bb>> = targets
        .iter()
        .map(|t| {
            atk.iter()
                .map(|&(_, u, _)| {
                    bitboard::reachable(bit(t.cell), u.stats().range, FULL) & !bit(t.cell)
                })
                .collect()
        })
        .collect();
    let mut found = None;
    let mut chosen = Vec::new();
    let mut try_plan = |chosen: &[(usize, u8)], b: &mut Budget| -> bool {
        match complete(
            base,
            spec,
            decided,
            friends,
            &atk,
            chosen,
            claimed,
            blockers,
            target_hexes,
            b,
        ) {
            Some(w) => {
                found = Some(w);
                true
            }
            None => false,
        }
    };
    let r = place_dfs(
        &targets,
        &order,
        0,
        0,
        0,
        &atk,
        &rings,
        0,
        // Decided units' hexes are already out of undecided attackers' ends
        // (and a decided attacker ends on its own).
        0,
        &mut chosen,
        b,
        &mut try_plan,
    );
    if r.is_none() {
        b.incomplete = true;
        why(if b.verifies == 0 { 2 } else { 1 });
    }
    found
}

/// Every target, alone: the attackers able to end on a distinct hex in its
/// ring deal at least its remaining health. Hexes held by staying enemies
/// or claimed by decided units cannot be used, and each hex holds one
/// attacker, so only the strongest `hexes` attackers count.
fn lethal_capacity(
    base: &MinionsState,
    spec: &TurnSpec,
    targets: &[assign::Target],
    atk: &[(u8, Unit, Bb)],
) -> bool {
    let _ = (base, spec);
    for t in targets {
        let mut ring_ends: Bb = 0;
        let mut dmg: Vec<u32> = Vec::new();
        for &(ai, d) in &t.reach {
            let (_, u, ends) = atk[ai];
            let ring = bitboard::reachable(bit(t.cell), u.stats().range, FULL) & !bit(t.cell);
            ring_ends |= ends & ring;
            dmg.push(d);
        }
        dmg.sort_unstable_by(|a, b| b.cmp(a));
        let slots = ring_ends.count_ones() as usize;
        let best: u32 = dmg.iter().take(slots).sum();
        if best < t.need {
            return false;
        }
    }
    true
}

/// Attackers onto distinct hexes covering every target (most constrained
/// first). Within one target the attackers are taken in increasing index,
/// so each set is tried once whatever the order. `Some(true)` when `found`
/// accepted a placement, `Some(false)` when exhausted, `None` on the cap.
#[allow(clippy::too_many_arguments)]
fn place_dfs(
    targets: &[assign::Target],
    order: &[usize],
    ti: usize,
    owed: u32,
    min_ai: usize,
    atk: &[(u8, Unit, Bb)],
    rings: &[Vec<Bb>],
    used: u64,
    hexes: Bb,
    chosen: &mut Vec<(usize, u8)>,
    b: &mut Budget,
    found: &mut dyn FnMut(&[(usize, u8)], &mut Budget) -> bool,
) -> Option<bool> {
    if ti == order.len() {
        if b.verifies == 0 {
            return None;
        }
        return Some(found(chosen, b));
    }
    if b.nodes == 0 {
        return None;
    }
    b.nodes -= 1;
    let t = &targets[order[ti]];
    let owed = if owed == 0 { t.need } else { owed };
    let mut unknown = false;
    for &(ai, dmg) in &t.reach {
        if ai < min_ai || used & (1 << ai) != 0 {
            continue;
        }
        let free = atk[ai].2 & rings[order[ti]][ai] & !hexes;
        for h in bitboard::cells(free) {
            let (nti, nowed, nmin) = if dmg >= owed {
                (ti + 1, 0, 0)
            } else {
                (ti, owed - dmg, ai + 1)
            };
            chosen.push((ai, h));
            let r = place_dfs(
                targets,
                order,
                nti,
                nowed,
                nmin,
                atk,
                rings,
                used | (1 << ai),
                hexes | bit(h),
                chosen,
                b,
                found,
            );
            chosen.pop();
            match r {
                Some(true) => return Some(true),
                None => unknown = true,
                Some(false) => {}
            }
            if b.nodes == 0 || b.verifies == 0 {
                return None;
            }
        }
    }
    if unknown {
        None
    } else {
        Some(false)
    }
}

/// Complete a placement into a full declaration and verify it: undecided
/// friends not placed stay, unless their hex is needed, in which case they
/// move to a free hex (every assignment of free hexes is tried).
#[allow(clippy::too_many_arguments)]
fn complete(
    base: &MinionsState,
    spec: &TurnSpec,
    decided: &[u8],
    friends: &[u8],
    atk: &[(u8, Unit, Bb)],
    chosen: &[(usize, u8)],
    claimed: Bb,
    blockers: Bb,
    target_hexes: Bb,
    b: &mut Budget,
) -> Option<TurnSpec> {
    let mut w = *spec;
    w.spawns = [0; NUM_CELLS];
    let mut used = claimed;
    let mut placed: Bb = 0;
    for &f in decided {
        placed |= bit(f);
    }
    for &(ai, h) in chosen {
        let a = atk[ai].0;
        used |= bit(h);
        if decided.contains(&a) {
            continue;
        }
        w.moves[a as usize] = crate::turn::offset_index(a, h)?;
        placed |= bit(a);
    }
    // Stayers first (their hexes are then taken), then the displaced.
    let mut displaced = Vec::new();
    let mut stayers: Bb = 0;
    for &f in friends {
        if has(placed, f) {
            continue;
        }
        if has(used, f) {
            displaced.push(f);
        } else {
            w.moves[f as usize] = 0;
            used |= bit(f);
            stayers |= bit(f);
        }
    }
    relocate(
        base,
        &mut w,
        displaced,
        used,
        stayers,
        blockers,
        target_hexes,
        b,
    )
}

/// Move every displaced friend to a hex of its own: an empty one, a target's
/// (free once it dies), or a staying friend's, which that friend then has to
/// leave in turn (a chain). Every choice is tried until one verifies.
#[allow(clippy::too_many_arguments)]
fn relocate(
    base: &MinionsState,
    w: &mut TurnSpec,
    displaced: Vec<u8>,
    used: Bb,
    stayers: Bb,
    blockers: Bb,
    target_hexes: Bb,
    b: &mut Budget,
) -> Option<TurnSpec> {
    let Some((&f, rest)) = displaced.split_first() else {
        if b.verifies == 0 {
            b.incomplete = true;
            return None;
        }
        b.verifies -= 1;
        return verify(base, w, b).then_some(*w);
    };
    let (_, u) = base.piece_at(f).expect("friend");
    if !base.can_move_piece(f) {
        return None;
    }
    let st = u.stats();
    let pass = if st.flying { FULL } else { FULL & !blockers };
    let reach = bitboard::reachable(bit(f), st.speed, pass) & !bit(f) & !blockers;
    let empty = reach & !used;
    let mut cands: Vec<u8> = bitboard::cells(empty & !target_hexes).collect();
    cands.extend(bitboard::cells(empty & target_hexes));
    cands.extend(bitboard::cells(reach & stayers));
    for h in cands {
        let k = crate::turn::offset_index(f, h)?;
        w.moves[f as usize] = k;
        let (next, st2) = if has(stayers, h) {
            let mut next = rest.to_vec();
            next.push(h);
            (next, stayers & !bit(h))
        } else {
            (rest.to_vec(), stayers)
        };
        let r = relocate(base, w, next, used | bit(h), st2, blockers, target_hexes, b);
        if has(stayers, h) {
            w.moves[h as usize] = 0;
        }
        if let Some(done) = r {
            return Some(done);
        }
        if b.verifies == 0 {
            b.incomplete = true;
            return None;
        }
    }
    None
}

/// Does the complete declaration `w` realize? The fast realizers, then the
/// exact search over its (fully decided) actions; an exhausted budget there
/// marks the plan search incomplete.
fn verify(base: &MinionsState, w: &TurnSpec, b: &mut Budget) -> bool {
    if known_realizable(base, w) {
        return true;
    }
    let budget = if b.verified < plan_env("PLAN_VERIFY_DEEP", VERIFY_DEEP) {
        plan_env("PLAN_VERIFY_BUDGET", VERIFY_BUDGET)
    } else {
        plan_env("PLAN_VERIFY_SHALLOW", VERIFY_SHALLOW)
    };
    b.verified += 1;
    match exact_realize(base, w, side_bb(base, 0), side_bb(base, 1), budget) {
        Exact::Found(..) => true,
        Exact::Impossible => false,
        Exact::Unknown => {
            b.incomplete = true;
            why(3);
            false
        }
    }
}

/// Why the plan model misses a known witness `w` of the query (debugging):
/// the first plan-search stage that rejects `w`'s kill set or placement.
pub(super) fn explain_miss(
    base: &MinionsState,
    spec: &TurnSpec,
    decided: Bb,
    fated: Bb,
    w: &TurnSpec,
) -> String {
    let friends: Vec<u8> = (0..NUM_CELLS as u8)
        .filter(|&c| base.piece_at(c).is_some_and(|(sd, _)| sd == 0))
        .collect();
    let decided_list: Vec<u8> = friends
        .iter()
        .copied()
        .filter(|&f| has(decided, f))
        .collect();
    let mut sp = *spec;
    let mut extras = Vec::new();
    for e in 0..NUM_CELLS as u8 {
        if base.piece_at(e).is_some_and(|(sd, _)| sd != 0) && !has(fated, e) {
            sp.targets[e as usize] = w.targets[e as usize];
            if w.targets[e as usize] != TARGET_STAY {
                extras.push(e);
            }
        }
    }
    let atk = attackers(base, &sp, &decided_list, 0, 0);
    let mut notes = format!(
        "extras {:?}; ",
        extras
            .iter()
            .map(|&e| crate::map::cell_name(e))
            .collect::<Vec<_>>()
    );
    for &f in &friends {
        let d = dest(w, f);
        match atk.iter().find(|a| a.0 == f) {
            Some(&(_, _, ends)) if !has(ends, d) => {
                notes += &format!(
                    "{}->{} not in ends; ",
                    crate::map::cell_name(f),
                    crate::map::cell_name(d)
                )
            }
            _ => {}
        }
    }
    let Some(targets) = kill_targets(base, &sp, &atk) else {
        return notes + "kill_targets: a target unreachable";
    };
    if !lethal_capacity(base, &sp, &targets, &atk) {
        return notes + "lethal_capacity rejects";
    }
    let a_: Vec<assign::Attacker> = atk
        .iter()
        .map(|&(c, _, _)| assign::Attacker {
            cell: c,
            attacks: 1,
        })
        .collect();
    if assign::hall_violation(&targets, &a_).is_some() {
        return notes + "hall rejects";
    }
    let mut b = Budget {
        nodes: NODE_CAP,
        verifies: VERIFY_CAP,
        sets: SETS_CAP,
        incomplete: false,
        verified: 0,
    };
    let r = placements(base, &sp, &decided_list, &friends, &mut b);
    notes += &format!(
        "placements -> {} (incomplete {}, verifies used {}, nodes used {}); targets {:?}\n",
        r.is_some(),
        b.incomplete,
        VERIFY_CAP - b.verifies,
        NODE_CAP - b.nodes,
        targets
            .iter()
            .map(|t| (crate::map::cell_name(t.cell), t.need, t.reach.len()))
            .collect::<Vec<_>>()
    );
    for &f in &friends {
        let (_, u) = base.piece_at(f).expect("friend");
        notes += &format!(
            "   F {} {:?} {} -> {}{}\n",
            crate::map::cell_name(f),
            u,
            if has(decided, f) { "decided" } else { "free" },
            crate::map::cell_name(dest(w, f)),
            if atk.iter().any(|a| a.0 == f) {
                " (attacker)"
            } else {
                ""
            }
        );
    }
    for e in 0..NUM_CELLS as u8 {
        if let Some((sd, u)) = base.piece_at(e) {
            if sd != 0 {
                notes += &format!(
                    "   E {} {:?} dmg {} {} {}\n",
                    crate::map::cell_name(e),
                    u,
                    base.damage_on(e),
                    if has(fated, e) { "fated" } else { "unfated" },
                    ["stay", "KILL", "UNSUMMON"][w.targets[e as usize] as usize]
                );
            }
        }
    }
    notes
}
