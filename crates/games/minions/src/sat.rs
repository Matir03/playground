//! Exact SAT check of a turn's attack phase (CaDiCaL, incremental).
//!
//! Is there a legal sequence of moves and attacks that takes every friendly
//! unit to its declared cell and gives every enemy its declared fate (stay,
//! killed, unsummoned)? [`TurnSat::new`] encodes a turn start once;
//! [`TurnSat::complete`] answers for a partial declaration (the decided
//! variables of a lattice node) as assumptions: a witness completing it, or
//! a refutation. No caps or deadlines: the answer is exact.
//!
//! The encoding. Per friend `u`: its final cell (one-hot over its cell and
//! the cells it can reach within its speed) and at most one attack target
//! (an enemy adjacent to its final cell: it attacks after moving, if it
//! moves; a lumbering unit that attacks does not move). Per enemy: its fate,
//! the total damage it takes (a weighted counter; the necromancer's unsummon
//! counts 1 against persistent units and removes the others), so that
//! killed = damage >= defense without an unsummon, stay = below it without
//! one, unsummoned = an unsummon after any other attacks, which stay below
//! defense. Timing: events are each friend's move, its attack and each
//! enemy's removal; every rule is a precedence -- move before attack, attack
//! before its target's removal, a destination held by an enemy (by a friend)
//! only after that enemy's removal (that friend's move), a non-flying path
//! through an enemy's cell only after its removal -- and a sequence exists
//! iff those precedences are acyclic (`before` variables, transitive and
//! antisymmetric). The one exception is the cyclic move: a destination held
//! by a friend may instead be entered *at once* with that friend's move, a
//! `tie` (the two moves unordered with each other and ordered alike against
//! every other event). A tied group is played as its rings, each one
//! [`MinionsMove::Cycle`], and its chains, in order. Overkill is harmless: attacks are not part of the
//! declaration, and one landing on a dead unit is simply not made.
//! [`verify`] replays a witness's schedule through the game rules.

use crate::bitboard::{bit, cells, dilate, has};
use crate::game::{Minions, MinionsMove, MinionsState};
use crate::map::NUM_CELLS;
use crate::turn::{
    offset_index, TurnSpec, MOVE_DEST, N_OFFSETS, TARGET_KILL, TARGET_STAY, TARGET_UNSUMMON,
};
use crate::units::{Attack, Unit};
use playground_core::Game;

/// A decided variable of a partial declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fix {
    /// The friend on `cell` moves by offset `k` (0: stays).
    Move { cell: u8, k: u8 },
    /// The enemy on `cell` gets `TARGET_*` fate `t`.
    Fate { cell: u8, t: u8 },
}

pub enum SatResult {
    /// A complete fates + moves declaration extending the fixes (spawns
    /// empty), and the order of its moves and attacks.
    Witness(TurnSpec, Vec<MinionsMove>),
    /// No completion: the fixes whose assumptions the refutation used.
    Refuted(Vec<Fix>),
}

struct Friend {
    cell: u8,
    unit: Unit,
    /// Final-cell candidates and their variables (the origin first).
    dest: Vec<(u8, i32)>,
    /// Attack candidates: (enemy index, variable).
    att: Vec<(usize, i32)>,
    ev_move: usize,
    ev_att: usize,
}

struct Enemy {
    cell: u8,
    unit: Unit,
    /// Fate variables (stay, kill, unsummon); 0: impossible.
    fate: [i32; 3],
    ev_rem: usize,
}

pub struct TurnSat {
    s: cadical::Solver,
    nv: i32,
    base: MinionsState,
    friends: Vec<Friend>,
    enemies: Vec<Enemy>,
    /// `before[x * n_ev + y]`: event x happens before event y.
    before: Vec<i32>,
    n_ev: usize,
    /// While encoding: (destination variable, friend there, its move event,
    /// the mover's move event) -- tied once every friend has its variables.
    pending_vacate: Vec<(i32, usize, usize, usize)>,
    /// The `tie` variables of pairs of friends' moves (indices, lower first).
    ties: Vec<(usize, usize, i32)>,
    pub calls: u64,
}

impl TurnSat {
    fn var(&mut self) -> i32 {
        self.nv += 1;
        self.nv
    }

    fn clause(&mut self, c: &[i32]) {
        self.s.add_clause(c.iter().copied());
    }

    fn b(&self, x: usize, y: usize) -> i32 {
        self.before[x * self.n_ev + y]
    }

    fn amo(&mut self, lits: &[i32]) {
        for i in 0..lits.len() {
            for j in i + 1..lits.len() {
                self.clause(&[-lits[i], -lits[j]]);
            }
        }
    }

    /// Encode the attack phase of the turn starting at `base` (canonical:
    /// the mover is side 0, nothing moved or attacked yet).
    pub fn new(base: &MinionsState) -> Self {
        debug_assert_eq!(base.side, 0);
        let mut t = TurnSat {
            s: cadical::Solver::new(),
            nv: 0,
            base: base.clone(),
            friends: Vec::new(),
            enemies: Vec::new(),
            before: Vec::new(),
            n_ev: 0,
            pending_vacate: Vec::new(),
            ties: Vec::new(),
            calls: 0,
        };
        let mut enemy_at = [usize::MAX; NUM_CELLS];
        for c in 0..NUM_CELLS as u8 {
            match base.piece_at(c) {
                Some((0, unit)) => t.friends.push(Friend {
                    cell: c,
                    unit,
                    dest: Vec::new(),
                    att: Vec::new(),
                    ev_move: 0,
                    ev_att: 0,
                }),
                Some((_, unit)) => {
                    enemy_at[c as usize] = t.enemies.len();
                    t.enemies.push(Enemy {
                        cell: c,
                        unit,
                        fate: [0; 3],
                        ev_rem: 0,
                    });
                }
                None => {}
            }
        }
        // Events.
        let (nf, ne) = (t.friends.len(), t.enemies.len());
        for (i, f) in t.friends.iter_mut().enumerate() {
            f.ev_move = 2 * i;
            f.ev_att = 2 * i + 1;
        }
        for (i, e) in t.enemies.iter_mut().enumerate() {
            e.ev_rem = 2 * nf + i;
        }
        t.n_ev = 2 * nf + ne;
        let n = t.n_ev;
        t.before = vec![0; n * n];
        for x in 0..n {
            for y in 0..n {
                if x != y {
                    t.before[x * n + y] = t.var();
                }
            }
        }
        for x in 0..n {
            for y in x + 1..n {
                let (a, b) = (t.b(x, y), t.b(y, x));
                t.clause(&[-a, -b]);
            }
        }
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    if x != y && y != z && x != z {
                        let (a, b, c) = (t.b(x, y), t.b(y, z), t.b(x, z));
                        t.clause(&[-a, -b, c]);
                    }
                }
            }
        }
        // Fates.
        for i in 0..ne {
            let unit = t.enemies[i].unit;
            let st = unit.stats();
            let stay = t.var();
            // The enemy necromancer is never killed in a lattice: a turn that
            // can kill it is terminal and has none.
            let kill = if st.necromancer { 0 } else { t.var() };
            let uns = if st.persistent { 0 } else { t.var() };
            let fs: Vec<i32> = [stay, kill, uns].into_iter().filter(|&v| v != 0).collect();
            t.clause(&fs);
            t.amo(&fs);
            t.enemies[i].fate = [stay, kill, uns];
        }
        // Friends: final cells, paths, attacks.
        let enemies_bb = base.side_bb(1);
        for fi in 0..nf {
            let (o, unit) = (t.friends[fi].cell, t.friends[fi].unit);
            let st = unit.stats();
            let stay = t.var();
            t.friends[fi].dest.push((o, stay));
            let mut cand: Vec<(u8, Vec<u64>)> = Vec::new();
            for k in 1..N_OFFSETS {
                let d = MOVE_DEST[o as usize][k];
                if d as usize >= NUM_CELLS {
                    continue;
                }
                let sets = blocker_sets(
                    o,
                    d,
                    st.speed,
                    if st.flying { 0 } else { enemies_bb },
                    &enemy_at,
                );
                if !sets.is_empty() {
                    cand.push((d, sets));
                }
            }
            let mut dvars = vec![stay];
            for (d, sets) in cand {
                let x = t.var();
                dvars.push(x);
                t.friends[fi].dest.push((d, x));
                let ev_m = t.friends[fi].ev_move;
                // The destination's occupant leaves first.
                match base.piece_at(d) {
                    Some((0, _)) => {
                        // That friend may not have its variables yet: tied below.
                        let v = t.friends.iter().position(|f| f.cell == d).unwrap();
                        let ev_v = t.friends[v].ev_move;
                        t.pending_vacate.push((x, v, ev_v, ev_m));
                    }
                    Some((_, _)) => {
                        let e = enemy_at[d as usize];
                        let (st_e, ev_r) = (t.enemies[e].fate[0], t.enemies[e].ev_rem);
                        let bb = t.b(ev_r, ev_m);
                        t.clause(&[-x, -st_e]);
                        t.clause(&[-x, bb]);
                    }
                    None => {}
                }
                // A path whose enemies are all removed first.
                if sets.iter().any(|&s| s == 0) {
                    continue;
                }
                let mut alts = vec![-x];
                for s in sets {
                    let p = t.var();
                    alts.push(p);
                    let mut m = s;
                    while m != 0 {
                        let e = m.trailing_zeros() as usize;
                        m &= m - 1;
                        let (st_e, ev_r) = (t.enemies[e].fate[0], t.enemies[e].ev_rem);
                        let bb = t.b(ev_r, ev_m);
                        t.clause(&[-p, -st_e]);
                        t.clause(&[-p, bb]);
                    }
                }
                t.clause(&alts);
            }
            t.clause(&dvars);
            t.amo(&dvars);
            // Attacks: an enemy adjacent to the final cell.
            let mut avars = Vec::new();
            for ei in 0..ne {
                let ec = t.enemies[ei].cell;
                let near: Vec<i32> = t.friends[fi]
                    .dest
                    .iter()
                    .filter(|&&(d, _)| has(dilate(bit(d)), ec))
                    .map(|&(_, v)| v)
                    .collect();
                if near.is_empty() || st.range != 1 {
                    continue;
                }
                let a = t.var();
                avars.push(a);
                t.friends[fi].att.push((ei, a));
                let mut c = vec![-a];
                c.extend(near);
                t.clause(&c);
                let (ev_m, ev_a, ev_r) = (
                    t.friends[fi].ev_move,
                    t.friends[fi].ev_att,
                    t.enemies[ei].ev_rem,
                );
                if st.lumbering {
                    t.clause(&[-a, stay]);
                } else {
                    let bb = t.b(ev_m, ev_a);
                    t.clause(&[-a, stay, bb]);
                }
                let st_e = t.enemies[ei].fate[0];
                let bb = t.b(ev_a, ev_r);
                t.clause(&[-a, st_e, bb]);
            }
            t.amo(&avars);
        }
        // Destinations held by friends: that friend moves first, or at once
        // (a tie: part of a cyclic move).
        for (x, v, ev_v, ev_m) in std::mem::take(&mut t.pending_vacate) {
            let vstay = t.friends[v].dest[0].1;
            let bb = t.b(ev_v, ev_m);
            let tie = t.tie(ev_m / 2, v);
            t.clause(&[-x, -vstay]);
            t.clause(&[-x, bb, tie]);
        }
        // One unit per final cell; a staying enemy holds its cell.
        for c in 0..NUM_CELLS as u8 {
            let mut lits: Vec<i32> = t
                .friends
                .iter()
                .flat_map(|f| f.dest.iter().filter(|&&(d, _)| d == c).map(|&(_, v)| v))
                .collect();
            if enemy_at[c as usize] != usize::MAX {
                lits.push(t.enemies[enemy_at[c as usize]].fate[0]);
            }
            t.amo(&lits);
        }
        // Damage and fates.
        for ei in 0..ne {
            let def = t.enemies[ei].unit.stats().defense as usize;
            let persistent = t.enemies[ei].unit.stats().persistent;
            let [stay, kill, uns] = t.enemies[ei].fate;
            let mut dmg: Vec<(i32, usize)> = Vec::new();
            let mut bounce: Vec<(i32, usize)> = Vec::new();
            for fi in 0..nf {
                for &(e, a) in &t.friends[fi].att {
                    if e != ei {
                        continue;
                    }
                    match t.friends[fi].unit.stats().attack {
                        Attack::Damage(d) => dmg.push((a, d as usize)),
                        Attack::Unsummon if persistent => dmg.push((a, 1)),
                        Attack::Unsummon => bounce.push((a, fi)),
                    }
                }
            }
            // A spared enemy is untouched (the declarations' convention:
            // damage-only attacks never help, damage is wiped at turn end).
            for &(a, _) in dmg
                .iter()
                .chain(&bounce.iter().map(|&(a, _)| (a, 0)).collect::<Vec<_>>())
            {
                t.clause(&[-stay, -a]);
            }
            let ge = t.counter(&dmg, def);
            for &(a, _) in &bounce {
                t.clause(&[-stay, -a]);
                if kill != 0 {
                    t.clause(&[-kill, -a]);
                }
            }
            t.clause(&[-stay, -ge]);
            if kill != 0 {
                t.clause(&[-kill, ge]);
            }
            if uns != 0 {
                let mut c = vec![-uns];
                c.extend(bounce.iter().map(|&(a, _)| a));
                t.clause(&c);
                t.clause(&[-uns, -ge]);
                // Its other attackers strike before the unsummon.
                for &(b, nfi) in &bounce {
                    let ev_n = t.friends[nfi].ev_att;
                    for fi in 0..nf {
                        if fi == nfi {
                            continue;
                        }
                        for &(e, a) in &t.friends[fi].att.clone() {
                            if e == ei {
                                let bb = t.b(t.friends[fi].ev_att, ev_n);
                                t.clause(&[-uns, -b, -a, bb]);
                            }
                        }
                    }
                }
            }
        }
        t
    }

    /// The tie variable of friends `u` and `v`'s moves, made on first use:
    /// neither before the other, and alike against every other event.
    fn tie(&mut self, u: usize, v: usize) -> i32 {
        let (u, v) = (u.min(v), u.max(v));
        if let Some(&(_, _, t)) = self.ties.iter().find(|&&(a, b, _)| a == u && b == v) {
            return t;
        }
        let t = self.var();
        self.ties.push((u, v, t));
        let (mu, mv) = (self.friends[u].ev_move, self.friends[v].ev_move);
        let (a, b) = (self.b(mu, mv), self.b(mv, mu));
        self.clause(&[-t, -a]);
        self.clause(&[-t, -b]);
        for z in 0..self.n_ev {
            if z == mu || z == mv {
                continue;
            }
            for (x, y) in [(mu, mv), (mv, mu)] {
                let (zx, zy) = (self.b(z, x), self.b(z, y));
                self.clause(&[-t, -zx, zy]);
                let (xz, yz) = (self.b(x, z), self.b(y, z));
                self.clause(&[-t, -xz, yz]);
            }
        }
        t
    }

    /// A literal true iff the weighted sum of `terms` reaches `cap`
    /// (sequential counter, both directions).
    fn counter(&mut self, terms: &[(i32, usize)], cap: usize) -> i32 {
        // prev[j - 1]: the sum so far >= j (j = 1..=cap); 0 = false.
        let mut prev = vec![0i32; cap];
        for &(a, w) in terms {
            let mut cur = vec![0i32; cap];
            for j in 1..=cap {
                let s = self.var();
                cur[j - 1] = s;
                let p = prev[j - 1];
                if p != 0 {
                    self.clause(&[-p, s]);
                }
                if j <= w {
                    self.clause(&[-a, s]);
                } else if prev[j - w - 1] != 0 {
                    let q = prev[j - w - 1];
                    self.clause(&[-a, -q, s]);
                }
                // s -> p or (a and sum-so-far >= j - w)
                let mut c = vec![-s, a];
                if p != 0 {
                    c.push(p);
                }
                self.clause(&c);
                if j > w {
                    let q = prev[j - w - 1];
                    let mut c = vec![-s];
                    if p != 0 {
                        c.push(p);
                    }
                    if q != 0 {
                        c.push(q);
                    }
                    self.clause(&c);
                }
            }
            prev = cur;
        }
        match prev[cap - 1] {
            0 => {
                let f = self.var();
                self.clause(&[-f]);
                f
            }
            v => v,
        }
    }

    /// A completion of `fixes`, or a refutation (exact).
    pub fn complete(&mut self, fixes: &[Fix]) -> SatResult {
        self.calls += 1;
        let mut assume: Vec<(i32, Fix)> = Vec::with_capacity(fixes.len());
        for &f in fixes {
            let lit = match f {
                Fix::Move { cell, k } => {
                    let d = MOVE_DEST[cell as usize][k as usize];
                    self.friends
                        .iter()
                        .find(|x| x.cell == cell)
                        .and_then(|x| x.dest.iter().find(|&&(c, _)| c == d).map(|&(_, v)| v))
                }
                Fix::Fate { cell, t } => self
                    .enemies
                    .iter()
                    .find(|x| x.cell == cell)
                    .and_then(|x| (x.fate[t as usize] != 0).then_some(x.fate[t as usize])),
            };
            match lit {
                Some(l) => assume.push((l, f)),
                None => return SatResult::Refuted(vec![f]),
            }
        }
        match self.s.solve_with(assume.iter().map(|&(l, _)| l)) {
            Some(true) => {
                let (spec, order) = self.model();
                SatResult::Witness(spec, order)
            }
            _ => SatResult::Refuted(
                assume
                    .iter()
                    .filter(|&&(l, _)| self.s.failed(l))
                    .map(|&(_, f)| f)
                    .collect(),
            ),
        }
    }

    fn val(&self, l: i32) -> bool {
        self.s.value(l) == Some(true)
    }

    /// The model's declaration and a schedule for it.
    fn model(&self) -> (TurnSpec, Vec<MinionsMove>) {
        let mut spec = TurnSpec::empty();
        let mut fdest = vec![0u8; self.friends.len()];
        for (i, f) in self.friends.iter().enumerate() {
            let &(d, _) = f
                .dest
                .iter()
                .find(|&&(_, v)| self.val(v))
                .expect("one final cell");
            fdest[i] = d;
            spec.moves[f.cell as usize] = offset_index(f.cell, d).expect("an offset");
        }
        for e in &self.enemies {
            let t = (0..3)
                .find(|&k| e.fate[k] != 0 && self.val(e.fate[k]))
                .expect("one fate");
            spec.targets[e.cell as usize] = [TARGET_STAY, TARGET_KILL, TARGET_UNSUMMON][t];
        }
        // Tied moves (union-find over the true ties): one event per group.
        let nf = self.friends.len();
        let mut root: Vec<usize> = (0..nf).collect();
        fn find(r: &mut [usize], mut x: usize) -> usize {
            while r[x] != x {
                r[x] = r[r[x]];
                x = r[x];
            }
            x
        }
        for &(u, v, t) in &self.ties {
            if self.val(t) {
                let (a, b) = (find(&mut root, u), find(&mut root, v));
                root[a] = b;
            }
        }
        // Events that happen, ordered by how many others precede them (a
        // tied group's members all have the same count).
        let mut evs: Vec<(usize, MinionsMove)> = Vec::new();
        for (i, f) in self.friends.iter().enumerate() {
            if fdest[i] == f.cell || find(&mut root, i) != i {
                // A tied group is played from its root, below.
            } else {
                let group: Vec<usize> = (0..nf)
                    .filter(|&j| fdest[j] != self.friends[j].cell && find(&mut root, j) == i)
                    .collect();
                for m in group_moves(&group.iter().map(|&j| (self.friends[j].cell, fdest[j])).collect::<Vec<_>>()) {
                    evs.push((f.ev_move, m));
                }
            }
            if let Some(&(e, _)) = f.att.iter().find(|&&(_, a)| self.val(a)) {
                evs.push((
                    f.ev_att,
                    MinionsMove::Attack {
                        from: fdest[i],
                        to: self.enemies[e].cell,
                    },
                ));
            }
        }
        // A group whose root stays: play it from its first mover.
        for i in 0..nf {
            let r = find(&mut root, i);
            if r == i || fdest[r] != self.friends[r].cell || fdest[i] == self.friends[i].cell {
                continue;
            }
            let first = (0..nf)
                .find(|&j| find(&mut root, j) == r && fdest[j] != self.friends[j].cell)
                .expect("a mover");
            if first != i {
                continue;
            }
            let group: Vec<usize> = (0..nf)
                .filter(|&j| fdest[j] != self.friends[j].cell && find(&mut root, j) == r)
                .collect();
            for m in group_moves(&group.iter().map(|&j| (self.friends[j].cell, fdest[j])).collect::<Vec<_>>()) {
                evs.push((self.friends[i].ev_move, m));
            }
        }
        let mut keyed: Vec<(usize, MinionsMove)> = evs
            .iter()
            .map(|&(x, m)| {
                let k = (0..self.n_ev)
                    .filter(|&y| y != x && self.val(self.b(y, x)))
                    .count();
                (k, m)
            })
            .collect();
        keyed.sort_by_key(|&(k, _)| k);
        (spec, keyed.into_iter().map(|(_, m)| m).collect())
    }

    pub fn base(&self) -> &MinionsState {
        &self.base
    }
}

/// Simultaneous moves `(origin, destination)` as legal actions: each one
/// whose destination no other member still stands on, in turn, then the
/// rings left as cyclic moves.
fn group_moves(group: &[(u8, u8)]) -> Vec<MinionsMove> {
    let mut left: Vec<(u8, u8)> = group.to_vec();
    let mut out = Vec::with_capacity(left.len());
    loop {
        let Some(i) = left.iter().position(|&(_, d)| !left.iter().any(|&(o, _)| o == d)) else {
            break;
        };
        let (from, to) = left.remove(i);
        out.push(MinionsMove::Move { from, to });
    }
    while let Some(&(o, _)) = left.first() {
        let mut ring = vec![o];
        let mut x = left.iter().find(|&&(a, _)| a == o).unwrap().1;
        while x != o {
            ring.push(x);
            x = left.iter().find(|&&(a, _)| a == x).expect("a ring").1;
        }
        left.retain(|&(a, _)| !ring.contains(&a));
        out.push(MinionsMove::cycle(&ring));
    }
    out
}

/// The decided variables of a turn under construction (a lattice node):
/// fates of the enemies, then moves of the friends, decided so far.
pub fn fixes_of(s: &crate::steps::StepState) -> Vec<Fix> {
    use crate::steps::Stage;
    let (nf, nm) = match s.stage {
        Stage::Fates => (s.idx, 0),
        Stage::Moves => (s.enemies.len(), s.idx),
        Stage::Spawns | Stage::Terminal => (s.enemies.len(), s.friends.len()),
    };
    let mut out = Vec::with_capacity(nf + nm);
    for &e in &s.enemies[..nf] {
        out.push(Fix::Fate {
            cell: e,
            t: s.spec.targets[e as usize],
        });
    }
    for &f in &s.friends[..nm] {
        out.push(Fix::Move {
            cell: f,
            k: s.spec.moves[f as usize],
        });
    }
    out
}

/// The minimal sets of enemies (bit masks of enemy indices) whose removal
/// opens some path of at most `speed` steps from `o` to `d` (empty: none
/// needed; no sets: unreachable). `blockers`: the enemy cells that block
/// (none for flying units).
fn blocker_sets(
    o: u8,
    d: u8,
    speed: u8,
    blockers: u128,
    enemy_at: &[usize; NUM_CELLS],
) -> Vec<u64> {
    let mut sets: Vec<u64> = Vec::new();
    fn walk(
        at: u8,
        d: u8,
        left: u8,
        mask: u64,
        seen: u128,
        blockers: u128,
        enemy_at: &[usize; NUM_CELLS],
        out: &mut Vec<u64>,
    ) {
        for n in cells(dilate(bit(at)) & !seen) {
            let m = if has(blockers, n) {
                mask | 1 << enemy_at[n as usize]
            } else {
                mask
            };
            if n == d {
                out.push(m);
            } else if left > 1 {
                walk(n, d, left - 1, m, seen | bit(n), blockers, enemy_at, out);
            }
        }
    }
    walk(o, d, speed, 0, bit(o), blockers, enemy_at, &mut sets);
    sets.sort_unstable();
    sets.dedup();
    let all = sets.clone();
    sets.retain(|&s| !all.iter().any(|&t| t != s && t & s == t));
    sets
}

/// Replay `order` from `base` through the game rules (skipping attacks on
/// units already gone) and check it realizes `spec`'s moves and fates.
pub fn verify(base: &MinionsState, spec: &TurnSpec, order: &[MinionsMove]) -> Result<(), String> {
    let mut s = base.clone();
    let mut bounced = [0u8; 5];
    for &m in order {
        if let MinionsMove::Attack { to, .. } = m {
            if s.piece_at(to).is_none_or(|(side, _)| side == 0) {
                continue;
            }
        }
        if !Minions::is_legal(&s, &m) {
            return Err(format!("illegal {m:?}"));
        }
        s = Minions::apply(&s, &m);
    }
    for c in 0..NUM_CELLS as u8 {
        if let Some((0, u)) = base.piece_at(c) {
            let d = MOVE_DEST[c as usize][spec.moves[c as usize] as usize];
            if s.piece_at(d) != Some((0, u)) {
                return Err(format!("friend from {c} not at {d}"));
            }
        }
        if let Some((1, u)) = base.piece_at(c) {
            let gone = s.piece_at(c).is_none_or(|(side, _)| side == 0);
            let want = spec.targets[c as usize];
            if gone != (want != TARGET_STAY) {
                return Err(format!("enemy {u:?} at {c}: fate {want}, gone {gone}"));
            }
            if want == TARGET_UNSUMMON {
                if let Some(k) = u.reinforcement_slot() {
                    bounced[k] += 1;
                }
            }
        }
    }
    // Unsummons, per unit type, exactly as declared.
    for k in 0..5 {
        let got = s.reinforcements[1][k] - base.reinforcements[1][k];
        if got != bounced[k] {
            return Err(format!(
                "slot {k}: {got} unsummoned, {} declared",
                bounced[k]
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::steps::{exact_realize, Exact, StepState};
    use rand::{Rng, SeedableRng};

    /// A random attack phase from `base` and the declaration it realizes.
    /// (None: it killed the enemy necromancer -- a terminal turn, which
    /// never has a lattice.)
    fn random_turn(
        base: &MinionsState,
        rng: &mut impl Rng,
    ) -> Option<(TurnSpec, Vec<MinionsMove>)> {
        let mut played = Vec::new();
        let mut removed = [TARGET_STAY; NUM_CELLS];
        let mut s = base.clone();
        // Where each friend that started on a cell is now.
        let mut at: Vec<(u8, u8)> = (0..NUM_CELLS as u8)
            .filter(|&c| matches!(base.piece_at(c), Some((0, _))))
            .map(|c| (c, c))
            .collect();
        for _ in 0..rng.gen_range(0..30) {
            let ms: Vec<MinionsMove> = Minions::legal_moves(&s)
                .into_iter()
                .filter(|m| !matches!(m, MinionsMove::EndPhase))
                .collect();
            if ms.is_empty() || s.winner != 0 {
                break;
            }
            let m = ms[rng.gen_range(0..ms.len())];
            // All at once: a cycle's members trade cells.
            let moved: Vec<(usize, u8)> = m
                .relocations()
                .filter_map(|(from, to)| at.iter().position(|x| x.1 == from).map(|i| (i, to)))
                .collect();
            for (i, to) in moved {
                at[i].1 = to;
            }
            played.push(m);
            let mut fate = None;
            if let MinionsMove::Attack { from, to } = m {
                let (_, au) = s.piece_at(from).unwrap();
                let (_, tu) = s.piece_at(to).unwrap();
                fate = Some((to, au == Unit::Necromancer && !tu.stats().persistent));
            }
            s = Minions::apply(&s, &m);
            if let Some((to, bounce)) = fate {
                if s.piece_at(to).is_none() {
                    removed[to as usize] = if bounce { TARGET_UNSUMMON } else { TARGET_KILL };
                }
            }
        }
        let mut spec = TurnSpec::empty();
        for &(o, d) in &at {
            spec.moves[o as usize] = offset_index(o, d).unwrap();
        }
        for c in 0..NUM_CELLS {
            spec.targets[c] = removed[c];
        }
        (s.winner == 0).then_some((spec, played))
    }

    fn fixes_of(
        base: &MinionsState,
        spec: &TurnSpec,
        mut keep: impl FnMut(u8) -> bool,
    ) -> (Vec<Fix>, u128, u128) {
        let (mut f, mut dec, mut fat) = (Vec::new(), 0u128, 0u128);
        for c in 0..NUM_CELLS as u8 {
            if !keep(c) {
                continue;
            }
            match base.piece_at(c) {
                Some((0, _)) => {
                    f.push(Fix::Move {
                        cell: c,
                        k: spec.moves[c as usize],
                    });
                    dec |= bit(c);
                }
                Some((_, _)) => {
                    f.push(Fix::Fate {
                        cell: c,
                        t: spec.targets[c as usize],
                    });
                    fat |= bit(c);
                }
                None => {}
            }
        }
        (f, dec, fat)
    }

    /// Cost of encoding a turn and of queries on it (random partial
    /// declarations from played turns, and perturbed ones).
    #[test]
    #[ignore]
    fn sat_speed() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let (mut enc, mut q, mut nq, mut worst, mut vars, mut clauses) =
            (0f64, 0f64, 0u32, 0f64, 0f64, 0f64);
        let mut n = 0;
        for g in 0..400u64 {
            let real = crate::selfplay::random_opening(22, 1.0884, &mut rng).0;
            let st = StepState::begin(&real, g);
            if crate::lattice::necro_kill(&st).is_ok() {
                continue;
            }
            let t0 = std::time::Instant::now();
            let mut ts = TurnSat::new(&st.base);
            enc += t0.elapsed().as_secs_f64();
            vars += f64::from(ts.s.num_variables());
            clauses += ts.s.num_clauses() as f64;
            n += 1;
            for _ in 0..10 {
                let Some((spec, _)) = random_turn(&st.base, &mut rng) else {
                    continue;
                };
                let p = rng.gen_range(0.0..1.0);
                let (fx, _, _) = fixes_of(&st.base, &spec, |_| rng.gen_bool(p));
                let mut fx2 = fx.clone();
                for f in fx2.iter_mut() {
                    if let Fix::Fate { t, .. } = f {
                        if rng.gen_bool(0.2) {
                            *t = if *t == TARGET_STAY {
                                TARGET_KILL
                            } else {
                                TARGET_STAY
                            };
                        }
                    }
                }
                for f in [fx, fx2] {
                    let t0 = std::time::Instant::now();
                    let _ = ts.complete(&f);
                    let e = t0.elapsed().as_secs_f64();
                    q += e;
                    worst = worst.max(e);
                    nq += 1;
                }
            }
        }
        eprintln!(
            "{n} turns: encode {:.2} ms ({:.0} vars, {:.0} clauses); {nq} queries {:.3} ms mean, worst {:.1} ms",
            enc / n as f64 * 1e3,
            vars / n as f64,
            clauses / n as f64,
            q / f64::from(nq) * 1e3,
            worst * 1e3
        );
    }

    /// Played declarations (and random subsets of them) are satisfiable and
    /// every witness replays; perturbed ones agree with the exact search
    /// wherever it concludes.
    /// A declared ring of moves completes, as one cyclic move; a unit may
    /// still not enter the cell of a friend who stays.
    #[test]
    fn sat_completes_cyclic_moves() {
        let units = [Unit::Zombie, Unit::Warg, Unit::Serpent, Unit::Ghost];
        for n in [2, 3, 4] {
            let (mut base, ring) = crate::turn::tests::bare_with_ring(n);
            let mut fixes = Vec::new();
            for i in 0..n {
                base.cells[ring[i] as usize] = 1 + units[i] as u8;
                let k = offset_index(ring[i], ring[(i + 1) % n]).unwrap();
                fixes.push(Fix::Move { cell: ring[i], k });
            }
            let mut sat = TurnSat::new(&base);
            match sat.complete(&fixes) {
                SatResult::Witness(spec, order) => {
                    verify(&base, &spec, &order).unwrap();
                    assert!(order.iter().any(|m| m.cycle_cells().len() == n), "{order:?}");
                }
                SatResult::Refuted(f) => panic!("ring of {n} refuted: {f:?}"),
            }
            // The last unit stays instead: nobody can enter its cell.
            let mut stuck = fixes.clone();
            stuck[n - 1] = Fix::Move { cell: ring[n - 1], k: 0 };
            assert!(matches!(sat.complete(&stuck), SatResult::Refuted(_)), "ring of {n}");
        }
    }

    #[test]
    fn sat_agrees_with_exact_search() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        let (mut sat_ok, mut agree, mut found, mut impossible, mut unknown) = (0, 0, 0, 0, 0);
        let mut false_refuted = 0;
        let n: u64 = std::env::var("SAT_TEST_POSITIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60);
        for g in 0..n {
            let real = crate::selfplay::random_opening(22, 1.0884, &mut rng).0;
            let base = StepState::begin(&real, g).base;
            if crate::lattice::necro_kill(&StepState::begin(&real, g)).is_ok() {
                continue;
            }
            let mut ts = TurnSat::new(&base);
            for _ in 0..6 {
                let Some((spec, played)) = random_turn(&base, &mut rng) else {
                    continue;
                };
                let p = rng.gen_range(0.0..1.0);
                let (fx, _, _) = fixes_of(&base, &spec, |_| rng.gen_bool(p));
                match ts.complete(&fx) {
                    SatResult::Witness(w, order) => {
                        verify(&base, &w, &order).unwrap_or_else(|e| {
                            panic!("bad witness: {e}\n{}\nfixes {fx:?}", base.render())
                        });
                        sat_ok += 1;
                    }
                    SatResult::Refuted(core) => {
                        panic!("a played declaration refuted\n{}\nfixes {fx:?}\nplayed {played:?}\ncore {core:?}", base.render())
                    }
                }
                // Perturbed: random other choices for a few variables.
                let mut q = spec;
                for c in 0..NUM_CELLS as u8 {
                    if rng.gen_bool(0.15) {
                        match base.piece_at(c) {
                            Some((0, _)) => q.moves[c as usize] = rng.gen_range(0..N_OFFSETS as u8),
                            // Only fates a lattice offers: no unsummon of a
                            // persistent unit, no necromancer kill.
                            Some((_, u)) => {
                                let st = u.stats();
                                let ok: Vec<u8> = [TARGET_STAY, TARGET_KILL, TARGET_UNSUMMON]
                                    .into_iter()
                                    .filter(|&t| {
                                        !(t == TARGET_UNSUMMON && st.persistent)
                                            && !(t == TARGET_KILL && st.necromancer)
                                    })
                                    .collect();
                                q.targets[c as usize] = ok[rng.gen_range(0..ok.len())];
                            }
                            None => {}
                        }
                    }
                }
                let keep: Vec<bool> = (0..NUM_CELLS).map(|_| rng.gen_bool(0.6)).collect();
                let (fx, dec, fat) = fixes_of(&base, &q, |c| keep[c as usize]);
                let fx: Vec<Fix> = fx
                    .into_iter()
                    .filter(|f| !matches!(f, Fix::Move { cell, k } if MOVE_DEST[*cell as usize][*k as usize] as usize >= NUM_CELLS))
                    .collect();
                let dec = fx.iter().fold(0u128, |b, f| match f {
                    Fix::Move { cell, .. } => b | bit(*cell),
                    _ => b,
                }) & dec;
                let s = ts.complete(&fx);
                if let SatResult::Witness(w, order) = &s {
                    eprintln!("order {order:?}");
                    verify(&base, w, order).unwrap_or_else(|e| {
                        panic!("bad witness: {e}\n{}\nfixes {fx:?}", base.render())
                    });
                }
                match exact_realize(&base, &q, dec, fat, 2_000_000) {
                    Exact::Found(ew, acts) => {
                        found += 1;
                        if let SatResult::Refuted(core) = &s {
                            panic!(
                                "exact found, SAT refuted\n{}\nfixes {fx:?}\ncore {core:?}\nexact actions {acts:?}\nexact moves {:?}\nexact targets {:?}",
                                base.render(),
                                (0..NUM_CELLS).filter(|&c| ew.moves[c] != 255).map(|c| (c, ew.moves[c], MOVE_DEST[c][ew.moves[c] as usize])).collect::<Vec<_>>(),
                                (0..NUM_CELLS).filter(|&c| ew.targets[c] != 0).map(|c| (c, ew.targets[c])).collect::<Vec<_>>()
                            );
                        }
                        agree += 1;
                    }
                    Exact::Impossible => {
                        impossible += 1;
                        if let SatResult::Witness(w, _) = &s {
                            // The witness replays (checked above), so the
                            // search refuted a realizable declaration: its
                            // false refutation, counted.
                            let full = exact_realize(
                                &base,
                                w,
                                base.side_bb(0),
                                base.side_bb(1),
                                5_000_000,
                            );
                            assert!(
                                !matches!(full, Exact::Impossible),
                                "the exact search refutes SAT's verified witness\n{}",
                                base.render()
                            );
                            false_refuted += 1;
                            continue;
                        }
                        agree += 1;
                    }
                    Exact::Unknown => unknown += 1,
                }
            }
        }
        eprintln!(
            "played: {sat_ok} satisfiable; perturbed: {agree} agree ({found} found, {impossible} impossible), {unknown} unknown; exact search's false refutations (SAT witness replays): {false_refuted}"
        );
        assert!(sat_ok > 100 && found > 20 && impossible > 20);
    }
}
