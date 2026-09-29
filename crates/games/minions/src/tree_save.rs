//! Saving shared-tree self-play trees (`steps-selfplay --save-trees DIR`) for
//! offline study and the web viewer (`web/src/games/minions/Trees.tsx`).
//!
//! One directory per batch (one tree, `--games` games, from their common
//! start to where the batch ended), holding:
//!
//! - `nodes.bin`: the full tree. Self-play prunes the tree to the subtrees
//!   under the live game positions after every move, so no whole tree ever
//!   exists in memory; each node is written when it leaves (with its final
//!   statistics), and the rest when the batch ends. Node ids are global to the
//!   batch and stable across the pruning. Header: `MSTT`, version u32, salt
//!   u64, state length u16, the start state (`Minions::encode_state`). Then
//!   [`NODE_BYTES`]-byte records: gid u32, parent gid u32 (`u32::MAX` at the
//!   root), visits u32, value f32, prior f32, step u8, kind u8 (0 leaf,
//!   1 expanded, 2 terminal, 3 dead), flags u8 (1 flip, 2 resolved), comp i8.
//!   Any node's state is the start's `StepState::begin(start, salt)` with the
//!   steps on its parent chain applied.
//! - `m{move}_g{game}.json`: what each game's move was decided on, taken at
//!   the move: the subtree under the game's position, its most visited
//!   expanded nodes (the played line always) with every live edge, and the
//!   board each exported node shows. (The full tree only has final
//!   statistics, which later moves' searches inflate along the played line.)
//! - `meta.json`: per move, each game's position (gid) before and after, and
//!   its view file; the node count written after each move.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::bfs::Tree;
use crate::game::{Minions, MinionsState};
use crate::steps_analysis::{board_at, to_pm};
use playground_core::Game;

pub const NODE_BYTES: usize = 24;
const VERSION: u32 = 1;

#[derive(Serialize)]
struct GameMove {
    game: usize,
    turn: u32,
    from: u32,
    /// `None`: the walk got stuck.
    to: Option<u32>,
    view: String,
}

#[derive(Serialize)]
struct MoveMeta {
    games: Vec<GameMove>,
    /// Nodes in `nodes.bin` once this move's pruning was written.
    nodes_written: u64,
}

#[derive(Serialize)]
struct Meta {
    batch: u32,
    games: usize,
    start: String,
    salt: u64,
    moves: Vec<MoveMeta>,
    nodes: u64,
    /// Why the batch ended: "games over", "enough records" or "stopped".
    ended: String,
}

/// One edge in a view: `[label, visits, value (or null), prior value, child
/// node index (or -1), played]`, values in [-1, 1] for the parent's mover.
#[derive(Serialize)]
struct ViewEdge(String, u32, Option<f32>, f32, i64, bool);

#[derive(Serialize)]
struct ViewNode {
    /// Parent's index in `nodes`, -1 at the view's root.
    p: i64,
    n: u32,
    q: f32,
    pq: f32,
    t: bool,
    /// The board this node shows (`GameState.payload` hex).
    b: String,
    e: Vec<ViewEdge>,
}

#[derive(Serialize)]
struct View {
    batch: u32,
    game: usize,
    #[serde(rename = "move")]
    mv: usize,
    turn: u32,
    /// Search value of the position for its mover, [-1, 1].
    value: f32,
    /// Nodes in the whole tree when the move was made.
    tree_nodes: usize,
    nodes: Vec<ViewNode>,
    /// The played root edge's index in `nodes[0].e`.
    chosen: Option<usize>,
}

pub struct TreeSaver {
    dir: PathBuf,
    batch: u32,
    games: usize,
    out: std::io::BufWriter<std::fs::File>,
    /// Per node of the current tree: its global id and its parent's.
    gid: Vec<u32>,
    pgid: Vec<u32>,
    next: u32,
    written: u64,
    moves: Vec<MoveMeta>,
    view_nodes: usize,
    start: String,
    salt: u64,
}

impl TreeSaver {
    pub fn new(
        root: &Path,
        batch: u32,
        games: usize,
        start: &MinionsState,
        salt: u64,
        view_nodes: usize,
    ) -> anyhow::Result<Self> {
        let dir = root.join(format!("b{batch:03}"));
        std::fs::create_dir_all(&dir)?;
        let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("nodes.bin"))?);
        let enc = Minions::encode_state(start);
        out.write_all(b"MSTT")?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&salt.to_le_bytes())?;
        out.write_all(&(enc.len() as u16).to_le_bytes())?;
        out.write_all(&enc)?;
        Ok(Self {
            dir,
            batch,
            games,
            out,
            gid: Vec::new(),
            pgid: Vec::new(),
            next: 0,
            written: 0,
            moves: Vec::new(),
            view_nodes,
            start: crate::eval::hex_state(start),
            salt,
        })
    }

    /// Give the nodes created since the last call their global ids.
    pub fn sync(&mut self, tree: &Tree) {
        for i in self.gid.len()..tree.nodes.len() {
            let p = tree.nodes[i].parent;
            self.gid.push(self.next);
            self.pgid.push(if p == u32::MAX {
                u32::MAX
            } else {
                self.gid[p as usize]
            });
            self.next += 1;
        }
    }

    pub fn gid(&self, n: u32) -> u32 {
        self.gid[n as usize]
    }

    /// Write game `game`'s view of move `mv`: the subtree under `from`, with
    /// the line to `to` (the position after the turn) marked as played.
    /// Returns the file name.
    pub fn view(
        &self,
        tree: &Tree,
        mv: usize,
        game: usize,
        from: u32,
        to: Option<u32>,
    ) -> anyhow::Result<String> {
        let name = format!("m{mv:03}_g{game:02}.json");
        let v = export_view(tree, from, to, self.view_nodes, self.batch, game, mv);
        std::fs::write(self.dir.join(&name), serde_json::to_vec(&v)?)?;
        Ok(name)
    }

    /// Record move `mv`: each game's `(game, from, to, view)`, positions as
    /// indices in the tree the move was made in.
    pub fn add_move(&mut self, tree: &Tree, games: Vec<(usize, u32, Option<u32>, String)>) {
        let games = games
            .into_iter()
            .map(|(game, from, to, view)| GameMove {
                game,
                turn: tree.state_of(from).real_base().turn as u32,
                from: self.gid(from),
                to: to.map(|t| self.gid(t)),
                view,
            })
            .collect();
        self.moves.push(MoveMeta {
            games,
            nodes_written: 0,
        });
    }

    /// Write every node outside `keep`'s subtrees (all of them for an empty
    /// `keep`), before `tree.retain(keep)`.
    pub fn write_outside(&mut self, tree: &Tree, keep: &[u32]) -> anyhow::Result<()> {
        self.sync(tree);
        let inside = crate::stepsp::inside_mask(tree, keep);
        let mut buf = Vec::with_capacity(NODE_BYTES * 4096);
        for (i, nd) in tree.nodes.iter().enumerate() {
            if inside[i] {
                continue;
            }
            buf.extend_from_slice(&self.gid[i].to_le_bytes());
            buf.extend_from_slice(&self.pgid[i].to_le_bytes());
            buf.extend_from_slice(&nd.visits.to_le_bytes());
            buf.extend_from_slice(&nd.value.to_le_bytes());
            buf.extend_from_slice(&nd.prior.to_le_bytes());
            buf.push(nd.step);
            buf.push(nd.kind);
            buf.push(u8::from(nd.flip) | u8::from(nd.resolved) << 1);
            buf.push(nd.comp as u8);
            self.written += 1;
            if buf.len() >= NODE_BYTES * 4096 {
                self.out.write_all(&buf)?;
                buf.clear();
            }
        }
        self.out.write_all(&buf)?;
        if let Some(m) = self.moves.last_mut() {
            m.nodes_written = self.written;
        }
        Ok(())
    }

    /// Carry the ids over `tree.retain`'s old-to-new `map`.
    pub fn remap(&mut self, map: &[u32]) {
        let kept = map.iter().filter(|&&m| m != u32::MAX).count();
        let (mut gid, mut pgid) = (vec![0; kept], vec![0; kept]);
        for (old, &m) in map.iter().enumerate() {
            if m != u32::MAX {
                gid[m as usize] = self.gid[old];
                pgid[m as usize] = self.pgid[old];
            }
        }
        self.gid = gid;
        self.pgid = pgid;
    }

    /// Write what is left of `tree` and the batch's metadata.
    pub fn finish(mut self, tree: &Tree, ended: &str) -> anyhow::Result<()> {
        if !tree.nodes.is_empty() {
            self.write_outside(tree, &[])?;
        }
        self.out.flush()?;
        let meta = Meta {
            batch: self.batch,
            games: self.games,
            start: self.start.clone(),
            salt: self.salt,
            moves: std::mem::take(&mut self.moves),
            nodes: self.written,
            ended: ended.to_string(),
        };
        std::fs::write(self.dir.join("meta.json"), serde_json::to_vec(&meta)?)?;
        Ok(())
    }
}

/// The subtree under `from` as a [`View`]: its most visited expanded nodes
/// (at most `max_nodes`, every node on the played line to `to` first), each
/// with all its live edges.
fn export_view(
    tree: &Tree,
    from: u32,
    to: Option<u32>,
    max_nodes: usize,
    batch: u32,
    game: usize,
    mv: usize,
) -> View {
    // The played line, `from` excluded.
    let mut played: HashSet<u32> = HashSet::new();
    if let Some(mut t) = to {
        while t != from && t != u32::MAX {
            played.insert(t);
            t = tree.nodes[t as usize].parent;
        }
    }
    let mut picked: Vec<u32> = Vec::new();
    let mut frontier = vec![from];
    while let Some(n) = frontier.pop() {
        for c in tree.children(n) {
            if tree.kind(c) == 1 && tree.nodes[c as usize].visits > 0 {
                picked.push(c);
                frontier.push(c);
            }
        }
    }
    picked.sort_by_key(|&c| {
        (
            !played.contains(&c),
            std::cmp::Reverse(tree.nodes[c as usize].visits),
        )
    });
    picked.truncate(max_nodes.saturating_sub(1));
    let mut keep: HashSet<u32> = picked.into_iter().collect();
    keep.insert(from);
    // Depth first from `from`, keeping only nodes whose parent made the cut.
    let mut index: HashMap<u32, usize> = HashMap::new();
    let mut order: Vec<u32> = Vec::new();
    let mut stack = vec![from];
    while let Some(n) = stack.pop() {
        index.insert(n, order.len());
        order.push(n);
        for c in tree.children(n).rev() {
            if keep.contains(&c) {
                stack.push(c);
            }
        }
    }
    let mut nodes = Vec::with_capacity(order.len());
    let mut chosen = None;
    for &n in &order {
        let nd = &tree.nodes[n as usize];
        let st = tree.state_of(n);
        let live: Vec<u32> = tree.children(n).filter(|&c| tree.kind(c) != 3).collect();
        let mut e = Vec::with_capacity(live.len());
        for (i, &c) in live.iter().enumerate() {
            let cn = &tree.nodes[c as usize];
            let flip = |v: f32| if cn.flip { 1.0 - v } else { v };
            let mut label = st.step_label(cn.step);
            if cn.flip {
                label.push_str("  ⟶ end of turn");
            } else if tree.kind(c) == 2 {
                label.push_str("  ⟶ game over");
            }
            if n == from && played.contains(&c) {
                chosen = Some(i);
            }
            e.push(ViewEdge(
                label,
                cn.visits,
                (cn.visits > 0 || cn.kind == 1).then(|| to_pm(tree.q_parent(c))),
                to_pm(flip(cn.prior)),
                index.get(&c).map_or(-1, |&x| x as i64),
                played.contains(&c),
            ));
        }
        nodes.push(ViewNode {
            p: if n == from {
                -1
            } else {
                index[&nd.parent] as i64
            },
            n: nd.visits,
            q: to_pm(nd.value),
            pq: to_pm(nd.prior),
            t: nd.kind == 2,
            b: crate::eval::hex_state(&board_at(tree, n)),
            e,
        });
    }
    View {
        batch,
        game,
        mv,
        turn: tree.state_of(from).real_base().turn as u32,
        value: to_pm(tree.nodes[from as usize].value),
        tree_nodes: tree.nodes.len(),
        nodes,
        chosen,
    }
}
