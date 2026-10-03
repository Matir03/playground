//! Saving shared-tree self-play trees (`steps-selfplay --save-trees DIR`) for
//! offline study and the web viewer (`web/src/games/minions/Trees.tsx`, served
//! by [`crate::tree_serve`]).
//!
//! One directory per batch (one tree, `--games` games, from their common
//! start to where the batch ended), holding:
//!
//! - `r{round}.bin`: the whole tree as it stood when round `round`'s moves
//!   were made (after the walks, before the pruning to the games' new
//!   positions), so every node's statistics are the ones that move was decided
//!   on. The pruned parts are almost all of each tree, so these add up to
//!   about the batch's distinct nodes; together they are the full tree (a
//!   node's last snapshot has its final statistics). `final.bin` holds what
//!   was left when the batch ended. Header: `MSTS`, version u32, node count
//!   u32; then [`NODE_BYTES`]-byte records in the tree's own order (parents
//!   first, siblings contiguous): gid u32 (stable across the snapshots), first
//!   child index u32, visits u32, value f32, prior f32, children u16, step u8,
//!   kind u8 (0 leaf, 1 expanded, 2 terminal, 3 dead), flags u8 (1 flip,
//!   2 resolved), comp i8. Values are for the node's own mover.
//! - `meta.json`: the start position and salt; per round its file, each game's
//!   position before and after the move (indices in that snapshot), and the
//!   snapshot's forest roots with their step paths from the start. Any node's
//!   state is `StepState::begin(start, salt)` with its root's path and then
//!   the steps below the root applied.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bfs::Tree;
use crate::game::{Minions, MinionsState};
use crate::steps::StepState;
use playground_core::Game;

pub const NODE_BYTES: usize = 26;
pub const VERSION: u32 = 2;

#[derive(Serialize, Deserialize, Clone)]
pub struct GameMove {
    pub game: usize,
    pub turn: u32,
    pub from: u32,
    /// `None`: the walk got stuck.
    pub to: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Root {
    pub idx: u32,
    /// Steps from the batch start, hex.
    pub path: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Round {
    pub file: String,
    pub nodes: u32,
    pub games: Vec<GameMove>,
    pub roots: Vec<Root>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Meta {
    pub version: u32,
    pub batch: u32,
    pub games: usize,
    /// `Minions::encode_state` of the start, hex.
    pub start: String,
    pub salt: u64,
    pub rounds: Vec<Round>,
    /// Distinct nodes the batch created.
    pub distinct: u64,
    /// Why the batch ended: "games over", "enough records" or "stopped"
    /// (empty while it runs).
    pub ended: String,
    /// The backup's power-mean exponent ("2", "inf"); absent in trees saved
    /// before 2026-10-02, which all used 2 except the hand-labelled minimax runs.
    #[serde(default)]
    pub power: Option<String>,
}

impl Meta {
    pub fn power(&self) -> f32 {
        self.power
            .as_deref()
            .and_then(|p| p.parse().ok())
            .unwrap_or(2.0)
    }
}

/// The board a node shows: the real position at a game position, a turn's
/// end or the game's end; otherwise the turn so far projected.
pub fn board_of(st: &StepState, real: bool) -> MinionsState {
    if real || st.is_terminal() {
        return st.real_base();
    }
    let b = st.view().board;
    if st.flipped {
        b.rotated()
    } else {
        b
    }
}

pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

pub fn unhex(s: &str) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(s.len() % 2 == 0, "odd hex length");
    (0..s.len() / 2)
        .map(|i| Ok(u8::from_str_radix(&s[2 * i..2 * i + 2], 16)?))
        .collect()
}

pub struct TreeSaver {
    dir: PathBuf,
    meta: Meta,
    /// Per node of the current tree: its global id.
    gid: Vec<u32>,
    next: u32,
    /// The current tree's forest roots: their step paths from the start.
    roots: HashMap<u32, Vec<u8>>,
    /// The last snapshot's file and metadata, being written in the
    /// background (self-play does not wait for the disk).
    writing: Option<std::thread::JoinHandle<anyhow::Result<()>>>,
}

impl TreeSaver {
    pub fn new(
        root: &Path,
        batch: u32,
        games: usize,
        start: &MinionsState,
        salt: u64,
        power: f32,
    ) -> anyhow::Result<Self> {
        let dir = root.join(format!("b{batch:03}"));
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            meta: Meta {
                version: VERSION,
                batch,
                games,
                start: hex(&Minions::encode_state(start)),
                salt,
                rounds: Vec::new(),
                distinct: 0,
                ended: String::new(),
                power: Some(if power.is_finite() {
                    power.to_string()
                } else {
                    "inf".into()
                }),
            },
            gid: Vec::new(),
            next: 0,
            roots: HashMap::from([(0, Vec::new())]),
            writing: None,
        })
    }

    fn sync(&mut self, tree: &Tree) {
        while self.gid.len() < tree.nodes.len() {
            self.gid.push(self.next);
            self.next += 1;
        }
    }

    /// Write the tree as it stands, as round `file`, with each game's
    /// `(game, from, to)` for the move just made.
    pub fn snapshot(
        &mut self,
        tree: &Tree,
        file: &str,
        games: &[(usize, u32, Option<u32>)],
    ) -> anyhow::Result<()> {
        self.sync(tree);
        let mut out = Vec::with_capacity(12 + tree.nodes.len() * NODE_BYTES);
        out.write_all(b"MSTS")?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&(tree.nodes.len() as u32).to_le_bytes())?;
        for (i, nd) in tree.nodes.iter().enumerate() {
            let mut r = [0u8; NODE_BYTES];
            r[0..4].copy_from_slice(&self.gid[i].to_le_bytes());
            r[4..8].copy_from_slice(&nd.first.to_le_bytes());
            r[8..12].copy_from_slice(&nd.visits.to_le_bytes());
            r[12..16].copy_from_slice(&nd.value.to_le_bytes());
            r[16..20].copy_from_slice(&nd.prior.to_le_bytes());
            r[20..22].copy_from_slice(&nd.nch.to_le_bytes());
            r[22] = nd.step;
            r[23] = nd.kind;
            r[24] = u8::from(nd.flip) | u8::from(nd.resolved) << 1;
            r[25] = nd.comp as u8;
            out.write_all(&r)?;
        }
        let mut roots: Vec<Root> = self
            .roots
            .iter()
            .map(|(&idx, p)| Root { idx, path: hex(p) })
            .collect();
        roots.sort_by_key(|r| r.idx);
        self.meta.rounds.push(Round {
            file: file.to_string(),
            nodes: tree.nodes.len() as u32,
            games: games
                .iter()
                .map(|&(game, from, to)| GameMove {
                    game,
                    turn: tree.state_of(from).real_base().turn as u32,
                    from,
                    to,
                })
                .collect(),
            roots,
        });
        // The metadata is rewritten every round (after its file), so a run
        // that dies mid-batch stays readable.
        self.meta.distinct = u64::from(self.next);
        let meta = serde_json::to_vec(&self.meta)?;
        self.wait()?;
        let (path, dir) = (self.dir.join(file), self.dir.clone());
        self.writing = Some(std::thread::spawn(move || {
            std::fs::write(path, out)?;
            write_meta(&dir, &meta)
        }));
        Ok(())
    }

    /// Wait for the snapshot being written, if any.
    fn wait(&mut self) -> anyhow::Result<()> {
        match self.writing.take() {
            Some(h) => h
                .join()
                .map_err(|_| anyhow::anyhow!("tree writer panicked"))?,
            None => Ok(()),
        }
    }

    /// Carry ids and root paths over `tree.retain(keep)`, whose old-to-new
    /// index map is `map`.
    pub fn retained(&mut self, tree: &Tree, keep: &[u32], map: &[u32]) {
        let mut roots = HashMap::new();
        for &k in keep {
            // `k`'s forest root, and the steps from it down to `k`.
            let mut r = k;
            while tree.nodes[r as usize].parent != u32::MAX {
                r = tree.nodes[r as usize].parent;
            }
            let mut p = self.roots.get(&r).cloned().unwrap_or_default();
            p.extend(tree.path(k));
            roots.insert(map[k as usize], p);
        }
        self.roots = roots;
        let kept = map.iter().filter(|&&m| m != u32::MAX).count();
        let mut gid = vec![0; kept];
        for (old, &m) in map.iter().enumerate() {
            if m != u32::MAX {
                gid[m as usize] = self.gid[old];
            }
        }
        self.gid = gid;
    }

    /// Write what is left of `tree` and the batch's metadata.
    pub fn finish(mut self, tree: &Tree, ended: &str) -> anyhow::Result<()> {
        if !tree.nodes.is_empty() {
            self.snapshot(tree, "final.bin", &[])?;
        }
        self.meta.ended = ended.to_string();
        self.meta.distinct = u64::from(self.next);
        self.wait()?;
        write_meta(&self.dir, &serde_json::to_vec(&self.meta)?)
    }
}

fn write_meta(dir: &Path, meta: &[u8]) -> anyhow::Result<()> {
    let tmp = dir.join(".meta.json.tmp");
    std::fs::write(&tmp, meta)?;
    std::fs::rename(tmp, dir.join("meta.json"))?;
    Ok(())
}
