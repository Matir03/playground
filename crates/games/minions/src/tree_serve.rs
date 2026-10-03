//! `steps-tree-serve`: nodes of saved self-play trees ([`crate::tree_save`])
//! on demand, for the web viewer (`web/src/games/minions/Trees.tsx`, proxied
//! by Vite under `/treeapi`). A round's tree has millions of nodes, far more
//! than a browser can take; the viewer asks for the nodes a reader opens.
//!
//! Plain HTTP/1.1 GET with JSON answers, one thread per request:
//!
//! - `/line?run=R&batch=B&round=K&game=G`: the nodes on game G's played line
//!   in round K (its position before the move to the one after), the first
//!   being the root the viewer shows.
//! - `/nodes?run=R&batch=B&round=K&game=G&idx=I,J,..`: those nodes.
//!
//! A node is `{i, n, q, pq, t, ok, b, e}`: its index in the snapshot, visits,
//! search and static value for its own mover in [-1, 1], terminal, whether
//! the replayed state offers exactly the saved children, board
//! (`GameState.payload` hex) and live edges `[label, visits, value or null,
//! static value, child index, played, expandable, child board]`, values for
//! the node's mover.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::game::Minions;
use crate::steps::StepState;
use crate::tree_save::{board_of, hex, unhex, Meta, NODE_BYTES};
use playground_core::Game;

struct Snap {
    raw: Vec<u8>,
    parent: Vec<u32>,
}

impl Snap {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)?;
        anyhow::ensure!(
            raw.len() >= 12 && &raw[..4] == b"MSTS",
            "{}: not a snapshot",
            path.display()
        );
        let n = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        anyhow::ensure!(
            raw.len() == 12 + n * NODE_BYTES,
            "{}: truncated",
            path.display()
        );
        let raw = raw[12..].to_vec();
        let mut s = Snap {
            raw,
            parent: vec![u32::MAX; n],
        };
        for i in 0..n as u32 {
            let (first, nch) = (s.first(i), s.nch(i));
            for c in first..first + nch {
                s.parent[c as usize] = i;
            }
        }
        Ok(s)
    }
    fn len(&self) -> usize {
        self.parent.len()
    }
    fn f(&self, i: u32, at: usize) -> [u8; 4] {
        let o = i as usize * NODE_BYTES + at;
        self.raw[o..o + 4].try_into().unwrap()
    }
    fn first(&self, i: u32) -> u32 {
        u32::from_le_bytes(self.f(i, 4))
    }
    fn visits(&self, i: u32) -> u32 {
        u32::from_le_bytes(self.f(i, 8))
    }
    fn value(&self, i: u32) -> f32 {
        f32::from_le_bytes(self.f(i, 12))
    }
    fn prior(&self, i: u32) -> f32 {
        f32::from_le_bytes(self.f(i, 16))
    }
    fn b(&self, i: u32, at: usize) -> u8 {
        self.raw[i as usize * NODE_BYTES + at]
    }
    fn nch(&self, i: u32) -> u32 {
        let o = i as usize * NODE_BYTES + 20;
        u32::from(u16::from_le_bytes([self.raw[o], self.raw[o + 1]]))
    }
    fn step(&self, i: u32) -> u8 {
        self.b(i, 22)
    }
    fn kind(&self, i: u32) -> u8 {
        self.b(i, 23)
    }
    fn flip(&self, i: u32) -> bool {
        self.b(i, 24) & 1 != 0
    }
    fn comp(&self, i: u32) -> i8 {
        self.b(i, 25) as i8
    }
}

/// Each child's share in its parent's backed-up value, as `Tree::recompute`
/// forms it: with a proven child (or `p = inf`) the best child of the
/// deciding class carries it all; otherwise every live, unproven child is
/// weighted `1 + descents` in the power mean. Dead children and ignored
/// proven losses get 0.
fn backup_weights(snap: &Snap, first: u32, nch: u32, p: f32) -> Vec<f32> {
    let kids = first..first + nch;
    let live = |c: &u32| snap.kind(*c) != 3;
    let c_par = |c: u32| {
        if snap.flip(c) {
            -snap.comp(c)
        } else {
            snap.comp(c)
        }
    };
    let q_par = |c: u32| {
        if snap.flip(c) {
            1.0 - snap.value(c)
        } else {
            snap.value(c)
        }
    };
    let mut w = vec![0f32; nch as usize];
    let Some(pick) = kids.clone().filter(live).map(c_par).max() else {
        return w;
    };
    if pick != 0 || p.is_infinite() {
        let best = kids
            .clone()
            .filter(|c| live(c) && c_par(*c) == pick)
            .max_by(|&a, &b| q_par(a).total_cmp(&q_par(b)));
        if let Some(b) = best {
            w[(b - first) as usize] = 1.0;
        }
        return w;
    }
    let mut total = 0f32;
    for c in kids.clone().filter(|c| live(c) && c_par(*c) == 0) {
        w[(c - first) as usize] = 1.0 + snap.visits(c) as f32;
        total += 1.0 + snap.visits(c) as f32;
    }
    w.iter_mut().for_each(|x| *x /= total.max(1.0));
    w
}

#[derive(Default)]
struct Cache {
    metas: HashMap<(String, String), Arc<Meta>>,
    /// Most recently used last.
    snaps: Vec<((String, String, usize), Arc<Snap>)>,
    states: HashMap<(String, String, usize, u32), StepState>,
}

const SNAPS_KEPT: usize = 3;
const STATES_KEPT: usize = 50_000;

struct Server {
    dir: PathBuf,
    cache: Mutex<Cache>,
}

fn to_pm(v: f32) -> f32 {
    2.0 * v - 1.0
}

impl Server {
    fn batch_dir(&self, run: &str, batch: &str) -> anyhow::Result<PathBuf> {
        // The names come from a URL: no path tricks.
        let ok = |s: &str| {
            !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        };
        anyhow::ensure!(ok(run) && ok(batch), "bad run or batch name");
        Ok(self.dir.join(run).join(batch))
    }

    fn meta(&self, run: &str, batch: &str) -> anyhow::Result<Arc<Meta>> {
        let key = (run.to_string(), batch.to_string());
        if let Some(m) = self.cache.lock().unwrap().metas.get(&key) {
            // A batch still being written grows; reread it once it has ended.
            if !m.ended.is_empty() {
                return Ok(m.clone());
            }
        }
        let path = self.batch_dir(run, batch)?.join("meta.json");
        let m: Meta = serde_json::from_slice(&std::fs::read(&path)?)?;
        anyhow::ensure!(
            m.version == crate::tree_save::VERSION,
            "{}: old format (static views only)",
            path.display()
        );
        let m = Arc::new(m);
        self.cache.lock().unwrap().metas.insert(key, m.clone());
        Ok(m)
    }

    fn snap(&self, run: &str, batch: &str, round: usize, meta: &Meta) -> anyhow::Result<Arc<Snap>> {
        let key = (run.to_string(), batch.to_string(), round);
        {
            let mut c = self.cache.lock().unwrap();
            if let Some(p) = c.snaps.iter().position(|(k, _)| *k == key) {
                let e = c.snaps.remove(p);
                let s = e.1.clone();
                c.snaps.push(e);
                return Ok(s);
            }
        }
        let r = meta
            .rounds
            .get(round)
            .ok_or_else(|| anyhow::anyhow!("no round {round}"))?;
        let s = Arc::new(Snap::load(&self.batch_dir(run, batch)?.join(&r.file))?);
        let mut c = self.cache.lock().unwrap();
        c.snaps.push((key, s.clone()));
        if c.snaps.len() > SNAPS_KEPT {
            c.snaps.remove(0);
        }
        Ok(s)
    }

    /// Node `i`'s state as the tree held it: its root's path from the start,
    /// then the steps below, each taken the way an expansion takes it
    /// (`bfs::build_one`: forced steps collapsed, which also resolves a lazy
    /// witness, then the child). A plain `apply` chain from the start
    /// diverges from the tree within a few steps.
    fn state(
        &self,
        key: (&str, &str, usize),
        meta: &Meta,
        snap: &Snap,
        i: u32,
    ) -> anyhow::Result<StepState> {
        let ck = |n: u32| (key.0.to_string(), key.1.to_string(), key.2, n);
        let mut steps = Vec::new();
        let mut a = i;
        let mut base = None;
        loop {
            if let Some(s) = self.cache.lock().unwrap().states.get(&ck(a)) {
                base = Some(s.clone());
                break;
            }
            if snap.parent[a as usize] == u32::MAX {
                break;
            }
            steps.push(snap.step(a));
            a = snap.parent[a as usize];
        }
        let mut s = match base {
            Some(s) => s,
            None => {
                let round = &meta.rounds[key.2];
                let root = round
                    .roots
                    .iter()
                    .find(|r| r.idx == a)
                    .ok_or_else(|| anyhow::anyhow!("node {a} is no saved root"))?;
                let start = Minions::decode_state(&unhex(&meta.start)?)?;
                let mut s = StepState::begin(&start, meta.salt);
                for v in unhex(&root.path)? {
                    s.collapse_forced();
                    s = s.child(v);
                }
                if snap.kind(a) == 1 {
                    s.collapse_forced();
                }
                self.cache.lock().unwrap().states.insert(ck(a), s.clone());
                s
            }
        };
        for &v in steps.iter().rev() {
            s.collapse_forced();
            s = s.child(v);
        }
        if snap.kind(i) == 1 {
            s.collapse_forced();
        }
        let mut c = self.cache.lock().unwrap();
        if c.states.len() > STATES_KEPT {
            c.states.clear();
        }
        c.states.insert(ck(i), s.clone());
        Ok(s)
    }

    /// The played line of `game` in `round`: node indices from its position
    /// before the move to the one after.
    fn played(meta: &Meta, snap: &Snap, round: usize, game: usize) -> Vec<u32> {
        let Some(gm) = meta.rounds[round].games.iter().find(|g| g.game == game) else {
            return Vec::new();
        };
        let mut line = Vec::new();
        if let Some(mut t) = gm.to {
            while t != gm.from && t != u32::MAX {
                line.push(t);
                t = snap.parent[t as usize];
            }
        }
        line.push(gm.from);
        line.reverse();
        line
    }

    fn node(
        &self,
        key: (&str, &str, usize),
        meta: &Meta,
        snap: &Snap,
        i: u32,
        played: &[u32],
    ) -> anyhow::Result<Value> {
        anyhow::ensure!((i as usize) < snap.len(), "no node {i}");
        let st = self.state(key, meta, snap, i)?;
        let root = snap.parent[i as usize] == u32::MAX;
        let (first, nch) = (snap.first(i), snap.nch(i));
        // The replay is faithful if it offers exactly the saved children.
        let ok = snap.kind(i) != 1 || {
            let mut opts = st.clone().collapse_forced();
            let mut saved: Vec<u8> = (first..first + nch).map(|c| snap.step(c)).collect();
            opts.sort_unstable();
            saved.sort_unstable();
            opts == saved
        };
        let weights = backup_weights(snap, first, nch, meta.power());
        let mut edges = Vec::new();
        for c in first..first + nch {
            // Dead children (an option whose lazily offered witness turned out
            // impossible) are listed too, so a lone live child does not look
            // forced and the parent's visits add up; they have no board.
            if snap.kind(c) == 3 {
                edges.push(json!([
                    format!("{}  ✕ impossible", st.step_label(snap.step(c))),
                    snap.visits(c),
                    Value::Null,
                    to_pm(if snap.flip(c) {
                        1.0 - snap.prior(c)
                    } else {
                        snap.prior(c)
                    }),
                    c,
                    false,
                    false,
                    Value::Null,
                    0.0,
                    true,
                ]));
                continue;
            }
            let flip = snap.flip(c);
            let side = |v: f32| to_pm(if flip { 1.0 - v } else { v });
            let mut label = st.step_label(snap.step(c));
            if flip {
                label.push_str("  ⟶ end of turn");
            } else if snap.kind(c) == 2 {
                label.push_str("  ⟶ game over");
            }
            let cs = st.child(snap.step(c));
            let board = hex(&Minions::encode_state(&board_of(&cs, flip)));
            let visits = snap.visits(c);
            edges.push(json!([
                label,
                visits,
                (visits > 0 || snap.kind(c) == 1).then(|| side(snap.value(c))),
                side(snap.prior(c)),
                c,
                played.contains(&c) && played.contains(&i),
                snap.kind(c) == 1,
                board,
                weights[(c - first) as usize],
                false,
            ]));
        }
        Ok(json!({
            "i": i,
            "n": snap.visits(i),
            "q": to_pm(snap.value(i)),
            "pq": to_pm(snap.prior(i)),
            "t": snap.kind(i) == 2,
            "ok": ok,
            "b": hex(&Minions::encode_state(&board_of(&st, root || snap.flip(i)))),
            "e": edges,
        }))
    }

    fn handle(&self, path: &str, q: &HashMap<String, String>) -> anyhow::Result<Value> {
        let get = |k: &str| {
            q.get(k)
                .map(String::as_str)
                .ok_or_else(|| anyhow::anyhow!("missing {k}"))
        };
        let (run, batch) = (get("run")?, get("batch")?);
        let round: usize = get("round")?.parse()?;
        let game: usize = get("game")?.parse()?;
        let meta = self.meta(run, batch)?;
        let snap = self.snap(run, batch, round, &meta)?;
        let key = (run, batch, round);
        let played = Self::played(&meta, &snap, round, game);
        let idx: Vec<u32> = match path {
            "/line" => played.clone(),
            "/nodes" => get("idx")?
                .split(',')
                .map(|x| x.parse::<u32>().map_err(anyhow::Error::from))
                .collect::<anyhow::Result<_>>()?,
            _ => anyhow::bail!("unknown path {path}"),
        };
        let nodes = idx
            .iter()
            .map(|&i| self.node(key, &meta, &snap, i, &played))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(json!({ "nodes": nodes, "tree_nodes": snap.len() }))
    }
}

fn respond(mut s: TcpStream, status: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(body);
}

fn serve_one(server: &Server, s: TcpStream) {
    let mut r = BufReader::new(match s.try_clone() {
        Ok(c) => c,
        Err(_) => return,
    });
    let mut line = String::new();
    if r.read_line(&mut line).is_err() {
        return;
    }
    // Skip the headers.
    loop {
        let mut h = String::new();
        match r.read_line(&mut h) {
            Ok(0) | Err(_) => break,
            Ok(_) if h == "\r\n" || h == "\n" => break,
            Ok(_) => {}
        }
    }
    let target = line.split_whitespace().nth(1).unwrap_or("/");
    let target = target.strip_prefix("/treeapi").unwrap_or(target);
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let q: HashMap<String, String> = query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_string(), v.replace("%2C", ",")))
        .collect();
    match server.handle(path, &q) {
        Ok(v) => respond(s, "200 OK", &serde_json::to_vec(&v).unwrap_or_default()),
        Err(e) => respond(
            s,
            "400 Bad Request",
            &serde_json::to_vec(&json!({ "error": e.to_string() })).unwrap_or_default(),
        ),
    }
}

pub fn serve(dir: &Path, addr: &str) -> anyhow::Result<()> {
    // Replays must enumerate options exactly as the self-play that saved the
    // trees did (the cycle runs with lazy witnesses).
    crate::steps::LAZY_WITNESS.store(true, std::sync::atomic::Ordering::Relaxed);
    let server = Arc::new(Server {
        dir: dir.to_path_buf(),
        cache: Mutex::new(Cache::default()),
    });
    let listener = TcpListener::bind(addr)?;
    eprintln!("steps-tree-serve: {} on {addr}", dir.display());
    for s in listener.incoming() {
        let Ok(s) = s else { continue };
        let server = server.clone();
        std::thread::spawn(move || serve_one(&server, s));
    }
    Ok(())
}
