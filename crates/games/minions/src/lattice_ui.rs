//! The lattice player as the UI's "net" seat (`serve --lattice-net`), and its
//! searches as JSON for the web panel (`web/src/games/minions/LatticePanel.tsx`,
//! proxied by Vite under `/latapi`, served on `--lattice-http`).
//!
//! Every search -- the seat's own turns and the panel's analyses -- is kept
//! (the last few) whole, tier-1 tree and lattices, so the panel can open any
//! node of either tier after the fact. Plain HTTP/1.1 GET, JSON answers:
//!
//! - `/last`: the seat's latest search (`{id}`, 0 before the first).
//! - `/analyze?state=<GameState.payload hex>&n1=N`: search that position
//!   (without playing) and return its summary.
//! - `/search?id=S`: a search's summary: `{id, n1, ms, turn, root, chosen}`.
//! - `/t1?id=S&t=T`: tier-1 node T: its board, visits, value and children
//!   (complete turns: label, visits, value, prior, static value, board),
//!   most visited first.
//! - `/lat?id=S&t=T&node=N`: node N of tier-1 node T's lattice (the search
//!   over T's turn): stage, statistics, its projected board, and its kids
//!   (commitments) with prior, net value, visits, search value and the
//!   complete turns' mean below each.
//!
//! Values are win probabilities in [0, 1] for the mover of the node shown.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::game::{Minions, MinionsMove, MinionsState};
use crate::lattice::{Kind, NONE};
use crate::steps::{Stage, StepState};
use crate::stepsp::LatticePlay;
use crate::tier1::Tier1;
use crate::tree_save::{board_of, hex, unhex};
use playground_core::Game;

/// The seat's player, set by `serve --lattice-net`.
static SEAT: Mutex<Option<Arc<LatticePlay>>> = Mutex::new(None);
/// The kept searches, oldest first, and the seat's latest.
static KEPT: Mutex<VecDeque<Arc<Mutex<Search>>>> = Mutex::new(VecDeque::new());
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static LAST_SEAT: AtomicU64 = AtomicU64::new(0);
const KEEP: usize = 8;

pub struct Search {
    pub id: u64,
    pub n1: u32,
    pub ms: u64,
    pub real: MinionsState,
    pub t: Option<Tier1>,
    /// The chosen turn's tier-1 node; or, for a turn that kills the enemy
    /// necromancer (no search), that turn's end.
    pub chosen: Option<u32>,
    pub kill: Option<StepState>,
}

pub fn configure(play: LatticePlay) {
    *SEAT.lock().unwrap() = Some(Arc::new(play));
}

pub fn is_configured() -> bool {
    SEAT.lock().unwrap().is_some()
}

fn seat() -> anyhow::Result<Arc<LatticePlay>> {
    SEAT.lock()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no lattice net configured"))
}

/// Search `real` with `n1` tier-1 visits and keep the search.
pub fn search(real: &MinionsState, n1: u32) -> anyhow::Result<Arc<Mutex<Search>>> {
    let play = seat()?;
    let salt = NEXT_ID.load(Relaxed);
    let start = StepState::begin(real, salt);
    let t0 = std::time::Instant::now();
    let mut s = Search {
        id: NEXT_ID.fetch_add(1, Relaxed),
        n1,
        ms: 0,
        real: real.clone(),
        t: None,
        chosen: None,
        kill: None,
    };
    let kill = if start.is_terminal() {
        None
    } else {
        crate::lattice::necro_kill(&start).ok()
    };
    if let Some(end) = kill {
        s.kill = Some(end);
    } else {
        let mut t = Tier1::new(start, play.params.clone());
        t.run_inflight(play.eval.as_ref(), n1, play.inflight)?;
        s.chosen = t.best_child();
        s.t = Some(t);
    }
    s.ms = t0.elapsed().as_millis() as u64;
    let s = Arc::new(Mutex::new(s));
    let mut kept = KEPT.lock().unwrap();
    kept.push_back(s.clone());
    while kept.len() > KEEP {
        kept.pop_front();
    }
    Ok(s)
}

/// The seat's turn from `real`: its actions, and the mover's value in [-1, 1].
pub fn plan(
    real: &MinionsState,
    max_visits: Option<u32>,
) -> anyhow::Result<(Vec<MinionsMove>, f64)> {
    let n1 = max_visits.unwrap_or(seat()?.n1);
    let s = search(real, n1)?;
    let s = s.lock().unwrap();
    LAST_SEAT.store(s.id, Relaxed);
    if let Some(end) = &s.kill {
        return Ok((crate::steps::turn_actions(real, &end.done)?, 1.0));
    }
    let t = s.t.as_ref().expect("searched");
    let c = s
        .chosen
        .ok_or_else(|| anyhow::anyhow!("no turn found\n{}", real.render()))?;
    let actions = crate::steps::turn_actions(real, &t.nodes[c as usize].state.done)?;
    Ok((actions, 2.0 * t.root_value() - 1.0))
}

fn find(id: u64) -> anyhow::Result<Arc<Mutex<Search>>> {
    KEPT.lock()
        .unwrap()
        .iter()
        .find(|s| s.lock().unwrap().id == id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("search {id} is no longer kept"))
}

fn board_hex(s: &MinionsState) -> String {
    hex(&Minions::encode_state(s))
}

/// A complete turn from `from` (a turn start) to `to`, compactly: moves
/// `Zc0→d1`, attacks `d1✕e2`, buys `$Z`, spawns `+Z b1`.
fn turn_label(from: &StepState, to: &StepState) -> String {
    use crate::map::cell_name;
    let real = from.real_base();
    let Ok(actions) = crate::steps::turn_actions(&real, &to.done) else {
        return "?".to_string();
    };
    let letter = |u: crate::units::Unit| u.stats().name.chars().next().unwrap_or('?').to_ascii_uppercase();
    let mut s = real.clone();
    let mut out = Vec::new();
    for m in &actions {
        let t = match *m {
            MinionsMove::Move { from, to } => {
                let u = s.piece_at(from).map_or('?', |(_, u)| letter(u));
                format!("{u}{}→{}", cell_name(from), cell_name(to))
            }
            MinionsMove::Attack { from, to } => format!("{}✕{}", cell_name(from), cell_name(to)),
            MinionsMove::Buy(u) => format!("${}", letter(u)),
            MinionsMove::Spawn { unit, at } => format!("+{} {}", letter(unit), cell_name(at)),
            MinionsMove::EndPhase => String::new(),
        };
        if !t.is_empty() {
            out.push(t);
        }
        s = Minions::apply(&s, m);
    }
    if out.is_empty() {
        "pass".to_string()
    } else {
        out.join("  ")
    }
}

fn summary(s: &Search) -> Value {
    json!({
        "id": s.id,
        "n1": s.n1,
        "ms": s.ms,
        "turn": s.real.turn,
        "side": s.real.side,
        "board": board_hex(&s.real),
        "kill": s.kill.as_ref().map(|e| turn_label(&StepState::begin(&s.real, 0), e)),
        "root": s.t.as_ref().map(|_| 0),
        "chosen": s.chosen,
        "t1_nodes": s.t.as_ref().map_or(0, |t| t.nodes.len()),
        "lattice_nodes": s.t.as_ref().map_or(0, |t| t
            .nodes
            .iter()
            .filter_map(|n| n.lattice.as_ref())
            .map(|l| l.nodes.len())
            .sum::<usize>()),
    })
}

fn t1_node(s: &Search, tn: u32) -> anyhow::Result<Value> {
    let t =
        s.t.as_ref()
            .ok_or_else(|| anyhow::anyhow!("no tier-1 search"))?;
    let nd = t
        .nodes
        .get(tn as usize)
        .ok_or_else(|| anyhow::anyhow!("no tier-1 node {tn}"))?;
    let pri = t.priors(tn);
    let lat = nd.lattice.as_ref();
    let mut kids: Vec<Value> = nd
        .children
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let ch = &t.nodes[c as usize];
            // Children's values are their own mover's: flip for this node's.
            let sv = lat.and_then(|l| l.nodes[ch.sink as usize].value);
            json!({
                "t": c,
                "label": turn_label(&nd.state, &ch.state),
                "n": ch.n,
                "q": (ch.n > 0).then(|| 1.0 - ch.q()),
                "prior": pri.get(i),
                "sv": sv,
                "terminal": ch.terminal.is_some() || ch.state.is_terminal(),
                "expanded": !ch.children.is_empty(),
                "board": board_hex(&ch.state.real_base()),
            })
        })
        .collect();
    kids.sort_by(|a, b| b["n"].as_u64().cmp(&a["n"].as_u64()));
    Ok(json!({
        "t": tn,
        "n": nd.n,
        "q": nd.q(),
        "side": nd.state.real_base().side,
        "turn": nd.state.real_base().turn,
        "board": board_hex(&nd.state.real_base()),
        "terminal": nd.terminal,
        "chosen": (tn == 0).then_some(s.chosen).flatten(),
        "lattice": lat.map(|l| json!({
            "nodes": l.nodes.len(),
            "expansions": l.stats.expansions,
            "sinks": l.stats.sinks,
            "descents": l.stats.descents,
        })),
        "children": kids,
    }))
}

fn stage_name(s: Stage) -> &'static str {
    match s {
        Stage::Fates => "fates",
        Stage::Moves => "moves",
        Stage::Spawns => "spawns",
        Stage::Terminal => "complete",
    }
}

fn lat_node(s: &mut Search, tn: u32, n: u32) -> anyhow::Result<Value> {
    let t =
        s.t.as_mut()
            .ok_or_else(|| anyhow::anyhow!("no tier-1 search"))?;
    let nd = t
        .nodes
        .get_mut(tn as usize)
        .ok_or_else(|| anyhow::anyhow!("no tier-1 node {tn}"))?;
    let l = nd
        .lattice
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("tier-1 node {tn} has no lattice (never searched)"))?;
    anyhow::ensure!((n as usize) < l.nodes.len(), "no lattice node {n}");
    let inner = l.nodes[n as usize].kind == Kind::Inner && l.nodes[n as usize].expanded;
    let st = inner.then(|| l.stats(n));
    let node = &l.nodes[n as usize];
    let state = &node.state;
    let kids: Vec<Value> = node
        .kids
        .iter()
        .enumerate()
        .map(|(k, kid)| {
            let (kn, kw) = st.as_ref().map_or((0.0, 0.0), |s| s.kids[k]);
            let (sn, sw) = st.as_ref().map_or((0.0, 0.0), |s| s.skids[k]);
            json!({
                "k": k,
                "var": kid.var,
                "cell": crate::map::cell_name(if state.flipped {
                    crate::map::rot(kid.cell as usize) as u8
                } else {
                    kid.cell
                }),
                "label": state.with_next(kid.cell).step_label(kid.choice),
                "prior": node.expanded.then(|| f64::from(kid.logp).exp()),
                "v": node.expanded.then_some(kid.v),
                "n": kn,
                "q": (kn > 0.0).then(|| kw / kn),
                "sn": sn,
                "sq": (sn > 0.0).then(|| sw / sn),
                "dead": kid.dead,
                "child": (kid.node != NONE).then_some(kid.node),
            })
        })
        .collect();
    Ok(json!({
        "node": n,
        "kind": match node.kind {
            Kind::Inner => "inner",
            Kind::Sink => "sink",
            Kind::Dead => "dead",
        },
        "stage": stage_name(state.stage),
        "expanded": node.expanded,
        "decided": node.key.mask.count_ones(),
        "n": st.as_ref().map(|s| s.n),
        "q": st.as_ref().and_then(|s| (s.n > 0.0).then(|| s.w / s.n)),
        "sn": st.as_ref().map(|s| s.sn),
        "sq": st.as_ref().and_then(|s| (s.sn > 0.0).then(|| s.sw / s.sn)),
        "value": node.value,
        "board": board_hex(&board_of(state, false)),
        "kids": kids,
    }))
}

fn handle(path: &str, q: &HashMap<String, String>) -> anyhow::Result<Value> {
    let num = |k: &str| -> anyhow::Result<u64> {
        q.get(k)
            .ok_or_else(|| anyhow::anyhow!("missing {k}"))?
            .parse::<u64>()
            .map_err(|e| anyhow::anyhow!("{k}: {e}"))
    };
    match path {
        "/last" => Ok(json!({ "id": LAST_SEAT.load(Relaxed) })),
        "/analyze" => {
            let real = Minions::decode_state(&unhex(q.get("state").map_or("", |s| s.as_str()))?)?;
            let n1 = q
                .get("n1")
                .and_then(|v| v.parse().ok())
                .unwrap_or(seat()?.n1);
            let s = search(&real, n1)?;
            let s = s.lock().unwrap();
            Ok(summary(&s))
        }
        "/search" => Ok(summary(&find(num("id")?)?.lock().unwrap())),
        "/t1" => t1_node(&find(num("id")?)?.lock().unwrap(), num("t")? as u32),
        "/lat" => lat_node(
            &mut find(num("id")?)?.lock().unwrap(),
            num("t")? as u32,
            num("node")? as u32,
        ),
        _ => anyhow::bail!("unknown path {path}"),
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

fn serve_one(s: TcpStream) {
    let mut r = BufReader::new(match s.try_clone() {
        Ok(c) => c,
        Err(_) => return,
    });
    let mut line = String::new();
    if r.read_line(&mut line).is_err() {
        return;
    }
    loop {
        let mut h = String::new();
        match r.read_line(&mut h) {
            Ok(0) | Err(_) => break,
            Ok(_) if h == "\r\n" || h == "\n" => break,
            Ok(_) => {}
        }
    }
    let target = line.split_whitespace().nth(1).unwrap_or("/");
    let target = target.strip_prefix("/latapi").unwrap_or(target);
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let q: HashMap<String, String> = query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    match handle(path, &q) {
        Ok(v) => respond(s, "200 OK", &serde_json::to_vec(&v).unwrap_or_default()),
        Err(e) => respond(
            s,
            "400 Bad Request",
            &serde_json::to_vec(&json!({ "error": e.to_string() })).unwrap_or_default(),
        ),
    }
}

/// Serve the panel's JSON on `addr` (blocks; run it on its own thread).
pub fn serve_http(addr: &str) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr)?;
    eprintln!("lattice panel: http on {addr}");
    for s in listener.incoming() {
        let Ok(s) = s else { continue };
        std::thread::spawn(move || serve_one(s));
    }
    Ok(())
}
