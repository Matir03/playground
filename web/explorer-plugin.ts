/** Serves the Minions data explorer's logs (`web/src/games/minions/Explorer.tsx`)
 *  straight from the training directory, read-only, under /explorer-data:
 *
 *  - `/index`: the sources: self-play generations with a `games/` dir
 *    (lattice-gen --game-log) and match logs in `explorer/*.jsonl`
 *    (steps-match --game-log);
 *  - `/<src>/summary`: one row per game (setup, result, search stats);
 *  - `/<src>/game/<i>`: one game in full;
 *  - `/<src>/t2`: a self-play generation's sampled rooted tier-2 records.
 *
 *  `<src>` is `sp<gen>` or `m<name>`. Root: $EXPLORER_ROOT, else the lattice537
 *  lineage. Parsed files are cached by path and mtime. */
import fs from "node:fs";
import path from "node:path";
import type { Plugin } from "vite";

const ROOT = process.env.EXPLORER_ROOT ?? path.resolve(__dirname, "../data/models/minions/lattice537");

type Game = Record<string, any>;
const cache = new Map<string, { mtime: number; games: Game[] }>();

function readLines(file: string): Game[] {
  const st = fs.statSync(file);
  const hit = cache.get(file);
  if (hit && hit.mtime === st.mtimeMs) return hit.games;
  const games: Game[] = [];
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try {
      games.push(JSON.parse(line));
    } catch {
      // A line still being written.
    }
  }
  cache.set(file, { mtime: st.mtimeMs, games });
  return games;
}

function spDir(gen: string) {
  return path.join(ROOT, "data", `gen${gen}`, "games");
}

function gamesOf(src: string): Game[] {
  if (src.startsWith("sp")) {
    const dir = spDir(src.slice(2));
    if (!fs.existsSync(dir)) return [];
    return fs
      .readdirSync(dir)
      .filter((f) => /^w\d+\.jsonl$/.test(f))
      .sort()
      .flatMap((f) => readLines(path.join(dir, f)));
  }
  if (src.startsWith("m")) {
    const f = path.join(ROOT, "explorer", `${src.slice(1)}.jsonl`);
    return fs.existsSync(f) ? readLines(f) : [];
  }
  return [];
}

/** A match turn's search, flattened to the self-play turn's shape. */
function norm(t: Game): Game {
  return t.search !== undefined ? { ...t, ...(t.search ?? {}), search: undefined } : t;
}

function summary(g: Game, i: number) {
  const turns: Game[] = (g.turns ?? []).map(norm);
  const searched = turns.filter((t) => t.cands && t.cands.length > 0);
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const top = (t: Game) => t.cands.reduce((a: Game, b: Game) => (b.n > a.n ? b : a));
  return {
    i,
    unfinished: !!g.unfinished,
    winner: g.winner ?? 0,
    a: g.a,
    b: g.b,
    a_first: g.a_first,
    pair: g.pair,
    start_turn: turns[0]?.turn ?? null,
    final_turn: g.final_turn ?? null,
    n_turns: turns.length,
    kids: mean(searched.map((t) => t.kids)),
    visits: mean(searched.map((t) => t.visits)),
    top_share: mean(searched.map((t) => top(t).n / Math.max(1, t.visits))),
    played_top: mean(searched.map((t) => (t.played === top(t).label ? 1 : 0))),
    picked_top: mean(searched.filter((t) => t.picked !== undefined).map((t) => (t.picked === top(t).label ? 1 : 0))),
    fallback: turns.filter((t) => t.anchor?.from === "fallback").length,
    anchored: turns.filter((t) => t.anchor).length,
    value_dev: mean(searched.map((t) => Math.abs((t.value ?? 0.5) - 0.5))),
    a_turns: turns.filter((t) => t.mover === "a").length,
    // Featured games (steps-match --shadow): both nets searched every turn.
    shadowed: turns.filter((t) => t.shadow).length,
    agree: mean(turns.filter((t) => t.shadow && t.played).map((t) => (t.shadow.played === t.played ? 1 : 0))),
    vdiff: mean(turns.filter((t) => t.shadow && t.value != null).map((t) => Math.abs(t.value - t.shadow.value))),
  };
}

function listSources() {
  const out: { src: string; kind: string; label: string; gen: number | null }[] = [];
  const data = path.join(ROOT, "data");
  if (fs.existsSync(data)) {
    for (const d of fs.readdirSync(data)) {
      const m = /^gen(\d+)$/.exec(d);
      if (m && fs.existsSync(path.join(data, d, "games"))) {
        out.push({ src: `sp${m[1]}`, kind: "selfplay", label: `self-play gen ${m[1]}`, gen: Number(m[1]) });
      }
    }
  }
  const ex = path.join(ROOT, "explorer");
  if (fs.existsSync(ex)) {
    for (const f of fs.readdirSync(ex)) {
      const m = /^(.*)\.jsonl$/.exec(f);
      if (!m) continue;
      const g = /(\d+)/.exec(m[1]);
      out.push({ src: `m${m[1]}`, kind: "match", label: m[1], gen: g ? Number(g[1]) : null });
    }
  }
  out.sort((x, y) => (y.gen ?? 0) - (x.gen ?? 0) || x.src.localeCompare(y.src));
  return out;
}

export function explorerData(): Plugin {
  return {
    name: "explorer-data",
    configureServer(server) {
      server.middlewares.use("/explorer-data", (req, res) => {
        const send = (body: unknown, code = 200) => {
          res.statusCode = code;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(body));
        };
        try {
          const parts = (req.url ?? "/").split("?")[0].split("/").filter(Boolean).map(decodeURIComponent);
          if (parts[0] === "index") return send(listSources());
          const [src, what, arg] = parts;
          if (!src || !/^(sp\d+|m[\w.-]+)$/.test(src)) return send({ error: "bad source" }, 400);
          if (what === "summary") return send(gamesOf(src).map(summary));
          if (what === "game") {
            const g = gamesOf(src)[Number(arg)];
            if (!g) return send({ error: "no such game" }, 404);
            return send({ ...g, turns: (g.turns ?? []).map(norm) });
          }
          if (what === "t2" && src.startsWith("sp")) {
            const dir = spDir(src.slice(2));
            const recs = fs.existsSync(dir)
              ? fs
                  .readdirSync(dir)
                  .filter((f) => /^t2_w\d+\.jsonl$/.test(f))
                  .sort()
                  .flatMap((f) => readLines(path.join(dir, f)))
              : [];
            return send(recs.slice(0, 2000));
          }
          send({ error: "unknown request" }, 404);
        } catch (e) {
          send({ error: String(e) }, 500);
        }
      });
    },
  };
}
