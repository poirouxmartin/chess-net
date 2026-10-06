"""Live HTML dashboard for training: regenerates checkpoints/dashboard.html
(+ curves PNG) from the training log. Open dashboard.html in a browser;
it refreshes itself every 30 s.

Usage: py -m chessnet.dashboard --log <abs path> --out-dir <abs path> [--loop 30]
"""
import argparse
import html
import json
import os
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .progress import parse_log, load_elo, parse_steps

BG = "#141417"
CARD = "#1d1d22"
TXT = "#d2d2d7"
DIM = "#8c8c96"
GREEN = "#50c878"
AMBER = "#e0aa3c"
RED = "#e06060"
ACCENT = "#7fb8e8"


def current_game(log_path):
    try:
        with open(log_path, errors="replace") as f:
            text = f.read()
    except OSError:
        return None, None, None
    m = list(re.finditer(r"Cycle (\d+)/\d+ \| Self-play \((\d+) games\)", text))
    cyc = int(m[-1].group(1)) if m else None
    total = int(m[-1].group(2)) if m else None
    g = list(re.finditer(r"(?:Game|Fast) (\d+)/(\d+)", text))
    game = int(g[-1].group(1)) if g else 0
    return cyc, game, total


def total_cycles(log_path, default=50):
    try:
        with open(log_path, errors="replace") as f:
            text = f.read()
    except OSError:
        return default
    m = re.search(r"Cycles:\s*(\d+)", text)
    return int(m.group(1)) if m else default


def gpu_info():
    try:
        # CREATE_NO_WINDOW: nvidia-smi runs every loop tick; without this
        # flag each poll flashes a console window open/closed.
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=15,
            creationflags=no_window).decode().strip().split(",")
        return {"util": out[0].strip(), "mem_used": out[1].strip(),
                "mem_total": out[2].strip(), "temp": out[3].strip()}
    except Exception:
        return None


def log_tail(path, n=10):
    try:
        with open(path, errors="replace") as f:
            text = f.read().replace("\r", "\n")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return lines[-n:]
    except OSError:
        return []


def log_freshness(path):
    try:
        return time.time() - os.path.getmtime(path)
    except OSError:
        return None


def recent_ckpts(out_dir, n=8):
    try:
        files = [(f, os.path.getmtime(os.path.join(out_dir, f)))
                 for f in os.listdir(out_dir)
                 if f.endswith((".pt", ".onnx"))]
    except OSError:
        return []
    files.sort(key=lambda t: -t[1])
    return [(f, time.strftime("%d %H:%M", time.localtime(mt))) for f, mt in files[:n]]


def pace_state(out_dir, game, total):
    """Persist (ts, game) to estimate games/min and cycle ETA."""
    sp = os.path.join(out_dir, ".dashboard_pace.json")
    import json
    now = time.time()
    prev = None
    try:
        with open(sp) as f:
            prev = json.load(f)
    except (OSError, ValueError):
        pass
    rate_txt, eta_txt = "—", "—"
    if prev and game is not None and total and game > prev.get("game", -1) and now > prev.get("ts", now):
        dg = game - prev["game"]
        dt = (now - prev["ts"]) / 60.0
        if dg > 0 and dt > 0.2:
            rate = dg / dt
            left = (total - game) / rate
            rate_txt = f"{rate:.1f} parties/min"
            eta_txt = f"≈{left:.0f} min restantes"
    try:
        with open(sp, "w") as f:
            json.dump({"ts": now, "game": game or 0}, f)
    except OSError:
        pass
    return rate_txt, eta_txt


def arrow(cur, prev, lower_is_better):
    if cur is None or prev is None:
        return ""
    try:
        d = cur - prev
    except TypeError:
        return ""
    if abs(d) < 1e-12:
        return '<span class="dim">●</span>'
    good = (d < 0) if lower_is_better else (d > 0)
    cls = "ok" if good else "bad"
    return f'<span class="{cls}">{"▼" if d < 0 else "▲"}</span>'


def verdict(cycles):
    """One-glance verdict: is it really learning?"""
    done = [c for c in cycles if "loss" in c and c.get("games", 1) > 0]
    if len(done) < 2:
        return "Démarrage — pas encore de comparaison possible."
    a, b = done[-2], done[-1]
    bits = []
    dl = b["loss"] - a["loss"]
    if dl < -0.02:
        bits.append(f"✅ loss en baisse ({a['loss']:.2f} → {b['loss']:.2f})")
    elif dl > 0.02:
        bits.append(f"⚠️ loss en hausse ({a['loss']:.2f} → {b['loss']:.2f})")
    else:
        bits.append("➖ loss stable")
    for c in (a, b):
        t = c.get("terms", {})
        tot = sum(t.values()) or 1
        c["_decisive"] = (t.get("mate", 0) + t.get("resign", 0)) / tot
    dd = b["_decisive"] - a["_decisive"]
    if b["_decisive"] > 0.05 and dd > 0.01:
        bits.append(f"✅ parties décisives en hausse ({b['_decisive']:.0%})")
    elif b["_decisive"] > 0.05:
        bits.append(f"✅ parties décisives : {b['_decisive']:.0%}")
    else:
        bits.append("⚠️ que des nulles — pas de signal")
    return " · ".join(bits)


TERM_COLORS = {"mate": "#50c878", "resign": "#7fb8e8", "stalemate": "#8c8c96",
               "fifty": "#5a5a66", "repetition": "#55555e",
               "material": "#6a6a72", "auto-draw": "#4a4a52", "plies-cap": "#3a3a40",
               "other-draw": "#333338"}


def load_seed(out_dir):
    try:
        with open(os.path.join(out_dir, "history_seed.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def load_history(out_dir):
    """Append-only per-cycle history (survives log overwrites)."""
    import csv
    rows = []
    try:
        with open(os.path.join(out_dir, "history.csv"), newline="") as f:
            for r in csv.DictReader(f):
                try:
                    terms = {}
                    for kv in (r.get("terms") or "").split():
                        if ":" in kv:
                            k, v = kv.split(":", 1)
                            terms[k] = int(v)
                    rows.append({"cycle": int(r["cycle"]), "games": int(r["games"]),
                                 "positions": int(r["positions"]),
                                 "avg_result": float(r["avg_result"]),
                                 "loss": float(r["loss"]), "vloss": float(r["vloss"]),
                                 "ploss": float(r["ploss"]), "lr": float(r["lr"]),
                                 "seconds": float(r["seconds"]), "terms": terms})
                except (ValueError, KeyError):
                    continue
    except OSError:
        pass
    return rows


def run_info(log_path):
    """(resume_base_name, saving_range, ) from the log header."""
    try:
        with open(log_path, errors="replace") as f:
            head = f.read(4000)
    except OSError:
        return None, None
    m = re.search(r"Resuming from (.+)", head)
    base = os.path.basename(m.group(1).strip()) if m else "from scratch"
    m = re.search(r"Cycles:\s*\d+ \(saving (\d+)\.\.(\d+)\)", head)
    rng = f"{m.group(1)}..{m.group(2)}" if m else ""
    return base, rng


def live_counters(log_path):
    """Latest per-game progress: (done, total, positions, terms_dict)."""
    try:
        with open(log_path, errors="replace") as f:
            text = f.read().replace("\r", "\n")
    except OSError:
        return None
    ms = list(re.finditer(r"(?:Game|Fast) (\d+)/(\d+) · ([\d,]+) pos(?: \[(.*)\])?", text))
    if not ms:
        return None
    m = ms[-1]
    terms = {}
    if m.group(4):
        for kv in m.group(4).split():
            if ":" in kv:
                k, v = kv.split(":", 1)
                try:
                    terms[k] = int(v)
                except ValueError:
                    pass
    return int(m.group(1)), int(m.group(2)), int(m.group(3).replace(",", "")), terms


def run_start(log_path):
    try:
        ts = os.path.getctime(log_path)
    except OSError:
        return None
    return ts


def run_started_txt(log_path):
    ts = run_start(log_path)
    if ts is None:
        return "?"
    dt = max(0.0, time.time() - ts)
    h, m = int(dt // 3600), int((dt % 3600) // 60)
    ago = f"{h}h{m:02d}" if h else f"{m}min"
    return f"{time.strftime('%d/%m %H:%M', time.localtime(ts))} (il y a {ago})"


def game_in_hist(cycles, cyc):
    return any(c.get("cycle") == cyc for c in cycles)


def load_tactics(out_dir):
    try:
        with open(os.path.join(out_dir, "tactics.json")) as f:
            return json.load(f).get("nets", {})
    except (OSError, ValueError):
        return {}


def save_curves(cycles, elos, seed, tactics, steps, out):
    if not cycles and not seed and not tactics and not steps:
        return
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=False)
    ax1, ax2, ax3, ax4 = axes.flat
    for ax in axes.flat:
        ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    done = [c for c in cycles if "loss" in c and c.get("games", 1) > 0]
    if seed:
        sx = [c["cycle"] for c in seed]
        ax1.plot(sx, [c["loss"] for c in seed], "x--", color="gray",
                 label="runs précédents (20 parties/cycle)", markersize=5)
    if done:
        xs = [c["cycle"] for c in done]
        ax1.plot(xs, [c["loss"] for c in done], "o-", color=GREEN, label="loss")
        ax1.plot(xs, [c["vloss"] for c in done], "s-", color=ACCENT, label="value")
        ax1.plot(xs, [c["ploss"] for c in done], "^-", color=AMBER, label="policy")
    # Live dense curves (one point per ~100 steps, always positive ploss /
    # vloss so they fit the log axis; the entropy-subtracted total can go
    # negative). Steps on a twin top axis so no scale is faked.
    if steps:
        sx = [p["step"] for p in steps]
        ax1b = ax1.twiny()
        ax1b.plot(sx, [p["ploss"] for p in steps], "-", color=AMBER,
                  alpha=0.55, linewidth=1, label="policy live")
        ax1b.plot(sx, [p["vloss"] for p in steps], "-", color=ACCENT,
                  alpha=0.55, linewidth=1, label="value live")
        ax1b.set_xlabel("pas (live)", fontsize=8)
        ax1b.tick_params(labelsize=7)
        ax1b.legend(fontsize=7, loc="upper right")
    ax1.set_ylabel("loss (log)")
    ax1.set_yscale("log")
    if ax1.get_legend_handles_labels()[1]:
        ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    # Aligned on non-empty windows only (flush row excluded everywhere).
    plot = [c for c in cycles if c.get("games", 1) > 0]
    if done:
        xs = [c["cycle"] for c in done]
    else:
        xs = [c["cycle"] for c in plot] or [0]
    if plot:
        ax2.plot([c["cycle"] for c in plot], [c["avg_result"] for c in plot],
                 "o-", color=GREEN)
    else:
        ax2.text(0.5, 0.5, "premiere mesure en cours",
                 transform=ax2.transAxes, ha="center", color="gray")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("avg_result")
    ax2.grid(True, alpha=0.25)
    melos = [e for e in elos if abs(e.get("elo", 0)) >= 10]
    if melos:
        ax2b = ax2.twinx()
        ax2b.plot(range(1, len(melos) + 1), [e["elo"] for e in melos])
        ax2b.set_ylabel("Elo (mesure #)", color="orange")
    terms = [c.get("terms", {}) for c in plot]
    keys = []
    for t in terms:
        for k in t:
            if k not in keys:
                keys.append(k)
    if not keys:
        ax3.text(0.5, 0.5, "repartition a venir",
                 transform=ax3.transAxes, ha="center", color="gray")
    bottoms = [0.0] * len(plot)
    for k in keys:
        vals = [t.get(k, 0) / max(1, sum(t.values())) for t in terms]
        ax3.bar(xs, vals, bottom=bottoms, label=k)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax3.set_ylabel("fins de partie")
    if keys:
        ax3.legend(fontsize=7, ncol=4)
    # Tactics: mates found by policy top-1 per net (instant strength signal).
    if tactics:
        names = list(tactics.keys())
        hits = [tactics[n].get("hits", 0) for n in names]
        tots = [tactics[n].get("total", 1) for n in names]
        xpos = range(len(names))
        ax4.bar(xpos, hits, color=GREEN, label="trouvés")
        ax4.bar(xpos, [t - h for h, t in zip(hits, tots)], bottom=hits,
                color="#3a3a40", label="ratés")
        ax4.set_xticks(list(xpos))
        ax4.set_xticklabels([n.replace("model_", "").replace(".pt", "") for n in names], fontsize=8)
        ax4.set_ylabel("mats/3")
        ax4.set_ylim(0, max(tots + [1]))
        ax4.legend(fontsize=7)
    else:
        ax4.text(0.5, 0.5, "tactics.json absent", ha="center", color="gray")
    ax4.set_xlabel("réseau")
    fig.tight_layout()
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


CSS = (
    "body{background:#141417;color:#d2d2d7;font-family:Consolas,monospace;margin:10px 20px;max-width:none}"
    "h1{font-size:18px;margin:0}.sub{color:#8c8c96;font-size:12px}"
    ".cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:8px 0}"
    ".card{background:#1d1d22;border:1px solid #2a2a30;border-radius:8px;padding:8px 12px}"
    ".card .k{font-size:11px;color:#8c8c96}.card .v{font-size:22px;font-weight:bold}"
    ".ok{color:#50c878}.bad{color:#e06060}.dim{color:#8c8c96}"
    "h2{color:#8fc9a0;font-size:14px;margin:12px 0 6px}"
    ".tbl{overflow-x:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:2px 10px 2px 0;text-align:right;white-space:nowrap}"
    "th{color:#8c8c96}pre{background:#1d1d22;padding:8px;font-size:11px;overflow-x:auto;margin:6px 0}"
    "img{width:100%;display:block}.bar{background:#2a2a30;border-radius:6px;height:12px;overflow:hidden;margin:3px 0 6px}"
    ".bar>div{background:#50c878;height:100%}.bar.total>div{background:#7fb8e8}"
    ".cols{display:grid;grid-template-columns:3fr 2fr;gap:0 24px;align-items:start}"
    ".cols>div{min-width:0}"
    "@media(max-width:750px){.cols{grid-template-columns:1fr}}"
    ".hero{display:flex;align-items:center;gap:20px;background:#1d1d22;border:1px solid #2a2a30;"
    "border-radius:10px;padding:10px 20px;margin:10px 0}"
    ".hero-elo{font-size:48px;font-weight:bold;line-height:1;white-space:nowrap}"
    ".hero-main{flex:1;min-width:0}"
    ".hero-sub{font-size:12px;color:#8c8c96}"
    ".hero-verdict{font-size:14px;font-weight:bold;margin:4px 0}"
)


def bar(pct):
    pct = max(0.0, min(100.0, pct))
    return f'<div class="bar"><div style="width:{pct:.1f}%"></div></div>'


def render(args, cycles, elos, cyc, game, total, n_cycles, gpu, rate_txt, eta_txt,
           fresh_s, totals, run_base, run_range, started_txt, live_txt=""):
    last = cycles[-1] if cycles else {}
    prev = cycles[-2] if len(cycles) >= 2 else {}
    loss = last.get("loss")
    res = last.get("avg_result")
    cards = (
        f"<div class='card'><div class='k'>LOSS (dernier cycle)</div>"
        f"<div class='v'>{loss:.4f} {arrow(loss, prev.get('loss'), True)}</div></div>"
        if loss is not None else "<div class='card'><div class='k'>LOSS</div><div class='v dim'>—</div></div>"
    )
    cards += (
        f"<div class='card'><div class='k'>RÉSULTAT self-play</div>"
        f"<div class='v'>{res:.3f} {arrow(res, prev.get('avg_result'), False)}</div></div>"
        if res is not None else "<div class='card'><div class='k'>RÉSULTAT</div><div class='v dim'>—</div></div>"
    )
    if elos:
        e = elos[-1]
        cards += (f"<div class='card'><div class='k'>ELO {html.escape(os.path.basename(e.get('a','?')))} vs "
                  f"{html.escape(os.path.basename(e.get('b','?')))}</div>"
                  f"<div class='v'>{e.get('elo',0):+.0f}</div></div>")
    else:
        cards += "<div class='card'><div class='k'>ELO</div><div class='v dim'>—</div></div>"
    if game and total:
        cards += (f"<div class='card'><div class='k'>RYTHME</div>"
                  f"<div class='v' style='font-size:16px'>{rate_txt}</div></div>"
                  f"<div class='card'><div class='k'>FIN DE CYCLE</div>"
                  f"<div class='v' style='font-size:16px'>{eta_txt}</div></div>")
    if gpu:
        cards += (f"<div class='card'><div class='k'>GPU</div>"
                  f"<div class='v' style='font-size:16px'>{gpu['util']}% · {gpu['temp']}°C</div></div>")
    tg, tp = totals
    cards += (f"<div class='card'><div class='k'>PARTIES TOTALES</div>"
              f"<div class='v'>{tg:,}</div></div>"
              f"<div class='card'><div class='k'>POSITIONS TOTALES</div>"
              f"<div class='v'>{tp:,}</div></div>")

    if cyc and game and total:
        prog = (f"<div>Cycle {cyc} — partie {game}/{total}</div>{bar(100.0*game/total)}"
                f"<div>Total : {len(cycles)}/{n_cycles} cycles</div>{bar(100.0*len(cycles)/max(1,n_cycles)).replace('bar', 'bar total')}")
    elif cyc:
        prog = f"<div>Cycle {cyc} — phase training</div>"
    else:
        prog = "<div class='dim'>En attente de données…</div>"

    if fresh_s is None:
        health = '<span class="bad">● log introuvable</span>'
    elif fresh_s < 600:
        health = f'<span class="ok">● vivant (log il y a {fresh_s:.0f} s)</span>'
    else:
        health = f'<span class="bad">● log silencieux depuis {fresh_s/60:.0f} min — vérifier le run !</span>'

    rows = "".join(
        f"<tr><td>{c['cycle']}</td><td>{c.get('games', 0)}</td><td>{c.get('positions', 0)}</td>"
        f"<td>{c.get('avg_result', float('nan')):.3f}</td><td>{c.get('loss', float('nan')):.4f}</td>"
        f"<td>{c.get('vloss', float('nan')):.4f}</td><td>{c.get('ploss', float('nan')):.4f}</td>"
        f"<td>{c.get('seconds', 0):.0f}s</td></tr>"
        for c in cycles[-8:] if c.get("games", 1) > 0)
    if elos:
        erows = "".join(
            f"<tr><td>{html.escape(os.path.basename(e.get('a', '?')))}</td>"
            f"<td>{html.escape(os.path.basename(e.get('b', '?')))}</td>"
            f"<td>{e.get('games', 0)}</td><td>{e.get('score', 0):.3f}</td>"
            f"<td>{e.get('elo', 0):+.0f}</td></tr>"
            for e in elos[-15:])
        elo = f"<table><tr><th>new</th><th>ref</th><th>games</th><th>score</th><th>elo</th></tr>{erows}</table>"
    else:
        elo = '<p class="dim">Pas de mesure Elo décisive pour l\'instant (que des nulles).</p>'
    ckpts = "".join(f"<tr><td style='text-align:left'>{html.escape(f)}</td><td>{mt}</td></tr>"
                    for f, mt in recent_ckpts(args.out_dir))
    tail = "\n".join(html.escape(ln) for ln in log_tail(args.log, 6))
    curves = ('<img src="training_progress.png" alt="courbes">'
              if os.path.exists(os.path.join(args.out_dir, "training_progress.png"))
              else '<p class="dim">Les courbes apparaissent dès la fin du premier cycle.</p>')
    now = time.strftime("%H:%M:%S")
    refresh = args.loop if args.loop else 30
    if elos:
        e = elos[-1]
        ev = e.get("elo", 0)
        ecls = "ok" if ev > 0 else ("bad" if ev < 0 else "dim")
        elo_big = f"<div class='hero-elo {ecls}'>{ev:+.0f}</div>"
        elo_sub = (f"ELO · {html.escape(os.path.basename(e.get('a', '?')))} vs "
                   f"{html.escape(os.path.basename(e.get('b', '?')))} "
                   f"({e.get('games', 0)} parties)")
    else:
        elo_big = "<div class='hero-elo dim'>—</div>"
        elo_sub = "ELO · en attente de matchs décisifs"
    hero = (f"<div class='hero'>{elo_big}<div class='hero-main'>"
            f"<div class='hero-sub'>{elo_sub}</div>"
            f"<div class='hero-verdict'>{verdict(cycles)}</div>"
            f"{prog}</div></div>")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<title>chessnet training</title><style>{CSS}</style></head><body>
<h1>♟ chessnet training</h1>
<div class="sub">{health} · maj {now} · refresh auto {refresh} s</div>
<div class="sub">{live_txt}</div>
{hero}
<div class="cards"><div class="card" style="min-width:300px"><div class="k">RÉSEAU ENTRAÎNÉ</div><div class="v" style="font-size:17px">{run_base} → {run_range}</div></div>
<div class="card" style="min-width:220px"><div class="k">DÉMARRÉ</div><div class="v" style="font-size:17px">{started_txt}</div></div></div>
<div class="cols">
<div>
<h2>Courbes</h2>
{curves}
</div>
<div>
<h2>Chiffres</h2>
<div class="cards">{cards}</div>
<h2>Cycles (8 derniers)</h2>
<div class="tbl"><table><tr><th>cyc</th><th>games</th><th>pos</th><th>result</th><th>loss</th><th>vloss</th><th>ploss</th><th>time</th></tr>
{rows}</table></div>
<h2>Elo</h2>
{elo}
<h2>Checkpoints récents</h2>
<div class="tbl"><table>{ckpts}</table></div>
<h2>Log</h2>
<pre>{tail}</pre>
</div>
</div>
</body></html>"""

def once(args):
    live = parse_log(args.log)
    elos = load_elo(args.out_dir)
    seed = load_seed(args.out_dir)
    tactics = load_tactics(args.out_dir)
    # Merge append-only history (survives restarts/log overwrites) with the
    # live log; history wins on duplicates.
    hist = load_history(args.out_dir)
    seen = {c["cycle"] for c in hist}
    cycles = sorted(hist + [c for c in live if c["cycle"] not in seen],
                    key=lambda c: c["cycle"])
    save_curves(cycles, elos, seed, tactics, parse_steps(args.log),
                os.path.join(args.out_dir, "training_progress.png"))
    cyc, game, total = current_game(args.log)
    live_cnt = live_counters(args.log)
    live_pos = 0
    if live_cnt:
        game, total, live_pos = live_cnt[0], live_cnt[1], live_cnt[2]
    n_cycles = total_cycles(args.log)
    gpu = gpu_info()
    rate_txt, eta_txt = pace_state(args.out_dir, game, total)
    fresh_s = log_freshness(args.log)
    run_base, run_range = run_info(args.log)
    started_txt = run_started_txt(args.log)
    cur_done = 0 if game_in_hist(cycles, cyc) else (game or 0)
    cur_pos = 0 if game_in_hist(cycles, cyc) else live_pos
    tg = sum(c.get("games", 0) for c in cycles) + cur_done
    tp = sum(c.get("positions", 0) for c in cycles) + cur_pos
    steps = parse_steps(args.log)
    if steps:
        s = steps[-1]
        live_txt = (f"● live pas {s['step']} · ploss {s['ploss']:.4f} · "
                    f"vloss {s['vloss']:.4f} · ent {s['ent']:.3f} · "
                    f"{s['games']} parties · buf {s['buf']:,}")
    else:
        live_txt = "○ en attente des premiers pas…"
    page = render(args, cycles, elos, cyc, game, total, n_cycles, gpu, rate_txt,
                  eta_txt, fresh_s, (tg, tp), run_base or "?", run_range or "",
                  started_txt, live_txt)
    with open(os.path.join(args.out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[{time.strftime('%H:%M:%S')}] dashboard: {len(cycles)} cycles, "
          f"cycle={cyc} game={game} gpu={gpu['util'] + '%' if gpu else '?'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="training_big.log")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--loop", type=int, default=0, help="regenerate every N s (0 = once)")
    args = ap.parse_args()
    while True:
        try:
            once(args)
        except Exception as e:  # never kill a monitoring loop on a parse hiccup
            print(f"dashboard error: {e}", flush=True)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
