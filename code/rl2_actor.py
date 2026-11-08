"""rl2_actor -- always-on cloud rollout actor for the split PPO loop (cloud plays, Mac learns).

    python3 rl2_actor.py ROOT --workers 56 --lockstep 2 --shard-games 8 --temp 1.0

ROOT/in/live/w_<it>.npz   latest learner weights (pushed by the Mac; highest <it> wins)
ROOT/in/league.json       learner's League table (opponent priors + PFSP win rates; snapshots map to ROOT/in/snaps/)
ROOT/out/it<IT>_<t>_<n>.pkl.z   finished games (zlib'd pickle of play_batch "games" lists), pulled + removed by the Mac
Workers never wait for the learner: every task uses whatever weights are newest when it is submitted, so the CPU
stays saturated. ROOT/out is capped at --max-out-games (oldest shards dropped) so a stalled learner cannot fill
the disk; the learner itself drops games whose behaviour weights are too stale.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BC_W", "64")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def latest_weights(live):
    ws = [f for f in os.listdir(live) if f.startswith("w_") and f.endswith(".npz")]
    if not ws:
        return None, -1
    f = max(ws, key=lambda s: int(s[2:-4]))
    return os.path.join(live, f), int(f[2:-4])


def load_league(path, snaps_dir):
    try:
        tab = json.load(open(path))
    except (OSError, ValueError):
        return None
    keys, w = [], []
    for k, v in tab.items():
        if v.get("prior", 0) <= 0:
            continue
        if k.startswith("SNAP:"):
            _, ad, p = k.split(":", 2)
            lp = os.path.join(snaps_dir, os.path.basename(p))
            if not os.path.exists(lp):
                continue
            spec = f"SNAP:{ad}:{lp}"
        else:
            spec = k
        keys.append((k, spec))
        w.append(v["prior"] * (1.0 - v.get("wr", 0.5) + 0.05) ** PFSP)
    if not keys:
        return None
    w = np.asarray(w, np.float64)
    return keys, w / w.sum()


PFSP = 1.0


def main():
    global PFSP
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--adapter", default="cnet2")
    p.add_argument("--workers", type=int, default=56)
    p.add_argument("--lockstep", type=int, default=2)
    p.add_argument("--shard-games", type=int, default=8)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--pfsp", type=float, default=1.0)
    p.add_argument("--max-out-games", type=int, default=1200)
    p.add_argument("--seed0", type=int, default=5000000)
    a = p.parse_args()
    PFSP = a.pfsp
    live = os.path.join(a.root, "in", "live"); snaps = os.path.join(a.root, "in", "snaps")
    out = os.path.join(a.root, "out"); tmp = os.path.join(a.root, "tmp")
    for d in (live, snaps, out, tmp):
        os.makedirs(d, exist_ok=True)
    import rl2_ppo_rollout as RO
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.workers)
    rng = np.random.default_rng(int(time.time()))
    seed = a.seed0 + int(time.time()) % 1000000 * 1000
    inflight = []
    buf, buf_it = [], None
    stats = {"t0": time.time(), "games": 0, "shards": 0, "dropped": 0}
    lg_t, lg = 0.0, None
    log = open(os.path.join(a.root, "actor.log"), "a")
    while True:
        wk, it = latest_weights(live)
        if wk is None:
            time.sleep(5)
            continue
        if time.time() - lg_t > 20:
            lg = load_league(os.path.join(a.root, "in", "league.json"), snaps) or lg
            lg_t = time.time()
        if lg is None:
            time.sleep(5)
            continue
        keys, pw = lg
        while len(inflight) < 2 * a.workers:
            games = []
            for j in range(a.lockstep):
                k, spec = keys[rng.choice(len(keys), p=pw)]
                games.append((spec, seed + j // 2, j % 2))
            seed += a.lockstep
            task = {"adapter": a.adapter, "wkey": wk, "games": games, "temp": a.temp, "greedy": False,
                    "record": True, "rng_seed": int(rng.integers(1 << 30))}
            inflight.append((it, keys, pool.apply_async(RO.play_batch, (task,))))
        still = []
        for item in inflight:
            it_b, keys_b, r = item
            if not r.ready():
                still.append(item)
                continue
            try:
                res = r.get()
            except Exception as e:      # one bad game must not stop the actor
                log.write(json.dumps({"t": round(time.time()), "err": repr(e)[:300]}) + "\n"); log.flush()
                continue
            spec2key = {s: k for k, s in keys_b}
            for g in res["games"]:
                g["opp"] = spec2key.get(g["opp"], g["opp"])      # report under the learner's league key
                g["beh_it"] = it_b
            if buf_it is not None and buf_it != it_b and buf:
                flush(out, tmp, buf, buf_it, stats)
                buf = []
            buf_it = it_b
            buf.extend(res["games"])
            if len(buf) >= a.shard_games:
                flush(out, tmp, buf, buf_it, stats)
                buf = []
        inflight = still
        cap(out, a.max_out_games, a.shard_games, stats)
        if time.time() - stats.get("tl", 0) > 60:
            stats["tl"] = time.time()
            el = time.time() - stats["t0"]
            log.write(json.dumps({"t": round(time.time()), "it": it, "games": stats["games"],
                                  "games_per_h": round(3600 * stats["games"] / max(el, 1)),
                                  "out_shards": len(os.listdir(out)), "dropped": stats["dropped"]}) + "\n")
            log.flush()
        time.sleep(0.5)


def flush(out, tmp, games, it, stats):
    name = f"it{it:06d}_{int(time.time() * 1000)}_{len(games)}.pkl.z"
    tp = os.path.join(tmp, name)
    with open(tp, "wb") as f:
        f.write(zlib.compress(pickle.dumps(games, protocol=pickle.HIGHEST_PROTOCOL), 1))
    os.replace(tp, os.path.join(out, name))
    stats["games"] += len(games); stats["shards"] += 1


def cap(out, max_games, shard_games, stats):
    fs = sorted(os.listdir(out))
    over = len(fs) - max(1, max_games // max(1, shard_games))
    for f in fs[:max(0, over)]:
        try:
            os.remove(os.path.join(out, f)); stats["dropped"] += 1
        except OSError:
            pass


if __name__ == "__main__":
    main()
