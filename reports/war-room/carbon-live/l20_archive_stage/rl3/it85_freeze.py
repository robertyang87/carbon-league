#!/usr/bin/env python3
"""it85 freeze watcher (copy-only; never touches the trainer process).

When the it85 train row lands and state_last.pt is stable, freeze copies
for the §17 checkpoint exam: state_frozen_it85.pt (+md5), ckpt_85.npz and
league.json copies, and dump the decision inputs (last train rows, EVAL
tail) into the status file for the §15/KL boundary decision.
"""
import hashlib
import json
import os
import subprocess
import time

RUN = "/root/carbon/data/rl3/snipe_n1"
TRAIN = os.path.join(RUN, "train.jsonl")
STATE = os.path.join(RUN, "state_last.pt")
STATUS = os.path.join(RUN, "it85_freeze_status.txt")


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    with open(STATUS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return out


def main():
    log("WATCHER_START it85 freeze armed (copy-only)")
    beat = time.time()
    while True:
        tr = rows(TRAIN)
        if tr and tr[-1].get("it", 0) >= 85:
            log(f"it85 row seen kl={tr[-1].get('kl')} kl_coef={tr[-1].get('kl_coef')} wr={tr[-1].get('winrate')}")
            break
        if time.time() - beat > 600:
            log(f"waiting it85 (last_it={tr[-1].get('it') if tr else None})")
            beat = time.time()
        time.sleep(30)
    # wait for state stability (two identical size+mtime reads 30 s apart)
    prev = None
    for _ in range(40):
        try:
            st = os.stat(STATE)
            sig = (st.st_size, st.st_mtime)
        except FileNotFoundError:
            sig = None
        if sig is not None and sig == prev:
            break
        prev = sig
        time.sleep(30)
    import torch
    st8 = torch.load(STATE, map_location="cpu", weights_only=False)
    log(f"state stable, embedded it={st8['it']}")
    if int(st8["it"]) != 85:
        log("ABORT embedded it != 85")
        return
    dst = os.path.join(RUN, "state_frozen_it85.pt")
    subprocess.run(["cp", "-f", STATE, dst], check=True)
    log(f"frozen state -> {dst} md5={md5(dst)} src_md5={md5(STATE)}")
    for name, new in (("ckpt_85.npz", "ckpt_85_frozen.npz"), ("league.json", "league_it85.json")):
        src = os.path.join(RUN, name)
        if os.path.exists(src):
            subprocess.run(["cp", "-f", src, os.path.join(RUN, new)], check=True)
            log(f"frozen {name} -> {new} md5={md5(src)}")
        else:
            log(f"WARN {name} not found")
    tr = rows(TRAIN)
    for r in tr[-6:]:
        log(f"row it={r['it']} kl={r['kl']} kl_coef={r['kl_coef']} wr={r['winrate']} games={r['games']}")
    ev = rows(os.path.join(RUN, "eval.jsonl"))
    log(f"eval rows present its={sorted(set(e['it'] for e in ev))}")
    for e in ev[-4:]:
        log(f"eval it={e['it']} opp={e['opp']} mean={e['mean']} se={e['se']} wins={e['wins']}")
    log("FREEZE_DONE")


if __name__ == "__main__":
    main()
