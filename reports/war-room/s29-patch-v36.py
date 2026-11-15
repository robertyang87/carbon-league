"""Patch the four h29 snaploops to v3.6: window walks FILE LINE NUMBERS.

Root cause (2026-11-13 ~02:58-03:42): nxt = max_seed - s0 + 2 indexes the
window by SEED while sed addresses the file by LINE, and the file is in
completion order (line != seed). Once a tag's stored seeds formed a
complete prefix [s0, max] with no gaps below max, the gap list went empty
and the window kept re-fetching the same 9 lines whose seeds were already
stored -> 0 new rows forever, nxt frozen. Four tags deadlocked exactly
this way (fableg1_m2 43, fableg1_echo2 41, m7bsv2r1_echo2 49,
m7bsv2g1_m2 48); the other four survived on gap fill alone.

Fix: persist per-tag linepos (next line number to fetch) in a sidecar
JSON; window = (linepos, linepos+8) capped at 200; advance linepos on
every trip that reaches the target, whether or not new rows landed. Gap
fill (missing seeds below max, <=2/trip) is unchanged. A full walk visits
every line once, so max necessarily reaches s0+199 and gap fill can then
complete the tag. Existing stores are reused; linepos starts at 1, so
already-walked lines are re-fetched once (dedup by seed, harmless).
"""
import re, sys

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
FILES = ["s29-snaploop-fable.py", "s29-snaploop-fableg1.py",
         "s29-snaploop-m7v2r1.py", "s29-snaploop-m7v2g1.py"]

OLD_WINDOW = """        if s0 is not None and have:
            gaps[focus] = [s for s in range(s0, have[-1] + 1) if s not in store[focus]][:2]
            nxt = have[-1] - s0 + 2
        else:
            gaps[focus] = []
            nxt = 1
        windows[focus] = (nxt, nxt + WINDOW - 1)"""

NEW_WINDOW = """        if s0 is not None and have:
            gaps[focus] = [s for s in range(s0, have[-1] + 1) if s not in store[focus]][:2]
        else:
            gaps[focus] = []
        # v3.6: window walks FILE LINE NUMBERS via persisted linepos.
        # Old nxt = max_seed - s0 + 2 froze once stored seeds formed a
        # complete prefix with no gaps below max (window re-fetched
        # already-stored lines, gap list empty) - four tags stalled
        # 2026-11-13 ~02:58-03:42. Line numbers advance on every reached
        # trip regardless of new rows; gap fill covers the rest.
        lp = linepos.get(focus, 1)
        if lp > 200:
            windows[focus] = (1, 0)  # window exhausted; gap fill only
        else:
            windows[focus] = (lp, min(lp + WINDOW - 1, 200))"""

OLD_ADVANCE = """            store.setdefault(tag, {}).update(new)
            save_store(store)"""

NEW_ADVANCE = """            store.setdefault(tag, {}).update(new)
            save_store(store)
            # v3.6: advance the line window on every reached trip
            if windows[tag][1] >= windows[tag][0]:
                linepos[tag] = windows[tag][1] + 1
                save_linepos(linepos)"""

OLD_SAVESTORE = """def save_store(store):
    with open(STORE, "w") as f:
        json.dump(store, f)"""

NEW_SAVESTORE = OLD_SAVESTORE + """

def load_linepos():
    try:
        with open(LINEPOS) as f:
            return {k: int(v) for k, v in json.load(f).items()}
    except Exception:
        return {}

def save_linepos(lp):
    with open(LINEPOS, "w") as f:
        json.dump(lp, f)"""

for name in FILES:
    p = f"{BASE}/{name}"
    src = open(p).read()
    m = re.search(r'STORE = SNAP \+ "/lines-(.+)\.json"', src)
    assert m, f"{name}: STORE line not found"
    stem = m.group(1)
    src2 = src.replace(f'STORE = SNAP + "/lines-{stem}.json"',
                        f'STORE = SNAP + "/lines-{stem}.json"\nLINEPOS = SNAP + "/linepos-{stem}.json"', 1)
    assert src2 != src, f"{name}: LINEPOS insert failed"
    src = src2
    for old, new, label in [(OLD_SAVESTORE, NEW_SAVESTORE, "helpers"),
                            (OLD_WINDOW, NEW_WINDOW, "window"),
                            (OLD_ADVANCE, NEW_ADVANCE, "advance")]:
        assert src.count(old) == 1, f"{name}: {label} pattern count={src.count(old)}"
        src = src.replace(old, new, 1)
    assert src.count("    store = load_store()") == 1, f"{name}: load_store call not unique"
    src = src.replace("    store = load_store()",
                       "    store = load_store()\n    linepos = load_linepos()", 1)
    assert src.count("v3.5 partial-log)\")") == 1, f"{name}: start-line tag not unique"
    src = src.replace("v3.5 partial-log)\")", "v3.5 partial-log + v3.6 linepos-window)\")", 1)
    open(p, "w").write(src)
    print(f"patched {name} (linepos-{stem}.json)")

print("OK - all four loops patched to v3.6")
