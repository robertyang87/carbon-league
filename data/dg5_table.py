"""dg5_table -- one row per dg5 epoch checkpoint from the slot_val / dg_offline / dagger_probe outputs."""
import json, os, re
D = os.path.dirname(os.path.abspath(__file__))
print("epoch | worst slot drop, all units (pts vs dg2) | heads >1pt down | relabelled agree % | extra stops (dg2: 468)")
for e in range(6):
    sv = open(os.path.join(D, f"slot_val_dg5_ep{e}.out")).read()
    rows = [json.loads("[" + m + "]") for m in re.findall(r"^\s+\d+: \[(.*?)\]", sv, re.M)]
    half = len(rows) // 2
    worst = min(n[3] - b[3] for b, n in zip(rows[:half], rows[half:])) if half else float("nan")
    gate = json.loads(sv.strip().splitlines()[-1]) if sv.strip().endswith("}") else {}
    off = open(os.path.join(D, f"dg_offline_dg5_ep{e}.out")).read()
    ag = re.search(r"relabelled planter decisions agree ([\d.]+)%", off)
    pr = open(os.path.join(D, f"dagger_probe_dg5_ep{e}.out")).read()
    st = re.search(r"we stayed / labeler moved: (\d+)", pr)
    print(f"{e} | {worst:+.2f} | {len(gate.get('info_heads_down_over_1pt', []))} | "
          f"{ag.group(1) if ag else '?'} | {st.group(1) if st else '?'}")
