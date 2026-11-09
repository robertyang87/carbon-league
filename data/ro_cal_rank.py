"""rank the rule-stand-in calibration runs by distance to the M-vs-us ladder numbers (32 games, steps 1-100):
M planted 9.4, seized 6.3, planters 19, t50 ~6; us (dg5 side) planted 7.2, seized 6.9, t50 ~3."""
import glob, json, statistics as S
rows = []
for f in glob.glob("bigeval_live/ro_cal*_*.jsonl"):
    R = [g for g in map(json.loads, open(f)) if g.get("margin") is not None]
    if len(R) < 90:
        continue
    o = lambda k, s="opp": S.mean(g["lx"][s].get(k, 0) or 0 for g in R)
    v = dict(Mpl=o("planted"), Msz=o("seized"), Mrec=o("pla_rec"), Mt50=o("tree50"), Upl=o("planted", "pol"), Usz=o("seized", "pol"), Ut50=o("tree50", "pol"),
             Mcc=S.mean(g["lx"]["opp"]["cash100"] + g["lx"]["opp"]["cargo100"] for g in R))
    d = (abs(v["Mpl"] - 9.4) / 2 + abs(v["Msz"] - 6.3) / 2 + abs(v["Mrec"] - 19) / 4 + abs(v["Mt50"] - 6) / 1.5
         + abs(v["Upl"] - 7.2) / 2 + abs(v["Usz"] - 6.9) / 2 + abs(v["Ut50"] - 3) / 1.5)
    rows.append((d, f.split("ro_")[-1][:-6], v))
rows.sort(key=lambda r: r[0])
print("dist  variant                          | M: planted seized planters t50 cash+cargo | us: planted seized t50")
for d, n, v in rows[:12]:
    print(f"{d:5.2f} {n:32s} | {v['Mpl']:5.1f} {v['Msz']:5.1f} {v['Mrec']:5.1f} {v['Mt50']:4.1f} {v['Mcc']:6.0f} | {v['Upl']:5.1f} {v['Usz']:5.1f} {v['Ut50']:4.1f}")
print(len(rows), "runs; target | 9.4 6.3 19 6 ~2177 | 7.2 6.9 3")
