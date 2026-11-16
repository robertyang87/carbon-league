#!/usr/bin/env python3
"""Build the adj2 opening league from the main-arm closeout league (5d59f0da).

Per PREREG_PPOU2_ADJ2_20261116.md §3:
- Base: closeout league verbatim (16 keys, priors total 7.45).
- Backbone pressure (capacity + confrontation): m9xj260r1, m9xj260g1 priors
  raised so backbone-only pressure share >= 19.46% of the NEW total.
- Echo2: key added at prior 0.0 (slot reserved). Admission is NOT pre-granted;
  Tier-1 (vs ckpt_50 baseline x1.20 conversion line + dead-slot direct check)
  decides. If admitted, its prior is set at the start boundary (0.45 in the
  filed arithmetic below) and the share re-checked.
- caragor1_ep19 stays a regular member at 0.275 (dead-pressure exited, does
  not re-enter the pressure set). Master2 is not a pool member.
"""
import hashlib
import json

SRC = "/home/hatch/workspace/carbon-live/ppou2/closeout_20261116/league.json"
DST = "/home/hatch/workspace/carbon-live/ppou2/adj2/league_adj2.json"
W = "/root/carbon/data/weights"
K_R1 = f"SNAP:cnet2:{W}/cnet2_m9xj260r1.npz#T2"
K_G1 = f"SNAP:cnet2:{W}/cnet2_m9xj260g1.npz#T2"
K_E2 = f"SNAP:cnet2:{W}/cnet2_m3w5L.npz#T2"

lg = json.load(open(SRC))
assert len(lg) == 16, len(lg)
base_total = sum(v["prior"] for v in lg.values())
assert abs(base_total - 7.45) < 1e-9, base_total

lg[K_R1]["prior"] = 0.90   # 0.45 -> 0.90 (capacity passer + confrontation #2)
lg[K_G1]["prior"] = 0.75   # 0.35 -> 0.75 (confrontation #1)
lg[K_E2] = {"prior": 0.0, "n": 0, "wr": 0.5, "margin": 0.0}  # reserved slot

total = sum(v["prior"] for v in lg.values())
pressure = lg[K_R1]["prior"] + lg[K_G1]["prior"]
share = pressure / total
print(f"new total prior: {total:.2f}")
print(f"backbone pressure: {pressure:.2f} -> share {share*100:.2f}% (line 19.46%)")
assert share >= 0.1946, share
# Echo2-admitted variant (prior 0.45 at the start boundary):
t2 = total + 0.45
p2 = pressure + 0.45
print(f"if Echo2 admitted @0.45: total {t2:.2f}, pressure {p2:.2f} -> {p2/t2*100:.2f}%")
assert p2 / t2 >= 0.1946

data = json.dumps(lg, indent=1).encode()
open(DST, "wb").write(data)
print("wrote", DST, "md5:", hashlib.md5(data).hexdigest(), "keys:", len(lg))
