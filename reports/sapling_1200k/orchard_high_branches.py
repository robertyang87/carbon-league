"""Find prelocked >50 opportunities across independent old maps, then run at most four paired roots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

import orchard_branches as branch

OUT=branch.HERE/"orchard_high_branches"


def schedule():
    first=branch.plan()
    used={c["id"] for c in first["maps"]}
    all_cards=json.loads((branch.HERE/"top5_h2h_data.json").read_bytes())["games"]
    recent=json.loads((branch.HERE/"latest5_inventory.json").read_bytes())["games"]
    pool=[c for c in all_cards if c["split"]=="train" and c["id"] not in used]
    if {c["seed"] for c in pool}&{c["seed"] for c in recent}:
        raise ValueError("new diagnostic map overlaps")
    cards=sorted(pool,key=lambda c:hashlib.sha256(("orchard-high-root-1|"+c["id"]).encode()).digest())[:24]
    rows=[]
    for idx,card in enumerate(cards):
        row={k:card[k] for k in ("id","seed","canonical_sha256")}
        row.update(seat=idx%2,group="design" if idx%4<2 else "reserved",center=105)
        rows.append(row)
    return {"pack":branch.SHA,"maps":rows,
            "selection":"24 fresh original training maps, no overlap with six first-phase maps or 26 recent diagnostic maps. SHA256(orchard-high-root-1|id) ordering. Four disjoint strata by prospective group and seat; six maps per stratum. Use first map in each stratum with a qualifying root regardless of any subsequent outcome. No refill beyond six; uninspected later maps stay uninspected.",
            "root":"Same t60..150 five-step grid, visible cash/information, highest public predicted net with strict value >50. Parent already wants a safe move. Staying must also satisfy the prefix safety and final joint check and change exactly one order. Original opposing command is shared. No true future information used in selection.",
            "budget":"Up to four paired roots (eight responsive complete trajectories), two CPUs. Same deterministic frozen hybrid on each seat and independent original interpreter verification as phase one. Assess design and reserved strata separately; no candidate may be promoted from two reserved maps alone."}


def lane(rows,paths,cpu):
    out={}
    for group in ("design","reserved"):
        examined=[]
        for card in rows:
            if card["group"]!=group:
                continue
            got=branch.work(card,paths,cpu,OUT,50.)
            examined.append(card["id"])
            if got:
                break
        out[group]={"examined":examined,"chosen":examined[-1] if got else None}
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--replays",required=True,type=Path)
    ap.add_argument("--cpus",nargs=2,type=int,default=[0,1]);args=ap.parse_args()
    if branch.arena.prior.digest(branch.PACK)!=branch.SHA:
        raise ValueError("pack identity mismatch")
    p=schedule();OUT.mkdir(exist_ok=True)
    fp=OUT/"plan.json"
    if fp.exists() and json.loads(fp.read_bytes())!=p:
        raise ValueError("prospective plan changed")
    if not fp.exists():branch.write(fp,p)
    paths={f.name.removesuffix(".json.gz"):f for f in args.replays.rglob("*.json.gz")}
    if not set(c["id"] for c in p["maps"])<=set(paths):raise ValueError("missing preregistered map")
    with ProcessPoolExecutor(2) as pool:
        fs=[pool.submit(lane,[c for c in p["maps"] if c["seat"]==s],paths,cpu) for s,cpu in enumerate(args.cpus)]
        outcomes=[f.result() for f in fs]
    branch.write(OUT/"coverage.json",outcomes)
