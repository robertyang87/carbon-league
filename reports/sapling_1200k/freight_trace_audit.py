"""Inspect the subsequent expert trajectory at each root where freight intervened."""
from collections import Counter
from pathlib import Path
import argparse
import gzip
import json
import sys

from cash_practice import audit, ROOT

sys.path.insert(0, str(ROOT / "code"))
import cnet2_feat as F


def main(practice, corpus, out):
    by_game = {}
    practice_rows = json.loads(practice.read_bytes())
    for change in practice_rows["changes"]:
        by_game.setdefault(change["id"], []).append(change)
    inventory = {r["id"]:r for r in json.loads((corpus/"manifest.json").read_bytes())["games"]}
    rows = []
    for gid, changes in by_game.items():
        gpath = corpus / (gid+".json.gz")
        if audit.digest(gpath) != inventory[gid]["sha256"]:
            raise ValueError(gid)
        steps = json.loads(gzip.decompress(gpath.read_bytes()))["steps"]
        for root in changes:
            s, t = root["seat"], root["step"]
            mine = steps[t][0]["observation"]["players"][s]
            obs = dict(player=s, step=t, players=steps[t][0]["observation"]["players"],
                       carbon=steps[t][s]["observation"]["carbon"])
            feat = F.Feat(s)
            x = feat.observe(obs)
            for slot, a0, a1 in root["swaps"]:
                uid = feat.own_uid[slot]
                cell, load, role = mine[2][uid]
                if role != "COLLECTOR":
                    raise ValueError(uid)
                expert = (steps[t+1][s].get("action") or {}).get(uid) or "NONE"
                def physical(a):
                    return int(F.CANON_MOVE[a]) if s else a
                a0, a1 = map(physical, (a0, a1))
                homes = list(mine[1].values())
                def dist(a):
                    return int(F.DIST[F.DEST[cell,a], homes].min())
                after = "still_carrying"
                horizon = min(t+12,len(steps)-1)
                for j in range(t+1,horizon+1):
                    currently = steps[j][0]["observation"]["players"][s][2].get(uid)
                    if currently is None:
                        after = "unit_disappeared"
                        break
                    previous = steps[j-1][0]["observation"]["players"][s][2][uid]
                    if currently[0] in homes and currently[1] == 0 and previous[1]>0:
                        after = "reached_own_home_and_unloaded"
                        break
                rows.append(dict(id=gid,seat=s,step=t,slot=slot,uid=uid,load=load,source_distance=dist(0),
                                 parent=F.MOVES[a0], proposed=F.MOVES[a1], expert=expert,
                                 parent_distance=dist(a0),proposed_distance=dist(a1),
                                 expert_followup=after,followup_turn=j-t))
    result=dict(scope="Actual 12-turn expert follow-up at changed same-prefix slots only. Later slots in a root can already have a counterfactual prefix. Alternative trajectories are unobserved.",
                practice_sha256=audit.digest(practice),slots=len(rows),
                agreement=dict(Counter("proposed" if r["expert"]==r["proposed"] else
                                       "parent" if r["expert"]==r["parent"] else "neither" for r in rows)),
                distance_change=dict(Counter("toward" if r["proposed_distance"]<r["parent_distance"] else
                                       "away" if r["proposed_distance"]>r["parent_distance"] else "same"
                                       for r in rows)),rows=rows)
    audit.shared.atomic_json(out,result)
    print(result["slots"],result["agreement"], result["distance_change"])


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("practice","corpus","out"):
        p.add_argument(name,type=Path)
    args=p.parse_args()
    main(args.practice,args.corpus,args.out)
