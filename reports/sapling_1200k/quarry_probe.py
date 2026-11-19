"""Test whether high-margin local quarry decisions appear in Top Five openings and middlegames."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import gzip
import json
import sys

from cash_practice import audit, np, ROOT
sys.path.insert(0, str(ROOT/"code"))
import cnet2_feat as F
from quarry_value import quarry_values


def main(corpus, out):
    meta = json.loads((corpus/"manifest.json").read_text())["games"]
    groups = {k:Counter() for k in ("stay","move")}
    examples = {k:[] for k in groups}
    by_game = []
    for game in sorted(meta,key=lambda g:g["id"]):
        packed = (corpus/(game["id"]+".json.gz")).read_bytes()
        assert sha256(packed).hexdigest() == game["sha256"]
        steps = json.loads(gzip.decompress(packed))["steps"]
        local = {k:Counter() for k in groups}
        for seat in (0,1):
            feat=F.Feat(seat)
            for t,frame in enumerate(steps[:-1]):
                obs=dict(step=t,player=seat,players=frame[0]["observation"]["players"],
                         carbon=frame[seat]["observation"]["carbon"])
                x=feat.observe(obs)
                freshness=t-feat.last_step[F.CANON if seat else F.GRID]
                ys=feat.labels(steps[t+1][seat].get("action") or {})
                for slot in range(10):
                    values=quarry_values(F,x,feat.aux,slot,freshness)
                    if values is None: continue
                    rank=np.argsort(values)
                    best=int(rank[-1])
                    second=max(0.,float(values[rank[-2]]))
                    if values[best] < 25. or values[best] - second < 12.: continue
                    actual=int(ys[slot])
                    # For a move recommendation, require actual economic work on
                    # the target rather than counting transit as mining.
                    uid=feat.own_uid[slot]
                    events=[]
                    first_workers=obs["players"][seat][2]
                    prev=first_workers[uid]
                    for j in range(t+1,min(len(steps),t+4)):
                        following=steps[j][0]["observation"]["players"][seat][2].get(uid)
                        cmd=(steps[j][seat].get("action") or {}).get(uid) or "NONE"
                        if following is None: break
                        if cmd=="NONE" and following[0]==prev[0] and following[1]>prev[1]:
                            events.append((j-t,following[1]-prev[1]))
                        prev=following
                    typ="stay" if best==0 else "move"
                    for c in (groups[typ],local[typ]):
                        c["roots"]+=1;c["expert_chose"]+=actual==best
                        if actual==best:
                            c["realized_next_harvest"]+=any(d==(1 if best==0 else 2) for d,_ in events)
                            c["realized_two_consecutive"]+=all(any(d==q for d,_ in events) for q in ((1,2) if best==0 else (2,3)))
                            c["gain_3_steps_when_selected"]+=sum(g for d,g in events)
                    if actual==best and len(events)>=2 and len(examples[typ])<3:
                        examples[typ].append(dict(gid=game["id"],step=t,seat=seat,uid=uid,
                                                  cargo=float(x["tok_cargo"][slot]),best=best,
                                                  value=float(values[best]),next_best=second,harvests=events))
        by_game.append(dict(gid=game["id"],stats=local))
    result=dict(scope="Seven archived Top Five games; both seats and all nonterminal roots. No counterfactual game outcome observed.",
                threshold="3-turn estimated gathered advantage versus second best >=12 and best value >=25; ambiguous team/visible enemy/late-return positions excluded.",
                games=by_game, summary=groups,examples=examples)
    audit.shared.atomic_json(out,result)
    print(groups,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("corpus",type=Path);p.add_argument("out",type=Path)
    args=p.parse_args();main(args.corpus,args.out)
