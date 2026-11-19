"""Independently check actor departures against exact expert commands before packaging quarry."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import gzip
import json
import sys

from cash_practice import audit,np
from freight_build import HERE,OUTPUT


def main(corpus, out):
    # The 10 lexicographically earliest frozen101 maps form a fixed check
    # distinct from the seven later discovery games. Some supplied V3's BC data.
    inventory = json.loads((HERE/"top5_h2h_data.json").read_text())["mutual_matches"]
    fixed = sorted(inventory,key=lambda g:g["id"])[:10]
    mod = audit.shared.module(OUTPUT,"quarry_actor_check")
    # Replace only the freight economic policy. Retain the exact frozen
    # parent's encode and decoded prefix, and inherited post-proposal joint veto.
    for n in ("quarry_value.py","quarry_runtime.py"):
        exec((HERE/n).read_text(),mod.__dict__)
    mod._freight_decide = mod.quarry_decide
    sys.path.insert(0, str(HERE.parents[1] / "code"))
    from materialize_top5_h2h import ARCHIVE, read_repo
    archive = {g["id"]:g for g in json.loads(read_repo(ARCHIVE/"sources.json"))["games"]}
    rows = []
    totals = Counter()
    for game in fixed:
        gid=game["id"]
        data=(corpus/(gid+".json.gz")).read_bytes()
        if sha256(data).hexdigest() != archive[gid]["packed_sha256"]:
            raise ValueError(gid)
        steps=json.loads(gzip.decompress(data))["steps"]
        local=Counter()
        for seat in (0,1):
            mod._FEAT=None
            for t,frame in enumerate(steps[:-1]):
                obs=dict(step=t,player=seat,players=frame[0]["observation"]["players"],
                         carbon=frame[seat]["observation"]["carbon"])
                if t==0:
                    mod.agent(obs)
                    continue
                x=mod._FEAT.observe(obs)
                # Economically impossible roots have no opportunity to alter an action.
                feat=mod._FEAT
                possible=(x["tok_mask"][:10] & (x["tok_kind"][:10]==0) & (x["tok_cargo"][:10]<=75) &
                          (feat.aux["est"][x["tok_pos"][:10]]>=.55))
                if t%3 or not possible.any() or x["left"]<18 or x["cash"]<60:
                    continue
                raw=mod._mixed_actions(x,feat.aux)
                local["examined_neural_roots"]+=1
                if not mod._FREIGHT_LAST:
                    continue
                ys=feat.labels(steps[t+1][seat].get("action") or {})
                for s,p,c in mod._FREIGHT_LAST:
                    assert int(raw[s])==c==0 and p!=0
                    uid=feat.own_uid[s]
                    expert=int(ys[s])
                    response="candidate" if expert==c else "parent" if expert==p else "neither"
                    local[response]+=1; local["changed"]+=1
                    rows.append(dict(id=gid,seat=seat,t=t,uid=uid,parent=p,expert=expert,
                                     source_d=int(feat.aux["dob"][x["tok_pos"][s]]),
                                     parent_d=int(feat.aux["dob"][mod.sys.modules["cnet2_feat"].DEST[x["tok_pos"][s],p]]),
                                     load=float(x["tok_cargo"][s]),resource=float(feat.aux["est"][x["tok_pos"][s]])*100))
        print(gid,local,flush=True)
        totals.update(local)
    audit.shared.atomic_json(out,dict(scope="Fixed first ten frozen101 Top5 games, both seats, every third nonterminal root. Disjoint from later7 discovery; overlaps source training corpus. Agreement cannot measure gain in game outcomes.",
            transfer_criterion="At least 20 direct actor overrides; more outright candidate matches than both all other actions combined on changed slots.",
            candidate_evidence_sufficient=totals["changed"]>=20 and totals["candidate"]>totals["parent"]+totals["neither"],
            source_sha256={n:audit.digest(HERE/n) for n in ("quarry_value.py","quarry_runtime.py")},
            inputs={g["id"]:g["canonical_sha256"] for g in fixed},totals=totals,changes=rows))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("corpus",type=Path);p.add_argument("out",type=Path)
    args=p.parse_args();main(args.corpus,args.out)
