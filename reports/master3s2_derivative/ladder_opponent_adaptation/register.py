"""Lock public game identities and geography before fitting or evaluating the opponent head."""
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent / "ladder"
BASE = LADDER / "prediction_feedback/package/candidate.py"
sha = lambda data: hashlib.sha256(data).hexdigest()


def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if str(p).endswith(".gz") else p.read_bytes())


def main():
    version = dict(v1="b99742a4a510a4de5dc99639b3c85604",v2="ec94599e5182be7d35eb14bc288625cd")
    cards = {}
    for ver, ident in version.items():
        for p in read(LADDER / f"post_v3_reconciliation/{ver}_pages.json.gz"):
            for m in p["data"]["matches"]:
                if m["state"] != "done":continue
                for g in m["result"]:
                    if g["game"]!=1:continue
                    seat = int(m["b"] == ident)
                    assert m[["a","b"][seat]]==ident
                    side=["a","b"][seat]
                    cards[m["id"]]=dict(match=m["id"],version=ver,seat=g["order"].index(side),official_score=g["scores"][side],opponent_id=m[["b","a"][seat]],opponent=m[["name_b","name_a"][seat]])
    paths={}
    for p in sorted(LADDER.rglob("*-1.json.gz")):
        match=p.name[:-10]
        if match in cards:
            paths.setdefault(match,p)
    rows=[]
    for mid,path in paths.items():
        j=read(path)
        if j["statuses"]!=["DONE","DONE"] or len(j["steps"])!=300:continue
        seat=cards[mid]["seat"]
        scores=j["rewards"]
        assert scores[seat]==cards[mid]["official_score"]
        excess=scores[seat]-scores[1-seat]
        terrain=sha(bytes(j["steps"][0][0]["observation"]["full_carbon"]))
        rows.append(cards[mid]|dict(path=str(path.relative_to(HERE.parent)),sha=sha(path.read_bytes()),terrain=terrain,margin=round(excess,2)))
    names={"安生N34","stephen-K4TacSearch-v6","doubao-flash-v1","ClaudeFable5.2-v7","claude-AlphaCarbon-Master3s2","哈基米 v3pro"}
    hold_ids={r["opponent_id"] for r in rows if r["opponent"] in names}
    assert len(hold_ids)>=4
    held=[r for r in rows if r["opponent_id"] in hold_ids]
    hold_terrains={r["terrain"] for r in held}
    fit=[r for r in rows if r["opponent_id"] not in hold_ids and r["terrain"] not in hold_terrains]
    assert len({r["terrain"] for r in rows})==len(rows)
    for r in held+fit:r["split"]="held" if r["opponent_id"] in hold_ids else "train"
    fixed=dict(parent=sha(BASE.read_bytes()),source="ds-pro pre-v3/post_v3_reconciliation official IDs, both physical seats, true DONE",
               steps=list(range(4,200,8)),procedure="frozen single fit, no hyperparameter grid or hold-driven selection. Each role normalized by source game. Smooth bounded contextual action residual plus role-specific logit scale normalized before softmax; equal role/map likelihood. Native opponent joint decoder same random seed. Alleged improvement only if overall held equal-role+map NLL and native top1 improve; report all cohorts and independent IDs. Physical 12 deterministic roots: first loss by match id of each held opponent, t44 and t108, per-method ABBA; same root, menus, work/depth, repeated real commands. No opponent/future order in prediction. Record exact opponent-root changes, wins separately & counterexamples. Risk groups: loaded collector >=100 and own-to-enemy collector/base <=3, or loaded collector nearing own base <=3; planter <=2 torus tiles from any tree; recruit-eligible base sequential legality scored separately.",
               fit_maps=len(fit),held_maps=len(held),held_ids=sorted(hold_ids),cards=sorted(fit+held,key=lambda x:x["match"]))
    raw=(json.dumps(fixed,indent=2,ensure_ascii=False)+'\n').encode()
    path=HERE/'protocol.json'
    if path.exists():assert path.read_bytes()==raw
    else:path.write_bytes(raw)
    print(sha(raw),len(fit),len(held),[(r['opponent'],r['margin']) for r in held])

if __name__=="__main__": main()
