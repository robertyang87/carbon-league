"""Freeze map groups independently of outcome; derive signed public features before joining future labels."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from ladder_frequency import fetch, previous_manifests, packed_replay, LATEST
from context import Game
from head import features


def bucket(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest(),"big")%5


def run():
    manifest_paths=previous_manifests(LATEST)
    cards={g["id"]:g for p in manifest_paths for g in fetch.load(p)["games"]}
    terrain={r["id"]:r["initial_terrain_u8_sha256"] for p in manifest_paths for r in
             (json.loads(line) for line in gzip.decompress((p.parent/"validation_maps.jsonl.gz").read_bytes()).splitlines())}
    joined=[]
    records=[json.loads(line) for line in gzip.decompress((HERE.parent/"frequency_games.jsonl.gz").read_bytes()).splitlines()]
    if len(cards)!=len(terrain) or len(cards)!=1528:
        raise ValueError("frozen corpus incomplete")
    for g in records:
        if not g["cases"]: continue
        gid=g["id"];raw=packed_replay(cards[gid])
        if fetch.sha(raw)!=g["replay_sha256"]: raise ValueError("replay fingerprint")
        game=Game(json.loads(gzip.decompress(raw)))
        cfg={k:game._data["configuration"][k] for k in ("size","episodeSteps","treeLifespan","recCollectorCost","recCollectorIncreaseCost")}
        board=terrain[gid]
        protected=any(bucket("recovery-rival-v1|"+g["names"][1-x["attack_seat"]])==0 for x in g["cases"])
        split="held_opponent" if protected else ("held_map" if bucket("recovery-map-v1|"+board)==1 else "train")
        for c in g["cases"]:
            s,t=c["attack_seat"],c["pre_t"]
            pre=[game._obs(u,s) for u in range(max(0,t-10),t+1)]
            X=features(pre,s,cfg)
            # Neither source metadata nor the selected worker id is available to the feature extractor.
            future=[game._obs(u,s,terminal=True)["players"][1-s][2] for u in range(t+1,min(game.turns,t+30)+1)]
            y=None if len(future)<30 else int(not any(any(w[2]=="COLLECTOR" for w in roster.values()) for roster in future))
            if y is not None and y!=int(c["first_real_collector_after_t"] is None or c["first_real_collector_after_t"]>t+30):
                raise ValueError("independent raw-roster label mismatch")
            diff=g["scores"][s]-g["scores"][1-s]
            joined.append({"game":gid,"map":board,"raw_sha256":g["replay_sha256"],"pre_t":t,"seat":s,
                           "actor":g["names"][s],"rival":g["names"][1-s],"result":"W" if diff>0 else ("L" if diff<0 else "D"),
                           "split":split,"public":X,"label_30":y})
    train_names={r["rival"] for r in joined if r["split"]=="train" and r["label_30"] is not None}
    for r in joined:r["rival_seen_in_fit"]=r["rival"] in train_names
    seen=defaultdict(set)
    for r in joined:seen[r["map"]].add(r["split"])
    if len(joined)!=214 or sum(r["label_30"] is not None for r in joined)!=209 or any(len(v)!=1 for v in seen.values()):
        raise ValueError("incomplete or overlapping map split")
    if any(r["rival_seen_in_fit"] for r in joined if bucket("recovery-rival-v1|"+r["rival"])==0):
        raise ValueError("held registration in training")
    out=HERE/"events.jsonl.gz"
    fetch.atomic(out,gzip.compress(b"".join((json.dumps(r,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode() for r in joined),mtime=0))
    fit_seal={"protocol_sha256":fetch.sha((HERE/"protocol.json").read_bytes()),"extractor_sha256":fetch.sha((HERE/"head.py").read_bytes()),
              "source_sha256":fetch.sha((HERE.parent/"frequency_games.jsonl.gz").read_bytes()),"events_sha256":fetch.sha(out.read_bytes()),
              "fit_maps":sorted({r["map"] for r in joined if r["split"]=="train" and r["label_30"] is not None}),
              "test_maps":sorted({r["map"] for r in joined if r["split"]!="train" and r["label_30"] is not None})}
    fetch.dump(HERE/"split.json",fit_seal)


if __name__=="__main__":
    run()
