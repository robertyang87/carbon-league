"""Verify the three real empty collisions and one-command alternatives in Max-V8 vs Master6."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from context import Corpus
from defense import DEST, MOVE
from master6_denial_cases import forks


def main():
    gid="314b9dd54841f457e3ae04b53371b757-1"
    corpus=Corpus(manifest=HERE/"manifest.json")
    game=corpus.open(gid)
    contacts=[]
    for t in (90,91,92):
        ps=game._obs(t,0)["players"]
        next_workers=game._obs(t+1,0)["players"][0][2]
        joint=game.labels(t,0,["actions"])["joint_commands_from_t"]
        gone=[(u,w) for u,w in ps[0][2].items() if w[2]=="COLLECTOR" and u not in next_workers]
        if len(gone)!=1:raise ValueError("expected one lost collector on each tick")
        uid,(pos,cargo,_)=gone[0];dst=DEST[pos][MOVE[joint[0].get(uid)]]
        hits=[(u,w) for u,w in ps[1][2].items() if DEST[w[0]][MOVE[joint[1].get(u)]]==dst]
        if len(hits)!=1 or pos!=dst or cargo or hits[0][1][1] or hits[0][1][2]!="COLLECTOR":
            raise ValueError("both agents must be empty collectors arriving at the focal cell")
        hid,h=hits[0]
        contacts.append(dict(t=t,own_id=uid,own_pos=pos,own_cargo=cargo,own_move=joint[0].get(uid),
                             enemy_id=hid,enemy_pos=h[0],enemy_cargo=h[1],enemy_move=joint[1].get(hid),contact=dst))
    jt=game.labels(90,0,["actions"])["joint_commands_from_t"]
    base=next(u for u in game._obs(90,0)["players"][0][1] if jt[0].get(u)=="RECPLANTER")
    last=contacts[-1]
    base_forks=forks(game,90,0,contacts[0]["own_id"],contacts[0]["enemy_id"],
                     [(0,base,action) for action in ("STAY","RECCOLLECTOR")])
    escape=forks(game,92,0,last["own_id"],last["enemy_id"],
                 [(0,last["own_id"],action) for action in ("UP","RIGHT","DOWN","LEFT")]+[(1,last["enemy_id"],"STAY")])
    for t in range(93,game.turns+1):
        cash,_,workers,trees=game._obs(t,0,terminal=True)["players"][0]
        if cash!=3.28 or len(workers)!=2 or any(w[2]!="PLANTER" or w[1]!=0 for w in workers.values()) or trees:
            raise ValueError(f"reported period was interrupted at {t}")
    out={"id":gid,"replay_sha256":corpus.rows[gid]["replay_sha256"],"contacts":contacts,"t90_base":base,
         "base_forks":base_forks,"t92_forks":escape,"unchanged_from_t93_through":game.turns,
         "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__),HERE.parent/"master6_denial_cases.py",HERE.parent/"context.py")}}
    (HERE/"maxv8_turns.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
