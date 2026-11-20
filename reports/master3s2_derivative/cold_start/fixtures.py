"""Build short public requests from registered, already-used training games."""
import gzip,hashlib,json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"opponent_policy"))
from original_policy import PROTOCOL,free_entry,sha,dump
def main():
    games=[c for c in PROTOCOL["games"] if c["split"]=="train"]
    maps=sorted({c["terrain"] for c in games})[:2]
    cards=[next(c for c in games if c["terrain"]==board and c["expert_seat"]==seat) for seat,board in enumerate(maps)]
    fixtures=[]
    for card in cards:
        root=85 if card["expert_seat"]==0 else 215
        history=[]
        for t,obs,cfg,order in free_entry.public_history(card):
            if t<=root:
                history.append(dict(observation=obs,wire=order))
        fixtures.append(dict(terrain=card["terrain"],seed=card["seed"],seat=card["expert_seat"],
                             source=card["sha256"],config={k:v for k,v in cfg.items() if k!="randomSeed"},history=history))
    dump(HERE/"requests.json.gz",fixtures)
if __name__=="__main__":main()
