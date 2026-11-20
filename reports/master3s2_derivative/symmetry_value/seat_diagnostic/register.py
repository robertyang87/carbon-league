"""Choose one root per signed trajectory, before either seat's value forward is computed."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/"opponent_policy/protocol.json"

def build():
    raw=SOURCE.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest!="16e77a290cfe1c22503981f34cff934aa3fccd6fc69b0ba41d9383e1e078e502":
        raise ValueError("Different underlying original-game registry")
    data=json.loads(raw)
    cards=data["games"]
    rows=[]
    panel={}
    ticks=[24,96,168,258]
    for split in ("train","val"):
        maps=sorted({r["terrain"] for r in cards if r["split"]==split})
        if len(maps)!={"train":24,"val":4}[split]:
            raise ValueError("Unexpected map partition")
        panel[split]=maps
        for i,terrain in enumerate(maps):
            seats=[(i+i//4)%2] if split=="train" else [0,1]
            for seat in seats:
                c,=[c for c in cards if c["terrain"]==terrain and c["expert_seat"]==seat]
                rows.append({**c,"step":ticks[i%4]})
    if set(panel["train"])&set(panel["val"]) or len(rows)!=32 or len({c["sha256"] for c in rows})!=32:
        raise ValueError("Training and held maps must be physically separate")
    return dict(schema="native-leaf-seat-consistency/1", registry_sha256=digest,
                examiner=data["source_master_sha256"], samples=rows,
                primary="At an actual reached, nonterminal official position, track the examiner's private carbon "
                        "from its public observations and realized previous orders. Supply those same beliefs to "
                        "both native 54-plane encoders with the search leaf's all-one mask. Previous players "
                        "and board are identical. Compute separate five global scalars per seat. No rotation.",
                offline_control="Repeat at the same reconstructed simulator state, replacing ONLY the shared "
                                "carbon vector with the official full carbon. This offline upper-information "
                                "diagnostic is unavailable to the deployed agent.",
                labels="One actual archived continuation to its verified official terminal per sampled game. "
                       "Target y=_terminal(own official score minus opponent official score). "
                       "Inspect native self v, minus other-seat v, and half their difference. No fitted heads, "
                       "no selection, and no interventions. These are observational labels.",
                summary="Equal map weights per split, averaging both held games within each map. For each "
                        "estimator report RMSE, MAE, signed error, exact sign accuracy (ties only match ties); "
                        "report own_v+opp_v discrepancy signed mean and mean absolute by split, per map and raw row. "
                        "Keep all registered roots and true early terminal/missing separately; no replacement. "
                        "Only 24 training and 4 held physical maps. No promotion criterion.")

if __name__=="__main__":
    path=HERE/"protocol.json"
    raw=(json.dumps(build(),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if path.exists() and path.read_bytes()!=raw:
        raise ValueError("Registration already differs")
    if not path.exists(): path.write_bytes(raw)
    print(hashlib.sha256(raw).hexdigest())
