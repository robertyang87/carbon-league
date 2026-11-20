"""Blind-to-branch registration of two already-examined, deployment-domain positions."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
HOSTED=D/"cold_start/development/hosted"
def sha(x):return hashlib.sha256(x).hexdigest()
def token(x):return sha(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
def build():
    book= json.loads((HOSTED/"settlement.json").read_bytes())
    assert book["records_reconciled"]
    ranked=json.loads((HERE.parent/"protocol.json").read_bytes())
    fit=json.loads((D/"terminal_value_pairs/protocol.json").read_bytes())
    g=sorted([x for x in book["games"] if x["seed"]==621758591],key=lambda x:x["seat"])
    assert len(g)==2 and [x["seat"] for x in g]==[0,1]
    tickets=[]
    for game in g:
        seat=game["seat"]; t=248
        blob=HOSTED/game["archive_path"]
        with zipfile.ZipFile(blob) as z:
            prefix=game["arm"]+"/"
            rb=z.read(prefix+"receipt.json"); turn=z.read(prefix+"turns.jsonl"); views=z.read(prefix+"observations.jsonl.gz")
        assert sha(rb)==game["receipt_sha256"] and sha(views)==game["public_observations_sha256"] and sha(turn)==json.loads(rb)["turns_sha256"]
        src=next(a for a in book["archives"] if a["id"]==game["archive_id"])
        assert sha(blob.read_bytes())==src["sha256"] and game["status"]=="complete" and game["official_transitions"]>t
        frames=gzip.decompress(views).splitlines(); frame=json.loads(frames[t]);obs=frame["observations"][seat]
        rng=int.from_bytes(hashlib.sha256(f"v2-action-domain-20261120/{game['terrain']}/{seat}/{t}".encode()).digest()[:4],"big")
        tickets.append(dict(seed=game["seed"],terrain=game["terrain"],seat=seat,step=t,
                        archive=src["sha256"],path=game["archive_path"],member=game["arm"],public_sha256=frame["public_observation_sha256"],
                        pre_root_orders_sha256=token([json.loads(l)["calls"][seat]["response"]["action"] for l in turn.splitlines()[:t]]),
                        public_income_lead=round(float(obs["players"][seat][0])-float(obs["players"][1-seat][0]),2),
                        policy_rng=rng,arm_order=["first","second"] if seat==0 else ["second","first"]))
    assert tickets[1]["public_income_lead"]==11.59 and tickets[0]["terrain"]==tickets[1]["terrain"]
    assert all(tickets[0]["terrain"]!=r["terrain"] for r in ranked["tickets"])
    return dict(schema="v2-action-ranking-retrospective-domain-check/1",source_settlement=sha((HOSTED/"settlement.json").read_bytes()),
        parent=ranked["base"],opponent=ranked["examiner"],paired_acquisition=sha((HERE.parent/"protocol.json").read_bytes()),
        learning_protocol=sha((D/"terminal_value_pairs/protocol.json").read_bytes()),physical_features=sha((D/"terminal_value_pairs/physical.py").read_bytes()),
        selection_reason="Retrospective selection. The previously exposed losing V2 seat 1 was ahead by 11.59 in public cash at t248, before divergence. Include the same terrain, seat 0, at that same step. Both original game outcomes already known. No fit, treatment tuning, weight selection, new maps, or replacement. This set represents one physical map.",
        selection_rule=ranked["selection"],features=fit["feature"],labels=ranked["labels"],tickets=tickets,
        runtime="One fixed CPU 3. During conditioning replay both signed public histories and recorded orders to t248. Reconstruct own private history with archived executed orders. Initialize independent role RNGs from registered policy_rng (+37*physical seat, uint32); simulate consumed root-policy draws. Original intended seed None and temperature 1, original wall deadline .58s. Calibrate local costs with first 3 public calls; true historic clocks and RNG unavailable. Trace verified longest depth8 score, complete and matching roll set. Exact ties retained, <2 distinct orders => skip with receipt. Both branches start from identical post-call native costs/RNGs and the single shared native opponent root command. Install learner-selected arm as actual private tracker intent. After one changed root, natural V2/true-opponent continuation, all actions and clocks, official score incl extinction. No repeat.",
        blind="Signed root captures separate from two-arm sealed outcomes. Publish a digest of each outcome without its numbers to the fitter, and keep outcomes unopened by predictor until the fitted weights or explicit final refusal to fit are frozen. Then report same-sign ranking by seat and signed prediction errors including ties, with no filtering. No promotion, no strength claims.")
if __name__=="__main__":
    p=HERE/"protocol.json";raw=(json.dumps(build(),sort_keys=True,indent=2)+"\n").encode()
    if p.exists() and p.read_bytes()!=raw: raise ValueError("Registered roots differ")
    if not p.exists():p.write_bytes(raw)
    print(sha(raw))
