"""Replay any completed root pair and recheck both public requests and both official scores."""
import gzip
import json
from pathlib import Path
from history import DERIVATIVE,HERE,advance,freight,setup,sha,views


def replay(path,card,ticket):
    raw=path.read_bytes()
    e=json.loads(gzip.decompress(raw))
    tape,acts,cfg,env,sim=setup(ticket,card)
    for moves in acts[:ticket["step"]]:
        advance(env,sim,moves)
    bank=[o["remainingOverageTime"] for o in views(env)]
    rows=e["rows"];s=ticket["seat"]
    for i,row in enumerate(rows,ticket["step"]):
        inputs=[freight.public(env,k,bank) for k in (0,1)]
        freight.check(i==row["t"] and [freight.digest(v) for v in inputs]==row["input_sha256"],"Unmatched public request")
        if i==ticket["step"]:
            freight.check(row["orders"][s]==e["choices"][e["arm"]] and row["orders"][1-s]==e["root_shared_original_order"],
                          "Wrong intervention or non-shared opponent reply")
        cash=advance(env,sim,row["orders"])
        flows=[dict(c) for c in freight.econ.cash_flow(sim.cash.events)]
        freight.check(cash==row["end_cash"] and flows==row["flow"],"Unmatched cash ledger")
        bank=[b-max(0,c["wall"]-1.) for b,c in zip(bank,row["clocks"])]
    scores=[st.reward for st in env.states]
    freight.check(env.done and scores==e["outcome"]["scores"] and e["outcome"]["margin"]==scores[s]-scores[1-s],"Wrong terminal score")
    return dict(file=path.name,sha256=sha(raw),transitions=len(rows),official_scores=scores,
                max_public_wall=max(c["wall"] for row in rows for c in row["clocks"]),
                official_margin=e["outcome"]["margin"])


if __name__=="__main__":
    import branches
    result=json.loads((HERE/"results.json").read_bytes())
    sel=json.loads((HERE/"selection.json").read_bytes())["selected"]
    cards=json.loads((DERIVATIVE/"opponent_policy/protocol.json").read_bytes())["games"]
    rows=[]
    for t in sel:
        card=next(c for c in cards if c["sha256"]==t["source_sha256"])
        for arm in ("parent","rotated"):
            rows.append(replay(HERE/f'{t["seed"]}_{t["seat"]}_{t["step"]}_{arm}.json.gz',card,t))
    branches.write("replay_check.json",dict(selected=len(sel),arms=rows,results_sha256=sha((HERE/"results.json").read_bytes())))
