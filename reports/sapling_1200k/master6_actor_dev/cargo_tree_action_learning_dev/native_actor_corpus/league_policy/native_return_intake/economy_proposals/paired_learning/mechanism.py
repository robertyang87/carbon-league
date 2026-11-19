"""Probe algebra on archival 460 step-12 features. Three artificial labels only."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import audit
import learn
import model


def fixture(ep, ticket, plan):
    seat=ep["seat"]
    pack=audit.LEAGUE/plan["behavior"]["package"]
    with audit.trajectory(ep,pack,plan["behavior"]["sha256"]) as (live,env,cfg,tape,actor):
        bank=[5.,5.]
        for raw in tape:
            t=env.states[0].observation.step
            obs=live.observe(env,seat,bank[seat])
            if t==12:
                recorded=next(r for r in ticket["rows"] if r["step"]==12)
                if audit.sha(live.encoded(obs))!=recorded["public_observation_sha256"]:
                    raise ValueError("archived feature root changed")
                cat=recorded["catalog"]
                cmds=[r["command"] for r in cat["orders"]]
                with audit.audit_menu.collect.attached(actor.modules):
                    x=actor.mod._FEAT.observe(obs)
                    phi=model.features(actor.mod,x,actor.mod._FEAT.aux,cmds)
                case={"options":[cmds[0],cmds[1]],"catalog":cat,"phi":phi,
                      "observation":obs,"files":{"archived_tape_sha256":ep["tape_sha256"]}}
                diff=phi[1]-phi[0]
                runs=[]
                for artificial in (1.,0.,-1.):
                    _,theta=learn.update(diff,artificial)
                    pred=learn.prediction(case,theta)
                    runs.append({"artificial_delta_u":artificial,
                                 "readout_norm":float(np.linalg.norm(theta)),
                                 "log_odds_change":pred["change_log_odds_B_over_A"],
                                 "A_and_B_ranks":[pred["new_ranks"][:2],pred["old_ranks"][:2]],
                                 "derivative_checks":learn.checks(diff,artificial)})
                first,zero,last=(r["log_odds_change"] for r in runs)
                if not first>0 or zero!=0 or last!=-first:
                    raise ValueError("wrong response to a reversed or tied synthetic label")
                return {"seed":ep["seed"],"seat":seat,"step":t,"public_sha256":recorded["public_observation_sha256"],
                        "full_menu_size":len(cmds),"runs":runs}
            actor.act(obs,cfg)
            line=json.loads(raw)
            env.step([c["response"]["action"] for c in line["calls"]])
            bank=line["overage_remaining_seconds"]
    raise ValueError("fixture root not found")


def main():
    plan,episodes=audit.inventory()
    raw=(HERE.parent/"roots.json.gz").read_bytes()
    data=json.loads(gzip.decompress(raw))
    rows=[]
    for ep in episodes:
        if ep["seed"]==621758460:
            ticket=next(t for t in data["tickets"] if (t["seed"],t["seat"])==(ep["seed"],ep["seat"]))
            rows.append(fixture(ep,ticket,plan))
    result={"protocol_sha256":learn.PLAN,"source_roots_sha256":learn.sha(raw),
            "meaning":"Synthetic reversal/zero validation on archived public native features only. No empirical label, fitted artifact, or match.",
            "source_code_sha256":{f:learn.sha((HERE/f).read_bytes()) for f in ("model.py","mechanism.py","learn.py")},
            "checks":rows}
    learn.seal(HERE/"mechanism.json",learn.encoded(result))
    return result


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
