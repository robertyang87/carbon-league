"""Reproduce all frozen actors on every actually played state and remove Sapling at the score source."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import responses as r


def run(arm,seat):
    plan=r.load(HERE/"protocol.json");seed=plan["seeds"][0];game=HERE/"games"/arm
    file=game/f"raw_{seed}_{seat}.json.gz";raw=r.load(file)
    wfile=game/f"wall_{seed}_{seat}.json.gz";wall=r.load(wfile)
    rec=r.load(game/f"receipt_{seed}_{seat}.json")
    assert not raw["errors"] and rec["official"]["statuses"]==["DONE","DONE"] and r.sha(file)==wall["raw_sha256"]==rec["game"]["raw_sha256"]
    assert all(r.sha(pack)==plan[name] for name,pack in r.PACK.items()) and raw["pack_sha256"]==plan[arm]
    models={name:r.live.module(p,"audit_"+name) for name,p in r.PACK.items()}
    models["zero_sapling"]=r.live.module(r.PACK["parent"],"audit_zero_sapling")
    z=models["zero_sapling"]
    assert z._HYBRID["student_weight"]==.25 and z._HYBRID["expert_weight"]==.75
    z._HYBRID=dict(z._HYBRID,student_weight=0.,expert_weight=1.)
    # Parent does not have the extra disagreement head. This arm also keeps the same
    # return decoder/guard. It can measure direct Sapling influence in the original.
    sim=r.FastCarbon(raw["game_configuration"]).reset(seed=seed)
    assert list(sim.carbon)==raw["initial_full_carbon"]
    phases=["early_0_89","mid_90_209","late_210_298"]
    metrics={k:{s:Counter() for s in phases+["all"]} for k in ("candidate_minus_parent","parent_minus_zero_sapling","candidate_minus_zero_sapling")}
    trace=[]
    for t,pair in enumerate(raw["actions"]):
        v=sim.obs(seat); cfg=raw["game_configuration"]
        cmds={name:r.live.clean(m.agent(v,cfg)) for name,m in models.items()}
        if cmds[arm]!=r.live.clean(pair[seat]):raise ValueError(f"actual serving mismatch {arm} seat {seat} t{t}")
        stage=phases[0 if t<90 else 1 if t<210 else 2]
        roles={k:"base" for k in v["players"][seat][1]}
        roles.update({k:"collector" if sim.workers[k].is_collector else "planter" for k in v["players"][seat][2]})
        trace.append(dict(t=t,role=roles,commands=cmds))
        for key,(left,right) in {"candidate_minus_parent":("candidate","parent"),"parent_minus_zero_sapling":("parent","zero_sapling"),"candidate_minus_zero_sapling":("candidate","zero_sapling")}.items():
            for s in (stage,"all"):
                out=metrics[key][s];out["roots"]+=1
                out["changed_joint_commands"]+=cmds[left]!=cmds[right]
                for k,role in roles.items():
                    out["present_"+role]+=1
                    out["changed_"+role]+=cmds[left].get(k,"NONE")!=cmds[right].get(k,"NONE")
        sim.step(*pair)
    scores=rec["official"]["official_reward_by_seat"]
    cash=r.cash_summary(raw,seat,0,raw["actions"],scores)
    calls=wall["calls"];ts=np.array([a["public_wall_seconds"] for a in calls])
    out=dict(seed=seed,actual_arm=arm,seat=seat,actual_score=scores,margin=scores[seat]-scores[1-seat],
             errors=raw["errors"],official_status=rec["official"]["statuses"],arm_sha256=raw["pack_sha256"],
             raw_sha256=r.sha(file),wall_sha256=r.sha(wfile),reproduced_steps=len(trace),cash=cash,metrics=metrics,
             original_examiner_calls=len(calls),original_native_clock=wall["summary"],
             original_public_seconds=dict(p50=float(np.median(ts)),p95=float(np.quantile(ts,.95)),max=float(ts.max())),
             conditioning="Every actual public observation along this arm's recorded path; independent actor history, own decoding prefix and unchanged return guard in all alternatives. Roles count existing units; full joint differences count steps.")
    r.write(HERE/f"audit_{arm}_{seat}.json",out)
    r.write(HERE/f"audit_{arm}_{seat}_actions.json.gz",trace)
    print(arm,seat,out["margin"],[k["all"]["changed_joint_commands"] for k in metrics.values()],flush=True)


if __name__=="__main__":run(sys.argv[1],int(sys.argv[2]))
