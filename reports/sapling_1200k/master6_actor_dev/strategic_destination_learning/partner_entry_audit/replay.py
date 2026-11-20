"""Replay one fixed historical terrain, both seats; issue counterfactual commands only off-board."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
CARGO=HERE.parents[1]/"cargo_tree_action_learning_dev"
sys.path.insert(0,str(CARGO/"native_actor_corpus/league_policy"))
import collect

SEED=621758346
IT50=collect.ROOT/"code/packs/ppou2_m_it50.py"
STUDENT=CARGO/"candidate.py"
NAMES=("native","native_zero_sapling","native_guard_off","old_8da4","old_zero_sapling","old_zero_no_return")
PAIRS=(("native","native_zero_sapling"),("native","native_guard_off"),("native","old_8da4"),
       ("old_8da4","old_zero_sapling"),("old_zero_sapling","old_zero_no_return"),
       ("native_guard_off","old_zero_no_return"),("native","old_zero_sapling"))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def zero_inject(player,student):
    """Execute a real 96-d Sapling trunk and autoregressive decoder at zero coefficient."""
    policy=player.modules["cnet2_np"].Policy
    old=policy.act
    metrics={"root_forwards":0,"active_slots":0,"different_student_argmax":0}
    def act(self,x,aux,*args,**kwargs):
        net=student.mod._NETS[1]
        tokens,ctx=net.encode(x,student.mod._HYBRID["style"])
        hidden=net.lin(ctx,"h0")
        student_prefix=player.modules["cnet2_feat"].Prefix(x,aux)
        previous=8
        metrics["root_forwards"]+=1
        existing=kwargs.get("hook")
        def observe(slot,scores,prefix,action):
            nonlocal hidden,previous
            unit,move,legal=student_prefix.feats(slot)
            encoded=np.concatenate((tokens[slot],ctx,unit,move.reshape(-1)))
            hidden=net.gru(net.lin(encoded,"din")+net.w["aemb.weight"][previous],hidden)
            head=(net.lin(hidden,"whead") if slot<10 else
                  np.concatenate((net.lin(hidden,"bhead"),np.zeros(2,np.float32))))
            sapling=head+net.lin(move,"V")@net.lin(hidden,"U")
            if not np.isfinite(sapling).all():raise ValueError("nonfinite student readout")
            combined=scores+0.*sapling
            if not np.array_equal(scores,combined):raise ValueError("zero residual changed native scores")
            metrics["active_slots"]+=1
            metrics["different_student_argmax"]+=int(np.argmax(np.where(legal,sapling,-np.inf))!=scores.argmax())
            out=existing(slot,combined,prefix,action) if existing is not None else None
            effective=int(action if out is None else out)
            student_prefix.advance(slot,effective)
            previous=effective+(5 if slot>=10 else 0)
            return out
        return old(self,x,aux,*args,**dict(kwargs,hook=observe))
    policy.act=act
    return metrics


def run(seat):
    folder=CARGO/"direct_original/games/candidate"
    raw_file=folder/f"raw_{SEED}_{seat}.json.gz"
    row_file=folder/f"row_{SEED}_{seat}.json"
    clock_file=folder/f"wall_{SEED}_{seat}.json.gz"
    raw=json.loads(gzip.decompress(raw_file.read_bytes()));card=json.loads(row_file.read_bytes())
    wall=json.loads(gzip.decompress(clock_file.read_bytes()))
    if raw["pack_sha256"]!=sha(STUDENT) or wall["raw_sha256"]!=sha(raw_file) or card["raw_sha256"]!=sha(raw_file):
        raise ValueError("historical package or action tape mismatch")
    cfg=raw["game_configuration"]
    np.random.seed(SEED)
    env=collect.make("carbon",configuration={**cfg,"randomSeed":SEED})
    if hashlib.sha256(bytes(env.states[0].observation.full_carbon)).hexdigest()!=raw["initial_terrain_u8_sha256"]:
        raise ValueError("official opening differs")
    arms={k:collect.Player(IT50 if k.startswith("native") else STUDENT) for k in NAMES}
    arms["native_guard_off"].mod.CONF["GUARD"]=False
    for k in ("old_zero_sapling","old_zero_no_return"):
        arms[k].mod._HYBRID.update(expert_weight=1.,student_weight=0.)
    arms["old_zero_no_return"].mod._RETURN_THETA[:]=0
    metrics=zero_inject(arms["native_zero_sapling"],arms["old_8da4"])
    rows=[]
    for t,logged in enumerate(raw["actions"]):
        pub=collect.public(env,seat)
        if env.done or pub["step"]!=t:raise ValueError("tape not aligned")
        cmds={}
        for name in ("old_8da4",*NAMES[:3],*NAMES[4:]):
            cmds[name]=arms[name].act(pub,cfg)
        if cmds["old_8da4"]!=logged[seat] or cmds["native"]!=cmds["native_zero_sapling"]:
            raise ValueError(f"public-entry parity failed at {seat}/{t}: logged={logged[seat]} rebuilt={cmds['old_8da4']}, native={cmds['native']} zero={cmds['native_zero_sapling']} stats={arms['native_zero_sapling'].mod._AG.stats}")
        actual=wall["calls"][t]
        if actual["step"]!=t:raise ValueError("clock tape out of order")
        rows.append({"step":t,"public_cash":pub["players"][seat][0],"workers":len(pub["players"][seat][2]),
                     "orders":cmds,"historical_examiner":{"depth":actual["max_depth"],
                     "wall":actual["public_wall_seconds"],"cpu":actual["process_cpu_seconds"]}})
        env.step(logged)
    if not env.done or len(rows)!=299 or [s.reward for s in env.states]!=raw["official_score_by_seat"]:
        raise ValueError("failed complete archived scored trajectory")
    stats={}
    for a,b in PAIRS:
        key=f"{a}/{b}"
        changed=[];units={}
        for row in rows:
            ca,cb=(row["orders"][k] for k in (a,b))
            if ca==cb:continue
            changed.append(row["step"])
            for uid in set(ca)|set(cb):
                if ca.get(uid)==cb.get(uid):continue
                role="base" if "recrtCenter" in uid else "worker"
                units[role]=units.get(role,0)+1
        stats[key]={"differing_roots":len(changed),"steps":changed,"physical_slots":units}
    return {"seat":seat,"archive_sha256":sha(raw_file),"row_sha256":sha(row_file),"clock_sha256":sha(clock_file),
            "historical_scores":raw["official_score_by_seat"],"historical_agent_call_seconds_by_seat":card["call_seconds_by_seat"],
            "historical_cpu_affinity":card["cpu_affinity"],"historical_opponent_clock":wall["summary"],
            "zero_sapling_forward":metrics,"arm_replay_wall_seconds":{k:sum(v.wall) for k,v in arms.items()},
            "arm_replay_cpu_seconds":{k:float(v.cpu) for k,v in arms.items()},"pairwise":stats,"rows":rows}


if __name__=="__main__":
    plan=json.loads((HERE/"protocol.json").read_bytes())
    if sha(IT50)!=plan["package"]["it50"] or sha(STUDENT)!=plan["package"]["8da4"]:raise ValueError("source changed")
    data=[run(seat) for seat in (0,1)]
    rows=[g.pop("rows") for g in data]
    raw=gzip.compress(json.dumps(rows,sort_keys=True,separators=(",",":"),allow_nan=False).encode(),mtime=0)
    (HERE/"commands.json.gz").write_bytes(raw)
    summary={"games":data,"traces_sha256":hashlib.sha256(raw).hexdigest(),"protocol_sha256":sha(HERE/"protocol.json"),
             "code_sha256":sha(Path(__file__)),"new_games":0,"cpu_affinity":sorted(__import__("os").sched_getaffinity(0))}
    (HERE/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
