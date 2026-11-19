"""Read back both branches and replay the issued policies, official returns and cash accounts."""
import json
from collections import Counter
import gzip
from run import HERE,CORPUS,CARGO,ROOT,OLD,PARENT,CHILD,BEGIN,END,ARMS,sha,digest,imp,source,env_for
from math import isclose


def one(case,protocol):
    legacy=imp("read_window",OLD/"run.py");k=legacy.kernel()
    proof=imp("check_native",OLD/"settle.py")
    econ=imp("cash_sources",CORPUS/"league_policy/natural_original/economy.py")
    folder=HERE/f"{case['seed']}_{case['seat']}";seat=case["seat"]
    common=k.R.load(folder/"shared.json.gz");trace=common["trace"];pfx=common["prefix"];fork=common["fork"]
    raw,_=source(case["seed"],seat)
    assert common["identity"]["protocol"]==sha(HERE/"protocol.json") and pfx[:BEGIN]==raw["actions"][:BEGIN]
    assert common["identity"]["parent"]==sha(PARENT) and common["identity"]["student"]==sha(CHILD)
    assert common["prefix_sha"]==digest(pfx)
    assert [r["t"] for r in trace]==list(range(len(trace))) and len(trace)==len(pfx)+(fork is not None)
    for t in range(BEGIN,len(trace)):
        r=trace[t]
        if fork is not None and t==fork["t"]:
            assert fork["enemy"]==r["proposed_enemy"] and fork["own"]==r["orders"] and r["orders"]["parent"]!=r["orders"]["student"]
        else:
            assert r["orders"]["parent"]==r["orders"]["student"] and pfx[t]==k.physical(r["orders"]["parent"],r["proposed_enemy"],seat)
    if fork:
        assert BEGIN<=fork["t"]<END and common["snapshot_sha"]==k.digest(gzip.decompress((folder/"original_root.pickle.gz").read_bytes()))
    commonclock=proof.clock_proof(common["clock"],0,len(trace))
    report={};tapes={}
    for arm in ARMS:
        path=folder/(arm+".json.gz");a=k.R.load(path)
        assert a["shared_sha"]==sha(folder/"shared.json.gz") and a["identity"]==common["identity"]
        assert a["arm"]==arm and digest(a["tail"])==a["tail_sha"]
        tape=pfx+a["tail"];tapes[arm]=tape
        assert len(tape)==a["transitions"] and len(a["tail"])==len(a["checks"])
        if fork:assert a["tail"][0][1-seat]==fork["enemy"]
        rt=proof.clock_proof(a["clock"],len(trace),len(tape))
        env=env_for(k,raw);sim=econ.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
        sim.cash=econ.CashEvents(sim.cash)
        assert digest(sim.carbon)==digest(list(env.states[0].observation.full_carbon)) or list(sim.carbon)==list(env.states[0].observation.full_carbon)
        models=legacy.models(k,dict(parent=PARENT,student=CHILD),"audit_"+arm)
        cfg=raw["game_configuration"]
        ledger={phase:[Counter(),Counter()] for phase in ("window","after","game")}
        modified=[];detail=[]
        for t,joint in enumerate(tape):
            obs=k.public(env,seat);op=k.public(env,1-seat)
            r=trace[t] if t<len(pfx) else a["checks"][t-len(pfx)]
            assert r["t"]==t and [digest(obs),digest(op)]==r["public"]
            if t<END:
                choices,_=legacy.commands(k,models,obs,cfg)
                assert choices==r["orders"]
                if t>=BEGIN and choices["parent"]!=choices["student"]:
                    modified.append(dict(t=t,units=proof.changed_units(choices["parent"],choices["student"])))
            else:
                act,_=k.own_call(models["parent"],obs,cfg);choices={"parent":act}
            assert joint[seat]==choices[arm if BEGIN<=t<END else "parent"]
            cash0=list(sim.cash);sim.cash.events.clear();env.step(joint);sim.step(*joint)
            values=econ.cash_flow(sim.cash.events)
            for side in (0,1):
                assert isclose(sum(values[side].values()),sim.cash[side]-cash0[side],abs_tol=1e-5)
                assert sim.cash[side]==env.states[0].observation.players[side][0]
                ledger["game"][side].update(values[side])
                if t>=BEGIN:ledger["window" if t<END else "after"][side].update(values[side])
            if t>=BEGIN:detail.append(dict(t=t,cash=list(sim.cash),events=list(sim.cash.events),
                                            trees=[len(sim.player_trees[s]) for s in (0,1)]))
        score,cash,status=k.score(env)
        assert score==a["official_scores"] and cash==a["physical_cash"] and status==a["official_status"]
        assert isclose(score[seat]-score[1-seat],a["margin"])
        ledger["game"][0]["terminal_adjustment"]=score[0]-cash[0];ledger["game"][1]["terminal_adjustment"]=score[1]-cash[1]
        for side in (0,1):assert isclose(80+sum(ledger["game"][side].values()),score[side],abs_tol=1e-5)
        out=folder/(arm+"_cash.json.gz");legacy.save(k,out,detail)
        report[arm]=dict(receipt_sha=sha(path),cash_detail_sha=sha(out),original_clock=rt,terminal_scores=score,
                         margin=a["margin"],by_physical_seat=ledger,distinct_free_orders=modified,
                         whole_game_sha=digest(tape),transitions=len(tape))
    pre=[t for t,(a,b) in enumerate(zip(tapes['parent'],tapes['student'])) if a[seat]!=b[seat]]
    en=[t for t,(a,b) in enumerate(zip(tapes['parent'],tapes['student'])) if a[1-seat]!=b[1-seat]]
    assert (pre[0] if pre else None)==(fork["t"] if fork else None) and (not en or pre and en[0]>pre[0])
    delta=report["student"]["margin"]-report["parent"]["margin"]
    return dict(seed=case["seed"],seat=seat,source_terrain=case["terrain"],shared_sha=sha(folder/"shared.json.gz"),
        shared_clock=commonclock,first_executed_own_difference=pre[:1],first_enemy_response_difference=en[:1],
        executed_own_difference_turns_in_window=[t for t in pre if BEGIN<=t<END],margin_delta=delta,
        improved=bool(pre and delta>0),arms=report)


if __name__=="__main__":
    p=json.loads((HERE/"protocol.json").read_bytes())
    results=[one(c,p) for c in p["cases"]]
    out=dict(protocol_sha=sha(HERE/"protocol.json"),cases=results,local_positive=all(r["improved"] for r in results))
    dest=HERE/"results.json";s=json.dumps(out,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)+"\n"
    if dest.exists() and dest.read_text()!=s:raise ValueError("previous ledger differs")
    dest.write_text(s)
    print([(r["first_executed_own_difference"],r["margin_delta"]) for r in results],out["local_positive"])
