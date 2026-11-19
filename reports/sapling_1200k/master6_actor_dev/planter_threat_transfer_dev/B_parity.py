"""Exercise both public package entry points on the exact registered old parents and ladder roots."""
from collections import Counter,defaultdict
import gzip
import json
from types import ModuleType

from pydantic_core import from_json

import build_B as build
import ladder_pairs as pairs
from registry import A,HERE,REPLAYS,read,sha,save
import risk

ROOT=HERE.parents[3]
EXCLUDED=frozenset(read(HERE/"parent_panel/panel.json")["blocked_new_pilot_seeds"])


def require(cond,msg):
    if not cond:
        raise ValueError(msg)


def module(source,name):
    mod=ModuleType(name)
    exec(compile(source,name,"exec"),vars(mod))
    return mod


def source_state(row):
    s=row["seat"]
    if "raw" in row:
        require(row["seed"] not in EXCLUDED,"fresh pilot is blocked before source I/O")
        p=ROOT/row["raw"]
        require(sha(p)==row["raw_sha256"],"different raw parent")
        tape=read(p)
        require(tape["candidate_seat"]==s and tape["seed"]==row["seed"] and
                tape["initial_terrain_u8_sha256"]==row["map"],"different parent terrain")
        cfg=tape["game_configuration"];sim=pairs.lots.FastCarbon(cfg).reset(seed=row["seed"])
        return sim,cfg,tape["actions"],"parent_original",row["seed"],row["map"]
    require(row["seed"] not in EXCLUDED,"fresh pilot seed in ladder panel")
    p=REPLAYS/(row["id"]+".json.gz")
    require(sha(p)==row["raw_sha256"],"different raw ladder game")
    tape=from_json(gzip.decompress(p.read_bytes()));frames=tape["steps"];init=frames[0][0]["observation"]
    require(build.sha(bytes(int(x) for x in init["full_carbon"]))==row["terrain"],"different ladder terrain")
    cfg=tape["configuration"];sim=pairs.lots.FastCarbon(cfg).load_observation(init)
    commands=[[p["action"] or {} for p in frame] for frame in frames[1:]]
    return sim,cfg,commands,"ladder",row["id"],row["terrain"]


def same_branch(actual,reference):
    return (actual["survived"]==reference["survived"] and actual["paid"]==reference["paid"]
            and abs(actual["expense"]-reference["expense"])<1e-6
            and all(abs(v-w)<1e-6 for v,w in zip(actual["post_cash"],reference["post_cash"],strict=True)))


def evaluate(row,lookup,a,b):
    sim,cfg,recorded,source,identity,terrain=source_state(row)
    s=row["seat"];expected={r["window"]:r["choice"] for r in lookup[source,identity,s]}
    require(set(expected)=={0,1},"missing pair of registered windows")
    emitted=set();audited=set();counts=Counter();results=[]
    run_sha=__import__("hashlib").sha256()
    for t,joint in enumerate(recorded[:236]):
        view=sim.obs(s)
        w=0 if 45<=t<=125 else 1 if 150<=t<=235 else None
        # Audit the first subsequent decision and the end of each used window to rule out reselection.
        must_sample=(t==0 or (w is not None and (w not in emitted or w not in audited or t in (125,235))))
        if must_sample:
            prior=0 if not t else len(a._PAY_TRACE)
            a_out=a.agent(view,cfg)
            b_out=b.agent({**view,"reveal":{"14":99999.},"full_carbon":[99999.]*225},cfg)
            require(a._PAY_TRACE==b._PAY_TRACE and a._PAY_USED==b._PAY_USED,("A and B selected different roots",source,identity,s,t))
            if len(a._PAY_TRACE)>prior:
                tr=a._PAY_TRACE[-1];probe=b._B_LAST; match=expected[w]
                require(tr["t"]==t and tr["window"]==w and w not in emitted and match is not None and
                        (match["t"],match["uid"],match["changed"],match["type"])==(t,tr["uid"],tr["changed"],tr["kind"]),
                        ("unexpected registered opportunity",source,identity,s,t))
                if source=="parent_original":
                    require(tr["base"]==joint[s], ("not the original parent trajectory",identity,s,t))
                require(probe["restore_parent"]==match["veto"] and
                        abs(probe["risk"]-match["death_prediction"])<1e-12 and
                        risk.category(probe["dc"],probe["dp"],probe["kind"])==
                        risk.category(match["enemy_collector_dist"],match["enemy_planter_dist"],match["type"]),
                        ("risk classification differs from independent table",identity,s,t))
                req=tr["base"] if match["veto"] else a_out
                require(req==b_out,("B joint differs from registered action",identity,s,t))
                root={}
                for name,actual,reference in (("A",a_out,match["branches"]["A"]),
                                               ("B",b_out,match["branches"]["parent" if match["veto"] else "A"])):
                    changed=[dict(v) for v in joint];changed[s]=actual
                    official=pairs.apply_branch(sim,s,tr["uid"],changed)
                    require(same_branch(official,reference),("official paired outcome disagrees",source,identity,s,t,name))
                    root[name]={"command":actual,"official":official}
                require({u for u in a_out.keys()|b_out.keys() if a_out.get(u)!=b_out.get(u)}
                        ==({tr["uid"]} if match["veto"] else set()),"nonfocal change")
                counts["selected"]+=1;counts["A_changed"]+=int(tr["changed"]);counts["B_veto"]+=int(match["veto"])
                counts["B_kept_A_changed"]+=int(tr["changed"] and not match["veto"])
                root.update(t=t,window=w,uid=tr["uid"],parent=tr["base"],B_trace=probe)
                results.append(root); emitted.add(w)
            else:
                require(a_out==b_out and b._B_LAST is None,("nonopportunity changed",source,identity,s,t))
                if w in emitted:
                    counts["post_selection_same_action_checks"]+=1;audited.add(w)
                counts["unchanged_other_calls"]+=1
            counts["entrypoint_calls_per_package"]+=1
            run_sha.update(json.dumps([t,a_out,b_out],sort_keys=True,separators=(",",":")).encode())
        else:
            a._FEAT.observe(view);b._FEAT.observe(view)
        sim.step(*joint)
    require(emitted=={w for w,r in expected.items() if r is not None},("missed original opportunity",source,identity,s))
    return {"source":source,"id":identity,"terrain":terrain,"seat":s,"raw_sha256":row["raw_sha256"],"counts":dict(counts),
            "no_opportunity_windows":2-len(emitted),"examined_commands_sha256":run_sha.hexdigest(),"roots":results}


def main():
    weights=read(HERE/"risk_weights.json");meta=read(HERE/"B_package.json")
    source=(HERE/"candidate_B.py").read_bytes();original=(A/"candidate.py").read_bytes()
    require(meta["B_sha256"]==build.sha(source) and meta["A_sha256"]==build.sha(original) and
            meta["risk_weights_sha256"]==sha(HERE/"risk_weights.json"),"package fingerprints differ")
    require(build.compose(weights)[0]==source,"B cannot regenerate from its source")
    table=read(HERE/"matched.json.gz")
    require(table["model_sha256"]==meta["risk_weights_sha256"],"offline table used different fitted weights")
    lookup=defaultdict(list)
    for r in table["rows"]: lookup[r["population"],r["id"],r["seat"]].append(r)
    rows=[*read(HERE/"parent_panel/panel.json")["games"],*read(HERE/"panel.json")["maps"]]
    require(len(rows)==66 and len(lookup)==66,"incomplete registered panel")
    a=module(original,"A_original_public_entry");b=module(source,"B_public_entry")
    allrows=[];cohorts=defaultdict(Counter)
    split_sha={"parent":sha(HERE/"parent_panel/panel.json"),"ladder":sha(HERE/"panel.json")}
    for row in rows:
        test=evaluate(row,lookup,a,b)
        allrows.append(test);cohorts[test["source"]].update(test["counts"])
    probes=[]
    for state in (0.,.5,1.):
        binary,receipt=build.compose({"risk":[state]*18})
        cm=module(binary,"B_boundary_fixture")
        tests=[cm._B_RISK.pick_parent(cm._B_WEIGHTS,changed,99,1,"seize") for changed in (False,True)]
        require(tests==[False,state>.5],"strict half threshold or unchanged-root control failed")
        probes.append({"risk":state,"restore":tests,"fixture_sha256":receipt["B_sha256"]})
    out={"pass":True,"B_sha256":meta["B_sha256"],"A_sha256":meta["A_sha256"],"weight_sha256":meta["risk_weights_sha256"],
         "offline_table_sha256":sha(HERE/"matched.json.gz"),"registered_panel_sha256":split_sha,
         "source_sha256":sha(HERE/"B_parity.py"),"new_games":0,"windows":len(table["rows"]),"map_seats":len(rows),
         "by_source":{k:dict(v) for k,v in cohorts.items()},"strict_boundary":probes,"games":allrows}
    path=HERE/"B_parity.json.gz"
    if path.exists():
        previous=read(path)
        require({k:v for k,v in previous.items() if k!="source_sha256"}==
                {k:v for k,v in out.items() if k!="source_sha256"},"replayed evidence differs")
        if previous["source_sha256"]!=out["source_sha256"]:
            payload=(json.dumps(out,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
            build.write(path,gzip.compress(payload,mtime=0))
    else:
        save(path,out)
    print(out["by_source"],out["B_sha256"])


if __name__=="__main__":
    main()
