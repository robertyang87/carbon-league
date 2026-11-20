"""Quantify estimate error only after deriving the two candidate inputs from the acting seat."""
import sys,time
import numpy as np
from visibility import D,HERE,sha,load,read,data,obs,report
from build import build

def run():
    mix=load(HERE/"candidate.py");p=load(D/"cold_start/candidate.py")
    cfg=read(HERE/"belief_protocol.json")
    assert build()==(HERE/"candidate.py").read_bytes() and sha((D/"ladder/ds-pro-v1-placement.json").read_bytes())==cfg["placement_sha256"]
    rows=[]
    for ordinal,c in enumerate(read(D/"ladder/ds-pro-v1-placement.json")["games"]):
        tape,h=data(c);s=c["candidate_seat"];tracks=[a.BeliefTracker(a._Bel,s) for a in (p,mix)]
        raw=[]
        for i,fr in enumerate(tape["steps"][:-1]):
            ob=obs(fr,s);steps=[];cost=[]
            for tr in tracks:
                start=time.perf_counter();z,known=tr.step(ob["players"],i,ob["carbon"]);cost.append(time.perf_counter()-start)
                steps.append((np.array(z),np.array(known,dtype=bool)))
            assert np.array_equal(steps[0][1],steps[1][1]);mask=steps[0][1]
            assert np.array_equal(steps[0][0][mask],steps[1][0][mask])
            if i==0:assert np.allclose(steps[0][0],steps[1][0],rtol=0,atol=2e-14)
            # Hidden board first referenced here, strictly after both forward predictions.
            truth=np.array(fr[0]["observation"]["full_carbon"]);v=[]
            for z,_ in steps:
                e=z-truth;v.extend([float(e@e),float(abs(e).sum()),float(e.sum())])
            raw.append(dict(t=i,unknown=int((~mask).sum()),
                  values=v,differences_over_1=int((abs(steps[0][0]-steps[1][0])>1).sum()),tracker_wall_ms=[1000*f for f in cost]))
            for tr in tracks:tr.set_actions(ob["players"],tape["steps"][i+1][s]["action"] or {})
        rows.append(dict(match=c["match_id"],seat=s,sha=h,rows=raw))
    report(HERE/"belief_rows.json.gz",rows)
    settle()

def settle():
    rows=read(HERE/"belief_rows.json.gz")
    phases=((0,49),(50,149),(150,298))
    cards=read(D/"ladder/ds-pro-v1-placement.json")["games"];groups=[]
    for ordinal,(g,c) in enumerate(zip(rows,cards,strict=True)):
        assert g["match"]==c["match_id"] and len(g["rows"])==299
        summary={}
        for k in phases:
            part=[r for r in g["rows"] if k[0]<=r["t"]<=k[1]];n=225*len(part)
            b=np.sum([r["values"] for r in part],axis=0)/n
            summary[str(k)]=dict(parent=dict(mse=float(b[0]),mae=float(b[1]),bias=float(b[2])),
                mixed=dict(mse=float(b[3]),mae=float(b[4]),bias=float(b[5])),unknown_fraction=sum(r["unknown"] for r in part)/n)
        groups.append(dict(ordinal=ordinal,match=g["match"],split="diagnostic" if ordinal<16 else "held",periods=summary))
    totals={}
    for split in ("diagnostic","held"):
        cohort=[r for r in groups if r["split"]==split];metric={}
        for k in phases:
            vs=[r["periods"][str(k)] for r in cohort]
            metric[str(k)]={"parent":{j:float(np.mean([q["parent"][j] for q in vs])) for j in ("mse","mae","bias")},
                            "mixed":{j:float(np.mean([q["mixed"][j] for q in vs])) for j in ("mse","mae","bias")},
                            "strict_mse_wins":sum(q["parent"]["mse"]>q["mixed"]["mse"] for q in vs)}
        totals[split]=metric
    held=totals["held"][str(phases[0])]
    costs=np.array([r["tracker_wall_ms"] for g in rows for r in g["rows"]])
    report(HERE/"belief_test.json",dict(candidate=sha(build()),maps=groups,by_split=totals,
         tracker_wall_ms={"percentiles":[50,95,99,100],"parent":np.percentile(costs[:,0],[50,95,99,100]).tolist(),"candidate":np.percentile(costs[:,1],[50,95,99,100]).tolist()},
         forecast_mse_gate=held["mixed"]["mse"]<held["parent"]["mse"] and held["strict_mse_wins"]>=9,
         changed_inputs_above_1=sum(r["differences_over_1"]>0 for g in rows for r in g["rows"])))
    print(totals,flush=True)
if __name__=="__main__":
    settle() if "--settle" in sys.argv else run()
