"""Independently check primary-capital and newly mined labels against official Board transitions and archive the public actor inputs."""
import gzip
import importlib.util
import json
from pathlib import Path
import numpy as np
from data import ACTOR, HERE, build, CACHE, OLD, dump, read, sha
from lots import measure
from pydantic_core import from_json
spec=importlib.util.spec_from_file_location("certified_lot_accountant",ACTOR/"lot_settlement_dev/lots.py")
cert=importlib.util.module_from_spec(spec);spec.loader.exec_module(cert)

def official_check(packet):
    g=packet["card"]; s=g["seat"];cfgraw=(build.CACHE/(g["id"]+".json.gz")).read_bytes()
    doc=from_json(gzip.decompress(cfgraw));cfg=doc["configuration"];frames=doc["steps"];T=len(frames)-1
    sim=cert.FastCarbon(cfg).reset(seed=g["seed"]);sim.load_observation(frames[0][0]["observation"])
    measured,_=measure(doc,s)
    computed={};payouts={}
    count=0;tested=0;err=0.;extreme=0
    for t in range(T):
        commands=[z["action"] or {} for z in frames[t+1]]
        if t<225:
            sim.step(*commands);continue
        pre=sim.full_observation()
        prior,trans,after=cert.one_step(sim,commands,s,True)
        if after["players"]!=frames[t+1][0]["observation"]["players"]:raise ValueError("official source differs")
        count+=1
        before_units=pre["players"][s][2];own_bases=list(pre["players"][s][1].values());settler=after["players"][s][2]
        old_tree={q for side in pre["players"] for q,_ in side[3].values()}
        oldbase={q for side in pre["players"] for q in side[1].values()}
        collisions={sim._move_nbr[cmd][pos] if cmd in sim._move_nbr else pos
            for who,owner,_,_,_ in sim.last_collisions for pos,_,kind in [pre["players"][owner][2].get(who,[-1,0,None])]
            for cmd in [commands[owner].get(who)] if pos>=0}
        for uid,(pos,cargo,kind) in before_units.items():
            if kind!="COLLECTOR":continue
            if uid in prior:
                flows,holder,fraction=trans[uid]
                direct=sum(prop for k,why,original,land,prop in flows if k=="own_direct" and why=="own_bank" and original==uid)
                computed[t,uid]=(direct,float(fraction) if holder==uid else 0.)
            if uid not in settler or commands[s].get(uid) not in (None,"NONE") or pos in old_tree|oldbase|collisions:continue
            n=sum(v[2]=="COLLECTOR" for v in after["players"][s][2].values())
            expected=pre["carbon"][pos]*max(0,cfg["initialCollectRate"]-n*cfg["collectDecreaseRate"])
            if abs(expected-measured[t,uid]["gross"])>1e-4:
                raise ValueError(("independent clean mining",g["id"],t,uid,expected,measured[t,uid]))
            tested+=1
    for t in reversed(range(225,T)):
        for (tt,u),(direct,f) in [(q,v) for q,v in computed.items() if q[0]==t]:
            payouts[t,u]=direct+f*payouts.get((t+1,u),0.)
            cargo=frames[t][0]["observation"]["players"][s][2][u][1]
            err=max(err,abs(payouts[t,u]*cargo-measured[t,u]["existing_direct"]))
            if err>1e-4:raise ValueError(("principal audit",g["id"],t,u,err))
    return {"id":g["id"],"physical_seat":s,"split":g["split"],"official_late_transitions":count,
            "direct_clean_new_mining_rule_checks":tested,"old_lot_cashflow_audits":len(payouts),"max_old_direct_error":err}

def main():
    packets=read(HERE/"settlements.json.gz")["games"]
    receipt={};chk=[]
    for part in ("train","held"):
        selected=[p for p in packets if p["card"]["split"]==part]
        for seat in (0,1):
            p=min((p for p in selected if p["card"]["seat"]==seat and p["sample"]),key=lambda z:z["card"]["id"])
            chk.append(official_check(p))
        arrays={}
        for p in selected:
            g=p["card"];file=(OLD if part=="train" else CACHE)/(g["id"]+".npz")
            with np.load(file) as z:
                for k in z.files:
                    if k in ("carbon","value"):continue
                    arrays.setdefault(k,[]).append(z[k])
        file=HERE/(part+"_inputs.npz")
        np.savez_compressed(file,**{k:np.concatenate(v) for k,v in arrays.items()})
        receipt[part]={"file":file.name,"sha256":sha(file.read_bytes()),"rows":sum(map(len,arrays["gid"])),
                       "fields":sorted(arrays),"recorded_labels":"y; Prefix uf/mf/legal for each slot use earlier recorded commands only"}
    dump(HERE/"label_audit.json",{"independent_official":chk,"public_input_receipts":receipt})
if __name__=="__main__":main()
