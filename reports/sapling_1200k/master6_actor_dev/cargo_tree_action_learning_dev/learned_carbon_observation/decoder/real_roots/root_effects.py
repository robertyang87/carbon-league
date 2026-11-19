"""Observe the executed intervention's immediate official effect only after the root and opponent move are sealed."""
import importlib.util
import numpy as np
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
CARBON=HERE.parents[1]
sys.path.insert(0,str(CARBON))
from common import ROOT,F,read,save,sha
sys.path.insert(0,str(CARBON/"active_roots"))
from select_roots import begin,public,digest


def all_assets(obs):
    p=obs["players"]
    return dict(cash=[s[0] for s in p],live_workers=[len(s[2]) for s in p],trees=[len(s[3]) for s in p],
                carried=[round(sum(w[1] for w in s[2].values()),6) for s in p])


def main():
    pick=importlib.util.spec_from_file_location("sealed_decoder_carbon_readout",HERE.parent/"candidate.py")
    policy=importlib.util.module_from_spec(pick)
    pick.loader.exec_module(policy)
    cases=[]
    for c in read(HERE/"roots.json")["cases"]:
        native=HERE/"native"/c["id"]
        receipt_path=native/"shared.json.gz"
        sh=read(receipt_path)
        states={}
        raw=read(ROOT/c["source"])
        uid=c["all_changed_uids"][0]
        for arm in ("parent","student"):
            env=begin(raw)
            feat=F.Feat(c["seat"])
            for joint in sh["prefix"]:
                feat.observe(public(env,c["seat"]))
                env.step(joint)
            s=c["seat"];t=c["t"];en=1-s
            o=public(env,s); x=feat.observe(o)
            assert digest({k:v for k,v in o.items() if k!="remainingOverageTime"})==c["public_obs_sha256"]
            physical=c["physical_changed_workers"][0]["destinations"][int(arm=="student")]
            full=env.states[0].observation.full_carbon
            a=all_assets(o);pre=o["players"][s][2][uid]
            _,new_aux=policy._carbon_views(x,feat.aux,feat,F,policy._CARBON_WEIGHTS)
            canonical=feat.cpos(physical)
            estimates=dict(encoder_only_native_carbon=float(feat.aux["est"][canonical]*100),
                           decoder_student_native_carbon=float(new_aux["est"][canonical]*100))
            resource=float(full[physical]); src_carbon=float(full[pre[0]])
            action=c[arm].get(uid,"NONE")
            pair=[{},{}];pair[s]=c[arm];pair[en]=sh["root"]["enemy"]
            env.step(pair)
            after=public(env,s); post=after["players"][s][2].get(uid)
            b=all_assets(after)
            states[arm]=dict(physical_action=action,destination=physical,observed_before=float(o["carbon"][physical]),
                    actually_visible_before=any((v[0]==physical if v[2]=="COLLECTOR" else physical in [int(z) for z in neighbor4(v[0])]) for v in o["players"][s][2].values()),
                    true_carbon_before_destination=resource,true_carbon_before_source=src_carbon,
                    true_carbon_after_destination=float(env.states[0].observation.full_carbon[physical]),
                    deployed_estimates=estimates,worker_before=pre,worker_after=post,immediate_personal_cargo_delta=None if post is None else post[1]-pre[1],
                    official_before=a,official_after=b)
        cases.append(dict(id=c["id"],seed=c["seed"],seat=c["seat"],t=c["t"],uid=uid,
            shared_sha=sha(receipt_path.read_bytes()),states=states))
    save(HERE/"root_effects.json",dict(method="Post-registration offline official one-step replay with the actual natural shared opponent command. Ground truth used only to explain scored transitions.",cases=cases))


def neighbor4(pos):
    r,c=divmod(pos,15)
    return (((r-1)%15)*15+c,r*15+(c+1)%15,((r+1)%15)*15+c,r*15+(c-1)%15)


if __name__=="__main__":main()
