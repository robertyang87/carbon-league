"""One additional locked prediction-only check after a newer archive arrived."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from pressure_return_train import HERE,ALL,OUT,sha,save,pressure_features,_PRESSURE_DIST,game_rows


def main(cache):
    model=json.loads((OUT/"model.json").read_bytes());coef=np.asarray(model["coefficients"])
    old=json.loads((OUT/"split.json").read_bytes())
    old_seeds=set(old["excluded_prior_26"])|{r["map"] for r in old["maps"]}
    path=ALL/"appendix_4bfdabf5/manifest.json";man=json.loads(path.read_bytes())
    rng=np.random.default_rng(531674);receipts=[];groups={}
    for g in man["games"]:
        if g["seed"] in old_seeds:continue
        old_seeds.add(g["seed"]);raw=(cache/g["path"]).read_bytes()
        if sha(raw)!=g["sha256"]:raise ValueError("appendix trajectory mismatch")
        env=json.loads(gzip.decompress(raw))
        if env["statuses"]!=["DONE","DONE"]:raise ValueError("unfinished replay")
        a=game_rows(0,env,{"lots":[],"enemy_base_refunds":[]}); obs=env["steps"][0][0]["observation"]
        hd=np.array([_PRESSURE_DIST[:,list(q[1].values())].min(axis=1) for q in obs["players"]])
        master=np.array([n=="claude-AlphaCarbon-Master6" for n in g["seat_names"]])[1-a["seat"]]
        receipts.append({"id":g["id"],"seed":g["seed"],"rows":len(a),"master6_victim_rows":int(master.sum())})
        for label,mask in (("all",np.ones(len(a),dtype=bool)),("master6_victim",master)):
            ids=np.flatnonzero(mask)
            if not len(ids):continue
            a2=a[rng.choice(ids,min(80,len(ids)),replace=False)]
            x=pressure_features(a2["hunter_cell"],a2["victim_cell"],a2["their_cargo"],a2["enemy_last_move"],
                                a2["enemy_toward_home"],a2["step"],hd[1-a2["seat"]]);y=a2["their_action"]
            item={"id":g["id"],"rows":len(y)}
            for method,z in (("fit",x@coef),("home_direction",x[:,:,1].astype(float)),
                             ("heading_and_home",.8*x[:,:,5]+.5*x[:,:,1])):
                z-=z.max(1,keepdims=True);prob=np.exp(z);prob/=prob.sum(1,keepdims=True)
                item[method+"_log_loss"]=float(-np.mean(np.log(prob[np.arange(len(y)),y])))
                item[method+"_top1"]=float((prob.argmax(1)==y).mean())
            groups.setdefault(label,[]).append(item)
    outputs={}
    for k,vals in groups.items():
        outputs[k]={"games":len(vals),"rows":sum(v["rows"] for v in vals),**{key:float(np.mean([v[key] for v in vals])) for key in
                    ("fit_log_loss","fit_top1","home_direction_log_loss","home_direction_top1",
                     "heading_and_home_log_loss","heading_and_home_top1")},"per_game":vals}
    save(OUT/"extra15_prediction.json",{"model_sha256":sha((OUT/"model.json").read_bytes()),
                                       "manifest_sha256":sha(path.read_bytes()),"receipts":receipts,"tests":outputs})
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--cache",type=Path,required=True);main(ap.parse_args().cache)
