import argparse,gzip,hashlib,importlib.util,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
from model import infer

HERE=Path(__file__).resolve().parent;D=HERE.parent
sys.path.insert(0,str(D/"opponent_rollout_calibration"))
sp=importlib.util.spec_from_file_location("prior_h2h_files",D/"opponent_rollout_calibration/study.py");old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):
    raw=p.read_bytes();return json.loads(gzip.decompress(raw) if p.suffix==".gz" else raw)
def save(p,obj):
    b=(json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(",",":"),sort_keys=True)+"\n").encode()
    p.write_bytes(gzip.compress(b,mtime=0) if p.suffix==".gz" else b)

def score_option(m,players,me,own_actual,enemy_actual,guess):
    enemy=1-me;worker=players[enemy][2];base=players[enemy][1];ours=players[me][2]
    our_dest={m.move(w[0],m.WIDX[own_actual.get(u,"NONE")]) for u,w in ours.items()}
    result=Counter(exact=int(guess==enemy_actual),roots=1)
    def rec(group,a,b):
        result[group+"_actual"]+=int(a);result[group+"_predicted"]+=int(b);result[group+"_true_positive"]+=int(a and b)
    for u,p in worker.items():
        a=enemy_actual.get(u,"NONE");b=guess.get(u,"NONE");spot,cargo,role=p
        dest_a=m.move(spot,m.WIDX[a]);dest_b=m.move(spot,m.WIDX[b])
        result["workers"]+=1;result["correct"]+=int(a==b)
        if cargo>=100 and role=="COLLECTOR":
            result["heavy"]+=1;result["heavy_correct"]+=int(a==b)
            rec("heavy_return", dest_a in base.values(), dest_b in base.values())
            rec("heavy_contact",dest_a in our_dest,dest_b in our_dest)
        rec("contact",dest_a in our_dest,dest_b in our_dest)
    result["bases"]=len(base);result["base_correct"]=sum(guess.get(u,"NONE")==enemy_actual.get(u,"NONE") for u in base)
    return dict(result)

def build_rows(cards):
    m=old.initialize();m._torch.set_num_threads(1)
    cfg=json.loads((D.parents[1]/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json").read_bytes())["configuration"]
    cfg={k:v["default"] for k,v in cfg.items() if isinstance(v,dict) and "default" in v};cfg["episodeSteps"]=300
    result=[]
    for ticket in cards:
        tape,_=old.read(ticket);s=ticket["seat"];tr=m.BeliefTracker(m._Bel,s);prev=None
        for t,(ob,commands) in enumerate(tape):
            estimate,_=tr.step(ob["players"],t,ob["carbon"])
            if t in range(8,297,6):
                # Current observed orders enter exclusively after the complete opponent menu and ratings have been returned.
                row=infer(m,ob,estimate,prev,cfg)
                metrics=[score_option(m,ob["players"],s,commands[s],commands[1-s],a) for a in row["menu"]]
                result.append(dict(row,seed=ticket["seed"],terrain=ticket["map"],seat=s,t=t,observed_opponent_order=commands[1-s],metrics=metrics))
            tr.set_actions(ob["players"],commands[s]);prev=ob["players"]
        print(ticket["seed"],s,len(result),flush=True)
    return result

def z_score(r,weight):
    prior=np.array(r["native_logit_sums"]);v=np.array(r["opponent_values"])
    return (prior-prior[0])/weight["prior_scale"],(v-v[0])/weight["value_scale"]
def choose(r,mode,w):
    if mode=="native":return 0
    p,v=z_score(r,w)
    if mode=="critic":return int(np.argmax(v))
    if mode=="prior":return int(np.argmax(p))
    if mode=="best_observed_in_menu":return int(np.argmax([a.get("correct",0) for a in r["metrics"]]))
    return int(np.argmax(p+float(mode)*v))
def stats(rs,mode,w):
    maps={};changed=0;k=Counter()
    for r in rs:
        selected=choose(r,mode,w);changed+=selected!=0
        metrics=r["metrics"][selected];maps.setdefault(r["terrain"],Counter()).update(metrics);k.update(metrics)
    return dict(totals=dict(k),changed=changed,worker_accuracy_map_mean=float(np.mean([v["correct"]/v["workers"] for v in maps.values()])),by_map={k:dict(v) for k,v in maps.items()})

def main(split):
    protocol=read(HERE/"protocol.json");cohort=read(D/"opponent_rollout_calibration/protocol.json");cards=[c for c in cohort["tickets"] if c["split"]==split]
    rows=build_rows(cards);out=HERE/(split+"_roots.json.gz")
    if out.exists():raise ValueError("Output already exists")
    save(out,rows)
    if split=="train":
        def scale(key):
            a=[v-r[key][0] for r in rows for v in r[key][1:]]
            return max(1.e-8,float(np.sqrt(np.mean(np.square(a)))))
        w=dict(prior_scale=scale("native_logit_sums"),value_scale=scale("opponent_values"),training_maps=sorted(set(r["terrain"] for r in rows)),
               source=sha(out.read_bytes()),candidate_weights=protocol["relative_critic_weights"])
        qualities={str(b):stats(rows,str(b),w)["worker_accuracy_map_mean"] for b in w["candidate_weights"]}
        w["beta"]=max(w["candidate_weights"],key=lambda b:qualities[str(b)]);w["training_scores"]=qualities
        save(HERE/"weights.json",w)
    else:
        w=read(HERE/"weights.json")
        assert not set(w["training_maps"])&set(r["terrain"] for r in rows)
    modes={k:stats(rows,v,w) for k,v in (("native","native"),("critic_only","critic"),("prior_only","prior"),("calibrated",str(w["beta"])),("hindsight_menu_upper_bound","best_observed_in_menu"))}
    timing={k:{lab:float(f) for lab,f in zip(("p50","p95","p99","max"),np.percentile([r["timings"][k] for r in rows],[50,95,99,100]))} for k in rows[0]["timings"]}
    doc=dict(cohort_sha=sha((D/"opponent_rollout_calibration/protocol.json").read_bytes()),method_sha=sha((HERE/"model.py").read_bytes()),
             weights_sha=sha((HERE/"weights.json").read_bytes()),raw_sha=sha(out.read_bytes()),rows=len(rows),histogram_K=dict(Counter(len(r["menu"]) for r in rows)),
             terminal_rows=sum(r["terminal_rows"] for r in rows),modes=modes,timings_seconds=timing)
    save(HERE/(split+".json"),doc)
    print(split,{k:(v['totals']['exact'],v['totals']['correct'],v['changed']) for k,v in modes.items()},w['beta'],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=["train","held"]);main(p.parse_args().split)
