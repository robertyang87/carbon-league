"""Compare V2's native root enemy collector readout before and after two fitted offsets."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import types

os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
import runtime

HERE=Path(__file__).resolve().parent
PARENT=HERE.parents[1]/"cold_start/candidate.py"
PROTOCOL=HERE/"protocol.json"
PREREG="1e63d22f2337f6cbf485fb62ab1849c0d05083edec933b687471bfb843294f00"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(ok,msg):
    if not ok:
        raise ValueError(msg)


def encode(obj):
    # Keep original roster insertion order; the live sampler uses it to break ties.
    return (json.dumps(obj,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode()


def save(path,raw):
    if path.exists():
        check(path.read_bytes()==raw,"Output identity changed: "+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as f:f.write(raw)


def import_game(card,path,split):
    raw=path.read_bytes()
    if split=="train":
        check(sha(raw)==card["original_gzip_sha256"],"Changed first-day replay")
    original=json.loads(gzip.decompress(raw))
    frames=original["steps"]
    check(len(frames)==int(original["configuration"]["episodeSteps"])==300 and original["statuses"]==["DONE","DONE"],
          "Expected all 299 two-sided official turns")
    # Opening full terrain is used only for the metadata overlap test, never as a policy input.
    full=frames[0][0]["observation"]["full_carbon"]
    check(len(full)==225 and all(float(v).is_integer() and 0<=v<=255 for v in full),"Unexpected initial board")
    seat=card["seat"]
    stripped=[]
    for t,(before,after) in enumerate(zip(frames,frames[1:])):
        shared=before[0]["observation"]
        own=before[seat]["observation"]
        check(own["player"]==seat and shared["step"]==t and len(own["carbon"])==225,"Wrong seat or replay time")
        stripped.append(dict(t=t,players=shared["players"],own_carbon=own["carbon"],
                             issued_after=[a["action"] or {} for a in after]))
    return dict(match=card["id"],replay_id=original["id"],game=card["game"],seat=seat,opponent=card["opponent"],
                terrain=sha(bytes(map(int,full))),source_gzip_sha256=sha(raw),frames=stripped)


def model():
    code=PARENT.read_bytes()
    check(sha(code)==json.loads(PROTOCOL.read_bytes())["parent"],"Wrong live baseline")
    m=types.ModuleType("verified_deployed_policy")
    exec(compile(code,"deployed_v2.py","exec"),m.__dict__)
    check(m._TEMP==1 and m._S_M==m._S_MMAX==1,"Wrong natural rollout settings")
    return m


def predictors(game,m,net):
    seat=game["seat"];enemy=1-seat
    tracker=m.BeliefTracker(m._Bel,seat)
    previous=None
    rows=[]
    for frame in game["frames"]:
        t=frame["t"];players=frame["players"]
        est,_=tracker.step(players,t,frame["own_carbon"])
        xo=m._quant(m.encode(players,t,enemy,est,[1.]*m.N,previous))[None]
        lw,lb,_=net(xo.astype(np.float32),m._SID)
        lw,lb=m._opponent_root_readout(xo,lw,lb,players,enemy)
        positions,phi=runtime.features(players,enemy,m._OPP_STEPS,m.FEAT_DIST)
        units=[(uid,u[1]) for uid,u in players[enemy][2].items() if u[2]=="COLLECTOR"]
        seed=int(1000003*(t+1)+17*seat)%2**31+7
        # Only after pre-action prediction and features are fixed, obtain the same-step opposing targets.
        issued=frame["issued_after"]
        target=np.asarray([m.WIDX[issued[enemy].get(uid) or "NONE"] for uid,_ in units],dtype=np.intp)
        rows.append((t,enemy,players,units,positions,phi,lw,lb,target,seed,issued[enemy]))
        tracker.set_actions(players,issued[seat]);previous=players
    return rows


def fit(games):
    datasets=[]
    for rows in games:
        datasets.append((np.concatenate([r[6][0,r[4]] for r in rows]).astype(float),
                         np.concatenate([r[5] for r in rows]).astype(float),np.concatenate([r[8] for r in rows])))
    def objective(theta):
        loss=.04*(theta@theta)/2;grad=.04*theta
        for z,phi,y in datasets:
            scores=z+phi@theta
            weights=softmax(scores,axis=1)
            loss+=np.mean(logsumexp(scores,axis=1)-scores[np.arange(len(y)),y])/len(games)
            weights[np.arange(len(y)),y]-=1
            grad+=np.einsum("na,nak->k",weights,phi)/(len(y)*len(games))
        return loss,grad
    answer=minimize(objective,np.zeros(2),jac=True,method="L-BFGS-B",bounds=[(-1.5,1.5)]*2,
                    options={"gtol":1e-9,"ftol":1e-12,"maxiter":100})
    check(answer.success and np.isfinite(answer.x).all(),str(answer.message))
    return answer.x.astype(np.float32)


def score(rows,m,theta):
    groups={name:dict(n=0,nll_sum_v2=0.,nll_sum_adjusted=0.,v2_correct=0,adjusted_correct=0,fixed=0,broken=0,
                      changed=0) for name in ("all","cargo_ge_200")}
    joint=[0,0]
    for t,enemy,players,units,pos,phi,lw,lb,y,seed,truth in rows:
        revised=runtime.adjust(lw.copy(),players,enemy,theta,m._OPP_STEPS,m.FEAT_DIST)
        check(np.array_equal(revised[0,pos],lw[0,pos]+phi@theta),"Training and serving logits diverge")
        candidates=[m._sample(players,enemy,z[0],lb[0],np.random.RandomState(seed),t) for z in (lw,revised)]
        joint=[v+int(candidate==truth) for v,candidate in zip(joint,candidates)]
        losses=[logsumexp(z[0,pos].astype(float),axis=1)-z[0,pos,y] for z in (lw,revised)]
        for i,(uid,cargo) in enumerate(units):
            target=truth.get(uid) or "NONE";a,b=[c.get(uid) or "NONE" for c in candidates]
            for name in ("all","cargo_ge_200") if cargo>=200 else ("all",):
                g=groups[name];g["n"]+=1;g["nll_sum_v2"]+=float(losses[0][i]);g["nll_sum_adjusted"]+=float(losses[1][i])
                g["v2_correct"]+=int(a==target);g["adjusted_correct"]+=int(b==target)
                g["fixed"]+=int(a!=target and b==target);g["broken"]+=int(a==target and b!=target);g["changed"]+=int(a!=b)
    for g in groups.values():
        n=g["n"];g["nll_v2"]=g.pop("nll_sum_v2")/n if n else None;g["nll_adjusted"]=g.pop("nll_sum_adjusted")/n if n else None
        g["net_correct"]=g["fixed"]-g["broken"]
    return dict(steps=len(rows),groups=groups,joint_orders_correct_v2=joint[0],joint_orders_correct_adjusted=joint[1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("train","held"))
    p.add_argument("replays",type=Path,nargs="*")
    a=p.parse_args();split=a.mode;plan=json.loads(PROTOCOL.read_bytes());check(sha(PROTOCOL.read_bytes())==PREREG,"Changed protocol")
    held=HERE/"held_tickets.json"
    cards=plan["train"] if split=="train" else json.loads(held.read_bytes())
    weight_file=HERE/"weights.json";program_sha=sha(Path(__file__).read_bytes());runtime_sha=sha((HERE/"runtime.py").read_bytes())
    if split=="held":
        frozen=json.loads(weight_file.read_bytes())
        check(frozen["protocol_sha256"]==PREREG and frozen["runtime_sha256"]==runtime_sha and
              frozen["program_sha256"]==program_sha and frozen["held_ticket_sha256"]==sha(held.read_bytes()),"Incomplete training freeze")
    check(len(a.replays) in (0,len(cards)),"One replay per registered map, in order")
    games=[]
    m=model();net=m._net()
    for idx,card in enumerate(cards):
        sanitized=HERE/split/(card["id"]+".json.gz")
        if a.replays:
            raw=a.replays[idx]
            check(card["id"] in raw.name,"Unbound match identifier in source input")
            game=import_game(card,raw,split)
            save(sanitized,gzip.compress(encode(game),mtime=0))
        else:
            game=json.loads(gzip.decompress(sanitized.read_bytes()))
        check(all(game[k]==card[k] for k in ("match","seat","opponent") if k!="match") and game["match"]==card["id"],"Wrong public history")
        games.append((game,predictors(game,m,net),sha(sanitized.read_bytes())))
    terrains={game["terrain"] for game,_,_ in games};check(len(terrains)==2,"Correlated terrain in panel")
    if split=="train":
        theta=fit([r for _,r,_ in games])
    else:
        check(not terrains.intersection(frozen["training_terrains"]),"Contaminated holdout")
        theta=np.array(frozen["coefficients"],dtype=np.float32)
    scores=[dict(match=game["match"],seat=game["seat"],opponent=game["opponent"],terrain=game["terrain"],
                 sanitized_sha256=digest,source_gzip_sha256=game["source_gzip_sha256"],**score(rows,m,theta)) for game,rows,digest in games]
    if split=="train":
        save(weight_file,encode(dict(parent=plan["parent"],protocol_sha256=PREREG,held_ticket_sha256=sha(held.read_bytes()),
             program_sha256=program_sha,runtime_sha256=runtime_sha,coefficients=theta.tolist(),dtype="float32",
             order=["collector_stay","cargo_scaled_home_progress"],training_terrains=sorted(terrains),
             training_histories=[g["sanitized_sha256"] for g in scores])))
    nll=[g["groups"]["all"]["nll_adjusted"]-g["groups"]["all"]["nll_v2"] for g in scores]
    net_correct=sum(g["groups"]["all"]["net_correct"] for g in scores)
    gates=dict(both_maps_noninferior=all(v<=1e-9 for v in nll),mean_nll_lower=sum(nll)<0,pooled_net_corrections_positive=net_correct>0)
    result=dict(split=split,parent=plan["parent"],weights_sha256=sha(weight_file.read_bytes()),protocol_sha256=PREREG,
                native_backend=type(net).__name__,games=scores,equal_game_mean_nll_delta=sum(nll)/2,
                net_corrections=net_correct,checks=gates,verdict="TRAIN" if split=="train" else "PASS" if all(gates.values()) else "FAIL")
    save(HERE/(split+"_result.json"),encode(result))
    print(result["verdict"],theta,gates,nll,net_correct)


if __name__=="__main__":main()
