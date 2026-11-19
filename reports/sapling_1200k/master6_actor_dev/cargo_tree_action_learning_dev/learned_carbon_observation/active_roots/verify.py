"""Verify frozen active-root proposals from an independent, public-entry official-prefix replay."""
import argparse
import gzip
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
CARBON = HERE.parent
ROOT = HERE.parents[5]
sys.path.insert(0, str(ROOT / "reports/sapling_1200k/root_recovery_branches"))
from branches import begin, public


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(obj):
    return sha(json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def load_pack(path, expected):
    assert sha(path.read_bytes()) == expected
    spec = importlib.util.spec_from_file_location("root_verify_"+expected[:12], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


MOVES = {"NONE":(0,0), "UP":(-1,0), "RIGHT":(0,1), "DOWN":(1,0), "LEFT":(0,-1)}
def land(pos, move):
    dr,dc=MOVES[move]
    return ((pos//15+dr)%15)*15+(pos%15+dc)%15


def distance(a,b):
    dr,dc=abs(a//15-b//15),abs(a%15-b%15)
    return min(dr,15-dr)+min(dc,15-dc)


def carbon_option(board, dest, bases):
    short=min(distance(dest,b) for b in bases)
    route=[p for p in range(225) if any(distance(dest,p)+distance(p,b)==short for b in bases)]
    ring=[land(dest,cmd) for cmd in ("UP","RIGHT","DOWN","LEFT")]
    return [float(board[dest]),float(board[ring].mean()),float(board[route].mean())]


def landings(own, orders):
    units={u:land(int(worker[0]),orders.get(u,"NONE")) for u,worker in own[2].items()}
    units.update({u:int(p) for u,p in own[1].items() if u in orders})
    return {tuple(sorted((a,b))) for a in units for b in units if a<b and units[a]==units[b]}


def replay_entry(path, expected, history, cfg):
    mod=load_pack(path,expected)
    for o in history[:-1]:
        mod.agent(o,cfg)
    F=sys.modules["cnet2_feat"]
    captured={}
    native=mod._mixed_actions
    def record(x,aux):
        captured["x"],captured["aux"]=x,aux
        return native(x,aux)
    mod._mixed_actions=record
    encode=mod._NETS[1].encode
    def capture_student(x,style):
        captured["encoded_map"]=x["maps"][0].copy()
        return encode(x,style)
    mod._NETS[1].encode=capture_student
    action=mod.agent(history[-1],cfg)
    mod._mixed_actions=native
    mod._NETS[1].encode=encode
    return mod,F,captured,action


def verify_prefix(mod,F,x,aux,orders,registered):
    labels=mod._FEAT.labels(orders)
    assert mod._FEAT.to_actions(labels)==orders
    pre=F.Prefix(x,aux)
    evidence=[]
    for slot in np.flatnonzero(x["tok_mask"][:F.OWN]):
        label=int(labels[slot]); _,_,legal=pre.feats(slot)
        assert legal[label]
        ref=registered[len(evidence)]
        assert (int(slot),mod._FEAT.own_uid[slot],label)==(ref["slot"],ref["uid"],ref["actual"])
        assert legal.tolist()==ref["legal"]
        assert (float(pre.cash),int(pre.slots),int(pre.plants))==(ref["cash"],ref["remaining_slots"],ref["plants"])
        assert np.flatnonzero(pre.reserved).tolist()==ref["claimed"]
        before=(float(pre.cash),int(pre.slots))
        pre.advance(slot,label)
        if slot>=10 and label>0:
            assert before[0]>=30 and before[1]>0 and before==(float(pre.cash+30),int(pre.slots+1))
        evidence.append([int(slot),label,legal.astype(int).tolist(),before[0],before[1],
                         np.flatnonzero(pre.reserved).tolist(),float(pre.cash),int(pre.slots)])
    return evidence


def forced_decode(mod, x, aux, focal, commands):
    native=mod._return_decode
    text=inspect.getsource(native)
    marker='action = int(np.argmax(np.where(legal, combined, -np.inf)))'
    assert text.count(marker)==1
    forced=mod._FEAT.labels(commands)
    record={}
    def choose(slot,branch,legal,logits):
        action=int(forced[slot] if slot<=focal else np.argmax(np.where(legal, logits,-np.inf)))
        assert legal[action]
        if slot==focal:
            record[branch]=(np.array(legal),np.array(logits))
        return action
    ns={**mod.__dict__,"verify_choose":choose}
    exec(compile(text.replace(marker,"action = verify_choose(slot,apply_residual,legal,combined)"),
                 "<verify-native-weighted-decode>","exec"),ns)
    branches={}
    def traced(x,aux,enc,resid):
        labels=ns["_return_decode"](x,aux,enc,resid)
        branches[resid]=labels
        return labels
    mod._return_decode=traced
    try:
        selected=mod._mixed_actions(x,aux)
    finally:
        mod._return_decode=native
    branch=next(k for k,v in branches.items() if v is selected)
    legal,scores=record[branch]
    prob=np.exp(scores.astype(float)[legal]-max(scores[legal]))
    norm=np.zeros(len(legal))
    norm[legal]=prob/sum(prob)
    alternate=int(forced[focal])
    return (mod._FEAT.to_actions(selected),dict(residual=branch,canonical_label=alternate,
              p=float(norm[alternate]),rank=1+int(np.sum(scores[legal]>scores[alternate])),
              scores=scores.tolist(),legal=legal.astype(int).tolist(),probs=norm.tolist()))


def verify(case, protocol, signed, train, phase):
    seed,seat,t=case["seed"],case["seat"],case["t"]
    card,idx=train[seed]
    assert (seat,phase)==((0,"collector"),(1,"planter"),(1,"collector"),(0,"planter"))[idx%4]
    assert card["path"]==case["source"] and card["sha256"]==case["source_sha256"] and card["terrain"]==case["terrain"]
    assert t in range(4,97,4) and (case["terrain"],seat,t) not in signed
    path=ROOT/card["path"]
    assert sha(path.read_bytes())==card["sha256"]
    raw=read(path)
    prefix=raw["actions"][:t]
    assert digest(prefix)==case["prefix_sha256"]
    env=begin(raw)
    history=[]
    for step in range(t+1):
        obs=public(env,seat)
        assert set(obs)=={"step","player","players","trees","carbon","remainingOverageTime"} and obs["step"]==step
        # Detach the screen before this referee might mutate its own state.
        history.append(json.loads(json.dumps(obs)))
        if step<t:
            env.step(prefix[step])
    public_hash=digest({k:v for k,v in history[-1].items() if k!="remainingOverageTime"})
    assert public_hash==case["public_obs_sha256"]
    assert case["root_enemy"]==raw["actions"][t][1-seat]
    assert raw["game_configuration"].get("randomSeed") is None
    p,_,cp,inc=replay_entry(CARBON.parent/"candidate.py",protocol["incumbent_sha256"],history,raw["game_configuration"])
    s,F,cs,free=replay_entry(CARBON/"candidate.py",protocol["carbon_student_sha256"],history,raw["game_configuration"])
    assert inc==case["incumbent"] and free==case["free_student"]
    assert np.array_equal(cp["x"]["maps"],cs["x"]["maps"])
    changed=cs["encoded_map"]!=cp["encoded_map"]
    mask=cs["x"]["maps"]
    eligible=(mask[2]==0) & ~(mask[14:18]>0).any(axis=0)
    assert not (changed & ~eligible).any()
    alt=case["alternative"]
    uid=case["focal_uid"]
    physical={u for u in set(inc)|set(alt) if inc.get(u,"NONE")!=alt.get(u,"NONE")}
    assert physical=={uid}
    own=history[-1]["players"][seat]
    assert own[2][uid][2].lower()==phase
    assert not (landings(own,alt)-landings(own,inc))
    d=[land(own[2][uid][0],a.get(uid,"NONE")) for a in (inc,alt)]
    assert list(case["physical_destinations"])==d
    board=[z["encoded_map"].reshape(15,15) for z in (cp,cs)]
    physical_boards=[(m[:,::-1] if seat else m).ravel().astype(float)*(100/255) for m in board]
    vectors=np.asarray([[carbon_option(m,dest,list(own[1].values())) for dest in d] for m in physical_boards])
    contrast=np.diff(vectors,axis=1).reshape(2,3)
    delta=contrast[1]-contrast[0]
    np.testing.assert_allclose(vectors[0],case["carbon"]["old"],atol=1e-6,rtol=0)
    np.testing.assert_allclose(vectors[1],case["carbon"]["student"],atol=1e-6,rtol=0)
    np.testing.assert_allclose(delta,case["carbon"]["contrast_change"],atol=1e-6,rtol=0)
    assert (max(abs(delta[0]),abs(delta[2])) if phase=="collector" else abs(delta[1]))>=1
    slot=s._FEAT.own_uid.index(uid)
    forced,prior=forced_decode(s,cs["x"],cs["aux"],slot,alt)
    assert forced==alt and prior["rank"]<=3 and prior["p"]>=.08
    assert prior["rank"]==case["prior"]["rank"]
    np.testing.assert_allclose(prior["scores"],case["prior"]["logits"],atol=1e-5,rtol=0)
    assert prior["legal"]==case["prior"]["legal"]
    assert prior["residual"]==(case["selected_paths"]["forced_student"]=="return")
    assert abs(prior["p"]-case["prior"]["p_alternative"])<1e-6
    victim_label=int(p._FEAT.labels(inc)[slot])
    assert abs(prior["probs"][victim_label]-case["prior"]["p_parent"])<1e-6
    checks=[verify_prefix(s,F,cs["x"],cs["aux"],a,case["budget_and_legality"][name])
            for a,name in ((inc,"incumbent"),(alt,"forced_student"))]
    cfg=raw["game_configuration"]
    alive=sum(age<cfg["treeLifespan"] for team in history[-1]["players"] for _,age in team[3].values())
    price=cfg["plantCost"]
    if len(own[3])>cfg["playerTreeProtectiveNumber"]:
        price+=cfg["plantCostInflationRatio"]*cfg["plantCostInflationBase"]**alive
    planter_budget=dict(public_cash=own[0],current_empty_ground_plant_cost=price,
                       seize_cost=cfg["seizeCost"],
                       focal_actions=[a.get(uid,"NONE") for a in (inc,alt)],
                       current_cash_covers_ground_price=own[0]>=price,
                       immediate_focal_plant_possible_by_action=[phase=="planter" and a.get(uid,"NONE")=="NONE" for a in (inc,alt)])

    return dict(seed=seed,seat=seat,t=t,role=phase,focal_uid=uid,public_sha256=public_hash,
                true_entry_prefix_calls_each=len(history),physical_destinations=d,contrast=delta.tolist(),
                public_economy=planter_budget,
                full_action_sha256=[digest(inc),digest(alt)],no_new_conflicts=True,legal_prefix_sha256=[digest(c) for c in checks],
                forced_final_native_prior=prior,previous_signed_root_overlap=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cpu",type=int,default=3)
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    protocol=read(HERE/"protocol.json")
    assert sha((CARBON/"protocol.json").read_bytes())==protocol["estimator_protocol_sha256"]
    train={c["seed"]:(c,i) for i,c in enumerate([c for c in read(CARBON/"protocol.json")["cards"] if c["split"]=="train"])}
    doc=read(HERE/"roots.json")
    assert sha((HERE/"protocol.json").read_bytes())==doc["protocol_sha256"]
    signed={(r["terrain"],r["seat"],r["t"]) for r in read(HERE.parents[2]/"joint_response_critic_dev/signed_roots.json.gz")["rows"]}
    seen=set()
    rows=[]
    for root in doc["cases"]:
        assert root["terrain"] not in seen
        seen.add(root["terrain"])
        phase=((0,"collector"),(1,"planter"),(1,"collector"),(0,"planter"))[train[root["seed"]][1]%4][1]
        rows.append(verify(root,protocol,signed,train,phase))
    assert len(rows)<=4
    out=dict(roots_sha256=sha((HERE/"roots.json").read_bytes()),protocol_sha256=doc["protocol_sha256"],
             verified_by="Official interpreter prefix, both real unforced public entries, actual frozen guarded decoder forced on the registered focal prefix; independent physical torus carbon and conflicts.",cases=rows)
    (HERE/"verification.json").write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("verified", len(rows), flush=True)

if __name__=="__main__":
    main()
