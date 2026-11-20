"""Check the new investment-only gate on named N34 roots and two older evaluation maps."""
import copy,json,gzip,importlib.util,os
from pathlib import Path
from probe import HERE,D,Game,ref,read,sha,module,COST,need,local_branches
from build_investment import build_investment
from ablation import patch

def test(path,seat,name,requested,new,old,fixed):
    raw=path.read_bytes();g=json.loads(gzip.decompress(raw));pub=Game(g);cfg=g["configuration"];tape=g["steps"]
    clean={k:v for k,v in cfg.items() if k!="randomSeed"}
    tracker=new.BeliefTracker(new._Bel,seat);hist={};prev=None;roots=[];gap=30;chosen=0
    init=tape[0][0]["observation"]["full_carbon"];ref.np.random.seed(cfg.get("randomSeed",0));env=ref.make("carbon",configuration=cfg);env.states[0].observation.full_carbon=copy.deepcopy(init)
    sim=ref.econ.FastCarbon(cfg).reset(full_carbon=init);sim.cash=ref.econ.CashEvents(sim.cash)
    for t in range(250):
        obs=pub._obs(t,seat);ps=obs["players"];need(env.states[0].observation.players==ps==sim.obs(0)["players"],"Lost physical history")
        eligibility=t in requested or not requested and t>=gap and chosen<2 and any(w[2]=="PLANTER" for w in ps[seat][2].values())
        if eligibility:
            rng=int(sha((path.name+"/"+str(t)+"/tree-v1").encode())[:8],16);hx={k:hist[t-k] for k in (3,5) if t-k in hist}
            def prepare(model):
                model._ST=dict(tr=copy.deepcopy(tracker),prev=copy.deepcopy(prev),rng=ref.np.random.RandomState(rng));model._S_COST[:]=COST
                model._TREE_HIST={seat:dict(hist)};model._TREE_LAST=None
            prepare(old);search=old._search;old._search=lambda ob,cf,est,ac,*args:ac
            try:raw_order=old.agent(obs,clean)
            finally:old._search=search
            trial=new._tree_assess(obs,clean,raw_order,hx)
            if t in requested or trial and trial.get("far_investment"):
                row=dict(t=t,private_rng=rng,plan=trial,base_order=raw_order);modes={}
                for key,m in (("old_menu_native_clock",old),("new_menu_native_clock",new),*fixed.items()):
                    prepare(m);(issued,clock),menu=ref.root_call(m,obs,clean)
                    choices=m._TREE_LAST
                    modes[key]=dict(K=clock["K"],H=clock["H"],aborted=clock["aborted"],issued=issued,menu=menu,wall=clock["wall"],
                        affected_options=choices["new"] if choices else [],selected_new=[] if not choices else [k for k in choices["new"] if choices["plan"]["roots"][k]==issued])
                row["modes"]=modes
                if trial and trial.get("far_investment"):
                    row["economic_branches"]=local_branches(env,sim,tape,t,seat,trial,new)
                roots.append(row);chosen+=1;gap=t+25
        cmd=[v["action"] for v in tape[t+1]];tracker.step(ps,t,obs["carbon"]);tracker.set_actions(ps,cmd[seat]);prev=ps;hist[t]=ps
        if t-5 in hist:del hist[t-5]
        sim.cash.events.clear();sim.step(*cmd);env.step(cmd,[])
    return dict(name=name,source=sha(raw),terrain=sha(bytes(init)),seat=seat,roots=roots)

def main():
    import torch
    os.environ["CARBON_DEBUG"]="1";torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from build import build
    old=build();new=build_investment();base=(D/"cold_start/candidate.py").read_bytes()
    mo,ne=module(old),module(new);fixed={k:module(patch(b)) for k,b in (("parent_fixed",base),("old_fixed",old),("new_fixed",new))}
    for m in (mo,ne,*fixed.values()):m._net()
    cases=[(D/"ladder/replays/eec452f455f16f82bb432ce057daa25c-1.json.gz",1,"N34_exposed",[137,142])]
    tickets=[c for c in read(D/"ladder/tree_payback/maps.json")["tickets"] if c["split"]=="held"][2:4]
    for i,c in enumerate(tickets):
        p=Path("/workspace/sapling-all-top5-20261119/replays")/(c["id"]+".json.gz")
        need(sha(p.read_bytes())==c["compressed_sha256"],"Incorrect archive");cases.append((p,i,"held_seen_by_payback_classifier",[]))
    rs=[]
    for p,s,k,t in cases:
        r=test(p,s,k,t,ne,mo,fixed);rs.append(r)
        print(k,[(v["t"],v["plan"] and v["plan"]["cell"],[(name,q["selected_new"]) for name,q in v["modes"].items()]) for v in r["roots"]],flush=True)
    doc=dict(source=sha(Path(__file__).read_bytes()),package=sha(new),ancestor=sha(old),cases=rs)
    (HERE/"investment_probe.json").write_text(json.dumps(doc,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
if __name__=="__main__":main()
