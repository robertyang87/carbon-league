"""Compare native choices at fixed full H8 and an eight-order maximum with shared private draws."""
import copy,gzip,json,os,sys
from pathlib import Path
from probe import D,HERE,Game,COST,build,sha,module,read,need,ref

def patch(blob):
    a=blob.decode().replace("\r\n","\n")
    for old,new in (("    cands = [act0]; keys","    most = 8\n    cands = [act0]; keys"),
        ("        return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT","        return False")):
        need(a.count(old)==1,"Missing fixed cost anchor");a=a.replace(old,new)
    return a.encode()

def reduce(m,which):
    if which in ("geometry","risk_only"):
        m._TREE_CASH["means"]=[0.,0.];m._TREE_CASH["coef"]=[[0.,0.]]*len(m._TREE_CASH["coef"])
    if which in ("geometry","cash_only"):
        m._TREE_LIB["predict"]=lambda features,head:.5

def main():
    import torch
    os.environ["CARBON_DEBUG"]="1";torch.set_num_threads(1);torch.set_num_interop_threads(1)
    probe=read(HERE/"probe.json");need(len(probe["cases"])==5,"Missing input roots")
    os.sched_setaffinity(0,{3})
    src=(D/"cold_start/candidate.py").read_bytes();blob=build()
    mods={k:module(patch(src if k=="parent" else blob)) for k in ("parent","geometry","risk_only","cash_only","both")}
    for k,m in mods.items():m._net();reduce(m,k)
    parent=mods["parent"];out=[]
    for case in probe["cases"]:
        p=(D/"ladder/replays"/case["archive"]) if case["group"]!="held_top5" else Path("/workspace/sapling-all-top5-20261119/replays")/case["archive"]
        need(sha(p.read_bytes())==case["archive_sha256"],"Source switch");g=read(p);game=Game(g);tape=g["steps"];seat=case["seat"]
        cfg={k:v for k,v in g["configuration"].items() if k!="randomSeed"};tickets={r["t"]:r for r in case["roots"]};tracker=parent.BeliefTracker(parent._Bel,seat)
        prev=None;hist={}
        for t in range(max(tickets)+1):
            obs=game._obs(t,seat);ps=obs["players"]
            if t in tickets:
                ticket=tickets[t]
                row=dict(t=t,case=case["group"],seat=seat,terrain=case["terrain"],modes={})
                for mode,m in mods.items():
                    m._ST=dict(tr=copy.deepcopy(tracker),prev=copy.deepcopy(prev),rng=ref.np.random.RandomState(ticket["p_rng"]));m._S_COST[:]=COST
                    if mode!="parent":m._TREE_HIST={seat:dict(hist)};m._TREE_LAST=None
                    (a,clk),menu=ref.root_call(m,obs,cfg)
                    p=None if mode=="parent" else m._TREE_LAST
                    row["modes"][mode]=dict(order=a,menu=menu,K=clk["K"],H=clk["H"],aborted=clk["aborted"],
                        selected_worker=None if p is None else p["plan"]["worker"],d=None if p is None else p["plan"]["route_distance"],
                        selected_extras=[] if p is None else [k for k in p["new"] if p["plan"]["roots"][k]==a])
                row["same_full_work"]=all(q["K"]==8 and q["H"]==8 and len(q["menu"])==8 and not q["aborted"] for q in row["modes"].values())
                out.append(row)
            actual=tape[t+1][seat]["action"];tracker.step(ps,t,obs["carbon"]);tracker.set_actions(ps,actual);prev=ps;hist[t]=ps
            if t-5 in hist:del hist[t-5]
        print(case["group"],flush=True)
    result=dict(script_sha=sha(Path(__file__).read_bytes()),registered_roots_input=sha((HERE/"probe.json").read_bytes()),
        parent_sha=sha(src),candidate_sha=sha(blob),fixed_K=8,fixed_H=8,decision_rule="one fixed modeled opposing root action shared across all candidates, complete all native H8 work; same public roots, frozen model, history, private draw; diagnostic clock admission override",rows=out)
    (HERE/"ablation.json").write_text(json.dumps(result,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
if __name__=="__main__":main()
