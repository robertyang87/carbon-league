"""Audit locked physical commands. Counterfactuals remain unlabelled unless exact archived arms match."""
from pathlib import Path
import gzip,hashlib,importlib.util,json,sys,zipfile
from order import Anchor
from labels import bind_terminal_returns

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
spec=importlib.util.spec_from_file_location("old_tree_verify",D/"tree_control_response/development/verify.py")
tree=importlib.util.module_from_spec(spec);spec.loader.exec_module(tree);pair=tree.pair
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix==".gz" else p.read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
def check(ok,why):
    if not ok:raise ValueError(why)
def hosted(p,prefix):
    with zipfile.ZipFile(p) as z:
        reg=json.loads(z.read(prefix+"registration.json"));rc=json.loads(z.read(prefix+"receipt.json"))
        cmd=z.read(prefix+"turns.jsonl");src=z.read(prefix+"observations.jsonl.gz")
    check(sha(cmd)==rc["turns_sha256"],"Changed action tape")
    return list(map(json.loads,cmd.splitlines())),list(map(json.loads,gzip.decompress(src).splitlines())),reg

def main():
    cfg=pair.base.econ.config();panel=read(D/"tree_control_response/development/results.json.gz")
    explore=D/"tree_control_response/development/exploratory";plan=read(explore/"protocol.json")
    res=dict(schema="isolated-parent-issued-planter-actions/1",method_sha256=sha((HERE/"order.py").read_bytes()),indirect_cases=[],confirmed_labels=[],N34=[],real_collision_witnesses={})
    def witness(anchor,archive):
        for entry in anchor.candidates().values():
            for reason in ("allied_destination","active_recruit_destination"):
                if reason in entry["reasons"] and reason not in res["real_collision_witnesses"]:
                    res["real_collision_witnesses"][reason]=dict(source=archive,root=anchor.receipt(),rejected=entry)
    for g in plan["games"]:
        n=g["ordinal"];section=panel["rows"][n];ticket=section["ticket"];s=g["seat"]
        frames,_=pair.tape(ticket);obs=json.loads(frames[g["step"]])["observations"][s]
        check(pair.base.digest(obs)==ticket["public_observation_sha256"],"Changed public state")
        cases={}
        for tag,name in (("P","parent"),("E","new")):
            a=g["arms"][tag]["order"]
            check(all(call["action"]==a for call in section["calls"] if call["mode"]==tag),"Mismatched native emitted command")
            branch=explore/f"root_{n}_{name}.json.gz";src=read(branch)
            check(src["rows"][0]["orders"][s]==a,"Mismatch at actually executed first move")
            cases[tag]=a
        pivot=ticket["preaction_opportunity"]["worker"];anchor=Anchor(obs,cases["P"],pivot,cfg);aud=anchor.audit(cases["E"])
        check(set(aud["reasons"])>={"focal_unchanged","other_entity_changed"},"Indirect change admitted")
        outcome=read(explore/f'{g["seed"]}_{s}_result.json')
        res["indirect_cases"].append(dict(map=g["seed"],seat=s,root=anchor.receipt(),excluded_candidate=aud,
            historical_whole_order_terminal_delta=outcome["terminal_margin_delta"],admissible_for_planter_cost_model=False))
    for end in (590,591):
        seed=621758000+end
        p=D/f"tree_value_labels/{seed}_root.json.gz";o=read(p);ticket=o["ticket"];s=ticket["seat"];t=ticket["step"]
        arc=D/f'cold_start/development/hosted/{ticket["archive"]}.zip';check(sha(arc.read_bytes())==ticket["archive_sha256"],"Different game")
        calls,frames,reg=hosted(arc,"cold_repair/");obs=frames[t]["observations"][s];actual=calls[t]["calls"][s]["response"]["action"]
        check(actual==o["local_root_calls"][s][0] and obs==o["public_root_by_seat"][s],"Historical action/private state")
        a=Anchor(obs,actual,ticket["uid"],cfg);keep=ticket["stay"] if actual==ticket["leave"] else ticket["leave"]
        aud=a.audit(keep);check(aud["accepted"],"True isolated intervention was rejected")
        settle=read(D/f"tree_value_labels/{seed}_result.json")
        branch_runs={}
        for kind in ("stay","leave"):
            tape=read(D/f"tree_value_labels/{seed}_{kind}.json.gz")
            check(tape["rows"][0]["actions"][s]==ticket[kind],"Wrong played physical branch");branch_runs[kind]=tape
        parent_arm="stay" if actual==ticket["stay"] else "leave"
        child_arm="leave" if parent_arm=="stay" else "stay"
        label=bind_terminal_returns(a,ticket,o["protocol"],branch_runs[parent_arm],branch_runs[child_arm])
        res["confirmed_labels"].append(dict(map=seed,root=a.receipt(),accepted=aud,receipt_sha256=sha(p.read_bytes()),
              cost_to_go_sample=label,delta_stay_minus_leave=settle["delta_stay_minus_leave"],actual_parent=ticket["historical"],
              terminal_gain_of_accepted_alternative=round(settle["delta_stay_minus_leave"]*(1 if keep==ticket["stay"] else -1),2)))
        for row,fr in zip(calls,frames):
            ob=fr["observations"][s];ac=row["calls"][s]["response"]["action"]
            for uid,w in ob["players"][s][2].items():
                if w[2]=="PLANTER" and len(res["real_collision_witnesses"])<2:witness(Anchor(ob,ac,uid,cfg),arc.name)
    gid="eec452f455f16f82bb432ce057daa25c";source=D/f"ladder/replays/{gid}-1.json.gz";g=read(source)
    sys.path.insert(0,str(D.parents[1]/"reports/sapling_1200k/all_top5"))
    from context import Game
    game=Game(g)
    for t,uid in ((137,"player-1-worker-22"),(137,"player-1-worker-24"),(142,"player-1-worker-24")):
        s=1;obs=game._obs(t,s);action=g["steps"][t+1][s]["action"];a=Anchor(obs,action,uid,g["configuration"])
        offer=a.receipt();check(any(r["accepted"] for r in offer["alternatives"].values()),"N34 has no physical alternative")
        res["N34"].append(dict(replay=gid,source_sha=sha(source.read_bytes()),root=offer,official_terminal_difference_for_alternative=None))
    check(set(res["real_collision_witnesses"])=={"allied_destination","active_recruit_destination"},"Missing real physical collision cases")
    dest=HERE/"evidence.json";out=(json.dumps(res,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
    dest.write_bytes(out)
    print('rejected:',[(x['map'],x['excluded_candidate']['changed']) for x in res['indirect_cases']])
    print('accepted:',[(x['map'],x['terminal_gain_of_accepted_alternative']) for x in res['confirmed_labels']])
    print('real_collision_witnesses:',[(k,v['root']['step'],v['rejected']['destination']) for k,v in res['real_collision_witnesses'].items()])
if __name__=="__main__":main()
