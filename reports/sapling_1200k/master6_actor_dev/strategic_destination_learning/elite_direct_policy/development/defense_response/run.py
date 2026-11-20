"""Run the two registered original-opponent continuations of the exposed defensive decision."""
import copy
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("public_defense_selector", HERE/"select.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)
n = selector.native
P = json.loads((HERE/"protocol.json").read_bytes())
FROZEN = "1031f78cff586896fd9c94adfbc2dedc3afdf1fbfc28f1eff58fa68db8a91690"
def read_worker(env, seat, uid):
    return copy.deepcopy(env.states[0].observation.players[seat][2].get(uid))

def check_team(module, obs, feat_before):
    F = module.modules["cnet2_feat"]
    x = feat_before.observe(obs)
    receipts = {}
    for arm in ("issued","alternative"):
        order = P["anchor_orders"][arm]
        y = feat_before.labels(order)
        n.need(feat_before.to_actions(y)==order, "Order cannot be encoded physically")
        prefix = F.Prefix(x,feat_before.aux)
        slots=[]
        for s,uid in enumerate(feat_before.own_uid):
            if uid is None:continue
            _,_,legal=prefix.feats(s)
            a=int(y[s]);n.need(legal[a],"Illegal conditional order")
            slots.append([uid,int(s),a,list(map(bool,legal))])
            prefix.advance(s,a)
        receipts[arm]=slots
    return receipts

def prepare():
    n.need(n.sha((HERE/"protocol.json").read_bytes())==FROZEN and selector.prepare()==P,
           "Registration or publicly computed choice changed")
    _, tape, _, _ = selector.read_source()
    source_bytes=subprocess.check_output(["git","-C",str(n.ROOT),"show",
        "06b939399dfcdd9ab913f634f4d282aed121ba7e:sources/master3s2/master3s2.py"])
    new = selector.ARENA.parent/"trained.py"
    n.need(n.sha(source_bytes)==P["original_sha256"] and n.sha(new.read_bytes())==P["student_sha256"],
           "Unexpected frozen player bytes")
    cfg=json.loads(gzip.decompress((n.CARGO/"direct_original/games/candidate/raw_621758346_1.json.gz").read_bytes()))["game_configuration"]
    cfg={**cfg,"agentTimeout":20,"actTimeout":1,"runTimeout":1500}
    public_cfg={k:v for k,v in cfg.items() if k!="randomSeed"}
    env=n.collect.begin(P["seed"],cfg)
    me=n.collect.Player(new)
    source_directory=tempfile.TemporaryDirectory(prefix="sapling-521-defense-")
    path=Path(source_directory.name)/"master3s2.py";path.write_bytes(source_bytes)
    opp=n.collect.Player(path)
    # Every live native snapshot verifies its module's file bytes, including when restoring successive arms.
    opp.source_directory=source_directory
    n.need(n.original_state.source_sha(opp.mod.__file__)==P["original_sha256"],"Native snapshot cannot verify its source")
    m=opp.mod
    n.need(m._TEMP==1 and m._SEED is None and m._SEARCH_OK and m._S_ABORT==.58,"Unexpected opponent runtime")
    bank=[5.,5.]; warm=[]; destinations=None
    root=P["root_step"]
    for t in range(root+1):
        obs=[n.public(env,s,bank) for s in (0,1)]
        hashes=[n.sha(n.encoded(o)) for o in obs]; row=tape[t]
        n.need(hashes==row["public_observation_sha256"] and all(c["status"]=="ok" for c in row["calls"]),
               "Official public inputs differ")
        if t: n.need(m._ST["tr"].last_dest==destinations,"Enemy detector retains a nonexecuted intent")
        if t==root:
            proof=check_team(me,obs[0],copy.deepcopy(me.mod._FEAT))
        own,mc=n.act(me,obs[0],public_cfg);smpl,oc=n.act(opp,obs[1],public_cfg,True)
        actual=[c["response"]["action"] for c in row["calls"]]
        n.need(own==actual[0] and m._ST["tr"].last_dest==n.targets(obs[1]["players"],smpl) and
               m._ST["prev"]==obs[1]["players"],"Warmed student/original path mismatch")
        m._ST["tr"].set_actions(obs[1]["players"],actual[1]);destinations=n.targets(obs[1]["players"],actual[1])
        n.need(m._ST["tr"].last_dest==destinations,"Unrecoverable opponent belief history")
        warm.append(dict(step=t,public_hashes=hashes,student_clock=mc,enemy_clock=oc,shadow_enemy_order=smpl,
                         used_archived_enemy_order=actual[1],corrected=smpl!=actual[1]))
        if t<root:
            env.step(actual);bank=row["overage_remaining_seconds"]
    c=row["calls"][1];r=c["response"]
    rclock=dict(wall=c["elapsed"],cpu=r["public_cpu_seconds"],K=r["search_last"]["candidates"],
                H=r["search_last"]["depth"],kind="one shared previously issued hosted simultaneous response")
    n.need(hashes==P["root_public_sha256"] and actual==[P["anchor_orders"]["issued"],P["anchor_orders"]["enemy"]],
           "Different registered root")
    n.save(HERE/"prefix.json.gz",dict(protocol=FROZEN,orig_sha=P["original_sha256"],host=socket.gethostname(),
           affinity=sorted(os.sched_getaffinity(0)),torch_threads=m._torch.get_num_threads(),rows=warm,full_prefix_legal=proof))
    return env,bank,me,opp,public_cfg,rclock,mc

def play():
    os.environ["CARBON_DEBUG"]="1"
    n.need(os.sched_getaffinity(0)=={0,1},"Diagnostic reserved CPUs 0,1")
    out=HERE/"tickets";out.mkdir(exist_ok=True)
    n.need(not list(out.iterdir()),"Ticket directory already has work; inspect before resuming")
    env,bank,me,opp,cfg,rclock,mc=prepare()
    orig=n.original_state.take(opp.mod)
    learner=copy.deepcopy(n.student_state(me.mod));sig=n.original_state.h(learner)
    ref=n.official_state.take(env,bank);physical=n.board(env)
    idx=[];root=P["root_step"]
    for pair in P["pairs"]:
        n.original_state.restore(opp.mod,orig)
        opp.mod._ST["rng"].set_state(np.random.RandomState(pair["stream_seed"]).get_state())
        initial=n.original_state.take(opp.mod)
        for arm in pair["order"]:
            bank=n.official_state.restore(env,ref);private=n.original_state.restore(opp.mod,initial)
            n.restore_student(me.mod,learner)
            n.need(n.board(env)==physical and n.original_state.h(n.student_state(me.mod))==sig,"Different simulator or student snapshot")
            ticket=out/f'{pair["pair"]}_{arm}.json.gz'
            with (out/(ticket.stem+".claim")).open("x") as f:f.write(FROZEN+"\n")
            rows=[]
            while not env.done:
                obs=[n.public(env,s,bank) for s in (0,1)];t=obs[0]["step"]
                hashes=[n.sha(n.encoded(o)) for o in obs]
                if t==root:
                    n.need(hashes==P["root_public_sha256"],"Root input differs")
                    orders=[P["anchor_orders"][arm],P["anchor_orders"]["enemy"]]
                    own_clock={**mc,"kind":"common public call for parent command; intervention is precomputed"}
                    enemy_clock=rclock
                else:
                    a,own_clock=n.act(me,obs[0],cfg);b,enemy_clock=n.act(opp,obs[1],cfg,True);orders=[a,b]
                before=n.board(env);prev=[p[0] for p in obs[0]["players"]]
                target=read_worker(env,1,P["target_uid"]);defender=read_worker(env,0,P["defender_uid"])
                clocks=[own_clock,enemy_clock];bal=[bank[s]-max(0.,clocks[s]["wall"]-1.) for s in (0,1)]
                env.step(orders)
                rows.append(dict(step=t,public_hashes=hashes,orders=orders,bank_before=bank,bank_after=bal,
                    cash_before=prev,cash_after=[p[0] for p in env.states[0].observation.players],
                    target_before=target,target_after=read_worker(env,1,P["target_uid"]),
                    defender_before=defender,defender_after=read_worker(env,0,P["defender_uid"]),
                    student_clock=own_clock,enemy_clock=enemy_clock,board_before=before,board_after=n.board(env)))
                bank=bal
            scores=[v.reward for v in env.states];margin=scores[0]-scores[1]
            n.need(len(rows)==P["terminal_step"]-root or env.done and len(rows)<P["terminal_step"]-root,"Invalid terminal boundary")
            n.save(ticket,dict(pair=pair["pair"],arm=arm,stream_seed=pair["stream_seed"],seed=P["seed"],seat=P["seat"],
                terrain=P["terrain"],root_board_sha256=physical,root_student_sha256=sig,root_private_sha256=private,
                root_public_sha256=P["root_public_sha256"],protocol=FROZEN,official_scores_by_seat=scores,
                margin=margin,rows=rows))
            idx.append(dict(pair=pair["pair"],arm=arm,margin=margin,sha256=n.sha(ticket.read_bytes())))
            print(pair["pair"],arm,scores,flush=True)
    n.save(HERE/"results.json",dict(protocol_sha256=FROZEN,runner_sha256=n.sha(Path(__file__).read_bytes()),
          prefix_sha256=n.sha((HERE/"prefix.json.gz").read_bytes()),tickets=idx,whole_games=0))

if __name__=="__main__":
    with (HERE/"active.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        play()
