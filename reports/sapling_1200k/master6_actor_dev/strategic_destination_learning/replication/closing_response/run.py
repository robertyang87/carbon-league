"""Play matched diagnostic suffixes of the single registered, previously inspected t288 root."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
STRAT=HERE.parents[1]
ROOT=next(p for p in HERE.parents if (p/".git").exists())
CARGO=STRAT.parent/"cargo_tree_action_learning_dev"
LEAGUE=CARGO/"native_actor_corpus/league_policy"
CHECKPOINT=LEAGUE/"native_return_intake/live_curriculum/inprocess_snapshot"
sys.path[:0]=[str(LEAGUE),str(CHECKPOINT)]
import collect
import official_state
import original_state
from zerosum_env.envs.carbon import idgen

P=json.loads((HERE/"protocol.json").read_bytes())
FROZEN="1f770df8ed4cb841edb22a7a382ead37d53021bb365a92cec968196e975f0cd2"
sha=lambda b:hashlib.sha256(b).hexdigest()


def need(x,msg):
    if not x:raise ValueError(msg)


def encoded(v):
    return (json.dumps(v,separators=(",",":"),allow_nan=False)+"\n").encode()


def save(path,obj):
    b=encoded(obj)
    if path.suffix==".gz":b=gzip.compress(b,compresslevel=6,mtime=0)
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_bytes(b);os.replace(tmp,path)


def public(env,s,bank):
    o=env.states[0].observation
    return dict(step=o.step,player=s,carbon=env.states[s].observation.carbon,players=o.players,
                trees=o.trees,remainingOverageTime=bank[s])


def board(env):
    # Physical state (plus official entity-ID sources) while excluding wall-clock and process streams.
    o=env.states[0].observation
    return sha(encoded([o.step,o.full_carbon,o.players,o.trees,dict(idgen._id_dict)]))


def targets(players,order):
    ret={}
    for uid,w in players[1][2].items():
        r,c=divmod(w[0],15);dr,dc={"NONE":(0,0),"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1)}[order.get(uid) or "NONE"]
        ret[uid]=(r+dr)%15*15+(c+dc)%15
    return ret


def act(player,obs,cfg,native=False):
    stats=player.mod._S_STATS if native else None
    old={k:stats[k] for k in ("single","changed","searched","aborted")} if native else {}
    t0=time.perf_counter();c0=time.process_time()
    a=player.act(obs,cfg)
    clock=dict(wall=time.perf_counter()-t0,cpu=time.process_time()-c0)
    if native:
        diff={k:int(stats[k]-v) for k,v in old.items()}
        if diff["single"]==1:
            n,d,rolls,internal=1,0,0,None
        else:
            need(diff["searched"] or diff["aborted"],"No real public search receipt")
            n,d,rolls,internal,*_=stats["last"]
        clock.update(K=int(n),H=int(d),rolls=int(rolls),internal=internal,**diff)
    return a,clock


def student_state(m):
    return (vars(m._FEAT),m._COUNTS,[n.trunk for n in m._NETS])


def restore_student(m,state):
    fs,counts,trunks=copy.deepcopy(state)
    vars(m._FEAT).clear();vars(m._FEAT).update(fs)
    m._COUNTS=counts
    for n,trunk in zip(m._NETS,trunks):n.trunk=trunk


def worker(env):
    return copy.deepcopy(env.states[0].observation.players[0][2].get("player-0-worker-47"))


def main():
    os.environ["CARBON_DEBUG"]="1"
    need(os.sched_getaffinity(0)=={0,1} and sha((HERE/"protocol.json").read_bytes())==FROZEN,"Wrong budget or plan")
    out=HERE/"tickets";out.mkdir(exist_ok=True)
    need(not any(out.iterdir()),"This register has already been touched")
    d=HERE.parent/"hosted";zip_path=d/"14588219048.zip"
    need(sha(zip_path.read_bytes())==P["zip_sha256"] and sha((STRAT/"bc.py").read_bytes())==P["bc_sha256"],"Sources drift")
    with zipfile.ZipFile(zip_path) as z:
        raw=z.read("bc/turns.jsonl");reg=json.loads(z.read("bc/registration.json"))
    tape=[json.loads(x) for x in raw.splitlines()]
    need(sha(raw)==P["turns_sha256"] and reg["physical_map_sha256"]==P["terrain"],"Bad archive")
    cfg=json.loads(gzip.decompress((CARGO/"direct_original/games/candidate/raw_621758346_1.json.gz").read_bytes()))["game_configuration"]
    cfg={**cfg,"agentTimeout":20,"actTimeout":1,"runTimeout":1500};pubcfg={k:v for k,v in cfg.items() if k!="randomSeed"}
    env=collect.begin(P["seed"],cfg);bank=[5.,5.]
    need(sha(bytes(env.states[0].observation.full_carbon))==P["terrain"],"Different map")
    real_bytes=subprocess.check_output(["git","-C",str(ROOT),"show","origin/main:sources/master3s2/master3s2.py"])
    need(sha(real_bytes)==P["original_sha256"],"Nonoriginal examiner")
    keep=tempfile.TemporaryDirectory(prefix="closing-response-original-");source=Path(keep.name)/"master3s2.py";source.write_bytes(real_bytes)
    enemy=collect.Player(source);bc=collect.Player(STRAT/"bc.py");m=enemy.mod
    need(m._TEMP==1 and m._SEED is None and m._S_ABORT==.58 and m._SEARCH_OK,"Nonstandard examiner")
    expected_prev=None;warm=[];states=[]
    for i in range(P["root_step"]+1):
        op=[public(env,s,bank) for s in (0,1)]
        hashes=[sha(encoded(o)) for o in op]
        need(tape[i]["public_observation_sha256"]==hashes and all(q["status"]=="ok" for q in tape[i]["calls"]),"Different archived public input")
        orders=[q["response"]["action"] for q in tape[i]["calls"]]
        if i:need(m._ST["tr"].last_dest==expected_prev,"Incorrect incoming enemy belief intents")
        own,sc=act(bc,op[0],pubcfg);need(own==orders[0],"BC differs on the observed input")
        guessed,ec=act(enemy,op[1],pubcfg,True)
        need(m._ST["tr"].last_dest==targets(op[1]["players"],guessed) and m._ST["prev"]==op[1]["players"],"Wrong native intent/players")
        m._ST["tr"].set_actions(op[1]["players"],orders[1]);expected_prev=targets(op[1]["players"],orders[1])
        need(m._ST["tr"].last_dest==expected_prev,"Cannot recover true executed enemy command")
        warm.append(dict(step=i,public_hashes=hashes,student_clock=sc,enemy_clock=ec,shadow_enemy_order=guessed,
                         used_archived_enemy_order=orders[1],corrected=guessed!=orders[1]))
        if i==P["root_step"]:
            need(hashes==P["root_public_sha256"] and orders==[P["anchor_orders"]["bc_order"],P["anchor_orders"]["enemy"]],"Wrong tactical intervention")
            hosted=tape[i]["calls"][1]
            root_enemy_clock=dict(wall=hosted["elapsed"],cpu=hosted["response"]["public_cpu_seconds"],
                        K=hosted["response"]["search_last"]["candidates"],H=hosted["response"]["search_last"]["depth"],
                        kind="already-issued hosted simultaneous root order; shared logically across arms")
            break
        env.step(orders);bank=tape[i]["overage_remaining_seconds"]
    save(HERE/"prefix.json.gz",dict(protocol=FROZEN,host=socket.gethostname(),affinity=sorted(os.sched_getaffinity(0)),
                orig_sha=P["original_sha256"],torch_threads=m._torch.get_num_threads(),rows=warm))
    ref=official_state.take(env,bank);start_board=board(env);starter=copy.deepcopy(student_state(bc.mod))
    signature=original_state.h(starter);origin=original_state.take(m)
    def reset():
        b=official_state.restore(env,ref)
        original_state.restore(m,origin)
        restore_student(bc.mod,starter)
        need(board(env)==start_board and original_state.h(student_state(bc.mod))==signature,"Root restore changed the board/BC history")
        return b
    for pair in P["pairs"]:
        reset();m._ST["rng"].set_state(np.random.RandomState(pair["stream_seed"]).get_state())
        pair_checkpoint=original_state.take(m)
        for arm in pair["order"]:
            bank=reset();private=original_state.restore(m,pair_checkpoint)
            need([sha(encoded(public(env,s,bank))) for s in (0,1)]==P["root_public_sha256"],"Public root changed")
            rows=[];path=out/f'{pair["pair"]}_{arm}.json.gz'
            (out/(path.stem+".claim")).write_text(f'{FROZEN} {arm}\n')
            while not env.done:
                i=env.states[0].observation.step
                op=[public(env,s,bank) for s in (0,1)]
                if i==P["root_step"]:
                    orders=[P["anchor_orders"][arm],P["anchor_orders"]["enemy"]]
                    sc=dict(wall=warm[-1]["student_clock"]["wall"],kind="common historical-root BC inference plus precomputed physical intervention")
                    ec=root_enemy_clock
                else:
                    a,sc=act(bc,op[0],pubcfg);b,ec=act(enemy,op[1],pubcfg,True);orders=[a,b]
                before=board(env);oldcash=[p[0] for p in env.states[0].observation.players];w=worker(env)
                rh=[sha(encoded(o)) for o in op];bal=[bank[s]-max(c["wall"]-1.,0.) for s,c in enumerate([sc,ec])]
                env.step(orders);cash=[p[0] for p in env.states[0].observation.players]
                rows.append(dict(step=i,public_hashes=rh,orders=orders,bank_before=bank,bank_after=bal,
                        cash_before=oldcash,cash_after=cash,worker47_before=w,worker47_after=worker(env),
                        student_clock=sc,enemy_clock=ec,board_before=before,board_after=board(env)))
                bank=bal
            scores=[p.reward for p in env.states];need(len(rows)==11 and rows[-1]["step"]==298,"Unexpected suffix length")
            result=dict(pair=pair["pair"],arm=arm,stream_seed=pair["stream_seed"],seed=P["seed"],seat=0,terrain=P["terrain"],
                protocol=FROZEN,root_public_sha256=P["root_public_sha256"],root_private_sha256=private,
                root_student_sha256=signature,root_board_sha256=start_board,rows=rows,official_scores_by_seat=scores,margin=scores[0]-scores[1])
            save(path,result)
            states.append(dict(pair=pair["pair"],arm=arm,margin=result["margin"],sha256=sha(path.read_bytes())))
            print(pair["pair"],arm,scores,result["margin"],flush=True)
    save(HERE/"results.json",dict(protocol_sha256=FROZEN,runner_sha256=sha(Path(__file__).read_bytes()),prefix_sha256=sha((HERE/"prefix.json.gz").read_bytes()),tickets=states,whole_games=0))

if __name__=="__main__":
    main()
