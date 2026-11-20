"""Intervene once in the root joint order; the same frozen controllers respond to the new boards."""
from collections import Counter
import copy
import gzip
import json
import os
from pathlib import Path

from history import DERIVATIVE, HERE, REGISTRATION, ROOT, advance, freight, setup, sha, source_cards, token, views


def write(name, value):
    out=HERE/name
    raw=(json.dumps(value,separators=(",",":"),sort_keys=True,allow_nan=False)+"\n").encode()
    if out.suffix==".gz":
        raw=gzip.compress(raw,mtime=0)
    freight.check(not out.exists() or out.read_bytes()==raw, "Refusing to overwrite a different branch")
    if not out.exists():
        temp=out.with_suffix(".partial")
        temp.write_bytes(raw)
        temp.replace(out)
    return sha(raw)


def restore_prefix(ticket, source=None, expected=None, wallet=None):
    seat,t=ticket["seat"],ticket["step"]
    if source is None:
        source=source_cards()[ticket["source_sha256"]]
    tape,actions,cfg,env,sim=setup(ticket,source)
    candidate=DERIVATIVE/"cold_start/candidate.py"
    original=ROOT/"data/master3s2_source.py"
    freight.check(sha(candidate.read_bytes())==REGISTRATION["base"] and
                  sha(original.read_bytes())==REGISTRATION["examiner"],"Frozen source changed")
    paths=[original,original]
    paths[seat]=candidate
    mods=[freight.load("paired_value_s"+str(s),p) for s,p in enumerate(paths)]
    for m in mods:
        freight.check(m._TEMP==1 and m._SEED is None and m._S_ABORT==.58 and m._RX and m._SEARCH_OK,
                      "Unexpected search runtime")
    for i in range(t):
        obs=views(env)
        step_orders=actions[i]
        for s,m in enumerate(mods):
            players=obs[s]["players"]
            if i in (0,1,2):
                freight.measured(m,obs[s],cfg)
                if i==0:
                    seed=int.from_bytes(bytes.fromhex(token([ticket["source_sha256"],s,"shared-referee-response"]))[:4],"big")
                    m._ST["rng"]=freight.np.random.RandomState(seed)
                    m._ST["rng"].random_sample(len(players[s][1])+len(players[s][2]))
            else:
                m._ST["tr"].step(players,i,obs[s]["carbon"])
                m._ST["rng"].random_sample(len(players[s][1])+len(players[s][2]))
            m._ST["tr"].set_actions(players,step_orders[s])
            m._ST["prev"]=players
        advance(env,sim,step_orders)
        if wallet is not None:
            for k,flow in enumerate(freight.econ.cash_flow(sim.cash.events)):
                wallet[k].update(flow)
    if expected is None:
        fingerprints=json.loads((HERE/"restored_sources.json").read_bytes())
        source=next(r for r in fingerprints["sources"] if r["source_sha256"]==ticket["source_sha256"])
        expected=source["roots"][str(t)]
    public=views(env)
    freight.check(expected["public_sha256_by_seat"]==[freight.digest(v) for v in public] and
                  expected["incoming_cash"]==list(sim.cash) and expected["prior_orders_sha256"]==token(actions[:t]),
                  "Both historical public views, cash and order prefix must agree")
    return mods,cfg,env,sim,public,expected


def run_pair(ticket, admission, ordinal):
    s,t=ticket["seat"],ticket["step"]
    mods,cfg,env,sim,obs,expected=restore_prefix(ticket)
    prefix_h=freight.original_state.h([m._ST for m in mods])
    # Two recorded public root orders are the experimental interventions. Restore the
    # comparator's private draw and measured initial cost for V2's root-side state.
    parent=mods[s]
    parent._ST["rng"]=freight.np.random.RandomState(ticket["policy_rng"])
    parent._S_COST[:]=admission["cost"]
    initial_rng=freight.original_state.h(parent._ST["rng"])
    self_reply,menu=freight.root_call(parent,obs[s],cfg)
    # Wall noise can change the public winner. The search-board generating menu and
    # both registered actions must reproduce before the selected experiment launches.
    freight.check(token(menu)==admission["menu_sha256"] and admission["parent"] in menu and admission["rotated"] in menu,
                  "Selected candidate menu did not reproduce in prefix recovery")
    return continue_pair(ticket,admission,["parent","rotated"] if ordinal%2==0 else ["rotated","parent"],
                         (mods,cfg,env,sim,obs,expected),self_reply,prefix_h,initial_rng,write)


def continue_pair(ticket, admission, order, state, self_reply, prefix_h, initial_rng, save):
    s,t=ticket["seat"],ticket["step"]
    mods,cfg,env,sim,obs,expected=state
    parent=mods[s]
    rival=freight.measured(mods[1-s],obs[1-s],cfg)
    bank=[v["remainingOverageTime"] for v in obs]
    official=freight.official_state.take(env,bank)
    shadow=copy.deepcopy(sim)
    snapshots=[copy.deepcopy(freight.original_state.dynamic(m)) for m in mods]
    configs=[(m._RX_CFG[0],copy.deepcopy(m._RX_CFG[1])) for m in mods]
    before=list(sim.cash)
    prefix=dict(training_root=ticket,time_policy="in-process wall; one logical CPU and shared host",
                host_cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
                host_memory_max=Path("/sys/fs/cgroup/memory.max").read_text().strip(),
                affinity=sorted(os.sched_getaffinity(0)), public_pair=expected, native_incoming_h=prefix_h,
                registered_v2_root_rng_sha=initial_rng,root_parent_recovery_choice=self_reply[0],
                root_shared_original_order=rival[0],choices=admission)
    stem=f"{ticket['seed']}_{s}_{t}"
    per_arm={}
    for arm in order:
        bank=freight.official_state.restore(env,official)
        sim=copy.deepcopy(shadow)
        for pid,m in enumerate(mods):
            for k,v in copy.deepcopy(snapshots[pid]).items():
                setattr(m,k,v)
            m._RX_CFG[:]=[configs[pid][0],copy.deepcopy(configs[pid][1])]
            freight.check(freight.original_state.h(freight.original_state.dynamic(m))==freight.original_state.h(snapshots[pid]),
                          "Native private state failed to restore")
        command=admission[arm]
        parent._ST["tr"].set_actions(obs[s]["players"],command)
        rows=[]
        while not env.done:
            inputs=[freight.public(env,k,bank) for k in (0,1)]
            i=inputs[0]["step"]
            if i==t:
                pair=[None,None];pair[s]=(command,self_reply[1]);pair[1-s]=rival
            else:
                pair=[freight.measured(m,inputs[k],cfg) for k,m in enumerate(mods)]
            moves=[r[0] for r in pair]
            clocks=[r[1] for r in pair]
            bank=[b-max(0,c["wall"]-1.) for b,c in zip(bank,clocks)]
            fingerprints=[freight.digest(v) for v in inputs]
            cash=advance(env,sim,moves)
            flow=[dict(c) for c in freight.econ.cash_flow(sim.cash.events)]
            rows.append(dict(t=i,input_sha256=fingerprints,orders=moves,clocks=clocks,end_cash=cash,flow=flow))
        scores=[float(st.reward) for st in env.states]
        outcome=dict(scores=scores,win=scores[s]>scores[1-s],margin=scores[s]-scores[1-s],
                     statuses=[st.status for st in env.states],transitions=len(rows),time_banks=bank)
        for label,horizon in (("first10",10),("terminal",len(rows))):
            accounts=[]
            for k in (0,1):
                totals=Counter()
                for row in rows[:horizon]:
                    totals.update(row["flow"][k])
                cash=rows[min(len(rows),horizon)-1]["end_cash"][k]
                freight.check(abs(sum(totals.values())-(cash-before[k]))<1e-5,"Unreconciled cash account")
                accounts.append(dict(cash=cash,change=cash-before[k],components=dict(totals)))
            outcome[label]=accounts
        evidence=save(stem+"_"+arm+".json.gz",dict(**prefix,arm=arm,outcome=outcome,rows=rows))
        per_arm[arm]=dict(**outcome,evidence_sha256=evidence)
        print(stem,arm,scores,flush=True)
    result=dict(root=[ticket["seed"],s,t],arms=per_arm)
    if "rotated" in per_arm:
        result["shifted_minus_parent_margin"]=per_arm["rotated"]["margin"]-per_arm["parent"]["margin"]
    return result
