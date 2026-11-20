"""Check team-level cash conservation and counterexamples with the archived Hajimi root."""
import copy
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import panel

HERE=Path(__file__).resolve().parent
PARENT=panel.D/"cold_start/candidate.py"
GID="3f0a1dfa0e1ac4a00a9f63f9cf644af9"
STEP=295
ME=1
DONOR="player-1-worker-37"
RECIPIENT="player-1-worker-9"


def main():
    package=panel.load("relay_mechanics",HERE/"candidate.py")
    ref=panel.load("official_referee",panel.D/"freight_response/run.py")
    source=panel.LADDER/"replays"/(GID+"-1.json.gz")
    raw=source.read_bytes();game=json.loads(gzip.decompress(raw));tape=game["steps"];cfg=game["configuration"]
    random.seed(cfg["randomSeed"]);np.random.seed(cfg["randomSeed"])
    env=ref.make("carbon",configuration=cfg);terrain=tape[0][0]["observation"]["full_carbon"]
    env.states[0].observation.full_carbon=copy.deepcopy(terrain)
    fast=ref.econ.FastCarbon(cfg).reset(full_carbon=terrain)
    def verify(engine):
        off=env.states[0].observation;ff=engine.obs(0)
        assert ff["players"]==off.players and ff["trees"]==off.trees and list(engine.carbon)==off.full_carbon
    for t in range(STEP):
        actions=[k["action"] or {} for k in tape[t+1]]
        fast.step(*actions);env.step(actions,[]);verify(fast)
        assert fast.obs(0)["players"]==tape[t+1][0]["observation"]["players"] and list(fast.carbon)==tape[t+1][0]["observation"]["full_carbon"]
    env.steps[:-1]=[None]*STEP;env.logs=[];frozen=ref.official_state.take(env,[0,0])
    parent=tape[STEP+1][ME]["action"]
    views=panel.Game(game)
    o=views._obs(STEP,ME);p=o["players"]
    propose=lambda ps,order,t=STEP:package._relay_choice(ps,ME,t,order)
    base=propose(p,parent)
    assert (base['donor'],base['recipient'],base['target'])==(DONOR,RECIPIENT,156)
    def focal(a):
        return a is not None and (a["donor"],a["recipient"],a["target"])==(DONOR,RECIPIENT,156)
    mutations={}
    for reason in ("equal_cargo","other_lighter_enemy","third_friendly_arrival","eligible_donor_can_bank"):
        pp=copy.deepcopy(p);aa=parent.copy()
        if reason=="equal_cargo":pp[ME][2][RECIPIENT][1]=pp[ME][2][DONOR][1]
        elif reason=="other_lighter_enemy":
            pp[0][2]["player-0-worker-43"][:2]=[141,0.]
        elif reason=="third_friendly_arrival":
            pp[ME][2]["player-1-worker-46"][0]=141;aa["player-1-worker-46"]="DOWN"
        else:aa[DONOR]="LEFT"  # The baseline itself already reaches the rendezvous.
        hit=focal(propose(pp,aa))
        mutations[reason]=dict(proposed_same_transaction=hit)
        assert not hit
    assert propose(p,parent,293) is None and propose(p,parent,298) is None
    assert {u for u in set(base["order"])|set(parent) if base["order"].get(u)!=parent.get(u)}=={DONOR}
    assert package._relay_leaf(o,cfg,None,None,parent,base,time.perf_counter()-1)==(None,"budget")
    rows=json.loads(gzip.decompress((HERE/"public_entries.json.gz").read_bytes()))
    h=next(r for r in rows if r['id']==GID and r['t']==STEP)
    def settle(ours):
        ref.official_state.restore(env,frozen);s=copy.deepcopy(fast);s.cash=ref.econ.CashEvents(s.cash)
        transitions=[]
        for t in range(STEP,299):
            acts=[dict(r["action"] or {}) for r in tape[t+1]]
            if t==STEP:acts[ME]=ours
            surviving=s.obs(0)["players"]
            ignored=[[uid for uid in at if uid not in surviving[side][1] and uid not in surviving[side][2]] for side,at in enumerate(acts)]
            s.cash.events.clear();s.step(*acts);env.step(acts,[]);verify(s)
            transitions.append(dict(state=t+1,public_joint_commands=acts,ignored_orders_after_divergence=ignored,wallet=list(s.cash),
               reconciled_team_delivery=[e for e in s.cash.events if e[0]=="own_delivery"],
               collisions=copy.deepcopy(s.last_collisions),remaining_workers=s.obs(0)["players"][1][2]))
        assert env.done and list(s.cash)==[side.reward for side in env.states]
        return dict(final_by_seat=list(s.cash),margin=round(s.cash[ME]-s.cash[1-ME],2),turns=transitions)
    cases={"historical_parent":settle(parent),"historical_parent_plus_relay":settle(base["order"])}
    for n,rep in enumerate(h['pairs']):
        trace=rep['True']['trace']
        assert trace['emitted'] and trace['proposal']['order']==rep['True']['action']
        cases['resampled_parent_'+str(n)]=settle(trace['parent'])
        cases['resampled_plus_relay_'+str(n)]=settle(trace['proposal']['order'])
    receipt=dict(raw_sha256=panel.digest(raw),candidate_sha256=panel.digest((HERE/"candidate.py").read_bytes()),
                 probe="Separate conditional outcome check after all on/off calls. Enemy and post-root actions fixed to those actually seen in this archive. A missing unit's order is ignored by the official rules.",
                 cross_checked_common_prefix=STEP,archived_official_scores=game["rewards"],root_interval_2_to_5=True,single_changed_worker=DONOR,
                 controls=mutations,expired_budget_returns_parent=True,source_observation_sha=panel.digest(o),cases=cases)
    assert cases["historical_parent"]["final_by_seat"]==game["rewards"]
    panel.save(HERE/"mechanics.json.gz",receipt)
    print({k:v['margin'] for k,v in cases.items()})

if __name__=="__main__":
    main()
