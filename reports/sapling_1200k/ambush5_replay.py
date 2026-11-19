"""Check the frozen live adapter against isolated start decisions and archive economy."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from ambush5_probe import arena,HERE,RUNS
from interception_cash import collision_events


def check(runs, candidate):
    expected=json.loads((HERE/'ambush5_exploratory_results.json').read_text())['sapling']['games']
    expected={(g['run'],g['seed'],g['seat']):{a['tick']:a for a in g['learned']} for g in expected}
    summaries=[]
    for folder in runs:
        ledger=json.loads((folder/'results.json').read_text())
        for g in ledger['games']:
            raw=json.loads(gzip.decompress((folder/g['raw_file']).read_bytes()))
            seat=g['candidate_seat']; summary=Counter(); rows=[]
            sim=arena.exam.FastCarbon(raw['game_configuration']).reset(seed=g['seed'])
            probes=expected.get((folder.name,g['seed'],seat))
            is_live=ledger['header']['pack_sha256']==arena.prior.digest(candidate)
            packed=arena.prior.shared.module(candidate,'frozen_ambush_probe')
            tagged={};previous={};active=None; timeline=[]; immediate=Counter()
            scratch=arena.exam.FastCarbon(raw['game_configuration'])
            for tick,joint in enumerate(raw['actions']):
                obs=sim.obs(seat)
                # On the deployed candidate's transcript, rerun the actual actor live on every step.
                if is_live:
                    action=packed.agent(obs,sim.config)
                    if action!=joint[seat]:
                        raise ValueError(f'live policy divergence at {tick}')
                    record=packed.agent.last_decision
                    base=packed.agent.last_baseline
                else:
                    base=joint[seat]
                    action,record=packed._ambush_update(obs,base)
                    if probes is not None and tick+5<=len(raw['actions']):
                        isolated=packed.choose(obs,previous,base,forecaster=packed._AMBUSH_MODEL)
                        oracle=probes.get(tick)
                        if (isolated is None)!=(oracle is None) or (isolated is not None and
                            any(isolated[k]!=oracle[k] for k in ('attacker','target','depth','destination','moves','forecast_probability'))):
                            raise ValueError(f'frozen initial choice mismatch at {tick}')
                        summary['frozen_choice_matches']+=1
                if record['changed'] and any(k!=record['attacker'] and (action.get(k) or None)!=(base.get(k) or None) for k in set(action)|set(base)):
                    raise ValueError('multiple worker decisions overridden')
                if record['phase']=='start':
                    active={'step':tick,'attacker':record['attacker'],'target':record['target'],'cargo_captured':0.,'cash':0.,'lost':0.,'unrecovered':0.}
                    plan=packed._AMBUSH_SEATS[seat]['plan']
                    active.update(destination=plan['destination'],depth=plan['depth'],attacker_origin=plan['origin'],
                                  enemy_origin=plan['target_origin'],enemy_starting_cargo=plan['target_cargo'],following=[])
                    rows.append(active)
                if record['phase']=='release':
                    active=None
                summary['roots']+=1;summary[record['phase']]+=1
                summary['changed']+=record['changed']
                before=sim.get_state()
                sim.step(*joint)
                if is_live:
                    ev=collision_events(before,sim,joint)
                    def receiver(events,side):
                        return sum(cargo for _,(owner,credit,cargo,route) in events.items() if owner!=side and
                                   credit==side and (route[0]=='worker' or route==('cash',side)))
                    if record['changed']:
                        control=list(joint)
                        control[seat]=base
                        scratch.set_state(before)
                        scratch.step(*control)
                        original=collision_events(before,scratch,control)
                        immediate['extra_enemy_cargo_received']+=receiver(ev,seat)-receiver(original,seat)
                        immediate['extra_own_cargo_given']+=receiver(ev,1-seat)-receiver(original,1-seat)
                        if active is not None:
                            active['changed_steps']=active.get('changed_steps',0)+1
                    income=[sim.cash[p]-before['cash'][p] for p in (0,1)]
                    timeline.append(dict(tick=tick,own_cash_before=before['cash'][seat],enemy_cash_before=before['cash'][1-seat],
                                         own_cash_after=sim.cash[seat],enemy_cash_after=sim.cash[1-seat],enemy_cash_swing=round(income[1-seat]-income[seat],4),
                                         own_live=len(sim.player_workers[seat]),enemy_live=len(sim.player_workers[1-seat]),
                                         taken=round(receiver(ev,seat),4),given=round(receiver(ev,1-seat),4),phase=record['phase'],attacker=record['attacker']))
                    def destination(uid):
                        if uid in ev:
                            return ev[uid][3]
                        if uid not in sim.workers:
                            return ('lost',None)
                        w=sim.workers[uid]
                        return ('cash',sim.base_at[w.pos][0]) if w.pos in sim.base_at else ('worker',uid)
                    def post(masses,route):
                        kind,who=route
                        for i,mass in masses:
                            if kind=='cash' and who==seat:
                                rows[i]['cash']+=mass
                            elif kind=='worker' and sim.workers[who].player==seat:
                                kept.setdefault(who,[]).append((i,mass))
                            elif kind=='unrecovered':
                                rows[i]['unrecovered']+=mass
                            else:
                                rows[i]['lost']+=mass
                    for raid in reversed(rows):
                        if tick-raid['step']>=5:
                            break
                        target=sim.workers.get(raid['target']); hunter=sim.workers.get(raid['attacker'])
                        raid['following'].append({'step_after':tick+1,'phase':record['phase'],
                                                  'hunter_pos':hunter.pos if hunter else None,
                                                  'target_pos':target.pos if target else None,'target_cargo':round(target.carbon,4) if target else None,
                                                  'target_on_enemy_base':bool(target and sim.base_at.get(target.pos,(None,))[0]==1-seat),
                                                  'separation':packed.DIST[hunter.pos][target.pos] if hunter and target else None})
                    kept={}
                    for uid,masses in tagged.items():
                        post(masses,destination(uid))
                    if active:
                        uid,wid=active['target'],active['attacker']
                        if uid in ev and wid in sim.workers:
                            _,credit,mass,where=ev[uid]
                            dest=packed.DEST[before['workers'][uid]['pos']][packed.intent.MOVE_ID[joint[1-seat].get(uid) or 'NONE']]
                            if credit==seat and sim.workers[wid].pos==dest and (where==('worker',wid) or where==('cash',seat)):
                                active['cargo_captured']+=mass
                                post([(len(rows)-1,mass)],where)
                    tagged=kept
                previous=obs['players'][1-seat][2]
            if arena.prior.hashed(sim.get_state())!=g['final_state_sha256']:
                raise ValueError('terminal state mismatch')
            if is_live:
                for masses in tagged.values():
                    for i,m in masses:
                        rows[i]['held_at_terminal']=rows[i].get('held_at_terminal',0.)+m
                for event in rows:
                    if abs(event['cargo_captured']-(event['cash']+event['lost']+event['unrecovered']+event.get('held_at_terminal',0)))>1e-5:
                        raise ValueError('cargo attribution does not balance')
            summaries.append(dict(run=folder.name,seed=g['seed'],seat=seat,actual_deployed=is_live,
                                  counts=summary,starts=rows if is_live else None,raw_sha256=g['raw_sha256'],
                                  net_immediate_transfers_if_only_changed_root_restored=dict(immediate) if is_live else None,
                                  largest_enemy_cash_jump_and_previous_five=(timeline[max(0,(peak:=max(range(len(timeline)),key=lambda j:timeline[j]['enemy_cash_swing']))-5):peak+1] if is_live else None)))
    return summaries


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',nargs='+',type=Path)
    p.add_argument('--pack',type=Path,default=HERE.parents[1]/'data/weights/sapling_1200k/soft_it50_v5_ambush5.py')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=check(a.run or [HERE/r for r in RUNS],a.pack)
    a.out.write_text(json.dumps({'pack_sha256':arena.prior.digest(a.pack),'games':result},separators=(',',':'))+'\n')
    print([(r['run'],r['seat'],r['counts']) for r in result])
