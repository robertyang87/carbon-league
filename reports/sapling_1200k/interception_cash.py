"""Follow identified collision cargo on the actually played intervention trajectory."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from interception_fit import sha, intent
from fastsim import FastCarbon


def collision_events(before, after, joint):
    ledger={}
    for uid,owner,amount,credited,winner_seat in after.last_collisions:
        if amount<=0:
            continue
        order=joint[owner].get(uid) or 'NONE'
        pos=int(intent.DEST[before['workers'][uid]['pos'],intent.MOVE_ID[order]])
        holder=next((wid for wid,w in after.workers.items()
                     if winner_seat is not None and w.player==winner_seat and w.pos==pos and w.is_collector),None)
        # A collector pays positive cargo at any occupied base after collisions.
        if winner_seat is not None and pos in after.base_at:
            route=('cash',after.base_at[pos][0])
        elif holder is not None:
            route=('worker',holder)
        elif credited is not None and winner_seat is None:
            route=('cash',credited)
        else:
            route=('unrecovered',None)
        ledger[uid]=(owner,credited,amount,route)
    return ledger


def changed_transfers(state, cfg, base, chosen, opponent, side):
    def run(actions):
        joint=[None,None];joint[side]=actions;joint[1-side]=opponent
        sim=FastCarbon(cfg);sim.set_state(state);sim.step(*joint)
        return collision_events(state,sim,joint)
    old,new=run(base),run(chosen)
    totals=Counter()
    for uid in set(old)|set(new):
        before=old.get(uid);after=new.get(uid)
        owner=(before or after)[0]
        credited=1-owner
        prior=before[2] if before and before[1]==credited else 0.
        actual=after[2] if after and after[1]==credited else 0.
        diff=actual-prior
        if owner!=side:
            totals['additionally_captured']+=max(diff,0.)
            totals['capture_forgone']+=max(-diff,0.)
        else:
            totals['own_transfer_prevented']+=max(-diff,0.)
            totals['additional_own_transfer']+=max(diff,0.)
    return {k:totals[k] for k in ('additionally_captured','capture_forgone','own_transfer_prevented','additional_own_transfer')}


def audit(raw, events):
    side=raw['candidate_seat']
    sim=FastCarbon(raw['game_configuration']).reset(raw['seed'])
    if sim.config['smeltCost']!=0:
        raise ValueError('cash attribution requires the actual zero smelting charge')
    active=defaultdict(Counter)
    ledger=defaultdict(Counter)
    for tick,joint in enumerate(raw['actions']):
        before=sim.get_state()
        old=active
        active=defaultdict(Counter)
        sim.step(*joint)
        actual=collision_events(before,sim,joint)
        def route(uid):
            if uid in actual:
                return actual[uid][3]
            w=sim.workers.get(uid)
            if w is None:return ('unrecovered',None)
            if w.pos in sim.base_at:return ('cash',sim.base_at[w.pos][0])
            return ('worker',uid)
        def carry(key,amount,destination):
            kind,who=destination
            if kind=='worker':
                if sim.workers[who].player==side:
                    active[who][key]+=amount
                else:
                    ledger[key]['subsequently_lost']+=amount
            elif kind=='cash' and who==side:
                ledger[key]['deposited_or_credited_to_our_cash']+=amount
            else:
                ledger[key]['subsequently_lost']+=amount
        for uid,tags in old.items():
            for key,mass in tags.items():
                carry(key,mass,route(uid))
        if tick not in events:
            continue
        choice=events[tick]
        if choice['selected_actions']!=joint[side]:raise ValueError('recorded interception differs')
        alternate=list(joint); alternate[side]=choice['base_actions']
        original=FastCarbon(raw['game_configuration'])
        original.set_state(before)
        original.step(*alternate)
        baseline=collision_events(before,original,alternate)
        for uid,(owner,credited,mass,destination) in actual.items():
            if owner==1-side and credited==side and baseline.get(uid,(None,None))[1]!=side:
                ledger['extra_capture']['identified']+=mass
                carry('extra_capture',mass,destination)
        for uid,(owner,credited,mass,_) in baseline.items():
            if owner!=side or credited!=1-side or actual.get(uid,(None,None))[1]==1-side:
                continue
            # One previously protected batch may be at risk repeatedly before deposit.
            tagged=max(0.,mass-sum(old.get(uid,{}).values()))
            ledger['protected_own']['repeat_exposure_excluded']+=mass-tagged
            ledger['protected_own']['identified']+=tagged
            carry('protected_own',tagged,route(uid))
    for tags in active.values():
        for key,mass in tags.items():
            ledger[key]['still_carried_at_terminal']+=mass
    for key,values in ledger.items():
        if abs(values['identified'] - sum(values[k] for k in ('subsequently_lost','deposited_or_credited_to_our_cash','still_carried_at_terminal'))) > 1e-5:
            raise ValueError(f'cargo did not reconcile for {key}')
    return dict(ledger)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--live',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    report=json.loads((a.run/'results.json').read_text());facts=json.loads(a.live.read_text())
    if not facts['actual_deployed'] or facts['ledger_sha256']!=sha((a.run/'results.json').read_bytes()):
        raise ValueError('need the actual candidate result and matching replay')
    by_key={(g['seed'],g['seat']):g for g in facts['games']}
    records=[]
    for g in report['games']:
        raw_bytes=(a.run/g['raw_file']).read_bytes()
        if sha(raw_bytes)!=g['raw_sha256']:raise ValueError('wrong game')
        events=by_key[(g['seed'],g['candidate_seat'])]['events']
        records.append(dict(seed=g['seed'],seat=g['candidate_seat'],raw_sha256=g['raw_sha256'],
                            traced=audit(json.loads(gzip.decompress(raw_bytes)),{e['step']:e for e in events})))
    a.out.write_text(json.dumps(dict(source_sha256=sha(Path(__file__).read_bytes()),games=records,
                     definition='Positive cargo identified at a changed joint root with original enemy command held fixed, followed on actual moves until our cash, loss, or terminal. Repeat exposure on the same retained cargo is excluded. This does not recover an alternate whole-game trajectory.'),indent=2)+'\n')


if __name__=='__main__':
    main()
