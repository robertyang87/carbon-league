"""Scan native v1 tapes in chronological order. Only earlier observations choose the first intervention."""
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

from build import BASE,HERE,ROOT,PLAN,sha,source

sys.path.insert(0,str(BASE/'root_recovery_branches'))
import live
from branches import clean
sys.path.insert(0,str(BASE))
from turning_point_ledger import FastCarbon, CashEvents

def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())

def retain(path,x):
    if path.exists():
        if read(path)!=x:
            raise ValueError(f'frozen result differs: {path}')
    else:
        live.save(path,x)

def exercise_tests(mod,obs,ticket,commands):
    """Same historical root; reduce public cash, or move one visible enemy collector adjacent."""
    obj=mod._OPTION
    actual,_=obj.confirm(ticket,commands)
    if not actual:
        raise ValueError('unexpected baseline eligibility')
    results={'unmodified':'exercise'}
    for tag in ('insufficient_cash','enemy_threat'):
        fork=copy.deepcopy(obj)
        hypothetical=copy.deepcopy(obs)
        seat=obs['player']
        if tag=='insufficient_cash':
            hypothetical['players'][seat][0]=max(0.,actual['price']-1)
        else:
            rival=hypothetical['players'][1-seat][2]
            if not rival:
                continue
            worker=next((k for k,v in rival.items() if v[2]=='COLLECTOR'),next(iter(rival)))
            available=set(mod._ROUTE_DEST[ticket['cell'],1:])
            occupied={u[0] for pl in hypothetical['players'] for k,u in pl[2].items() if k!=worker}
            dest=next((int(p) for p in sorted(available) if p not in occupied),None)
            if dest is None:
                continue
            rival[worker][0]=dest
        fork.obs=hypothetical
        fork.market.observe(hypothetical,seat,hypothetical['carbon'],obj.market.cfg,obj.t)
        q,reason=fork.confirm(ticket,commands)
        if q:
            raise ValueError('unsafe branch was not cancelled')
        results[tag]=reason
    return results


def inspect(path, branch_root=False):
    raw=read(path);s=raw['candidate_seat'];seed=raw['seed'];cfg=raw['game_configuration']
    model=live.module(HERE/'agent.py','option_diagnostic')
    sim=FastCarbon(cfg).reset(seed=seed)
    sim.cash=CashEvents(sim.cash)
    reasons=Counter()
    changes=[];first=None;prev_obs=None;prev_commands=None;receipts=[]
    for t,acts in enumerate(raw['actions']):
        obs=sim.obs(s);current=clean(acts[s]);controller=model._OPTION
        if t==0:
            if clean(model.agent(obs,cfg))!=current:
                raise ValueError('frozen parent start differs')
        else:
            controller.observe(obs,cfg)
            previous=controller.pending
            uid=previous['uid'] if previous else None
            q,reason=controller.confirm(previous,current) if previous else (None,'no_previous')
            if q and current.get(uid,'NONE')!='NONE':
                # Run the real two-backbone decoder only where both computed policies could differ.
                # At every other root the same per-seat Feat object still consumes its observation.
                issued=clean(model._option_original_agent(obs,cfg))
                last=controller.last
                diff=sorted(k for k in current.keys()|issued.keys() if current.get(k)!=issued.get(k))
                if last['committed'] and (diff!=[uid] or uid in issued or last['parent_move']!=current[uid]):
                    raise ValueError('deployed action differs from the tested one-planter intervention')
                if not last['committed'] and diff:
                    raise ValueError('parent did not reproduce')
                if last['committed']:
                    row={'t':t,'worker':uid,'cell':previous['cell'],'parent':current[uid],
                         'alternative':'NONE','ticket':previous,'quote':q}
                    changes.append(row)
                    if first is None:
                        first={**row,'branch_tests':exercise_tests(model,obs,previous,current),
                               'previous_actual_command':prev_commands[uid],
                               'observed_before_landing':prev_obs,'public_at_exercise':obs,
                               'root_parent_joint':current,'root_option_joint':issued,
                               'root_opponent_fixed':acts[1-s]}
                reason=last['reason']
            else:
                model._FEAT.observe(obs)
                reason=reason or 'parent_already_stays'
            reasons[reason]+=1
            # A later record is always a distinct shadow decision along the original trajectory.
            # Never propagate an unplayed command or an imaginary budget into the next root.
            controller.pending=controller.propose(current)
        previous_cash=sim.cash[:]
        sim.cash.events.clear()
        sim.step(*acts)
        if t and (q is not None and reason=='parent_already_stays'):
            booked=[e for e in sim.cash.events if e[0] in ('plant','seize') and e[1]==s and e[3]==uid]
            if booked:
                receipts.append({'t':t,'uid':uid,'quote':q['price'],'charged':-booked[0][2]})
        prev_obs=obs;prev_commands=current
    if any(abs(r['quote']-r['charged'])>1e-7 for r in receipts):
        raise ValueError('public execution quote differs from a historical exact charge')
    result={'seed':seed,'seat':s,'source':str(path.relative_to(ROOT)),'source_sha256':sha(path.read_bytes()),
            'completed_transitions':len(raw['actions']),'counts':dict(reasons),'actual_paid_parent_confirmations':receipts,
            'hypothetical_changes_along_original':changes}
    if branch_root and first:
        result['first_root']=first
    return result


def main():
    plan=read(PLAN)
    pkg=source()
    if pkg!=(HERE/'agent.py').read_bytes():
        raise ValueError('stale candidate')
    areas=[('terminal_return_fit/dev_learned',621461220,0),('terminal_return_fit/dev_learned',621461220,1)]
    expected={(r['seed'],r['seat']) for r in plan['branch_selection']}
    for item in plan['branch_selection']:
        for seat in (0,1):
            areas.append((item['source'],item['seed'],seat))
    rows=[];registered=[]
    for area,seed,seat in areas:
        raw=HERE.parent/area/f'raw_{seed}_{seat}.json.gz'
        if read(raw)['pack_sha256']!=plan['parent_pack_sha256']:
            raise ValueError('unexpected parent of archive')
        row=inspect(raw,(seed,seat) in expected)
        first=row.pop('first_root',None)
        if first:
            full=HERE/f'root_{seed}_{seat}.json.gz'
            retain(full,first)
            registered.append({'seed':seed,'seat':seat,'t':first['t'],'source':row['source'],'source_sha256':row['source_sha256'],
                               'root':full.name,'root_sha256':sha(full.read_bytes()),'worker':first['worker'],
                               'cell':first['cell'],'old':first['parent'],'paired_rng_seed':804210+seat})
        rows.append(row)
    retain(HERE/'mechanism.json',{'package_sha256':sha(pkg),'definition':'later triggers are per-root diagnostics on frozen parent tapes',
                                  'games':rows,'selected_roots':registered,'both_preassigned_roots':len(registered)==2})
    print([(r['seed'],r['seat'],r['t']) for r in registered],flush=True)
if __name__=='__main__':
    main()
