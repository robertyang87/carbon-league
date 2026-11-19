"""Reprice only these completed proxy tapes; separate delivery, collision and tree cash."""
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from world import World

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from turning_point_ledger import CashEvents


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)


def account(start,commands,official,seat):
    world=copy.deepcopy(start)
    sim=world._sim
    sim.cash=CashEvents(sim.cash)
    initial=list(sim.cash); totals=[Counter(),Counter()]
    for pair in commands:
        before=list(sim.cash);sim.cash.events.clear();world.advance(pair)
        writes=[Counter(),Counter()]
        for row in sim.cash.events:
            k,i,value=row[:3];writes[i][k]+=value
        for i in (0,1):
            if abs(sum(writes[i].values())-(sim.cash[i]-before[i]))>1e-6:
                raise ValueError('unexplained cash write')
            totals[i].update(writes[i])
    ending=world.terminal
    if not ending or any(ending[k]!=official[k] for k in ('terrain_sha256','steps','termination','result_by_seat','extinct_by_seat')):
        raise ValueError('terminal identity or outcome differs')
    if any(abs(a-b)>.005001 for a,b in zip(ending['scores'],official['scores'])):
        raise ValueError('official physical rewards differ')
    for i in (0,1):
        if abs(sum(totals[i].values())+initial[i]-sim.cash[i])>1e-6:
            raise ValueError('whole interval account mismatch')
        totals[i]['terminal_rule_adjustment']=official['scores'][i]-sim.cash[i]
    def record(i):
        return {'initial_cash':initial[i],'final_cash':sim.cash[i],'official_score':official['scores'][i],
                'cash_writes':dict(sorted(totals[i].items())),
                'own_carrier_banked':totals[i]['own_delivery'],'tree_absorption':totals[i]['tree_absorption'],
                'base_and_tree_collision_cash':totals[i]['base_collision_cash']+totals[i]['tree_collision_cash'],
                'paid_recruitment':-totals[i]['hire_collector']-totals[i]['hire_planter']}
    return {'focal':record(seat),'enemy':record(1-seat),'official_margin':official['margin_by_seat'][seat],
            'official_u':official['u_by_seat'][seat],'start_step':start.step,'end_step':world.step}


def contrast(base,new):
    delta={}
    for side in ('focal','enemy'):
        a,b=base[side],new[side]
        delta[side]={k:b[k]-a[k] for k in ('official_score','own_carrier_banked','tree_absorption',
                                           'base_and_tree_collision_cash','paid_recruitment')}
        ca,cb=a['cash_writes'],b['cash_writes']
        delta[side]['cash_writes']={k:cb.get(k,0.)-ca.get(k,0.) for k in sorted(ca.keys()|cb.keys())}
    margin=new['official_margin']-base['official_margin']
    if abs(delta['focal']['official_score']-delta['enemy']['official_score']-margin)>1e-7:
        raise ValueError('margin decomposition')
    delta['official_margin']=margin
    return delta


def main():
    if not (HERE/'result.json').is_file():
        raise RuntimeError('wait for complete independent source receipts')
    final=read(HERE/'result.json');maps=read(HERE/'map_declaration.json')
    bind=read(HERE/'binding.json'); package=read(HERE/'package.json')
    if final['package']!=package or final['training_sha256']!=sha(HERE/'update.json'):
        raise ValueError('final result does not match the trained update')
    outputs={'source_result_sha256':sha(HERE/'result.json'),'cash_scanner_sha256':sha(BASE/'turning_point_ledger.py'),
             'source_sha256':sha(Path(__file__)),'training':[],'development':[],
             'scope':'economy/raid proxy responses only; training figures are post-branch cash events. Multiple draws of an identical command share one realized deterministic continuation. No additional games.'}
    signed={(r['seed'],r['seat'],r['arm']):r for r in final['held']}
    for seed in maps['training']:
        for seat in (0,1):
            path=HERE/'training'/f'{seed}_{seat}.json.gz';log=read(path);r=log['root']
            if (r['seed'],r['seat'])!=(seed,seat) or r['binding']!=bind:
                raise ValueError('wrong training tape')
            source=ROOT/r['source']
            if sha(source)!=r['source_sha256']:
                raise ValueError('training prefix source changed')
            parent=read(source)
            start=World(seed)
            for cmds in parent['actions'][:r['t']]:start.advance(cmds)
            if start.step!=r['t'] or start.terrain_sha256!=r['terrain']:
                raise ValueError('branch prefix mismatch')
            branches={}
            for key,outcome in log['outcomes'].items():
                if outcome['actions'][0][seat]!=json.loads(key):
                    raise ValueError('wrong sampled execution')
                branches[key]=account(start,outcome['actions'],outcome['official'],seat)
            greedy=branches[r['greedy']]
            outputs['training'].append({'seed':seed,'terrain':r['terrain'],'seat':seat,'t':r['t'],'opponent':r['opponent'],
               'tape_sha256':sha(path),'draw_multiplicity':dict(Counter(r['draws'])),'greedy_key':r['greedy'],
               'cash_after_root_including_root':branches,'versus_root_greedy':{k:contrast(greedy,v) for k,v in branches.items()}})
    for seed in maps['new_proxy_development']:
        for seat in (0,1):
            pair={}
            for arm in ('parent','candidate'):
                path=HERE/'development'/f'{arm}_{seed}_{seat}.json.gz';r=read(path)
                if (r['seed'],r['seat'],r['arm'])!=(seed,seat,arm) or r['binding']!=bind or r['package']!=package or signed[seed,seat,arm]['official']!=r['official']:
                    raise ValueError('development tape differs from registered final result')
                pair[arm]=account(World(seed),r['actions'],r['official'],seat)
                pair[arm]['tape_sha256']=sha(path)
            outputs['development'].append({'seed':seed,'seat':seat,'opponent':r['opponent'],'arms':pair,
                'candidate_minus_parent':contrast(pair['parent'],pair['candidate'])})
    target=HERE/'return_accounts.json'
    payload=(json.dumps(outputs,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if target.exists() and target.read_bytes()!=payload:
        raise ValueError('immutable output changed')
    tmp=target.with_suffix('.partial');tmp.write_bytes(payload);tmp.replace(target)
    print('registered training roots',len(outputs['training']),'paired development map/seats',len(outputs['development']),'SHA256',sha(target))


if __name__=='__main__':
    main()
