"""Freeze affordable pay-vs-defer actions and public root features before either terminal-response arm."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from source_games import BASE,DIR,HERE,PLAN,ROOT,sha,trial
OPTION=HERE.parent/'planter_option'
sys.path.insert(0,str(BASE/'root_recovery_branches'))
import live
from branches import clean
sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon
sys.path.insert(0,str(BASE/'planter_routes_dev'))
from features import PlanterRouteState,route_value

SOURCE=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
MODEL=trial.read(OPTION/'value_weights.json')
MOVES=['NONE','UP','RIGHT','DOWN','LEFT']

def load(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())

def retain(path,x):
    if path.exists():
        if load(path)!=x:raise ValueError(f'immutable training artifact changed {path}')
    else:
        live.save(path,x)

def bind_capture(module):
    """Use the original parent selection, recording which of its existing decoding paths it used."""
    text=SOURCE.read_text()
    if sha(SOURCE)!=PLAN['parent_pack_sha256']:
        raise ValueError('changed v1')
    start=text.rindex('\ndef _mixed_actions(')
    select=text[start:]
    if select.count('return baseline')!=3 or select.count('return candidate')!=1:
        raise ValueError('unrecognized v1 decoder')
    select=select.replace('return baseline','return _capture(x,aux,enc,baseline,False)').replace('return candidate','return _capture(x,aux,enc,candidate,True)')
    def capture(x,aux,encoded,labels,mode):
        module._root=(x,aux,encoded,labels,mode)
        return labels
    module._capture=capture
    decoder=text[text.index('def _return_decode('):text.index('\n    return labels',text.index('def _return_decode('))+len('\n    return labels')]
    decoder=decoder.replace('_return_decode(x, aux, encoded, apply_residual)','_forced_decode(x, aux, encoded, apply_residual, target_slot, target_action)')
    a,b=decoder.index('        if not apply_residual:'),decoder.index('        labels[slot] = action')
    decoder=decoder[:a]+'''        if slot==target_slot:
            if slot>=10 or x['tok_kind'][slot]!=1 or not legal[target_action]:
                return None
            action=target_action
'''+decoder[b:]
    exec(compile(select+'\n'+decoder,'v1-training-root-decode','exec'),module.__dict__)


def route(upstream,item):
    """Compare complete legal joint commands. Ranking is independent of the opponent's simultaneous order."""
    s,t,uid=item['seat'],item['t'],item['uid']
    sim=FastCarbon(upstream['game_configuration']).reset(seed=upstream['seed'])
    net=live.module(SOURCE,'training_v1_root')
    bind_capture(net)
    market=PlanterRouteState()
    for k,acts in enumerate(upstream['actions'][:t+1]):
        obs=sim.obs(s)
        market.observe(obs,s,obs['carbon'],upstream['game_configuration'],k)
        if k==0 or k==t:
            base=clean(net.agent(obs,upstream['game_configuration']))
            if base!=clean(acts[s]):
                raise ValueError('parent diverges from the recorded actual command')
        else:
            net._FEAT.observe(obs)
        if k<t:
            sim.step(*acts)
    x,aux,enc,original,mode=net._root
    slot=next((i for i,v in enumerate(net._FEAT.own_uid[:10]) if v==uid),None)
    if slot is None or int(x['tok_kind'][slot])!=1:
        raise ValueError('selected worker missing from actual decision')
    current=base.get(uid,'NONE')
    if current!=item['parent_command']:
        raise ValueError('opportunity index disagrees with parent')
    if current!='NONE':
        ordered=['NONE']
    else:
        def key(cmd):
            a=item['actions'][cmd]
            feat=market.features(a['cell'])
            score=route_value(MODEL,feat[0])[2] if feat else float('-inf')
            return (a['enemy_collector_distance']<=1,a['enemy_planter_distance']<=1,-score,MOVES.index(cmd))
        ordered=sorted(MOVES[1:],key=key)
    for cmd in ordered:
        if item['actions'][cmd]['friendly_conflict']:
            continue
        physical=MOVES.index(cmd)
        F=sys.modules['cnet2_feat'];label=int(F.CANON_MOVE[physical]) if s else physical
        other=net._forced_decode(x,aux,enc,mode,slot,label)
        if other is None or np.flatnonzero(other!=original).tolist()!=[slot]:
            continue
        collisions=net._return_conflicts(x,other)
        if collisions-net._return_conflicts(x,original) or any(slot in p for p in collisions):
            continue
        actual=clean(net._FEAT.to_actions(other))
        if actual.get(uid,'NONE')!=cmd or {key for key in set(base)|set(actual) if actual.get(key)!=base.get(key)}!={uid}:
            raise ValueError('not an isolated physical planter command')
        return {'v1':base,'alternative':actual,'paid_arm':'parent' if current=='NONE' else 'alternative',
                'old':current,'new':cmd,'public_at_root':obs,'true_prefix_slot':slot,
                'candidate':item,'ranked_alternatives_in_order':ordered}
    return None


def ranking(row,map_i,window):
    s,t,uid=row['seat'],row['t'],row['uid']
    i=4*map_i+2*s+window
    targets=[(i>>shift)&1 for shift in (2,1,0)]
    prefer_stay=(map_i+s+window)%2==0
    mismatch=sum(a!=b for a,b in zip(row['stratum_bits'],targets))
    key=f"tree-pair-v1|{row['seed']}|{s}|{t}|{uid}"
    return (mismatch,(row['parent_command']=='NONE')!=prefer_stay,hashlib.sha256(key.encode()).hexdigest())


def select():
    references=trial.read(HERE/'source_receipt.json')
    ledger=HERE/'opportunities.jsonl.gz'
    records=[json.loads(z) for z in gzip.open(ledger,'rt')]
    pool=[];selected=[];empty=[]
    for g in references['games']:
        seed,s=g['seed'],g['seat'];m=PLAN['train_maps'].index(seed);raw=load(ROOT/g['raw_path'])
        for wi,(lo,hi) in enumerate(PLAN['windows']):
            rows=[r for r in records if (r['seed'],r['seat'])==(seed,s) and lo<=r['t']<=hi and
                  r['actions']['NONE']['payable_site'] and r['actions']['NONE']['can_pay_now'] and
                  not r['actions']['NONE']['friendly_conflict']]
            ranked=sorted(rows,key=lambda r:ranking(r,m,wi));cases=Counter(''.join(map(str,r['stratum_bits'])) for r in rows)
            used=None
            for i,row in enumerate(ranked):
                result=route(raw,row)
                if not result:
                    continue
                uid=row['uid'];t=row['t'];case=HERE/f'case_{seed}_{s}_{wi}'
                case.mkdir(exist_ok=True)
                retain(case/'root.json.gz',result)
                price=row['actions']['NONE']['price']
                used={'seed':seed,'seat':s,'window':wi,'step':t,'uid':uid,'cell':row['position'],'price':price,
                      'source':g['raw_path'],'source_sha256':g['raw_sha256'],'case_directory':case.name,
                      'root_sha256':sha(case/'root.json.gz'),'paid_arm':result['paid_arm'],'old':result['old'],'new':result['new'],
                      'source_stratum':list(row['stratum_bits']),'expected_stratum':[( (4*m+2*s+wi)>>b)&1 for b in (2,1,0)],
                      'rank_in_window':i+1,'rng_seed':813700+4*m+2*s+wi}
                selected.append(used)
                break
            pool.append({'seed':seed,'seat':s,'window':wi,'eligible_planter_roots':len(rows),'by_stratum':cases,'usable_pair':bool(used)})
            if used is None:
                empty.append([seed,s,wi])
    doc={'protocol_sha256':sha(HERE/'protocol.json'),'source_receipt_sha256':sha(HERE/'source_receipt.json'),
         'all_opportunities_sha256':sha(ledger),'covered_maps':PLAN['train_maps'],'windows':pool,'cases':selected,'empty_windows':empty,
         'opponent_future_or_terminal_used_in_selection':False,'selector_sha256':sha(Path(__file__))}
    retain(HERE/'selection.json',doc)
    print([(d['seed'],d['seat'],d['window'],d['step'],d['old'],d['new']) for d in selected],flush=True)

if __name__=='__main__':
    select()
