"""Freeze both sides of the rollout role assignment, then score saved original branch endings."""
from collections import Counter, defaultdict
from functools import partial
import gzip
import inspect
import json
from pathlib import Path
from statistics import mean
import sys
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
sys.path.insert(0,str(BRIDGE))
import bridge
import outcome_audit

ROLE_KEYS = ('hidden','bias','head')


def split_source():
    """Change only the two enemy decode sites in the old two-stream public simulation."""
    fn=inspect.getsource(bridge.play_public_pair)
    replacements=(
        ('def play_public_pair(cfg, obs, views, root_enemy_x, candidates, weights):',
         'def play_with_opponent(cfg, obs, views, root_enemy_x, candidates, weights, enemy_weights):'),
        ('decode(views[op], root_enemy_x, weights, jitter(',
         'decode(views[op], root_enemy_x, enemy_weights, jitter('),
        ('decode(track[s], x, weights, jitter(',
         'decode(track[s], x, weights if s == seat else enemy_weights, jitter('))
    for old,new in replacements:
        if fn.count(old)!=1:raise ValueError('ambiguous public rollout source: '+old)
        fn=fn.replace(old,new)
    return fn


def split_kernel():
    ns=dict(vars(bridge))
    exec(compile(split_source(),str(Path(__file__)), 'exec'),ns)
    return ns['play_with_opponent']


def weights(path):
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)==set(ROLE_KEYS)
        out={k:z[k] for k in ROLE_KEYS}
    m=out['hidden'].shape[1]
    assert out['hidden'].shape==(83,m) and out['bias'].shape==(m,) and out['head'].shape==(m,1)
    assert all(np.isfinite(a).all() for a in out.values())
    return out


def scoped_scores(roots):
    totals=Counter()
    for r in roots:
        for m in r['matches']:
            totals.update(m)
    return dict(totals)


def measure(command, actual, roles, loaded):
    assert command.keys()|actual.keys()<=roles.keys()
    def eq(u):return command.get(u,'NONE')==actual.get(u,'NONE')
    c={'joint_hit':int(all(eq(u) for u in roles)), 'joint_n':1}
    for role in ('collector','planter','base','loaded'):
        ids=loaded if role=='loaded' else [u for u,v in roles.items() if v==role]
        c[role+'_n']=len(ids);c[role+'_hit']=sum(eq(u) for u in ids)
    return c


def summarize(rows):
    by=defaultdict(list)
    for r in rows:by[r['terrain']].append(r)
    maps={}
    for g,v in sorted(by.items()):
        score={}
        for mode in ('old_h8','new_h8','parent','new_h1'):
            score[mode]=mean(r['arms'][r['picks'][mode]]['margin'] for r in v)
        maps[g]={'n':len(v),**score,'new_vs_parent':score['new_h8']-score['parent'],
                 'new_vs_old':score['new_h8']-score['old_h8']}
    return {'n':len(rows),'maps':maps,
            'new_vs_parent':mean(v['new_vs_parent'] for v in maps.values()),
            'new_vs_old':mean(v['new_vs_old'] for v in maps.values()),
            'positive_maps_vs_parent':sum(v['new_vs_parent']>1.e-7 for v in maps.values()),
            'positive_maps_vs_old':sum(v['new_vs_old']>1.e-7 for v in maps.values()),
            'branch_wdl':{a:dict(Counter(r['arms'][r['picks'][a]]['wdl'] for r in rows)) for a in ('old_h8','new_h8','parent')},
            'new_matches':scoped_scores(rows),
            'old_matches':scoped_scores([{'matches':r['old_matches']} for r in rows])}


def audit_roots(forms):
    previous=bridge.load(BRIDGE/'outcome_audit.json')
    assert previous['bridge_sha256']==bridge.sha(BRIDGE/'bridge.py')
    rows=previous['roots']
    registrations={str(d.relative_to(bridge.ROOT)) for recs in bridge.registered_cases().values() for t,idx,p,d in recs}
    assert len(rows)==64 and registrations=={r['case'] for r in rows}
    # Latest full pre-decision history may be reconstructed from the old recorded actions.
    # Root enemy and following actions/returns are absent from the function call to the search.
    with patch.object(bridge,'play_public_pair',partial(split_kernel(),enemy_weights=forms)):
        opened=outcome_audit.public_input(sorted(registrations))
    evidence=[]
    for old in rows:
        case=old['case'];root=bridge.load(bridge.ROOT/case/'root.json.gz')
        actual=old['actual_enemy'];roles=old['enemy_roles'];enemy=1-old['seat']
        public=root['public_at_root']
        assert public['step']==old['t'] and public['player']==old['seat']
        loaded=sorted(u for u,v in public['players'][enemy][2].items() if v[2]=='COLLECTOR' and v[1]>=100)
        assert all(roles[u]=='collector' for u in loaded)
        q=opened[case]
        first=[v['root_enemy_command'] for v in q['samples']]
        picks={h:('alternative' if q['mean_delta'][h]>0 else 'parent') for h in ('h8','h1')}
        for arm,receipt in old['arms'].items():
            signed=bridge.ROOT/case/('live_'+arm+'.json')
            assert bridge.sha(signed)==receipt['receipt_sha256']
            obj=bridge.load(signed);assert obj['action_sha256']==receipt['tape_sha256']
            assert obj['source_sha256']==outcome_audit.MASTER and obj['root_actions'][enemy]==actual
            assert abs(obj['margin']-receipt['margin'])<1.e-7
        evidence.append({'case':case,'terrain':old['terrain'],'batch':old['batch'],'seat':old['seat'],'step':old['t'],
                         'heavy_collectors':loaded,'actual_enemy':actual,'predicted_enemy':first,
                         'old_predicted_enemy':old['predicted_enemy'],'new_forecast_delta':q['mean_delta'],
                         'old_forecast_delta':old['forecast_delta'],'picks':{'new_h8':picks['h8'],'new_h1':picks['h1'],
                                                                      'old_h8':old['policies']['h8']['arm'],'parent':'parent'},
                         'arms':old['arms'],'matches':[measure(v,actual,roles,loaded) for v in first],
                         'old_matches':[measure(v,actual,roles,loaded) for v in old['predicted_enemy']]})
    return evidence


def verify_disjoint(roots, fitting):
    from hashlib import sha256
    terrains={r['terrain'] for r in roots}
    assert len(terrains)==16 and len(roots)==64
    arena={}
    for cohort,count in (('first_win_dev',8),('first_win_blend_dev',4)):
        f=bridge.ACTOR/cohort/'clearance.json';d=bridge.load(f);boards=set(d['physical_boards'].values())
        assert len(boards)==count and not boards.intersection(terrains|fitting)
        arena[cohort]={'n':len(boards),'receipt_sha256':bridge.sha(f),'private_terrain_set_sha256':sha256(json.dumps(sorted(boards)).encode()).hexdigest()}
    assert not terrains.intersection(fitting)
    return {'prior_native_sources':len(terrains),'fit_or_model_selection':len(fitting),
            'distinct_prior_native_cases':len(roots),'as_of_two_registered_new_cohorts':arena,'intersection':0}


def run(fitting):
    wt=HERE/'opponent.npz'
    oldpath=BRIDGE/'rollout.npz'
    evidence=audit_roots(weights(wt))
    departments={'all': summarize(evidence),'held':summarize([r for r in evidence if r['batch']>6]),
                 'discovery':summarize([r for r in evidence if r['batch']<=6])}
    assert len(departments['held']['maps'])==4 and departments['held']['n']==16
    total=departments['all'];old=total['old_matches'];new=total['new_matches']
    assert old['joint_hit']==0 and old['joint_n']==128 and old['planter_hit']==118 and old['planter_n']==240
    tests={k:new[k+'_hit']>old[k+'_hit'] for k in ('planter','loaded','joint')}
    req={'two_of_three_root_metrics_strictly_gain':sum(tests.values())>=2,
         'held_new_at_least_old':departments['held']['new_vs_old']>=-1.e-7,
         'held_positive_against_no_search_maps_at_least_two':departments['held']['positive_maps_vs_parent']>=2}
    fast=split_source()
    doc={'source_sha256':bridge.sha(Path(__file__)),'opponent_sha256':bridge.sha(wt),
         'old_own_and_opponent_sha256':bridge.sha(oldpath),'original_terminal_audit_sha256':bridge.sha(BRIDGE/'outcome_audit.json'),
         'rewritten_rollout_method_sha256':__import__('hashlib').sha256(fast.encode()).hexdigest(),
         'training_fit_receipt_sha256':bridge.sha(HERE/'fit.json'),'training_source_receipt_sha256':bridge.sha(HERE/'sources.json'),'protocol_sha256':bridge.sha(HERE/'protocol.json'),
         'separation':verify_disjoint(evidence,set(fitting)),'sections':departments,'criteria':req,'component_improved':tests,
         'pass':all(req.values()),'roots':evidence,'new_native_games':0,
         'unit':'Opportunity = exact role-specific command in each frozen CRN stream, so two dependent samples per root. 64 historical intervention roots in 16 previously examined maps. Final margin comparison is equally weighted by physical terrain.'}
    bridge.save(HERE/'transfer.json.gz',doc)
    return doc



def signed_inputs():
    sources=bridge.load(HERE/'sources.json')
    assert bridge.sha(HERE/'protocol.json')==sources['protocol_sha256']
    assert bridge.sha(BRIDGE/'rollout.npz')==sources['old_weights_sha256']
    boards={r['terrain'] for r in sources['records']}
    seeds={r['seed'] for r in sources['records']}
    assert bridge.load(HERE/'fit.json')['weights_sha256']==bridge.sha(HERE/'opponent.npz')
    assert len(boards)==len(seeds)==8
    assert len({r['terrain'] for r in sources['records'] if r['split']=='fit'})==6
    assert len({r['terrain'] for r in sources['records'] if r['split']=='rollout_hold'})==2
    for cohort in ('first_win_dev','first_win_blend_dev'):
        other=bridge.load(bridge.ACTOR/cohort/'clearance.json')['physical_boards']
        assert not (set(map(int,other)) & seeds or set(other.values()) & boards)
    regs=bridge.load(BRIDGE/'outcome_audit.json')['roots']
    assert not ({r['seed'] for r in regs} & seeds or {r['terrain'] for r in regs} & boards)
    return boards


if __name__=='__main__':
    run(signed_inputs())
