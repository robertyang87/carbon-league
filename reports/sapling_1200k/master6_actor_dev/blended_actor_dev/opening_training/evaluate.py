"""Run the new first-turn head and its off-switch against archived same-history commands."""
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

import build

HERE=Path(__file__).resolve().parent
SERVING=HERE.parent/'serving'
CACHE=Path('/workspace/sapling-all-top5-20261119/replays')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def agent(name):
    spec=importlib.util.spec_from_file_location('opening_normal_'+name,build.MATERIALIZED)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if name=='disabled':
        m._OPENING_ENABLED=False
    return m


def public(frame,seat):
    shared=frame[0]['observation']
    return dict(step=shared['step'],trees=shared['trees'],players=shared['players'],player=seat,carbon=frame[seat]['observation']['carbon'])


def clean(order):
    return {k:v for k,v in order.items() if v and v!='NONE'}


def pair(commands,bases):
    return [commands.get(b['uid'],'NONE') for b in bases]


def run():
    plan=json.loads((HERE/'protocol.json').read_bytes());fit=json.loads((HERE/'fit.json').read_bytes())
    info=json.loads(gzip.decompress((HERE/'panel.json.gz').read_bytes()))
    byid={g['id']:g for g in info['games']}
    prior=collections.Counter(tuple(pair(g['action_by_uid'],g['bases'])) for g in info['games'] if g['split']=='train')
    majority=sorted(prior,key=lambda k:(-prior[k],k))[0]
    release=build.build()
    models={k:agent(k) for k in ('enabled','disabled')}
    result=[];ns=0;nsdiff=0;t0off=0;program=sha(Path(__file__).read_bytes())
    for manifest in ('both','exposed_loss'):
        prev=json.loads((SERVING/f'probe_{manifest}.json').read_bytes())
        assert prev['quantized_packages']['packages']['candidate']['deployed_sha256']==plan['outside_step_zero_reference']
        raw=(SERVING/f'probe_{manifest}_roots.json.gz').read_bytes();assert sha(raw)==prev['roots_sha256']
        for g in json.loads(gzip.decompress(raw)):
            replay_raw=(CACHE/(g['id']+'.json.gz')).read_bytes();assert sha(replay_raw)==g['source_sha256']
            card=byid[g['id']];seat=g['seat'];assert card['seat']==seat and card['terrain_sha256']==g['terrain']
            replay=json.loads(gzip.decompress(replay_raw));frames=replay['steps']
            wanted={r['t']:r for r in g['roots']}; tracing=[];roots=[]
            for t in range(max(wanted)+1):
                obs=public(frames[t],seat)
                if t not in wanted:
                    for m in models.values(): m._FEAT.observe(obs)
                    continue
                given=wanted[t];demo=clean(frames[t+1][seat]['action'] or {})
                assert demo==given['demonstration_command']
                orig=given['arms']['candidate']['actual_command']
                after=clean(models['enabled'].agent(obs,replay['configuration']))
                off=clean(models['disabled'].agent(obs,replay['configuration']))
                if t==0:
                    t0off+=int(off!=orig)
                    modes={'trained':after,'disabled':off,'active':orig,'earlier_actor':given['arms']['active']['actual_command'],
                           'standalone_ce_control':given['arms']['control']['actual_command'],'demonstration':demo}
                    assert clean(card['action_by_uid'])==demo
                    roots={k:pair(v,card['bases']) for k,v in modes.items()}
                    # Rebuild the legal prefix including provisional recruitment charges.
                    m=models['enabled'];feat=m._FEAT;F=sys.modules['cnet2_feat']
                    fresh=F.Feat(seat);x=fresh.observe(obs)
                    for arm,commands in modes.items():
                        prefix=F.Prefix(x,fresh.aux);y=fresh.labels(commands)
                        assert fresh.to_actions(y)==commands
                        for slot in np.flatnonzero(x['tok_mask'][:F.OWN]):
                            assert prefix.feats(int(slot))[2][y[slot]], (g['id'],arm,slot)
                            prefix.advance(int(slot),int(y[slot]))
                else:
                    ns+=1;nsdiff+=int(after!=orig or off!=orig)
                    tracing.append(dict(t=t,unchanged_enabled=after==orig,unchanged_disabled=off==orig))
            result.append({k:g[k] for k in ('id','terrain','seat','split','opponent','wdl')}|
                          {'opening':roots,'nonopening_same_history':tracing})
    groups={}
    for split in ('val','holdout','diagnostic_seen_loss'):
        selected=[g for g in result if g['split']==split];summary={}
        for a in ('trained','disabled','active','earlier_actor','standalone_ce_control','train_majority'):
            p=[list(majority) if a=='train_majority' else g['opening'][a] for g in selected]
            y=[g['opening']['demonstration'] for g in selected]
            summary[a]={'maps':len(y),'exact':sum(v==truth for v,truth in zip(p,y)),
                        'composition':sum(sorted(v)==sorted(truth) for v,truth in zip(p,y)),
                        'orders':dict(collections.Counter('/'.join(v) for v in p))}
        groups[split]=summary
    check=plan['gate'];per={}
    for s in ('val','holdout'):
        a=groups[s];n=a['trained']['maps']
        per[s]={'composition':a['trained']['composition']/n>=check['each_panel_composition_match_minimum'],
                'joint':a['trained']['exact']/n>=check['each_panel_exact_joint_match_minimum'],
                'gain':(a['trained']['exact']-a['active']['exact'])/n>=check['each_panel_exact_joint_gain_over_active_minimum']}
    seats={k:sum(g['opening']['trained']!=g['opening']['active'] for g in result if g['seat']==k and g['split'] in ('val','holdout')) for k in (0,1)}
    passed=(all(all(v.values()) for v in per.values()) and nsdiff==0 and t0off==0 and all(seats.values()))
    out={'protocol_sha256':sha((HERE/'protocol.json').read_bytes()),'source_sha256':program,'release':release,'groups':groups,
         'train_ordered_pair_counts':{'/'.join(k):v for k,v in prior.items()},'off_at_t0_changed':t0off,
         'nonopening_probes':ns,'nonopening_differences_vs_frozen':nsdiff,'changed_new_by_seat':seats,
         'gate_components':per,'decision':'PASS' if passed else 'STOP','games':result}
    path=HERE/'normal_entry.json.gz';path.write_bytes(gzip.compress(json.dumps(out,ensure_ascii=False,sort_keys=True,allow_nan=False).encode(),mtime=0))
    print({k:out[k] for k in ('groups','off_at_t0_changed','nonopening_probes','nonopening_differences_vs_frozen','gate_components','decision')},flush=True)


if __name__=='__main__':
    run()
