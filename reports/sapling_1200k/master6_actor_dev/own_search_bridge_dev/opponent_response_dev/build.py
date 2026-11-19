"""On passage of the precommitted historical threshold, bind the new opponent to both enemy decode sites."""
import argparse
import base64
from collections import Counter
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transfer

bridge=transfer.bridge
ORIGINAL=HERE.parent
DEFAULT=Path('/workspace/sapling-h8-response-rollout.py')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make(target):
    result=bridge.load(HERE/'transfer.json.gz')
    if not result['pass'] or not all(result['criteria'].values()):
        raise ValueError('registered transfer gate did not pass')
    assert result['protocol_sha256']==bridge.sha(HERE/'protocol.json')
    assert result['opponent_sha256']==bridge.sha(HERE/'opponent.npz')
    old=load(ORIGINAL/'build.py','signed_selfrollout_builder')
    release=bridge.load(ORIGINAL/'release.json')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'parent.py'
        assert old.make(path)==release
        src=path.read_text()
    registered_fn=old.functions((ORIGINAL/'bridge.py').read_text())['play_public_pair']
    count=src.count(registered_fn)
    if count!=1:raise ValueError('old embedded rollout unexpected')
    encoded=base64.b64encode((HERE/'opponent.npz').read_bytes()).decode()
    replacement='_OPP_WEIGHTS=dict(np.load(io.BytesIO(base64.b64decode('+repr(encoded)+')),allow_pickle=False))\n'+transfer.split_source().strip()
    src=src.replace(registered_fn,replacement)
    old_call='verdict=play_public_pair(cfg,view,views,_BR_X,cmds,_BR_WEIGHTS)'
    assert src.count(old_call)==1
    src=src.replace(old_call,'verdict=play_with_opponent(cfg,view,views,_BR_X,cmds,_BR_WEIGHTS,_OPP_WEIGHTS)')
    data=src.encode()
    compile(data,str(target),'exec');assert len(data)<=10_000_000
    if target.exists() and target.read_bytes()!=data:raise ValueError('occupied by other build')
    if not target.exists():target.write_bytes(data)
    out={'sha256':bridge.sha(target),'bytes':len(data),'old_signed_pack_sha256':release['sha256'],
         'actual_actor_parent':release['parent_sha256'],'opponent_sha256':result['opponent_sha256'],
         'own_rollout_sha256':result['old_own_and_opponent_sha256'],'build_sha256':bridge.sha(Path(__file__))}
    return out


def normalized(c):
    return {k:v for k,v in c.items() if v and v!='NONE'}


def verify(target):
    result=bridge.load(HERE/'transfer.json.gz')
    checks={'transfer_sha256':bridge.sha(HERE/'transfer.json.gz'),'source_sha256':bridge.sha(Path(__file__)),
            'original_games_opened':0,'transfer_pass':result['pass']}
    if not result['pass']:
        checks['status']='SKIPPED_AFTER_REGISTERED_TRANSFER_GATE'
        checks['failed_components']=[k for k,v in result['criteria'].items() if not v]
        bridge.save(HERE/'entry_check.json',checks);return checks
    entry=make(target)
    assert bridge.sha(HERE/'opponent.npz')==entry['opponent_sha256']
    own=transfer.weights(ORIGINAL/'rollout.npz'); opponent=transfer.weights(HERE/'opponent.npz')
    cand=load(target,'tested_response_public')
    for k in transfer.ROLE_KEYS:
        assert np.array_equal(own[k],cand._BR_WEIGHTS[k]) and np.array_equal(opponent[k],cand._OPP_WEIGHTS[k])
    frozen=load(ORIGINAL/'build.py','old_frozen_agent_rebuilder')
    with tempfile.TemporaryDirectory() as td:
        back=Path(td)/'old.py'; frozen.make(back);selfnet=load(back,'verified_h8_identity')
    fp=bridge.load(ORIGINAL/'qualification.json.gz')
    boards=transfer.signed_inputs() | {r['terrain'] for r in bridge.load(ORIGINAL/'outcome_audit.json')['roots']}
    for cohort in ('first_win_dev','first_win_blend_dev'):
        boards|=set(bridge.load(bridge.ACTOR/cohort/'clearance.json')['physical_boards'].values())
    roots=[];before_equal=0;witness=0
    for g in sorted(fp['cases'],key=lambda g:(g['seed'],g['seat'])):
        assert g['terrain'] not in boards
        available=[r for r in g['opportunities'] if r.get('status')=='completed']
        if not available:continue
        last=min(r['t'] for r in available)
        raw=bridge.load(bridge.ROOT/g['source']);s=g['seat']
        assert bridge.sha(bridge.ROOT/g['source'])==g['sha256']
        assert raw['initial_terrain_u8_sha256']==g['terrain']
        sim=bridge.FastCarbon(raw['game_configuration']).reset(seed=g['seed'])
        for t,joint in enumerate(raw['actions'][:last+1]):
            ob=sim.obs(s); old=selfnet.agent(ob,sim.config); cmd=cand.agent(ob,sim.config)
            rec=cand._BR_LAST;prev=selfnet._BR_LAST
            assert normalized(prev['base'])==normalized(rec['base'])
            if rec['status']=='completed':
                compared=rec['comparison'];views=[None,None];views[s]=cand._FEAT;views[1-s]=cand._BR_OP
                ours,both=[rec['base'],rec['alternate']],cmp_samples(compared)
                chk=transfer.split_kernel()(sim.config,ob,views,cand._BR_X,ours,own,opponent)
                assert cmp_samples(chk)==both
                selected=ours[int(chk['mean_delta']['h8']>0)]
                assert normalized(cmd)==normalized(selected)
                old_response=transfer.split_kernel()(sim.config,ob,views,cand._BR_X,ours,own,own)
                contrast=cmp_samples(old_response)!=both
                witness+=contrast
                roots.append({'seed':g['seed'],'seat':s,'t':t,'status':'completed','result':normalized(cmd),
                              'old_h8':normalized(old),'alternate_chosen':chk['mean_delta']['h8']>0,
                              'opponent_changed_forecast':contrast,'new_root_enemy':[v['root_enemy_command'] for v in chk['samples']]})
            else:
                assert normalized(cmd)==normalized(rec['base'])
                before_equal+=1
            sim.step(*joint)
    checks.update(release=entry,completed_probes=roots,passed_opponent_change_witness=witness,
                  unchanged_or_fallback_calls=before_equal)
    checks['status']='PASS' if witness and {r['seat'] for r in roots}=={0,1} else 'INCOMPLETE_NATURAL_ENTRY_COVERAGE'
    bridge.save(HERE/'entry_check.json',checks)
    return checks


def cmp_samples(d):
    return d['samples'],d['mean_delta']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT)
    a=p.parse_args();print(verify(a.out)['status'])
