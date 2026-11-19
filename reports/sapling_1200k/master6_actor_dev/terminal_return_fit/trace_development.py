"""Trace frozen actors on both actual learned-agent development histories and check every emitted command."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np

from build import HERE, ROOT, PLAN, digest, source, save
import trial
sys.path.insert(0,str(ROOT/"code"))
from fastsim import FastCarbon


def attach_monitor(ns, original_source):
    start=original_source.index('def _return_decode(')
    end=original_source.index('    return labels\n',start)+len('    return labels\n')
    decoder=original_source[start:end]
    cue='        if apply_residual and slot < 10'
    record='''        prior_top = int(np.argmax(np.where(legal,combined,-np.inf)))
        expert_top, student_top = [int(np.argmax(np.where(legal,v,-np.inf))) for v in scores]
'''
    if decoder.count(cue)!=1 or decoder.count('        labels[slot] = action\n')!=1:
        raise ValueError('unrecognized frozen decoder')
    decoder=decoder.replace("    states = [list(entry)", "    _root_inputs[:] = [(x,aux)]\n    states = [list(entry)",1)
    decoder=decoder.replace(cue,record+cue).replace('        labels[slot] = action\n','''        _recorded[bool(apply_residual)][slot] = (expert_top,student_top,prior_top,action)
        labels[slot] = action
''')
    exec(compile(decoder,'instrumented_decoder.py','exec'),ns)


def run():
    snapshot=trial.read(HERE/'learned_checkpoint.json')
    if snapshot!=trial.compute_fit():
        raise ValueError('training identity does not reproduce')
    arm=snapshot['learned_pack']
    expected=trial.read_arm('learned',arm['pack_sha256'],trial.manifest_sha(arm))
    raw_src=source(snapshot['theta'])
    named=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
    if digest(raw_src)!=arm['pack_sha256'] or (named.exists() and named.read_bytes()!=raw_src):
        raise ValueError('frozen model package disagrees')
    ns={'__name__':'frozen_development_residual','__file__':'submission.py'}
    text=raw_src.decode()
    exec(compile(text,'submission.py','exec'),ns)
    ns['_root_inputs']=[]
    attach_monitor(ns,text)
    output=[]
    for item in expected:
        seat,seed=item['seat'],item['seed']
        p=HERE/'dev_learned'/f'raw_{seed}_{seat}.json.gz'
        record=json.loads(gzip.decompress(p.read_bytes()))
        sim=FastCarbon(record['game_configuration']).reset(seed=seed)
        counts=Counter();turns=[];indices=[]
        for t,joint in enumerate(record['actions']):
            obs=sim.obs(seat)
            ns['_recorded']={False:{},True:{}}
            issued=ns['agent'](obs,record['game_configuration'])
            archived={u:a for u,a in joint[seat].items() if a not in (None,'NONE')}
            if issued!=archived:
                raise ValueError(f"observer output disagrees with actual frozen package at {seat}/{t}")
            feat=ns['_FEAT'];actual=feat.labels(issued)
            base,maybe=ns['_recorded'][False],ns['_recorded'][True]
            if set(base)!=set(maybe):
                raise ValueError('candidate prefix incomplete')
            basepath=all(v[3]==actual[s] for s,v in base.items())
            candpath=all(v[3]==actual[s] for s,v in maybe.items())
            if not basepath and not candpath:raise ValueError('ambiguous emitted joint plan')
            path=base if basepath else maybe
            alt_changes=any(r[3]!=base[s][3] for s,r in maybe.items())
            counts['all_roots']+=1
            counts['safety_veto_roots']+=int(basepath and alt_changes)
            counts['residual_changed_roots']+=int(not basepath)
            previous=np.array(actual)
            for s,q in base.items():previous[s]=q[3]
            after=feat.to_actions(actual);before=feat.to_actions(previous)
            details=[]
            feature={}
            alias=ns['sys'].modules['cnet2_feat']
            if not basepath:
                x,aux=ns['_root_inputs'][0]
                prefix=alias.Prefix(x,aux)
                for token in sorted(path):
                    _,move,legal=prefix.feats(token)
                    if not legal[int(actual[token])]:
                        raise ValueError('actual command illegal under its own full prefix')
                    feature[token]=move
                    prefix.advance(token,int(actual[token]))
            for slot,(expert,student,mixture,command) in path.items():
                uid=feat.own_uid[slot]
                worker=obs['players'][seat][2].get(uid)
                role='base' if slot>=10 else worker[2].lower()
                counts['active_'+role]+=1
                if expert!=student:
                    key=('expert' if command==expert else 'sapling' if command==student else 'other')
                    counts['distinct_choices_'+role]+=1
                    counts['actual_'+key+'_'+role]+=1
                if int(actual[slot])!=base[slot][3]:
                    if role!='collector':
                        raise ValueError('protected role altered')
                    direct=(int(actual[slot])!=mixture)
                    counts['direct_residual_changes' if direct else 'downstream_prefix_changes']+=1
                    p0,load,_=worker
                    def situation(label,command):
                        raw_move=alias.MOVES.index(command)
                        f=feature[slot][label]
                        return {'landing':int(alias.DEST[p0,raw_move]),'home':bool(f[5]),
                            'danger_lighter_or_equal_enemy':bool(f[1]),'heavier_rival_in_reach':bool(f[2]),
                            'friendly_reservation':float(f[0]),'distance_home':int(round(float(f[10])*14)),
                            'tree_tax_feature':float(f[13])}
                    details.append({'parent_geometry_on_enacted_prefix':situation(base[slot][3],before.get(uid,'NONE')),
                                  'actual_geometry_on_enacted_prefix':situation(command,after.get(uid,'NONE')),'slot':slot,'uid':uid,'physical_cell':p0,'public_cargo':load,
                                  'parent':before.get(uid,'NONE'),'actual':after.get(uid,'NONE'),
                                  'source':'residual_on_enacted_prefix' if direct else 'earlier_collector_prefix',
                                  'expert_label_on_enacted_prefix':expert,'sapling_label_on_enacted_prefix':student,
                                  'mixed_label_before_residual_on_enacted_prefix':mixture,'actual_label':command})
            if bool(details)!=bool(not basepath):
                raise ValueError('change counter and returned commands disagree')
            if details:indices.append(t)
            turns.append({'t':t,'applied_change':bool(details),'veto':bool(basepath and alt_changes),'changes':details})
            sim.step(*joint)
        output.append({'seat':seat,'seed':seed,'raw_sha256':digest(p.read_bytes()),'counts':dict(counts),
                       'changed_turns':indices,'turns':turns})
    path=HERE/'development_actions.json.gz'
    data={'agent_sha256':arm['pack_sha256'],'observer_sha256':digest(Path(__file__).read_bytes()),
          'scope':'Every frozen learned-agent root on its own actual development trajectory. Parent and expert/student contrasts are on the same public root and actual prefix; no separate hypothetical future is identified.',
          'physical_commands':'NONE denotes omitted command; numeric slot label uses the actor canonical orientation.',
          'games':output}
    from build import atomic
    atomic(path,gzip.compress((json.dumps(data,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
    print([(g['seat'],g['counts']) for g in output],flush=True)


if __name__=='__main__':
    run()
