#!/usr/bin/env python3
"""S1 R4 read-only gate protocol probes. Actual AST-extracted repository functions;
synthetic teacher/terminal tables and workers only. No weights/training/remote jobs.
Run: python probe_method_r4.py --repo /path/to/frozen/repo --out /tmp/evidence
"""
import argparse, ast, contextlib, copy, gzip, hashlib, io, json, math, os
from pathlib import Path
from statistics import NormalDist
import sys, tempfile, time, types
from collections import Counter
import numpy as np

def load_functions(path):
    wanted={'md5_of','game_id_of','list_replays','sample_scene_specs','_q_scene','_anchor_task',
            't_ci','_t_crit','bootstrap_ci','cluster_stats','cluster_bootstrap_ci',
            '_sanitize','_atomic_dump','_verify_pilot_artifact','main'}
    constants={'CANON','LAYERS_Q','LAYERS_DU','DU_OPENING_SHARE','Q_PASS','Q_FAIL','DU_PASS','ANCHOR_N'}
    tree=ast.parse(path.read_text())
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted
          or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)]
    ns=dict(globals(),HERE=str(path.parent),_W={})
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns)
    return ns

def attach_teacher(ns,Q,P,terminal_by_game=None):
    Q=np.asarray(Q,float);P=np.asarray(P,float)
    q=np.sqrt(P)*np.exp((Q-Q.max())/.05);q/=q.sum()
    arms=[np.array([i],dtype=np.int64) for i in range(len(q))]
    teacher={'q':q,'Q_mean':Q,'arms':arms,'entropy':float(-(q*np.log(q)).sum()),
             'agree_greedy':int(np.argmax(q))==0,'wall_ms':1.,'guard':{}}
    ns['T']=types.SimpleNamespace(evaluate_root=lambda *a,**kw:teacher,
                                 TEACHER_GUARD_CFG={'GUARD':True,'GUARD_NARROW':True,'GUARD_CASH':40.,'LOWW':2})
    ns['extract_scene']=lambda s:dict(s,x_me=None,aux_me=None,x_opp=None,aux_opp=None,fm=None,fo=None)
    ns['_fresh_sim']=lambda s:None;ns['_ns']=lambda f:None;ns['_W']['pol']=None
    calls=[]
    def continuation(pol,scene,arm,seed,teacher):
        calls.append({'game':scene['game'],'arm':int(arm[0]),'seed':seed,'style':teacher})
        values=Q if terminal_by_game is None else terminal_by_game(scene['game'])
        return float(values[int(arm[0])])
    ns['continue_game']=continuation
    return teacher,calls

def run_main(ns,mode,specs,tau=1.,pilot=False,pilot_artifact=None,deltas=None,train_temp=None,replay_lengths=None):
    with tempfile.TemporaryDirectory(prefix='s1r4-method-') as td:
        p=Path(td);(p/'weights.npz').write_bytes(b'deterministic-no-model-fixture')
        games=sorted({s['game'] for s in specs})
        (p/'split.json').write_text(json.dumps({mode:games}))
        (p/'ds.txt').write_text('training_game_not_in_fixture\n')
        original_list=ns['list_replays'];original_sampler=ns['sample_scene_specs']
        if replay_lengths is None:
            ns['list_replays']=lambda *a:games
            ns['sample_scene_specs']=lambda *a,**kw:copy.deepcopy(specs)
        else:
            for game,n in replay_lengths.items():
                with gzip.open(p/(game+'.json.gz'),'wt') as fh:json.dump({'steps':[[{},{}] for _ in range(n)]},fh)
        ns['_du_scene']=lambda x:None
        def workers(fn,tasks,workers):
            if fn in (ns['_q_scene'],ns['_anchor_task']):return [fn(t) for t in tasks]
            return [dict(s,du=float(d),u_guided=.5+float(d)/2,u_greedy=.5-float(d)/2,
                         teacher_calls=1,teacher_ms=1.) for (s,_),d in zip(tasks,deltas)]
        ns['run_tasks']=workers
        argv=['gate','--weights',str(p/'weights.npz'),'--replays',td,'--split-manifest',str(p/'split.json'),
              '--ds-games-file',str(p/'ds.txt'),'--train-data-version','fixture-v1','--mode',mode,
              '--behavior-temp',str(tau),'--out',str(p/'result.json')]
        if train_temp is not None:argv+=['--train-temp',str(train_temp)]
        if pilot:argv+=['--pilot']
        if pilot_artifact:
            (p/'pilot.json').write_text(json.dumps(pilot_artifact));argv+=['--pilot-artifact',str(p/'pilot.json')]
        oldargv=sys.argv;sys.argv=argv
        try:
            with contextlib.redirect_stdout(io.StringIO()):rc=ns['main']()
        finally:sys.argv=oldargv;ns['sample_scene_specs']=original_sampler;ns['list_replays']=original_list
        result=json.loads((p/'result.json').read_text())
        return {'return_code':rc,'formal':result['prereg']['formal'],
                'prereg':{k:result['prereg'][k] for k in ('train_temp','behavior_temp','certified_target','n_scenes')},
                'summary':result['q_gate' if mode=='q' else 'du_gate'],
                'derived_scene_game_soft':ns['cluster_stats'](result['rows'],'soft_gain_train') if mode=='q' else None}

def specs_for(prefix,n):
    return [{'game':f'{prefix}_{i:03d}','step':step,'me':0} for i in range(n) for step in (50,150,220)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();src=Path(a.repo)/'code/stage1_teacher_gate.py';out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ns=load_functions(src);specs=specs_for('confirm',50)
    teacher,calls=attach_teacher(ns,[.80,.83]+[.76]*14,[.10,.09]+[.81/14]*14)
    old_negative=run_main(ns,'q',specs,tau=1.)
    assert old_negative['summary']['verdict']=='FAIL' and old_negative['return_code']==1
    assert len(calls)==30*16
    calls.clear();behavior_only=run_main(ns,'q',specs,tau=2.)
    assert behavior_only['summary']['verdict']=='FAIL'
    assert behavior_only['summary']['q_soft_gain_mean']<0<behavior_only['summary']['q_soft_gain_behavior_mean']
    assert behavior_only['prereg']['train_temp']==1. and len(calls)==30*16
    calls.clear();training_tempered=run_main(ns,'q',specs,tau=2.,train_temp=2.)
    assert training_tempered['summary']['verdict']=='PASS'
    assert training_tempered['summary']['q_soft_gain_train_mean']>0
    assert training_tempered['summary']['anchor']['soft']['game_mean']>0
    assert training_tempered['prereg']['train_temp']==2. and len(calls)==30*16

    pilot_specs=specs_for('pilot',30);pilot_ds=np.repeat([.035,.045]*15,3)
    pilot=run_main(ns,'du',pilot_specs,pilot=True,deltas=pilot_ds)
    artifact=pilot['summary']['pilot_artifact'];assert artifact['locked_N_games']==50 and artifact['grade']=='formal'
    deltas=np.repeat([.24]*25+[-.16]*25,3)
    cluster=run_main(ns,'du',specs,pilot_artifact=artifact,deltas=deltas)
    assert cluster['summary']['verdict']=='FAIL' and cluster['summary']['du_ci95_t_cluster'][0]<0

    selected=np.random.default_rng(995310).choice(len(specs),30,replace=False)
    multiplicity=Counter(specs[int(i)]['game'] for i in selected)
    def true_us(game):return [.75,.95] if multiplicity.get(game,0)>1 else [.90,.78]
    teacher2,calls2=attach_teacher(ns,[.80,.83],[.55,.45],true_us)
    anchor_weighting=run_main(ns,'q',specs);anch=anchor_weighting['summary']['anchor']
    assert anchor_weighting['summary']['verdict']=='INDETERMINATE'
    assert anch['hard']['root_mean']>0>anch['hard']['game_mean']
    assert anch['soft']['root_mean']>0>anch['soft']['game_mean']
    attach_teacher(ns,[.80,.83],[.55,.45],lambda g:[.80,.83])
    positive=run_main(ns,'q',specs);assert positive['summary']['verdict']=='PASS'
    attach_teacher(ns,[.80,.83],[.55,.45],lambda g:[.83,.80])
    negative_anchor=run_main(ns,'q',specs);assert negative_anchor['summary']['verdict']=='FAIL'

    # R21 residual closure: unequal root counts generated by the real sampler from
    # allowed short and full-length replay shells. Teacher q and its scores are
    # generated by the actual formula per source game. Independent terminal
    # continuations are deliberately positive in ALL games, isolating the scene
    # soft-gain aggregation used by the verdict.
    def table(qs,ps):
        Q=np.array(qs);P=np.array(ps);q=np.sqrt(P)*np.exp((Q-Q.max())/.05);q/=q.sum()
        return {'q':q,'Q_mean':Q,'arms':[np.array([i]) for i in range(len(q))],
                'entropy':float(-(q*np.log(q)).sum()),'agree_greedy':False,'wall_ms':1.,'guard':{}}
    tp=table([.80,.84],[.55,.45]);tn=table([.80,.83]+[.76]*14,[.10,.09]+[.81/14]*14)
    lengths={**{f'pos_{i:03d}':300 for i in range(25)},**{f'neg_{i:03d}':90 for i in range(75)}}
    ns['T']=types.SimpleNamespace(evaluate_root=lambda sim,me,pol,x,*args,**kw:tp if x.startswith('pos') else tn,
                                 TEACHER_GUARD_CFG={'GUARD':True,'GUARD_NARROW':True,'GUARD_CASH':40.,'LOWW':2})
    ns['extract_scene']=lambda s:dict(s,x_me=s['game'],aux_me=None,x_opp=None,aux_opp=None,fm=None,fo=None)
    ns['_fresh_sim']=lambda s:None;ns['_ns']=lambda f:None
    actual_calls=[]
    def real_outcome_table(pol,scene,arm,seed,teacher):
        actual_calls.append((scene['game'],scene['step'],int(arm[0])))
        return .75 if int(arm[0])==0 else .85
    ns['continue_game']=real_outcome_table
    mixed_specs=[{'game':g,'step':50,'me':0} for g in lengths]
    residual=run_main(ns,'q',mixed_specs,replay_lengths=lengths)
    assert residual['summary']['n']==150 and residual['summary']['n_games']==100
    assert residual['summary']['q_soft_gain_train_root_mean'] > 0 > residual['summary']['q_soft_gain_train_mean']
    assert residual['summary']['q_soft_gain_train_mean'] == residual['derived_scene_game_soft']['game_mean']
    assert residual['summary']['q_soft_gain_train_root_mean'] == residual['derived_scene_game_soft']['root_mean']
    assert residual['summary']['anchor']['soft']['game_mean']>0
    assert residual['summary']['verdict']=='FAIL' and residual['return_code']==1
    independently_weighted=(25*float(tp['q']@tp['Q_mean']-.8)+75*float(tn['q']@tn['Q_mean']-.8))/100
    assert abs(independently_weighted-residual['derived_scene_game_soft']['game_mean'])<1e-12

    # Original sampler full-length opening fraction remains unchanged.
    saved_gzip=ns['gzip'];payload=json.dumps({'steps':[[{},{}] for _ in range(300)]})
    ns['gzip']=types.SimpleNamespace(open=lambda path:io.StringIO(payload))
    sampled=ns['sample_scene_specs']([f'/fixture/g_{i:05d}.json.gz' for i in range(10000)],ns['LAYERS_DU'],995300,opening_share=.1)
    ns['gzip']=saved_gzip;nopen=sum(r['step']<40 for r in sampled);assert abs(nopen/len(sampled)-.1)<1e-4

    # Execute real pilot generation to check both named power fields, not just
    # recalculate an external formula. They are explicitly normal approximations.
    powers=[]
    for sigma in (.01,.1,.36):
        ds=np.repeat(.03+np.tile([-1.,1.],15)*sigma*math.sqrt(29/30),3)
        r=run_main(ns,'du',pilot_specs,pilot=True,deltas=ds)
        art=r['summary']['pilot_artifact'];s=art['sigma_game'];n=art['locked_N_games'];se=s/math.sqrt(n)
        sig=NormalDist().cdf(.02/se-1.96);conj=NormalDist().cdf((.02-max(.02,1.96*se))/se)
        assert abs(art['power_significance_only_at_locked']-sig)<1e-12
        assert abs(art['power_conjunction_approx_at_locked']-conj)<1e-12
        powers.append({k:art[k] for k in ('sigma_game','locked_N_games','power_significance_only_at_locked','power_conjunction_approx_at_locked','power_note')})
    import subprocess
    evidence={'reviewed_ref':subprocess.check_output(['git','-C',a.repo,'rev-parse','HEAD'],text=True).strip(),
      'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
      'scope':'Actual AST-extracted _q_scene/_anchor_task/main/statistics; replay-length residual uses original list_replays and sampler; teacher and terminal tables are explicit fixtures. No real model-quality claim.',
      'original_tau1_negative':old_negative,'behavior_tau2_default_training':behavior_only,
      'declared_train_temp2':training_tempered,'all_arm_anchor_check':{'n_anchor':30,'n_arms':16,'calls_each_temperature_run':480},
      'original_cluster_case':cluster,'original_anchor_point_estimate_case':anchor_weighting,
      'positive_control':positive,'negative_anchor_control':negative_anchor,
      'closed_scene_soft_aggregation':{'actual_main':residual,'corpus':{'positive_games':25,'positive_steps':300,'positive_roots_each':3,'negative_games':75,'negative_steps':90,'negative_roots_each':1},
        'positive_teacher':{k:tp[k].tolist() for k in ('q','Q_mean')},'negative_teacher':{k:tn[k].tolist() for k in ('q','Q_mean')},
        'independent_terminal_table':'For every game, U_g=.75 and U_other=.85; positive anchors isolate the soft scene-score criterion.',
        'exact_game_mean_from_corpus':independently_weighted,'actual_anchor_continuations':len(actual_calls)},
      'opening_fraction':{'n_games':10000,'n_roots':len(sampled),'n_opening':nopen,'share':nopen/len(sampled)},'power_fields_actual_pilots':powers}
    (out/'results.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'old_tau1':old_negative['summary']['verdict'],'behavior_tau2':behavior_only['summary']['verdict'],'training_tau2':training_tempered['summary']['verdict'],'old_anchor':anchor_weighting['summary']['verdict'],'closed_scene_soft':residual['summary']['verdict'],'soft_root_mean':residual['summary']['q_soft_gain_train_root_mean'],'soft_game_mean':residual['derived_scene_game_soft']['game_mean'],'output':str(out/'results.json')},indent=2))

if __name__=='__main__':main()
