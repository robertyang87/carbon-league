#!/usr/bin/env python3
"""S1 R2 read-only gate protocol probes. Actual AST-extracted repository functions;
synthetic teacher/terminal tables and workers only. No weights/training/remote jobs.
Run: python probe_method_r2.py --repo /path/to/frozen/repo --out /tmp/evidence
"""
import argparse, ast, contextlib, copy, gzip, hashlib, io, json, math, os
from pathlib import Path
from statistics import NormalDist
import sys, tempfile, time, types
from collections import Counter
import numpy as np

def load_functions(path):
    wanted={'md5_of','game_id_of','sample_scene_specs','_q_scene','_anchor_task',
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

def run_main(ns,mode,specs,tau=1.,pilot=False,pilot_artifact=None,deltas=None):
    with tempfile.TemporaryDirectory(prefix='s1r2-method-') as td:
        p=Path(td);(p/'weights.npz').write_bytes(b'deterministic-no-model-fixture')
        games=sorted({s['game'] for s in specs})
        (p/'split.json').write_text(json.dumps({mode:games}))
        (p/'ds.txt').write_text('training_game_not_in_fixture\n')
        ns['list_replays']=lambda *a:games
        original_sampler=ns['sample_scene_specs'];ns['sample_scene_specs']=lambda *a,**kw:copy.deepcopy(specs)
        ns['_du_scene']=lambda x:None
        def workers(fn,tasks,workers):
            if fn in (ns['_q_scene'],ns['_anchor_task']):return [fn(t) for t in tasks]
            return [dict(s,du=float(d),u_guided=.5+float(d)/2,u_greedy=.5-float(d)/2,
                         teacher_calls=1,teacher_ms=1.) for (s,_),d in zip(tasks,deltas)]
        ns['run_tasks']=workers
        argv=['gate','--weights',str(p/'weights.npz'),'--replays',td,'--split-manifest',str(p/'split.json'),
              '--ds-games-file',str(p/'ds.txt'),'--train-data-version','fixture-v1','--mode',mode,
              '--behavior-temp',str(tau),'--out',str(p/'result.json')]
        if pilot:argv+=['--pilot']
        if pilot_artifact:
            (p/'pilot.json').write_text(json.dumps(pilot_artifact));argv+=['--pilot-artifact',str(p/'pilot.json')]
        oldargv=sys.argv;sys.argv=argv
        try:
            with contextlib.redirect_stdout(io.StringIO()):rc=ns['main']()
        finally:sys.argv=oldargv;ns['sample_scene_specs']=original_sampler
        result=json.loads((p/'result.json').read_text())
        return {'return_code':rc,'formal':result['prereg']['formal'],
                'summary':result['q_gate' if mode=='q' else 'du_gate']}

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
    calls.clear();tempered=run_main(ns,'q',specs,tau=2.)
    assert tempered['summary']['verdict']=='PASS'
    assert tempered['summary']['q_soft_gain_mean']<0<tempered['summary']['q_soft_gain_behavior_mean']
    assert len(calls)==30*16

    # Original cluster example: real Stage A output locks 50 games, then
    # disjoint Stage B has 50 games with three correlated roots each.
    pilot_specs=specs_for('pilot',30)
    pilot_ds=np.repeat([.035,.045]*15,3)
    pilot=run_main(ns,'du',pilot_specs,pilot=True,deltas=pilot_ds)
    artifact=pilot['summary']['pilot_artifact'];assert artifact['locked_N_games']==50
    deltas=np.repeat([.24]*25+[-.16]*25,3)
    cluster=run_main(ns,'du',specs,pilot_artifact=artifact,deltas=deltas)
    assert cluster['summary']['verdict']=='FAIL' and cluster['summary']['du_ci95_t_cluster'][0]<0

    # Actual default anchor selection has unequal roots per source game.
    # Multiplicity is held fixed before evaluation in this deterministic fixture.
    selected=np.random.default_rng(995310).choice(len(specs),30,replace=False)
    multiplicity=Counter(specs[int(i)]['game'] for i in selected)
    def true_us(game):
        return [.75,.95] if multiplicity.get(game,0)>1 else [.90,.78]
    teacher2,calls2=attach_teacher(ns,[.80,.83],[.55,.45],true_us)
    anchor_weighting=run_main(ns,'q',specs)
    anch=anchor_weighting['summary']['anchor']
    assert anchor_weighting['summary']['verdict']=='PASS'
    assert anch['hard']['root_mean']>0>anch['hard']['game_mean']
    assert anch['soft']['root_mean']>0>anch['soft']['game_mean']

    # True all-game soft positive/negative cases for baseline control.
    attach_teacher(ns,[.80,.83],[.55,.45],lambda g:[.80,.83])
    positive=run_main(ns,'q',specs);assert positive['summary']['verdict']=='PASS'
    attach_teacher(ns,[.80,.83],[.55,.45],lambda g:[.83,.80])
    negative_anchor=run_main(ns,'q',specs);assert negative_anchor['summary']['verdict']=='FAIL'

    # Original sampler runs over virtual full-length replay files.
    payload=json.dumps({'steps':[[{},{}] for _ in range(300)]})
    ns['gzip']=types.SimpleNamespace(open=lambda path:io.StringIO(payload))
    sampled=ns['sample_scene_specs']([f'/fixture/g_{i:05d}.json.gz' for i in range(10000)],ns['LAYERS_DU'],995300,opening_share=.1)
    nopen=sum(r['step']<40 for r in sampled)
    assert abs(nopen/len(sampled)-.1)<1e-4

    powers=[]
    for sigma in [.01,.10,.36]:
        raw=math.ceil((140*sigma)**2);n=max(50,min(200,raw));se=sigma/math.sqrt(n)
        reported=NormalDist().cdf(.02/se-1.96)
        actual_approx=NormalDist().cdf((.02-max(.02,1.96*se))/se)
        powers.append({'sigma_game':sigma,'N_raw':raw,'N_locked':n,'true_effect':.02,
                       'reported_significance_power':reported,'composite_gate_power_normal_approx':actual_approx})

    evidence={'reviewed_ref':'7b025ae58f83250265e75a5cccf274f63ca7d0a3',
        'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'scope':'Actual _q_scene/_anchor_task/main/samplers/statistics; teacher and real-terminal outcome tables are explicit deterministic stubs; no real model quality claim.',
        'original_negative_soft_tau1':old_negative,
        'raw_training_q_negative_tau2_behavior_positive':tempered,
        'all_arm_anchor_check':{'calls_per_run':30*16,'n_anchor':30,'n_arms':16,'all_candidates_continued':True},
        'original_cluster_example_after_real_pilot_artifact':cluster,
        'pilot_fixture':{'n_games':30,'locked_N_games':50,'same_code_weights_config':True,'games_disjoint_from_confirmation':True},
        'anchor_point_estimate_counterexample':{'actual_main':anchor_weighting,
            'anchor_roots':30,'anchor_unique_games':len(multiplicity),
            'source_game_multiplicity':dict(multiplicity),
            'terminal_rule':'For selected games with multiplicity>1, U_g=.75,U_q=.95; for singleton games U_g=.90,U_q=.78; teacher q selects arm1, all U in normal win range.',
            'soft_weight':float(teacher2['q'][1])},
        'positive_control':positive,'negative_terminal_anchor_control':negative_anchor,
        'opening_full_length':{'n_games':10000,'n_roots':len(sampled),'n_opening':nopen,'share':nopen/len(sampled)},
        'power_semantics':powers}
    (out/'results.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'tau1':old_negative['summary']['verdict'],'tau2':tempered['summary']['verdict'],
                      'cluster':cluster['summary']['verdict'],'anchor_weighting':anchor_weighting['summary']['verdict'],
                      'anchor_root_soft':anch['soft']['root_mean'],'anchor_game_soft':anch['soft']['game_mean'],
                      'opening_share':nopen/len(sampled),'output':str(out/'results.json')},indent=2))

if __name__=='__main__':main()
