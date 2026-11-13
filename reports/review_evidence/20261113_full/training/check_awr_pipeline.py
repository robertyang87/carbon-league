"""AWR endpoint/weight probes. Synthetic policy functions, real generator/scorer/feature builder.
No optimizer, network training, cloud task or remote operation. Set CARBON_REPO and optional CARBON_REVIEW_OUT.
"""
import ast, contextlib, copy, gzip, io, json, os, subprocess, sys, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as Fn
repo=Path(os.environ['CARBON_REPO']).resolve();sys.path.insert(0,str(repo/'code'));torch.set_num_threads(1)
os.environ['BC_DEVICE']='cpu';os.environ['AWR_OPP_CASH']='0';os.environ.pop('AWR_FINAL',None);os.environ.pop('AWR_TAG',None)
import awr_gen as G
import awr_weights as W
import eval_v53 as E
import cnet2_build as B

def recruit(o,c,different=False):
    if o['step'] not in (0,1):return {}
    bases=list(o['players'][o['player']][1]);return {bases[o['step']%2 if different else 0]:'RECCOLLECTOR'}
def opponent(o,c):return recruit(o,c,True)
def noact(o,c):return {}
G.STEPS=299
result={'head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'authority':'Same two-step legal recruitment sequence as data/recheck_labels.py: original env ends at step2, rewards[-299,50], margin -349.',
        'authority_reused':True,'scope':'AWR generator runs unchanged 299-step loop with trivial deterministic agents; no real training or policy evaluation.'}
with tempfile.TemporaryDirectory(prefix='carbon-r5-awr-') as td:
    root=Path(td);corpus=root/'generated';(corpus/'replays').mkdir(parents=True)
    with patch.object(E,'load_agent_fn',lambda p:recruit if p.endswith('ours.py') else opponent):
        bad=G.play((str(corpus),'ours.py','opponent.py',0,0,0))
    with patch.object(E,'load_agent_fn',lambda p:noact if p.endswith('ours.py') else opponent):
        good=G.play((str(corpus),'ours.py','opponent.py',2,0,0))
    (corpus/'awr.jsonl').write_text('\n'.join(json.dumps(x) for x in (bad,good))+'\n')
    replay=json.load(gzip.open(corpus/bad['file']))['steps']
    scored=W.rescore(str(corpus))
    shard=root/'post_terminal.npz';task=(str(corpus/bad['file']),[0],0,[2],[bad['reward']],str(shard))
    build=B.build_game(task)
    with np.load(shard) as d:
        result['terminal_semantics']={'generator_reward':bad['reward'],'replay_states':len(replay),
            'last_step':replay[-1][0]['observation']['step'],'all_status_active':all(s[p]['status']=='ACTIVE' for s in replay for p in (0,1)),
            'scorer_reward2':scored[0]['reward2'],'misclassified_as_mirror':scored[0]['mirror'],
            'builder_return':list(build),'built_rows':len(d['step']),'post_terminal_rows':int((d['step']>=2).sum()),
            'built_value_unique':np.unique(d['value']).tolist(),'expected_official_margin':-349.0}
    # Replay-score distribution fixture, not a natural-run frequency estimate. The actual scorer uses only final state.
    # 40 seed groups of 5: all margins +100 except one -100, all nonmirror. No direct w injection.
    c=root/'weight_distribution';(c/'replays').mkdir(parents=True);ds=root/'ds';ds.mkdir();games=[]
    endobs=copy.deepcopy(replay[-1][0]['observation'])
    for i in range(200):
        o=copy.deepcopy(endobs);o['players'][0][0]=0. if i==0 else 100.;o['players'][1][0]=100. if i==0 else 0.
        key=f'awr_fixture_{i}';fn='replays/'+key+'.json.gz'
        # The other states are placeholders: this probe makes no legality claim about these fixture trajectories.
        with gzip.open(c/fn,'wt') as f:json.dump({'steps':[None]*299+[[{'observation':o}]]},f)
        games.append(dict(key=key,file=fn,seed=10000+i//5,k=i%5,side=0,reward=-100. if i==0 else 100.))
    (c/'awr.jsonl').write_text('\n'.join(json.dumps(g) for g in games)+'\n')
    (ds/'meta.json').write_text(json.dumps({'games':[{'key':g['key']} for g in games]}))
    np.save(ds/'gid.npy',np.arange(200,dtype=np.int32))
    kind=np.zeros((200,24),np.int8);mask=np.zeros((200,24),bool);mask[:,0]=True
    y=np.full((200,12),-1,np.int8);y[:,0]=0
    np.save(ds/'tok_kind.npy',kind);np.save(ds/'tok_mask.npy',mask);np.save(ds/'y.npy',y)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');env.pop('AWR_CAR_ANCHOR',None);env.pop('AWR_ESS_MIN',None);env.pop('AWR_WMAX',None)
    p=subprocess.run([sys.executable,str(repo/'code/awr_weights.py'),str(ds),str(c)],env=env,text=True,capture_output=True,check=True)
    target=root/'anchor'
    q=subprocess.run([sys.executable,str(repo/'code/awr_collector_anchor.py'),str(ds),str(target)],env=env,text=True,capture_output=True,check=True)
    w=torch.from_numpy(np.load(target/'w.npy'));uw=torch.from_numpy(np.load(target/'uw.npy'))
    tree=ast.parse((repo/'code/cnet2_train.py').read_text());outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_cached')
    fn=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='batch_loss')
    class FixedNet:
        def ctx_from(self,p,t):return p
        def decode(self,b,t,c):return torch.zeros((len(b['y']),12,5))
    ns=dict(torch=torch,Fn=Fn,net=FixedNet(),REF=None)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'real_cached_batch_loss','exec'),ns)
    batch=dict(w=w.half().float(),uw=uw.half().float(),y=torch.from_numpy(y).long(),teacher=torch.zeros(200,dtype=torch.long))
    loss,_=ns['batch_loss'](batch,None,None)
    result['weight_precision']={'actual_awr_weights_exit':p.returncode,'actual_awr_weights_stdout':p.stdout.strip(),
        'actual_anchor_exit':q.returncode,'actual_anchor_stdout':q.stdout.strip(),'min_w':float(w.min()),'max_w':float(w.max()),
        'max_uw':float(uw.max()),'raw_weights_finite':bool(torch.isfinite(w).all() and torch.isfinite(uw).all()),
        'cached_uw_finite':bool(torch.isfinite(batch['uw']).all()),'actual_cached_loss':str(float(loss)),
        'fixed_hyperparameters':'AWR_ESS_MIN=0.2, AWR_WMAX=20, anchor SCALE=1 (defaults)',
        'reachability_limit':'Actual scorer/weight/anchor/cached loss entry chain; constructed score distribution, not actual training-data min/max or validated replay trajectories.',
        'guarantee':'ESS and max clipping impose no positive lower bound on exp(A/beta).'}
text=json.dumps(result,ensure_ascii=False,indent=2)+'\n';print(text,end='')
if os.environ.get('CARBON_REVIEW_OUT'):
    out=Path(os.environ['CARBON_REVIEW_OUT']);out.mkdir(parents=True,exist_ok=True);(out/'awr-results.json').write_text(text)
