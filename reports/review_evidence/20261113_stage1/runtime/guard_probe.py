"""Portable guard extraction/driver probes with deterministic tiny-network fixtures.
Set CARBON_REPO, CARBON_REVIEW_OUT; business source is never modified.
"""
import ast,copy,gzip,hashlib,json,os,subprocess,sys,types
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
repo=Path(os.environ['CARBON_REPO']).resolve();out=Path(os.environ.get('CARBON_REVIEW_OUT','/tmp/carbon-stage1-runtime')).resolve();out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(repo/'code'))
import numpy as np,torch
from cnet2_model import CNet2
import cnet2_feat as F,cnet2_search as S,cnet2_np as NP,rl2_ppo_cnet2 as A
import stage0_teacher as T
from fastsim import FastCarbon
from eval_v53 import CFG
torch.set_num_threads(1);torch.manual_seed(8731)
net=CNet2(C=8,NB=1,D=8,NH=2,L=1,E=4,G=2)
w={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};w['meta']=np.array(json.dumps({'cfg':net.cfg,'fixture':'untrained tiny model for control flow only'}))
np.savez(out/'tiny_heads.npz',**w)
oldsrc=subprocess.check_output(['git','-C',str(repo),'show','bc1a84d4:code/cnet2_search.py'],text=True)
old=types.ModuleType('guard_baseline');exec(compile(oldsrc,'baseline_search','exec'),old.__dict__)
factory=next(n for n in ast.parse(oldsrc).body if isinstance(n,ast.FunctionDef) and n.name=='make_search_agent')
oldfn=[n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name in ('f_own_uids','bankrupt_guard')]
conf=dict(S.DEFAULTS,SEARCH=False,ROLL='self',GUARD=True,GUARD_NARROW=True,GUARD_CASH=40.,LOWW=2,TEACHER=1,GUARD_LOG=True)
base=FastCarbon(CFG).reset(seed=91).full_observation(0)
base['players'][0][0]=20.;base['players'][1][0]=100.
base['players'][0][2]={'c1':[100,20,'COLLECTOR'],'p2':[130,0,'PLANTER']}
base['players'][1][2]={'e1':[101,0,'COLLECTOR']}
def obs(step,own=None,cash=20.):
 x=copy.deepcopy(base);x['step']=step;x['players'][0][0]=cash
 if own is not None:x['players'][0][2]=copy.deepcopy(own)
 return x
states=[obs(0),obs(1,{'c1':[100,20,'COLLECTOR']}),obs(2,{'c1':[100,20,'COLLECTOR']}),obs(3,{},100.),obs(4,{'born':[69,0,'COLLECTOR']},20.),obs(0)]
# Force default stay for a full-agent old/new parity test that actually fires moves and recruitment.
w_stay={k:(v.copy() if hasattr(v,'copy') else v) for k,v in w.items()}
for k in ['whead.weight','whead.bias','bhead.weight','bhead.bias','U.weight','U.bias']:w_stay[k][...]=0
w_stay['whead.bias'][0]=5.;w_stay['bhead.bias'][0]=5.
aold=old.make_search_agent(conf,weights=w_stay,rweights=w_stay);anew=S.make_search_agent(conf,weights=w_stay,rweights=w_stay)
rows=[];alias=anew.state['guard_base']
for x in states:
 a=aold(copy.deepcopy(x),CFG);b=anew(copy.deepcopy(x),CFG)
 row={'step':x['step'],'actions_equal':a==b,'stats_equal':aold.stats==anew.stats,'actions':b,'guard_base_equal':aold.state['guard_base']==anew.state['guard_base'],'guard_base':sorted(anew.state['guard_base']),'alias_retained':anew.state['guard_base'] is alias,'old_lostw':aold.state.get('lostw'),'old_prevw':sorted(aold.state.get('prevw',[])),'stats':copy.deepcopy(anew.stats)};rows.append(row)
 assert row['actions_equal'] and row['stats_equal'] and row['guard_base_equal'] and row['alias_retained'],row
# Standalone function extraction compares all trigger outputs, state and logs using genuine decode_b.
g=S.BankruptcyGuard(conf);stats_new={};stats_old={};f=F.Feat(0);st={'f':f,'T_now':1,'lostw':0,'prevw':set(),'guard_base':set()};pol=S.CachePolicy(w_stay)
ns=dict(np=np,F=F,st=st,stats=stats_old,cfg=conf,pol=pol,decode_b=S.decode_b);exec(compile(ast.Module(body=oldfn,type_ignores=[]),'baseline_guard','exec'),ns)
synth=[]
for i,o in enumerate(states):
 if o['step']==0:st.update(lostw=0,prevw=set());g.reset_game();f=F.Feat(0);st['f']=f
 x=f.observe(o);y,lg,_=pol.act(x,f.aux,teacher=1);st['guard_base']=set();g.reset_step()
 a=ns['bankrupt_guard'](x,f.aux,y.copy(),lg);b=g.apply(x,f.aux,y.copy(),lg,pol,f.own_uid,1,stats_new)
 r={'i':i,'step':o['step'],'labels_equal':bool(np.array_equal(a,b)),'stats_equal':stats_old==stats_new,'state_equal':st['lostw']==g.lostw and st['prevw']==g.prevw and st['guard_base']==g.guard_base,'snapshot':{'lostw':g.lostw,'prevw':sorted(g.prevw),'guard_base':sorted(g.guard_base)}};synth.append(r);assert all(r[k] for k in ('labels_equal','stats_equal','state_equal'))
# Snapshot isolation, step/game reset, and restore alias boundary.
g=S.BankruptcyGuard(conf);alias=g.guard_base;g.guard_base.add(10);g.prevw={'x'};g.lostw=2;snap=g.snapshot();clone=g.copy();clone.guard_base.add(11);clone.prevw.add('z');clone.lostw=5
isolation=g.guard_base=={10} and g.prevw=={'x'} and g.lostw==2
g.reset_step();step_alias=alias is g.guard_base and not alias;g.guard_base.add(11);g.restore(snap);restore_alias=alias is g.guard_base;g.reset_game()
resets={'copy_isolated':isolation,'reset_step_preserves_alias':step_alias,'restore_preserves_alias':restore_alias,'alias_after_restore':sorted(alias),'guard_base_after_reset_game':sorted(g.guard_base),'lostw_after_reset_game':g.lostw,'prevw_after_reset_game':sorted(g.prevw),'current_call_sites_restore_only_new_branch_guards':True}
# Greedy Actor vs NP equivalence under the actual default style (no temperature/schedule drift).
actor=A.Cnet2Actor(w);actor.teacher=1;actor.sched=False;pol0=NP.Policy(w);greedy=[]
for seat in (0,1):
 feat=F.Feat(seat)
 for step in (0,1,50,239,240,298):
  o=obs(step);o['player']=seat;x=feat.observe(o);ya,*_=actor._decode(x,feat.aux,np.random.default_rng(9),1.,True);yb,_,_=pol0.act(x,feat.aux,teacher=1)
  greedy.append({'seat':seat,'step':step,'same_labels':bool(np.array_equal(ya,yb))});assert greedy[-1]['same_labels']
# Actual teacher API cannot inherit the driver's persistent guard history.
class StayPolicy(S.CachePolicy):
 def act(self,x,aux,teacher=1,forced=None,want_u=False,**kw):
  labels=np.where(x['tok_mask'][:F.OWN],0,-1).astype(np.int64)
  if forced is not None:labels=np.asarray(forced).copy()
  logits=np.tile(np.array([5,0,0,0,0],np.float32),(F.OWN,1));logits[~x['tok_mask'][:F.OWN]]=-1e9
  result=(labels,logits,0.)
  return (*result,.5) if want_u else result
p=StayPolicy(w_stay);f=F.Feat(0);persistent=S.BankruptcyGuard(T.TEACHER_GUARD_CFG)
for o in states[:2]:
 x=f.observe(o);y,lg,_=p.act(x,f.aux);persistent.reset_step();guarded=persistent.apply(x,f.aux,y.copy(),lg,p,f.own_uid,1,{})
root=FastCarbon(CFG);root.load_observation(states[1]);fm=F.Feat(0);fo=F.Feat(1);xm=fm.observe(root.obs(0));xo=fo.observe(root.obs(1))
r=T.evaluate_root(root,0,p,xm,fm.aux,xo,fo.aux,types.SimpleNamespace(seat=0,own_uid=fm.own_uid),types.SimpleNamespace(seat=1,own_uid=fo.own_uid),teacher=1,K=2,M=1,H=0)
rootguard={'history_lostw':persistent.lostw,'persistent_guard_action':f.to_actions(guarded),'teacher_greedy_arm_action':fm.to_actions(r['arms'][0]),'teacher_selected_action':fm.to_actions(r['arms'][int(np.argmax(r['q']))]),'teacher_guard_stats':r['guard'],'q':r['q'].tolist(),'limitation':'Deterministic StayPolicy isolates guard-state wiring; actual evaluate_root/FastCarbon/Feat/BankruptcyGuard; K2 M1 H0, no trained strength claim'}
result={'reviewed_sha':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'baseline':'bc1a84d4','fixture':'seed8731 untrained C8 NB1 D8 NH2 L1 E4 G2, stay heads only for guard trigger tests','full_agent_parity':rows,'standalone_parity':synth,'reset_copy_alias':resets,'actor_np_default_greedy':greedy,'guided_history_counterexample':rootguard,'official_heads_v2_available':False,'official_replays_available':False}
(out/'guard-probe-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'full_parity_cases':len(rows),'standalone_cases':len(synth),'greedy_parity_cases':len(greedy),'guard_triggers':anew.stats,'reset':resets,'history_counterexample':rootguard},indent=2))
# Stage the existing A3 synth test unchanged against explicitly synthetic substitutes.
accept=out/'acceptance_surrogate';(accept/'orig').mkdir(parents=True,exist_ok=True);(accept/'replays').mkdir(exist_ok=True)
(accept/'test_s12_a3.py').write_bytes((repo/'automation/acceptance_s1/test_s12_a3.py').read_bytes());(accept/'orig/cnet2_search.py').write_text(oldsrc)
np.savez(accept/'heads_v2.npz',**w_stay)
with gzip.open(accept/'replays/synthetic.json.gz','wt') as z:json.dump({'steps':[[{'observation':obs(t)},{'observation':dict(obs(t),player=1)}] for t in range(101)]},z)
e=dict(os.environ,PYTHONPATH=str(repo/'code'),PYTHONDONTWRITEBYTECODE='1')
cp=subprocess.run([sys.executable,'-B',str(accept/'test_s12_a3.py'),'synth'],env=e,text=True,capture_output=True)
(out/'acceptance-surrogate.json').write_text(json.dumps({'exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr,'limitation':'Existing test source unchanged; tiny untrained weights and synthetic positions substitute unavailable canonical heads/corpus; not a rerun of reported 63 real positions.'},indent=2)+'\n');print('existing A3 synth with surrogate:',cp.returncode,cp.stdout)
