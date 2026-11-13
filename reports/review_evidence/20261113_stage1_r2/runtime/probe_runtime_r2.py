"""S1-R2 guard state wiring checks. Read-only business code; tiny untrained fixture.
CARBON_REPO points to 7b025ae5 checkout; CARBON_REVIEW_OUT is a scratch directory.
Runs archived original probe, then passes real snapshots and traces short gate continuations.
"""
import contextlib,copy,hashlib,io,json,os,runpy,sys,types
from pathlib import Path
repo=Path(os.environ['CARBON_REPO']).resolve();out=Path(os.environ['CARBON_REVIEW_OUT']).resolve();out.mkdir(parents=True,exist_ok=True)
original=repo/'reports/review_evidence/20261113_stage1/runtime/guard_probe.py'
with contextlib.redirect_stdout(io.StringIO()) as log:
 ns=runpy.run_path(str(original))
(out/'original.stdout.txt').write_text(log.getvalue())
np=ns['np'];F=ns['F'];S=ns['S'];T=ns['T'];w=ns['w_stay'];FastCarbon=ns['FastCarbon'];CFG=ns['CFG']
import stage1_teacher_gate as G
# Original lost-worker fixture: snapshot before or after observing the current root must both work.
root=ns['root'];fm=ns['fm'];fo=ns['fo'];xm=ns['xm'];xo=ns['xo'];p=ns['p'];persistent=ns['persistent']
snap=persistent.snapshot();snap0=copy.deepcopy(snap)
args=(root,0,p,xm,fm.aux,xo,fo.aux,types.SimpleNamespace(seat=0,own_uid=fm.own_uid),types.SimpleNamespace(seat=1,own_uid=fo.own_uid))
r=T.evaluate_root(*args,teacher=2,K=2,M=1,H=0,guard_snaps=(snap,None))
expected=ns['f'].to_actions(ns['guarded']);actual=fm.to_actions(r['arms'][0]);assert actual==expected and snap==snap0
history={'persistent_action':expected,'teacher_greedy_arm':actual,'input_snapshot_unchanged':snap==snap0,'lostw':snap['lostw'],'match':actual==expected}
# In-place restore fixes aliases while keeping snapshots independent.
g=S.BankruptcyGuard(T.TEACHER_GUARD_CFG);alias=g.guard_base;g.restore({'lostw':2,'prevw':{'a'},'guard_base':{10}});alias_ok=alias is g.guard_base and alias=={10};g.reset_game();assert alias_ok and alias==set()
# A real short simulator trace loses a planter on the first step, then checks that each new
# teacher call inherits the actual guard history on both seats. Inference is controlled,
# but evaluate_root, continuation, guards, Feat and FastCarbon remain unmodified.
class ScriptPolicy(S.CachePolicy):
 def act(self,x,aux,teacher=1,forced=None,want_u=False,**kw):
  labels=np.where(x['tok_mask'][:F.OWN],0,-1).astype(np.int64)
  for s in range(10):
   if labels[s]>=0 and int(x['tok_pos'][s]) in (123,131):labels[s]=2 if int(x['tok_pos'][s])==123 else 4
  logits=np.full((F.OWN,5),-20.,np.float32)
  for s in range(F.OWN):
   if labels[s]>=0:logits[s,labels[s]]=20.
   else:logits[s,:]=-1e9
  for s in (10,11):
   if labels[s]>=0:logits[s,3:]=-1e9
   if x['cash']<30:logits[s,1:]=-1e9
  if forced is not None:labels=np.asarray(forced).copy()
  result=(labels,logits,0.)
  return (*result,.5) if want_u else result
p2=ScriptPolicy(w)
orig_eval=T.evaluate_root;events=[]
def traced_eval(*a,**kw):
 sim,me=a[0],a[1];ss=copy.deepcopy(kw.get('guard_snaps'))
 result=orig_eval(*a,**kw)
 events.append({'step':sim.step_count,'me':me,'teacher':kw['teacher'],'snaps':[{'lostw':s['lostw'],'prevw':sorted(s['prevw'])} if s else None for s in ss],'greedy':F.Feat.to_actions(a[7],result['arms'][0])})
 assert kw.get('guard_snaps')==ss
 return result
T.evaluate_root=traced_eval
traces=[]
try:
 for me in (0,1):
  obs=copy.deepcopy(ns['states'][0]);obs['step']=295
  # Both seats can lose units in the trace, avoiding assumptions about seat-loop order.
  obs['players'][me][0]=20.;obs['players'][me][2]={'c1':[100,20,'COLLECTOR'],'p2':[130,0,'PLANTER']}
  obs['players'][1-me][0]=20.;obs['players'][1-me][2]={'e1':[101,0,'COLLECTOR'],'killer':[131,0,'COLLECTOR']}
  sim=FastCarbon(CFG);sim.load_observation(obs);fs={0:F.Feat(0),1:F.Feat(1)};xs={s:fs[s].observe(sim.obs(s)) for s in (0,1)}
  scene={'full':sim.full_observation(0),'step':295,'me':me,'game':'controlled','fm':fs[me],'fo':fs[1-me],'x_me':xs[me],'x_opp':xs[1-me],'aux_me':fs[me].aux,'aux_opp':fs[1-me].aux}
  start=len(events);stats={};u=G.continue_game(p2,scene,None,900,teacher=2,guided_KMH={'K':2,'M':1,'H':0},stats=stats)
  ev=events[start:];assert all(e['teacher']==2 for e in ev)
  # At root296 the live guard still carries previous step workers; root application
  # notices the loss, and next call297 carries lostw=1. No branch loss leaks into it.
  assert len(ev)==4,ev
  assert ev[0]['snaps'][0]['lostw']==0 and ev[0]['snaps'][0]['prevw']==[]
  assert ev[1]['snaps'][0]['prevw']==['c1','p2'],ev
  assert ev[2]['snaps'][0]['lostw']==1 and ev[3]['snaps'][0]['lostw']==1,ev
  traces.append({'me':me,'u':u,'events':ev,'stats':stats})
finally:T.evaluate_root=orig_eval
# Actual guided_match entry, four diagnostic steps from a controlled starting state.
# The external exam opponent is a deterministic local stub; no remote/exam files are read.
import stage0_guided_match as GM
old_sim,old_cp,old_opp=GM.FastCarbon,GM.CachePolicy,GM.load_opp_agent
class InitialSim(FastCarbon):
 def reset(self,*args,**kwargs):
  if getattr(self,'_controlled_load',False):return super().reset(*args,**kwargs)
  super().reset(*args,**kwargs)
  seed=kwargs.get('seed',0);me=seed%2
  o=copy.deepcopy(ns['states'][0]);o['step']=0
  o['players'][me][0]=20.;o['players'][me][2]={'c1':[100,20,'COLLECTOR'],'p2':[130,0,'PLANTER']}
  o['players'][1-me][0]=20.;o['players'][1-me][2]={'e1':[101,0,'COLLECTOR'],'killer':[131,0,'COLLECTOR']}
  self._controlled_load=True
  self.load_observation(o)
  self._controlled_load=False
  return self
def local_opp(o,*args):
 workers=o['players'][o['player']][2]
 return {'killer':'LEFT'} if 'killer' in workers and workers['killer'][0]==131 else {}
GM.FastCarbon=InitialSim;GM.CachePolicy=lambda weights: ScriptPolicy(weights);GM.load_opp_agent=lambda name:local_opp
T.evaluate_root=traced_eval;match_traces=[]
try:
 for seed in (0,1):
  start=len(events);rec=GM.play_one((str(out/'tiny_heads.npz'),'local','guided',seed,4));ev=events[start:]
  assert len(ev)==4,(rec,ev)
  assert all(e['teacher']==2 and e['snaps'][1] is None for e in ev)
  assert ev[1]['snaps'][0]['prevw']==['c1','p2'],ev
  assert ev[2]['snaps'][0]['lostw']==1 and ev[3]['snaps'][0]['lostw']==1,ev
  match_traces.append({'seat':seed,'events':ev,'completed_diagnostic_steps':rec['steps']})
finally:
 GM.FastCarbon,GM.CachePolicy,GM.load_opp_agent=old_sim,old_cp,old_opp
 T.evaluate_root=orig_eval

# Same style-2 default decode is equivalent for Actor and NumPy; special schedules not claimed.
actor=ns['A'].Cnet2Actor(ns['w']);actor.teacher=2;actor.sched=False;pol0=ns['NP'].Policy(ns['w']);parity=[]
for seat in (0,1):
 feat=F.Feat(seat)
 for step in (0,1,50,239,240,298):
  o=ns['obs'](step);o['player']=seat;x=feat.observe(o);a,*_=actor._decode(x,feat.aux,np.random.default_rng(9),1.,True);b,_,_=pol0.act(x,feat.aux,teacher=2);same=bool(np.array_equal(a,b));assert same;parity.append({'seat':seat,'step':step,'equal':same})
result={'review_sha':'7b025ae58f83250265e75a5cccf274f63ca7d0a3','original_snapshot_counterexample_closed':history,'restore_alias_fixed':alias_ok,'live_gate_two_seat_traces':traces,'actor_np_style2':parity,'guided_match_two_seat_traces':match_traces,'scope':'Real guard/teacher/gate/simulator with controlled tiny-network inference; no strength or full-match claim. Gate roots start fresh as declared; original optional no-snapshot call remains fresh by contract.'}
result['source_sha256']={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in ('code/stage0_teacher.py','code/stage0_guided_match.py','code/stage1_teacher_gate.py','code/cnet2_search.py','reports/review_evidence/20261113_stage1/runtime/guard_probe.py')}
(out/'runtime-r2-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'snapshot_counterexample_closed':history,'restore_alias_fixed':alias_ok,'gate_trace_seats':len(traces),'style2_parity':len(parity)},indent=2))
