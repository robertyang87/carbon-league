"""Paired root-head forecasts against actual independently issued actions on eight old boards."""
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent; D=HERE.parents[1]; ROOT=D.parents[1]
REG='71624ab5125ae5e5a61a44560db4317ccd4df0bd7975924ee8b2dcf12571c5cb'
ORIGINAL='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
def sha(x):return hashlib.sha256(x).hexdigest()
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def save(p,obj):
 raw=(json.dumps(obj,separators=(',',':'),allow_nan=False,ensure_ascii=False)+'\n').encode()
 if p.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
 p.write_bytes(raw);return sha(raw)
def digest(x):return sha((json.dumps(x,separators=(',',':'),allow_nan=False)+'\n').encode())
def logp(z):return z-np.logaddexp.reduce(z,axis=1,keepdims=True)
def collect(card,steps,nets,econ,cfg):
 p=D/card['rel'];assert sha(p.read_bytes())==card['archive_sha256'];arm=card['arm']
 with zipfile.ZipFile(p) as z:
  rc=json.loads(z.read(arm+'/receipt.json'));rg=json.loads(z.read(arm+'/registration.json'))
  if not rc['counted'] or rc['official_transitions']!=299:
   return [],dict(status='incomplete',seed=card['seed'],seat=card['seat'],turns=rc['official_transitions'],source_status=rc['status'])
  trs=z.read(arm+'/turns.jsonl');turns=trs.splitlines();frames=gzip.decompress(z.read(arm+'/observations.jsonl.gz')).splitlines()
 assert sha(trs)==rc['turns_sha256'] and rg['physical_map_sha256']==card['terrain'] and rg['candidate_seat']==card['seat'] and rg['seed']==card['seed']
 assert rg['files']['opponent']['sha256']==ORIGINAL and rc['initialization'][1-card['seat']]['response']['original']==dict(_TEMP=1.,_SEED=None,_SEARCH_OK=True,_S_ABORT=.58)
 sim=econ.FastCarbon(cfg).reset(seed=card['seed']);sim.cash=econ.CashEvents(sim.cash);assert sha(bytes(map(int,sim.carbon)))==card['terrain']
 parent,new=nets; trackers=[parent.BeliefTracker(parent._Bel,s) for s in (0,1)]; prev=[None,None];out=[]
 for t,(ob,line) in enumerate(zip(frames,turns,strict=True)):
  ob=json.loads(ob);line=json.loads(line);pub=ob['observations'];assert ob['step']==line['step']==t and ob['public_observation_sha256']==line['public_observation_sha256']
  assert all(a['status']=='ok' for a in line['calls']);cmd=[a['response']['action'] for a in line['calls']]
  assert [digest(a) for a in pub]==line['public_observation_sha256']
  perspective=[trackers[s].step(pub[s]['players'],t,pub[s]['carbon']) for s in (0,1)]
  assert all(sim.obs(s)['players']==pub[s]['players'] and sim.obs(s)['carbon']==pub[s]['carbon'] for s in (0,1))
  updates=[]
  if t in steps:
   players=pub[0]['players'];positions=[{uid:parent.move(w[0],parent.WIDX[cmd[s].get(uid,'NONE')]) for uid,w in players[s][2].items()} for s in (0,1)]
   for s in (0,1):
    src='master3s2' if s!=card['seat'] else 'historical_subject'
    est,known=perspective[s];x=parent._quant(parent.encode(players,t,s,est,known,prev[s]))
    nw,nb=parent._POLICY.forward(x,parent._SID,own_root=True);cw,cb=new._root_rotation_policy(x,new._SID)
    pred=[(logp(nw.astype(float)),logp(nb.astype(float))),(logp(cw.astype(float)),logp(cb.astype(float)))]
    cash,bases,workers,_=players[s]; unitrows=[];roster=len(workers)
    for uid,pos in bases.items():
     act=cmd[s].get(uid,'NONE')
     unitrows.append((uid,pos,'base','base',parent.BIDX[act],0.,cash>=30. and roster<10))
     if act!='NONE':roster+=1;cash-=30.
    for uid,(pos,load,kind) in workers.items():
     forced=kind=='COLLECTOR' and parent.go_home(pos,load,parent._probs(nw[pos]),x,set(bases.values()),set(players[1-s][1].values()),t,parent._HOME) is not None
     unitrows.append((uid,pos,'worker',kind,parent.WIDX[cmd[s].get(uid,'NONE')],load,not forced))
    for uid,pos,role,kind,y,cargo,eligible in unitrows:
     ref,cand=[logits[int(role=='base')][pos] for logits in pred]; enemy_contact=role=='worker' and positions[s][uid] in positions[1-s].values()
     updates.append(dict(seed=card['seed'],seat=s,source=src,group=card['group'],t=t,uid=uid,role=role,kind=kind,y=y,eligible=bool(eligible),contact=bool(enemy_contact),cargo=cargo,
       ref_loss=float(-ref[y]),new_loss=float(-cand[y]),ref_top=int(ref.argmax()),new_top=int(cand.argmax())))
  sim.cash.events.clear();sim.step(*cmd)
  events={(s,detail[0]):(kind,val) for kind,s,val,*detail in sim.cash.events if kind in ('plant','seize','own_delivery') and (kind!='own_delivery' or val>0.)}
  for r in updates:
   post=sim.obs(r['seat'])['players'][r['seat']][3]; before=pub[0]['players'][r['seat']][3]
   event,amount=events.get((r['seat'],r['uid']),(None,0.));r['realized']=event;r['immediate_cash_attributed']=float(amount)
   if r['kind']=='PLANTER' and r['y']==0:
    for tree in post.keys()-before.keys():
     if sim.obs(0)['trees'][tree][0]==r['uid']:
      r['realized']='seize' if tree in pub[0]['players'][1-r['seat']][3] else 'plant'
   out.append(r)
  for s in (0,1):trackers[s].set_actions(pub[s]['players'],cmd[s]);prev[s]=pub[s]['players']
 assert np.allclose(list(sim.cash),rc['official_scores_by_seat'])
 return out,dict(status='complete',seed=card['seed'],seat=card['seat'],role_source_sha=rg['files']['candidate']['sha256'],units=len(out),economic_terminal_verified=True)

def metrics(rows):
 if not rows:return None
 m={}
 for k in ('ref','new'):
  m[k+'_nll']=float(np.mean([r[k+'_loss'] for r in rows]));m[k+'_top1']=float(np.mean([r[k+'_top']==r['y'] for r in rows]))
 return dict(n=len(rows),maps=len({r['seed'] for r in rows}),**m)
def groups(rows):
 output={}
 for cohort in ('previous_hold','external','both'):
  for source in ('master3s2','historical_subject'):
   raw=[r for r in rows if r['source']==source and (cohort=='both' or r['group']==cohort)]
   elig=[r for r in raw if r['eligible']]
   by_map={}
   for seed in sorted({r['seed'] for r in elig}):
    rs=[r for r in elig if r['seed']==seed]; roles={role:metrics([r for r in rs if r['role']==role]) for role in ('worker','base')}
    by_map[str(seed)]={k:float(np.mean([v[k] for v in roles.values() if v])) for k in ('ref_nll','new_nll','ref_top1','new_top1')}
   summ={k:float(np.mean([r[k] for r in by_map.values()])) for k in ('ref_nll','new_nll','ref_top1','new_top1')}
   def subset(test):
    rs=[r for r in raw if test(r)]
    return dict(all_issued=metrics(rs),eligible=metrics([r for r in rs if r['eligible']]))
   output[cohort+'/'+source]={'map_role_equal':summ,'map_role_stats':by_map,'pooled_by_role':{k:subset(lambda r:r['role']==k or r['kind']==k) for k in ('base','worker','COLLECTOR','PLANTER')},
      'real_transitions':{k:subset(lambda r: r['contact'] if k=='contact' else r['realized']==k) for k in ('contact','plant','seize','own_delivery')},
      'eligible_by_seat':{s:metrics([r for r in elig if r['seat']==s]) for s in (0,1)}}
 return output

def main():
 regp=HERE/'protocol.json';assert sha(regp.read_bytes())==REG;reg=json.loads(regp.read_bytes())
 pp=D/'advantage_policy_update/v2_bc.py';np_=HERE.parent/'submission.py'
 assert sha(pp.read_bytes())==reg['baseline'] and sha(np_.read_bytes())==reg['candidate']
 parent=imp('rotate_baseline',pp);new=imp('rot_180',np_);parent._torch.set_num_threads(1)
 econ=imp('role_cash_ledger',ROOT/'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/natural_original/economy.py')
 assert sha(econ.SOURCE_CFG.read_bytes())==econ.CFG_HASH
 cfg=json.loads(gzip.decompress(econ.SOURCE_CFG.read_bytes()))['game_configuration']
 result=[];cards=[]
 for c in reg['tickets']:
  rows,card=collect(c,set(reg['steps']),(parent,new),econ,cfg);result+=rows;cards.append(card);print(c['seed'],c['seat'],card['status'],len(rows),flush=True)
 hh=save(HERE/'unit_rows.json.gz',result)
 save(HERE/'scores.json',dict(protocol_sha256=REG,rows_sha256=hh,baseline=reg['baseline'],candidate=reg['candidate'],receipts=cards,stats=groups(result)))
if __name__=='__main__':main()
