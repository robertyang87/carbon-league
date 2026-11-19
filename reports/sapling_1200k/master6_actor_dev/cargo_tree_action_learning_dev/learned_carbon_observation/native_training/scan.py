"""Census the two frozen public policies, running every archived training history without an opponent call."""
import argparse
from collections import Counter
from pathlib import Path
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent;LEARN=HERE.parent
sys.path.insert(0,str(LEARN))
from common import F,ROOT,public_frames,read,sha
from verify_entry import PACKS,run_pack,compact

REG=HERE/'protocol.json'
COMMAND=('NONE','UP','RIGHT','DOWN','LEFT')
def save(path,obj):
    import gzip
    b=compact(obj)
    if path.suffix=='.gz':b=gzip.compress(b,mtime=0)
    if path.exists():assert path.read_bytes()==b
    else:
        tmp=path.with_suffix('.part');tmp.write_bytes(b);tmp.replace(path)
def view(state,s):
    fields=state['players']; own=fields[s]
    trees={p:(i,age) for i,h in enumerate(fields) for p,age in h[3].values()}
    bases={p:i for i,h in enumerate(fields) for p in h[1].values()}
    return fields,own,trees,bases
def classes(t):
    return 'opening' if t<100 else 'middle' if t<200 else 'end'

def context(uid,role,cmd,own_action,obs,feat,cfg,slot):
    s=obs['player'];fields,own,trees,bases=view(obs,s);workers=own[2]
    pos,cargo,real_role=workers[uid];assert role==real_role
    dst=int(F.DEST[pos,COMMAND.index(cmd)]);nb=F.DEST[dst,1:];dist=F.DIST
    en=[(w,c,r) for w,c,r in fields[1-s][2].values()]
    nearby=[(c,r) for w,c,r in en if dist[w,dst]<=1]
    own_dest={u:int(F.DEST[p,COMMAND.index(own_action.get(u,'NONE'))]) for u,(p,_,_) in workers.items()}
    coll=sorted(u for u,z in own_dest.items() if u!=uid and z==dst)
    owned=trees.get(dst);plant_cost=cfg['plantCost']
    if len(own[3])>cfg['playerTreeProtectiveNumber']:
        plant_cost+=cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**sum(age<cfg['treeLifespan'] for _,age in trees.values())
    spend_before=0.
    for u,p in own[1].items():
        if own_action.get(u) in ('RECCOLLECTOR','RECPLANTER'):spend_before+=cfg.get('recCollectorCost',30)
    # Price and funds are actionable pre-root proxies. Same-turn resolution orders, other plant purchases and enemy moves remain uncertain.
    can_buy=(role=='PLANTER' and cmd=='NONE' and dst not in bases and (owned is None or owned[0]!=s))
    paid=cfg['seizeCost'] if owned else plant_cost
    econ=dict(deposit=bool(role=='COLLECTOR' and cargo>0 and bases.get(dst)==s),
              harvest=bool(role=='COLLECTOR' and cmd=='NONE' and dst not in bases and dst not in trees and obs['carbon'][dst]>0),
              return_distance=int(dist[dst,list(own[1].values())].min()) if role=='COLLECTOR' and cargo>0 else None,
              lighter_enemy_threat=bool(role=='COLLECTOR' and any(r=='COLLECTOR' and c<=cargo for c,r in nearby) and cargo>0),
              heavier_enemy_target=bool(role=='COLLECTOR' and any(r=='COLLECTOR' and c>cargo for c,r in nearby)),
              friendly_conflict=bool(coll),purchase=('seize' if owned else 'plant') if can_buy and own[0]>=paid else None,
              site_tree='own' if owned and owned[0]==s else 'enemy' if owned else 'none',
              contest=bool(nearby) if role=='PLANTER' else None)
    estimate=float(feat.est[dst]) if role=='COLLECTOR' else float(np.mean(feat.est[nb]))
    return dict(command=cmd,destination=dst,signals=econ,belief_richness=estimate,
                current_direct=bool(feat.last_step[dst]==obs['step']),nearby_enemy_units=len(nearby),friendly_conflicts=coll,
                own_tree_age=None if owned is None else owned[1],current_wallet=float(own[0]),at_action_fee=float(paid) if can_buy else None,
                cash_after_base_requests=float(own[0]-spend_before) if can_buy else None)

def inspect(obs,feat,xa,cfg,actions,recorded):
    s=obs['player'];t=obs['step'];own=obs['players'][s];workers=own[2];commands={}
    for name,a in actions.items():
        labels=feat.labels(a);prefix=F.Prefix(xa,feat.aux)
        assert set(a.keys())<=set(own[1])|set(workers)
        for slot in np.flatnonzero(xa['tok_mask'][:F.OWN]):
            _,_,legal=prefix.feats(slot)
            assert legal[labels[slot]], (t,s,slot,name)
            prefix.advance(slot,int(labels[slot]))
    for uid in sorted(own[1].keys()|workers.keys()):
        a,b=[v.get(uid,'NONE') for v in actions.values()]
        if a==b:continue
        if uid not in workers:
            commands[uid]=dict(role='BASE',parent=a,student=b,qualifies=False,different=['base_decision'])
            continue
        p,cargo,role=workers[uid];slot=feat.own_uid.index(uid)
        aa,bb=[context(uid,role,cmd,act,obs,feat,cfg,slot) for cmd,act in zip((a,b),actions.values())]
        diffs=[k for k in aa['signals'] if aa['signals'][k]!=bb['signals'][k]]
        if abs(aa['belief_richness']-bb['belief_richness'])>=1:diffs.append('estimated_richness')
        trees=view(obs,s)[2];base=view(obs,s)[3]
        surface='base' if p in base else ('own_tree' if trees[p][0]==s else 'enemy_tree') if p in trees else 'empty'
        commands[uid]=dict(role=role,pos=p,cargo=float(cargo),surface=surface,was_logged=recorded.get(uid,'NONE'),parent=aa,student=bb,qualifies=bool(diffs),different=diffs)
    return commands

def run(card,reg,cpu):
    out=HERE/(str(card['seed'])+'.json.gz')
    if out.exists():
        val=read(out);assert val['identity']==reg['student'] and val['registration']==sha(REG.read_bytes()) and val['card']==card;return val
    f=[[],[]]; time0=time.perf_counter();joints=[]
    for obs,_,joint,cfg in public_frames(card):
        for s in (0,1):f[s].append(obs[s])
        joints.append(joint)
    result=[]
    for s,frames in enumerate(f):
        arms={}
        for k in ('incumbent','student'):
            # New independent process per model/seat. No executable Master3s2 code is loaded.
            packet=run_pack(PACKS[k],'timing',cpu,frames,cfg)
            assert packet['ready']['pack_sha256']==reg['parent' if k=='incumbent' else k]
            arms[k]=[x['action'] for x in packet['rows']];assert len(arms[k])==len(frames)
        feat=F.Feat(s);events=[];counts=Counter();stages={k:Counter() for k in ('opening','middle','end')};roles=Counter();rows=[]
        for t,obs in enumerate(frames):
            assert obs['player']==s and obs['step']==t
            stage=classes(t);x=feat.observe(obs);own=obs['players'][s]
            base,student=arms['incumbent'][t],arms['student'][t]
            n={'turns':1,'workers':len(own[2]),'bases':len(own[1]),'collectors':sum(w[2]=='COLLECTOR' for w in own[2].values()),'planters':sum(w[2]=='PLANTER' for w in own[2].values()),'changed_turns':int(base!=student)}
            event=None
            if base!=student:
                changed=inspect(obs,feat,x,cfg,{'parent':base,'student':student},joints[t][s])
                real=[u for u,v in changed.items() if v['role']!='BASE'];eco=[u for u in real if changed[u]['qualifies']]
                n.update(changed_worker_turns=int(bool(real)),eligible_turns=int(bool(eco)),changed_workers=len(real),eligible_workers=len(eco))
                roles.update(v['role'] for v in changed.values())
                for u in eco: n['eligible_'+changed[u]['role'].lower()]=n.get('eligible_'+changed[u]['role'].lower(),0)+1
                event=dict(t=t,phase=stage,visible_sha256=sha(compact(obs)),recorded_own=joints[t][s],recorded_enemy=joints[t][1-s],
                           parent=base,student=student,changed=changed,eligible_workers=eco,
                           incumbent_matches_recorded_joint_own=(base==joints[t][s]),student_matches_recorded_joint_own=(student==joints[t][s]))
                events.append(event)
            for c in (counts,stages[stage]):c.update(n)
            rows.append([sha(compact(obs)),base,student])
        result.append(dict(seat=s,counts=counts,stage=stages,changed_roles=roles,events=events,all_public_commands=rows))
    rec=dict(identity=reg['student'],parent=reg['parent'],registration=sha(REG.read_bytes()),card=card,seats=result,wall_s=time.perf_counter()-time0,
             scope='Historical joint actions were used only to advance the next observation. No counterfactual or new game result was produced.')
    save(out,rec);return rec

def main(cpu):
    reg=read(REG)
    assert reg['training_registration']==sha((LEARN/'protocol.json').read_bytes())
    assert reg['parent']==sha(PACKS['incumbent'].read_bytes()) and reg['student']==sha(PACKS['student'].read_bytes())
    old=read(LEARN/'protocol.json')
    assert reg['training_cards']==[c for c in old['cards'] if c['split']=='train'] and len(reg['training_cards'])==10
    assert not {c['terrain'] for c in reg['training_cards']}&set(reg['heldout_terrains'])
    for card in reg['training_cards']:
        rec=run(card,reg,cpu);print(card['seed'],[(s['counts']['turns'],s['counts'].get('eligible_turns',0)) for s in rec['seats']],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cpu',type=int,default=2);main(ap.parse_args().cpu)
