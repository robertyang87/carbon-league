"""Audit every own public pre-action state on the four maps; fork only the current simultaneous transition."""
import gzip,inspect,json,sys
from collections import Counter
from pathlib import Path
from review import HERE, EXP, reconcile
sys.path.insert(0,str(EXP))
from collect import policy, D, TO, DIRECTIONS
from analyze import FastCarbon,CashEvents
import numpy as np

def load_scored_model(path):
    net=policy.load(path);capture={}
    source=inspect.getsource(net._return_decode)
    marker='action = int(np.argmax(np.where(legal, combined, -np.inf)))'
    if source.count(marker)!=1:raise ValueError('release decoder changed')
    def keep(branch,slot,scores,legal,chosen):
        z=np.stack(scores);idx=np.argsort(-np.where(legal,z,-np.inf),axis=1,kind='stable')
        capture.setdefault(branch,{})[slot]=(idx[0,0].item(),idx[1,0].item(),chosen,
                                              [int(np.flatnonzero(row==chosen)[0])+1 for row in idx])
    ns=dict(vars(net),insert_trace=keep)
    exec(source.replace(marker,marker+'\n        insert_trace(apply_residual,slot,scores,legal,action)'),ns)
    net._return_decode=ns['_return_decode']
    return net,capture

def dist(x,y):return int(D[x,y])
def dest(pos,cmd):return int(TO[pos,DIRECTIONS.index(cmd)])
def labels(working,slot,act):
    substitute=working.labels({});substitute[slot]=act
    return working.to_actions(substitute).get(working.own_uid[slot],'STAY')

def alt_outcome(sim,state,s,uid,new,orders):
    sim.set_state(state);pair=[dict(o) for o in orders]
    if new=='STAY':pair[s].pop(uid,None)
    else:pair[s][uid]=new
    sim.cash=CashEvents(sim.cash);sim.step(*pair)
    cash=sim.cash.events
    return dict(credited=round(sum(e[2] for e in cash if e[0]=='own_delivery' and e[1]==s and e[3]==uid),5),
                bought=bool(any(e[0]=='plant' and e[1]==s and e[3]==uid for e in cash)),
                survived=uid in sim.workers,held=round(sim.workers[uid].carbon,5) if uid in sim.workers else 0,
                cashmargin=round(sim.cash[s]-sim.cash[1-s],3))

def kinds(view,uid,actual,other):
    s=view['player'];me,enemy=view['players'][s],view['players'][1-s]
    p,c,role=me[2][uid];h=min(dist(p,q) for q in me[1].values())
    en=list(enemy[2].values());near=min((dist(p,w[0]) for w in en),default=99)
    targets={dest(p,actual),dest(p,other)}
    contest=min((dist(z,w[0]) for z in targets for w in en if w[2]=='COLLECTOR' and w[1]<=c),default=99)
    tree={pos for pl in (me,enemy) for pos,age in pl[3].values()}
    base={pos for pl in (me,enemy) for pos in pl[1].values()};bs=set(me[1].values())
    soil=sum(view['carbon'][int(n)] for n in TO[p,1:]) if role=='PLANTER' else None
    tags=[]
    if role=='COLLECTOR':
        if c>=100 and h<=3: tags.append('laden_near_home')
        if c>=100 and h==1 and targets&bs: tags.append('deposit_now')
        if c>=40 and contest<=1:tags.append('loaded_under_threat')
        if other!=actual and 'STAY' in (actual,other) and view['carbon'][p]>=30 and contest>1 and p not in (tree|base):tags.append('visible_safe_harvest')
    elif role=='PLANTER' and p not in (tree|base) and near>=2 and soil>=120 and 'STAY' in (other,actual) and 90<=view['step']<190: tags.append('safe_rich_plant_now')
    return tags,dict(cell=p,cargo=c,home=h,near_enemy=near,contest=contest,visible_soil=soil,visible_cell=view['carbon'][p])

def run(card,net,cap):
    tape=reconcile.read(reconcile.ROOT/card['path']);s=card['seat'];cfg=tape['game_configuration']
    live=FastCarbon(cfg).reset(seed=card['seed']);fork=FastCarbon(cfg)
    detail=[];counts=Counter();trace=[]
    for t,pair in enumerate(tape['actions']):
        view=live.obs(s);cap.clear();issued=net.agent(view,cfg);counts['verified_root']+=1
        if issued!=pair[s]:raise ValueError(('full-team release disagreement',card['seed'],t,s,issued,pair[s]))
        feat=net._FEAT;y=feat.labels(issued)
        route=next((b for b in (True,False) if b in cap and all(y[k]==v[2] for k,v in cap[b].items())),None)
        if route is None:raise ValueError('lost release branch')
        st=None;all_slots=[]
        owner=view['players'][s][2]
        for slot,(a,b,c,chosen_ranks) in cap[route].items():
            uid=feat.own_uid[slot]
            cmds=[labels(feat,slot,action) for action in (a,b,c)] if slot<10 else [a,b,c]
            all_slots.append([uid,*cmds,*chosen_ranks]);counts['active_slots']+=1
            if a!=b:counts['two_model_disagree']+=1
            if slot>=10 or a==b:continue
            counts['worker_disagree']+=1
            alt=cmds[1] if b!=c else cmds[0]
            tags,cues=kinds(view,uid,cmds[2],alt)
            if not tags:continue
            other_dest={dest(w[0],issued.get(wid,'STAY')) for wid,w in owner.items() if wid!=uid}
            physical_targets={dest(cues['cell'],cmds[2]),dest(cues['cell'],alt)}
            conflict=bool((physical_targets-{dest(cues['cell'],cmds[2])})&other_dest)
            if st is None:st=live.get_state()
            choices={cmd:alt_outcome(fork,st,s,uid,cmd,pair) for cmd in set(cmds)}
            detail.append(dict(t=t,uid=uid,role=owner[uid][2],tags=tags,pre=cues,it50=cmds[0],sapling=cmds[1],sent=cmds[2],
                               conflict=conflict,one_step=choices,opponent_commands=pair[1-s]))
        trace.append(all_slots)
        live.step(*pair)
    return dict(seed=card['seed'],seat=s,arm=card['arm'],raw_sha256=card['raw_sha256'],counts=counts,
                slots_order=['uid','it50_top','sapling_top','sent','rank_of_sent_it50','rank_of_sent_sapling'],
                every_step=trace,opportunities=detail)

def main():
    games={}
    for cohort in (EXP/'direct_original/accounts', HERE):
        for f in cohort.glob('*_621*_[01].json'):
            card=reconcile.read(f);games[card['seed'],card['seat'],card['arm']]=card
    for arm,pack in [('candidate',EXP/'candidate.py'),('parent',EXP.parent/'deployed_mix_policy_gradient_dev/full_episode/selected.py')]:
        net,cap=load_scored_model(pack)
        for k,c in sorted(games.items()):
            if k[-1]!=arm:continue
            path=HERE/f"rank_{arm}_{c['seed']}_{c['seat']}.json.gz"
            if path.exists():continue
            x=run(c,net,cap)
            data=json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode();tmp=path.with_suffix('.tmp')
            tmp.write_bytes(gzip.compress(data,mtime=0));tmp.replace(path)
            print(k,x['counts'],len(x['opportunities']),flush=True)
if __name__=='__main__':main()
