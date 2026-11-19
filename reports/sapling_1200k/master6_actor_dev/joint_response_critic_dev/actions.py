"""Legality and state/whole-team-action features. Live code accesses only the acting seat and its Sapling activations."""
import hashlib
import sys
import numpy as np

JRC_ROLES=('collector','planter','base')


def _jrc_capacity(obs,cfg,orders,F):
    """Return concrete resource and geometry violations under publicly knowable simultaneous own intents."""
    s=obs['player'];players=obs['players'];me=players[s];positions={};occupied={}
    trees={p:(j,age) for j,pl in enumerate(players) for p,age in pl[3].values()};bases={p for pl in players for p in pl[1].values()}
    life=cfg['treeLifespan']
    price=cfg['plantCost'] if len(me[3])<=cfg['playerTreeProtectiveNumber'] else cfg['plantCost']+cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**sum(a<life for _,a in trees.values())
    from collections import Counter
    counts=Counter(w[2] for pl in players for w in pl[2].values());cash=me[0];hires=0;bad=set();outlay=0
    for uid,cell in me[1].items():
        order=orders.get(uid)
        if order not in ('RECCOLLECTOR','RECPLANTER'): continue
        typ='Collector' if order=='RECCOLLECTOR' else 'Planter';cost=cfg['rec'+typ+'Cost']+counts[typ.upper()]*cfg['rec'+typ+'IncreaseCost']
        if hires+len(me[2])>=cfg['workerLimit'] or cash<cost:bad.add(('hire',uid)); continue
        cash-=cost;outlay+=cost;hires+=1;occupied.setdefault(cell,[]).append(uid)
    due=[]
    for uid,(p,_,who) in me[2].items():
        cmd=orders.get(uid,'NONE');z=int(F.DEST[p,F.MOVES.index(cmd)]);positions[uid]=z;occupied.setdefault(z,[]).append(uid)
        if who=='PLANTER' and cmd=='NONE' and p not in bases:
            tree=trees.get(p)
            if tree is None or tree[1]>=life:due.append((uid,price))
            elif tree[0]!=s:due.append((uid,cfg['seizeCost']))
    if sum(v for _,v in due)>cash+1e-8:
        bad.update(('pay',uid) for uid,_ in due)
    for lst in occupied.values():
        for i,a in enumerate(lst):
            for b in lst[:i]:bad.add(('same_ally_destination',*sorted([a,b])))
    return bad,outlay+sum(v for _,v in due)


def _jrc_context(obs,cfg,feat,x,aux,student_tokens):
    F=sys.modules['cnet2_feat']; s=obs['player']; me=obs['players'][s]; opp=obs['players'][1-s]
    live=np.flatnonzero(x['tok_mask'][:F.OWN]);roles=np.array([2 if i>=10 else int(x['tok_kind'][i]) for i in live]);tokens=student_tokens[live]
    # These fixed channels are the exact independently trained student activations at the same public root.
    anchors=np.tanh(np.asarray(tokens)[:,[0,3,7,11,15,19,23,31]]/3)
    st=np.array([obs['step']/299, (299-obs['step'])/299,
        np.log1p(max(me[0],0))/9,np.log1p(max(opp[0],0))/9,len(me[3])/20,len(opp[3])/20,
        len(me[2])/10,len(opp[2])/10,
        sum(w[1] for w in me[2].values())/500,sum(w[1] for w in opp[2].values())/500],dtype=np.float32)
    return F,live,roles,anchors,st


def _jrc_repr(ctx,obs,cfg,feat,x,aux,labels):
    F,live,roles,emb,st=ctx;prefix=F.Prefix(x,aux)
    whole=feat.to_actions(labels)
    # Re-encode every downstream teammate under this actual new prefix.
    details=[]; valid=True
    for k,slot in enumerate(live):
        a=int(labels[slot]);unit,moves,legal=prefix.feats(slot)
        if a<0 or a>=len(legal) or not legal[a]: valid=False;break
        role=int(roles[k]);pos=int(x['tok_pos'][slot])
        d=int(F.DEST[pos,a]) if role!=2 else pos
        move_alt=a if role!=2 else min(a,2)
        numeric=np.r_[unit,moves[move_alt],np.eye(5,dtype=np.float32)[a],float(aux['dob'][d])/14,float(aux['dob'][d]-aux['dob'][pos])/14,
                      np.log1p(max(0,float(x['tok_cargo'][slot])))/7,float(aux['en_pl'][d]),
                      np.asarray(emb[k])*float(a!=0),np.asarray(emb[k])*float(a==0)]
        details.append((role,numeric));prefix.advance(slot,a)
    if not valid:return None
    a=np.zeros((3,len(details[0][1])),dtype=np.float32)
    for role,vec in details:a[role]+=vec
    a/=np.array([[6.],[4.],[2.]])
    bad,outlay=_jrc_capacity(obs,cfg,whole,F)
    arr=np.r_[st,a.ravel(),min(5,len(bad))/5,np.log1p(max(0,outlay))/7].astype(np.float32)
    if not np.isfinite(arr).all():raise ValueError('nonfinite public joint embedding')
    return arr,bad


def _jrc_options(ctx,obs,cfg,feat,x,aux,labels,branches=None):
    F,live,roles,_,_=ctx
    base=_jrc_repr(ctx,obs,cfg,feat,x,aux,labels)
    if base is None:raise ValueError('live Sapling parent failed own prefix')
    items=[(labels,base[0],'parent')];seen={tuple(labels)}
    if branches is None:
        for role,slot in zip(roles,live):
            for a in range(3 if role==2 else 5):
                if a==labels[slot]:continue
                alt=labels.copy();alt[slot]=a
                got=_jrc_repr(ctx,obs,cfg,feat,x,aux,alt)
                if got is not None and got[1].issubset(base[1]):
                    items.append((alt,got[0],JRC_ROLES[role]))
    else:
        for lab,role in branches:
            if tuple(lab) in seen:continue
            got=_jrc_repr(ctx,obs,cfg,feat,x,aux,lab)
            if got is not None and got[1].issubset(base[1]):
                items.append((lab,got[0],role));seen.add(tuple(lab))
    return items
