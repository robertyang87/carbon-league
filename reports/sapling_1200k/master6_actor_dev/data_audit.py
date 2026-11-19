"""Independently compare sparse rows to the full standard builder and document observable action coverage."""
import gzip,json,tempfile,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import data_build as b
from cnet2_build import _build_game

def run():
    split=json.loads((b.HERE/'data_split.json').read_bytes())['games']
    receipt=json.loads(gzip.decompress((b.HERE/'data_receipts.json.gz').read_bytes()));cards={g['id']:g for g in split}
    counts=defaultdict(Counter);regions=defaultdict(lambda:defaultdict(Counter));match=[]
    for g in receipt['games']:
        f=b.DEFAULT/g['file'];c=cards[g['id']];s=g['split']
        if b.sha(f.read_bytes())!=g['sha256']:raise ValueError('shard identity')
        z=np.load(f);y=z['y'];legal=z['legal'];mf=z['mf'];kind=z['tok_kind'][:,:y.shape[1]]
        actionable=(y>=0)&(legal.sum(-1)>=2);ri,si=np.nonzero(actionable);a=y[ri,si]
        if not legal[ri,si,a].all():raise ValueError('illegal supervised command')
        chosen=mf[ri,si,a];bounds=mf[ri,si];can=legal[ri,si];worker=si<10
        role=kind[ri,si];collector=worker&(role==0);planter=worker&(role==1);cargo=z['tok_cargo'][ri,si]
        def pair(key,opp,response):
            counts[s][key+'_opportunity']+=int(opp.sum());counts[s][key+'_demonstrated']+=int((opp&response).sum())
        risk=bounds[:,:,1]>0;safe=can&~risk
        pair('loaded_safe_escape',collector&(cargo>=30)&risk[:,0]&safe.any(1),chosen[:,1]==0)
        pair('loaded_bank',collector&(cargo>=30)&((bounds[:,:,5]>0)&can).any(1),chosen[:,5]>0)
        pair('approach_grabbable_enemy_cargo',collector&((bounds[:,:,2]>0)&can).any(1),chosen[:,2]>0)
        pair('planter_move_onto_enemy_tree',planter&(bounds[:,0,9]==0)&((bounds[:,:,9]>0)&can).any(1),chosen[:,9]>0)
        pair('planter_stay_on_enemy_tree',planter&(bounds[:,0,9]>0),a==0)
        pair('protect_owned_tree',planter&(bounds[:,0,8]>0)&(bounds[:,0,4]>0),a==0)
        pair('base_recruit_choice',~worker,a>0)
        final=ri[z['step'][ri]>=270];counts[s]['late_roots_270_or_after']+=int((z['step']>=270).sum())
        counts[s]['late_actionable_slots']+=len(final)
        regions[s]['wdl'][c['wdl']]+=1;regions[s]['role_seat'][str(c['seat'])]+=1
    with tempfile.TemporaryDirectory() as name:
        for part in ('train','val','holdout'):
            for seat in (0,1):
                c=next(g for g in split if g['split']==part and g['seat']==seat)
                key=c['id'];path=Path(name)/(key+'.npz')
                gid,n,err=_build_game((str(b.CACHE/(key+'.json.gz')),[seat],c['gid'],[2],[c['margin']],str(path)))
                if err or n!=c['turns']:raise ValueError('reference build failed')
                with np.load(path) as full,np.load(b.DEFAULT/(key+'.npz')) as sparse:
                    if set(full)!=set(sparse):raise ValueError('schema drift')
                    ids=sparse['step'].astype(int)
                    diff=[k for k in full if not np.array_equal(full[k][ids],sparse[k])]
                    if diff:raise ValueError(f'different teacher-forced fields {key}: {diff}')
                match.append({'id':key,'split':part,'seat':seat,'roots':len(ids),'fields':len(b.FIELDS)})
    data={'full_builder_parity':match,'action_opportunities':counts,'map_coverage':regions,
          'scope':'Each opening is a different conditional study in the true expert prefix. Danger/possible contact is reachable under public moves, not a certain hit. A planter on a hostile tree must still pay and survive. Labels are actions, not confirmed payments. The late-root threshold is t>=270.'}
    (b.HERE/'data_audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':run()
