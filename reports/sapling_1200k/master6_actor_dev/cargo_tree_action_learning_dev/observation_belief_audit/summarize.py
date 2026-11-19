"""Summarize observable alternatives without crediting historic returns to unexecuted actions."""
from collections import defaultdict,Counter
import json
import numpy as np
from audit import HERE,load,save,gitfile,digest

def mean_or_none(a):
    return float(np.mean(a)) if len(a) else None

def format3(v):
    return '—' if v is None else f'{v:.3f}'

def compute():
    games=load(HERE/'cell_groups.json.gz');meta=load(HERE/'opportunities.json.gz');vs=np.load(HERE/'ranks.npz')['values'].astype(float)
    assert len(games)==16 and len(meta)==len(vs)
    cells=defaultdict(lambda:np.zeros(5));counter=Counter()
    def group(card,pov,stage):
        side='Sapling_side' if pov==card['seat'] else 'Master3s2_side'
        return ('all','map/'+card['terrain'],'stage/'+stage,'physical/'+str(pov),'policy/'+side,'source/'+card['origin']+'/'+card['arm'])
    for g in games:
        for s,phase,zone,mask,n,*rates in g['cell_groups']:
            for bucket in group(g['card'],s,phase):
                cells[bucket,zone,mask]+=np.array([n,*rates])
    cellrows=[dict(pop=k[0],zone=k[1],mask=k[2],n=int(v[0]),sap_raw=v[1]/v[0],master_raw=v[2]/v[0],sap_input=v[3]/v[0],master_input=v[4]/v[0]) for k,v in cells.items()]
    lo=np.array([z[6] for z in meta]); leg=np.array([z[7] for z in meta]); feasible=np.array([z[8] for z in meta]);obs=np.array([z[9] for z in meta]);N=len(vs)
    ia=np.arange(N);acts=np.arange(5);tri=np.triu_indices(5,1)
    rr=[];ex=[]
    numeric=['jointly_informative','target_ambiguous','true_order_comparisons','strict_rank_flips','flips_towards_truth','flips_away_from_truth',
             'sap_pair_credits','master_pair_credits','topsets_differ','strict_top_disjoint','sap_top_credit','master_top_credit',
             'sap_true_regret','master_true_regret','logged_choice_is_top_sap','logged_choice_is_top_master','logged_truth_regret']
    card_index=np.array([z[0] for z in meta]);seats=np.array([z[1] for z in meta]);ts=np.array([z[2] for z in meta]);roles=np.array([z[5] for z in meta]);srcs=np.array([g['card']['origin']+'/'+g['card']['arm'] for g in games])
    terrains=np.array([g['card']['terrain'] for g in games])[card_index]
    arm_seat=np.array([g['card']['seat'] for g in games])[card_index]
    stage=np.choose(np.minimum(ts//100,2),['opening','middle','end'])
    groups={'all':np.ones(N,bool)}
    groups.update({'map/'+m:terrains==m for m in np.unique(terrains)})
    groups.update({'stage/'+m:stage==m for m in np.unique(stage)})
    groups.update({'physical/'+str(i):seats==i for i in (0,1)})
    groups.update({'policy/Sapling_side':seats==arm_seat,'policy/Master3s2_side':seats!=arm_seat})
    groups.update({'source/'+p:srcs[card_index]==p for p in np.unique(srcs)})
    for name,mask in [('motor',leg),('economic',leg&feasible)]:
        legal=(mask[:,None]&(1<<acts))>0
        y,sa,ma=vs.transpose(1,0,2)
        anyopt=legal.any(1);nopt=legal.sum(1)
        def maxval(v):
            return np.where(anyopt,np.where(legal,v,-np.inf).max(1),0)
        def tops(v):
            return legal & np.isclose(v,maxval(v)[:,None],rtol=0,atol=1e-5)
        sy,ss,sm=[tops(v) for v in (y,sa,ma)]
        st=np.maximum(1,ss.sum(1));mt=np.maximum(1,sm.sum(1));ym=maxval(y)
        dy,ds,dm=[v[:,tri[0]]-v[:,tri[1]] for v in (y,sa,ma)]; tr=np.abs(dy)>1e-5; comp=tr&legal[:,tri[0]]&legal[:,tri[1]]
        sign=lambda v:np.sign(np.where(np.abs(v)>1e-5,v,0))
        ys,ass,ams=map(sign,(dy,ds,dm));flips=comp&(ass*ams==-1)
        informative=comp.any(1)
        loglegal=legal[ia,lo]
        scalar={'jointly_informative':informative,'target_ambiguous':(sy.sum(1)>1)&informative,
                'true_order_comparisons':comp.sum(1),'strict_rank_flips':flips.sum(1),'flips_towards_truth':(flips&(ams==ys)).sum(1),'flips_away_from_truth':(flips&(ass==ys)).sum(1),
                'sap_pair_credits':((ass*ys+1)*.5*comp).sum(1),'master_pair_credits':((ams*ys+1)*.5*comp).sum(1),
                'topsets_differ':(~(ss==sm).all(1))&informative,'strict_top_disjoint':~(ss&sm).any(1)&informative,
                'sap_top_credit':((ss&sy).sum(1)/st)*informative,'master_top_credit':((sm&sy).sum(1)/mt)*informative,
                'sap_true_regret':(ym-(ss*y).sum(1)/st)*informative,'master_true_regret':(ym-(sm*y).sum(1)/mt)*informative,
                'logged_choice_is_top_sap':ss[ia,lo]&informative&loglegal,'logged_choice_is_top_master':sm[ia,lo]&informative&loglegal,
                'logged_truth_regret':(ym-y[ia,lo])*informative*loglegal}
        for role in ('COLLECTOR','PLANTER'):
            take=roles==role
            for scope,scope_mask in groups.items():
                b=take&scope_mask
                raw={k:float(v[b].sum()) for k,v in scalar.items()}
                raw.update(legal_lt2=int((b&(nopt<2)).sum()),legal_ge2=int((b&(nopt>=2)).sum()),fewer_than_two_distinct_truth_values=int((b&(nopt>=2)&~informative).sum()),
                           all_options_currently_direct=int((b&informative&((mask&~obs)==0)).sum()),eligible_logged_choices=int((b&informative&loglegal).sum()),
                           opportunities=int(b.sum()))
                den=raw['jointly_informative'];pc=raw['true_order_comparisons']
                rr.append(dict(scope=scope,role=role,panel=name,**raw,sap_expected_regret=raw['sap_true_regret']/den if den else None,master_expected_regret=raw['master_true_regret']/den if den else None,
                    sap_top_accuracy=raw['sap_top_credit']/den if den else None,master_top_accuracy=raw['master_top_credit']/den if den else None,
                    sap_pair_accuracy=raw['sap_pair_credits']/pc if pc else None,master_pair_accuracy=raw['master_pair_credits']/pc if pc else None))
            if name=='economic':
                gain=scalar['sap_true_regret']-scalar['master_true_regret']
                for tag,values in [('improved',gain),('worsened',-gain)]:
                    valid=np.flatnonzero(take&informative&scalar['strict_top_disjoint']&(values>1))
                    if len(valid):
                        i=int(valid[np.argmax(values[valid])]); card=games[meta[i][0]]['card']
                        ex.append(dict(case=tag,seed=card['seed'],track=card['arm'],registered_seat=card['seat'],pov=int(seats[i]),step=int(ts[i]),role=role,uid=meta[i][4],
                            options=legal[i].tolist(),directions=['NONE','UP','RIGHT','DOWN','LEFT'],logged=int(lo[i]),
                            truth=y[i].tolist(),sap=sa[i].tolist(),master=ma[i].tolist(),regret_reduction=float(gain[i])))
    old=json.loads(gitfile('data/belief_mae_results.json'));oldinfo={}
    for k,snaps in old.items():
        cols=[int(round(225*r['coverage'])) for r in snaps]
        assert all(abs(n/225-r['coverage'])<=.000051 for n,r in zip(cols,snaps))
        oldinfo[k]=dict(snapshots=len(snaps),possible_cells=225*len(snaps),included_cells=sum(cols),coverage=sum(cols)/(225*len(snaps)),
            reported_unweighted_snapshot_mae=float(np.mean([r['mae_est'] for r in snaps if r['mae_est'] is not None])))
    save(HERE/'results.json',dict(sources=[g['card'] for g in games],seat_frames=2*sum(g['turns'] for g in games),worker_decisions=N,old_two_games=oldinfo,
        cell_error=cellrows,ranking=rr,counterexamples=ex))
    return games,cells,cellrows,rr,oldinfo,ex

if __name__=='__main__':compute()
