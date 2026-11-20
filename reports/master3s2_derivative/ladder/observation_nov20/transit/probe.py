"""Ten-step transit and booked-cash probe. Run with repository root as first argument."""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(sys.argv[1]); HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reports/sapling_1200k/all_top5'))
import economy_extract as E
PROTO=json.loads((HERE/'protocol.json').read_bytes())
L=ROOT/'reports/master3s2_derivative/ladder'; O=L/'observation_nov20'
identity={}
for filename in ('1c_test.json.gz','ansheng.json.gz','distinct_opponents.json.gz'):
    for g in json.loads(gzip.decompress((O/filename).read_bytes()))['games']:
        identity[g['opponent']]=(g['match'],g['seat'],g['score'])
assert set(identity)==set(PROTO['games'])
def distance(a,b,size):
    x,y=abs(a%size-b%size),abs(a//size-b//size)
    return min(x,size-x)+min(y,size-y)
def extract(players,seat,size):
    # This function never receives the replay, board or future actions.
    sources={}
    for uid,(pos,load,role) in players[seat][2].items():
        if role=='COLLECTOR' and load>0:
            dist=min(distance(pos,base,size) for base in players[seat][1].values())
            sources[uid]=dict(cell=pos,mass=load,dist=dist)
    return sources
def diff(x):return float(x[0]-x[1])
rows=[];meta={}; events=[]; delayed=[]
for opp in PROTO['games']:
    match,seat,score=identity[opp]; path=L/'replays'/(match+'-1.json.gz')
    raw=path.read_bytes(); env=json.loads(gzip.decompress(raw));frames=env['steps'];cfg=env['configuration']
    parties=(seat,1-seat);ps=[row[0]['observation']['players'] for row in frames];size=cfg['size']
    assert len(ps)==300 and [env['rewards'][s] for s in parties]==score
    wallet,_,receipts,_=E.reconstruct(frames,cfg)
    # This second pass is retrospective only. Its results are never passed to extract().
    sim=E.FastCarbon(cfg).load_observation(frames[0][0]['observation'])
    collisions=[]
    for t,frame in enumerate(frames[1:],1):
        sim.step(*[(p['action'] or {}) for p in frame])
        for uid,victim,amount,recipient,winner in sim.last_collisions:
            if amount>0 and winner==1-victim:
                event=dict(t=t,amount=float(amount),uid=uid,winner=winner,opponent=opp)
                if 250<t<=299:collisions.append(event);events.append(event)
    meta[opp]=dict(match=match,seat=seat,score=score,raw_sha256=hashlib.sha256(raw).hexdigest(),near_terminal_steals=collisions)
    for t in range(250,290):
        feats=[extract(ps[t],s,size) for s in parties]
        reachable=[{u:w for u,w in side.items() if w['dist']<=10} for side in feats]
        transit=[sum(w['mass'] for w in hold.values()) for hold in reachable]
        gross=[sum(w['mass'] for w in hold.values()) for hold in feats]
        gap=[ps[t][s][0] for s in parties];endgap=[ps[t+10][s][0] for s in parties]
        labels=[sum(tick[s][0] for tick in wallet[t:t+10]) for s in parties]
        data=dict(opp=opp,t=t,bank_gap=diff(labels),forecast_transit=diff(transit),current_cash_gap=diff(gap),next_cash_gap=diff(endgap),eligible=transit,total_carried=gross,
                  eligible_count=[len(v) for v in reachable],positive_load_count=[len(v) for v in feats],future_bank=labels,
                  max_dist=max([w['dist'] for d in feats for w in d.values()],default=0),conversion={})
        for look in (3,5,10):
            sd=[]
            for s,d in zip(parties,reachable):
                receipts_by_id=collections.defaultdict(list)
                for j in range(t,t+look):
                    for ss,uid,amount in receipts[j]:
                        if ss==s:receipts_by_id[uid].append((j+1,amount))
                paid=[u for u in receipts_by_id if u in d]
                right=[(u,receipts_by_id[u][0][1]-d[u]['mass']) for u in paid]
                if look==10:
                    for u,net in right:
                        if net>0.01:
                            delayed.append(dict(opp=opp,t=t,seat=s,uid=u,old_mass=d[u]['mass'],first_receipt_state=receipts_by_id[u][0][0],first_receipt=receipts_by_id[u][0][1],difference=net))
                sd.append(dict(reachable_paid_n=len(paid),reachable_total_n=len(d),paying_ids_start_mass=sum(d[u]['mass'] for u in paid),
                         eligible_mass=sum(w['mass'] for w in d.values()),first_receipts_from_paying_ids=sum(receipts_by_id[u][0][1] for u in paid),
                         repeated_on_same=sum(sum(a for _,a in receipts_by_id[u][1:]) for u in paid),
                         others_bank=sum(sum(a for _,a in vals) for u,vals in receipts_by_id.items() if u not in d),
                         first_paid_over_initial_count=sum(v>0.01 for _,v in right)))
            data['conversion'][look]=sd
        for j in range(2):
            dec=data['conversion'][10][j]
            assert abs(sum(dec[k] for k in ('first_receipts_from_paying_ids','repeated_on_same','others_bank'))-labels[j])<0.001
        rows.append(data)

def mat(rs,fields):
    return np.asarray([[1.,*[r[k] for k in fields]] for r in rs])
coefs={}
for key in PROTO['games']:
    train=[r for r in rows if r['opp']!=key]; held=[r for r in rows if r['opp']==key]
    y=[r['bank_gap'] for r in train]; coefs[key]={}
    for label,fields in [('cash_lomo',['current_cash_gap']),('transit_lomo',['forecast_transit']),('cash_transit_lomo',['current_cash_gap','forecast_transit'])]:
        coef=np.linalg.lstsq(mat(train,fields),y,rcond=None)[0];coefs[key][label]=coef.tolist()
        for row,val in zip(held,mat(held,fields)@coef):row[label]=float(val)
    for row in held:row['no_increment']=0.
keys=['no_increment','cash_lomo','forecast_transit','transit_lomo','cash_transit_lomo']
def metric(actual,pred):
    y=np.asarray(actual);p=np.asarray(pred);non=np.abs(y)>.005;use=non & (np.abs(p)>.005)
    return dict(mae=float(np.mean(np.abs(y-p))),rmse=float(np.sqrt(np.mean((y-p)**2))),bias=float(np.mean(p-y)),
        non_ties=int(sum(non)),proposed=int(sum(use)),correct_sign=int(np.sum(np.sign(p[use])==np.sign(y[use]))),
        correlation=float(np.corrcoef(p,y)[0,1]) if len(y)>1 and p.std()>0 and y.std()>0 else None)
def reduce(rs):
    out={'n':len(rs),'maps':len({r['opp'] for r in rs}),'deposit':{},'future_wallet':{}}
    for key in keys:
        out['deposit'][key]=metric([r['bank_gap'] for r in rs],[r[key] for r in rs])
        out['future_wallet'][key]=metric([r['next_cash_gap'] for r in rs],[r[key]+r['current_cash_gap'] for r in rs])
    out['load_coverage']=dict(qualified_mass=sum(sum(r['eligible']) for r in rs),all_mass=sum(sum(r['total_carried']) for r in rs),
        max_distance=max(r['max_dist'] for r in rs),qualified=sum(sum(r['eligible_count']) for r in rs),all_positive=sum(sum(r['positive_load_count']) for r in rs))
    out['conversion']={}
    for h in (3,5,10):
        cols=[el for r in rs for el in r['conversion'][h]]
        out['conversion'][h]={k:sum(c[k] for c in cols) for k in cols[0]}
    return out
sets={**{g:[g] for g in identity},'all':list(identity),'ansheng':[g for g in identity if '安生' in g],'other_distinct':[g for g in identity if g not in ['1C-Test'] and '安生' not in g]}
out=dict(protocol=PROTO,games=meta,per_game_lomo_coefficients=coefs,summaries={},disjoint_summaries={},largest_observed_increase_to_first_bank=sorted(delayed,key=lambda x:-x['difference'])[:12],roots=rows)
for label,allowed in sets.items():
    rs=[r for r in rows if r['opp'] in allowed];out['summaries'][label]=reduce(rs)
    out['disjoint_summaries'][label]=reduce([r for r in rs if (r['t']-250)%10==0])
with gzip.GzipFile(filename=str(HERE/'evidence.json.gz'),mode='wb',mtime=0) as f:f.write(json.dumps(out,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())
for label in (*PROTO['games'],'ansheng','other_distinct','all'):
    q=out['summaries'][label];m=q['deposit'];print(label,q['n'],*[round(m[k]['mae'],1) for k in keys],m['forecast_transit']['correct_sign'],m['forecast_transit']['proposed'],round(m['forecast_transit']['correlation'],2),[round(q['future_wallet'][k]['mae'],1) for k in ('no_increment','forecast_transit')],flush=True)
print(out['summaries']['all']['load_coverage'],out['summaries']['all']['conversion'])
print('nonoverlap',[(k,round(v['mae'],2)) for k,v in out['disjoint_summaries']['all']['deposit'].items()])
print('collisions',[(e['opponent'],e['t'],round(e['amount'],1)) for e in events]);print('gains',out['largest_observed_increase_to_first_bank'][:3])
