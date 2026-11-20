"""Independently reload the four complete arms and compare paired, seat-1 cash flows."""
import gzip
import hashlib
import json
from math import ceil
from pathlib import Path

HERE=Path(__file__).resolve().parent

def actual(path, expected_sha=None):
    b=path.read_bytes()
    if expected_sha:assert hashlib.sha256(b).hexdigest()==expected_sha
    return json.loads(gzip.decompress(b) if path.suffix=='.gz' else b)


def summary():
    plan=actual(HERE/'protocol.json');res=[]
    for t in plan['tickets']:
        n=t['seed'];f=actual(HERE/f'{n}_pair.json')
        arms={k:actual(HERE/f'{n}_{k}.json.gz',f[k]['evidence_sha256']) for k in ('mixed','double')}
        for x in arms.values():
            assert x['root_public']==t['root_observation_hash']
            assert x['rows'][0]['public_sha256']==t['root_observation_hash']
            assert x['rows'][0]['orders']==x['actual_joint_opening']
            assert x['result']['statuses']==['DONE','DONE']
        l,r=arms['mixed'],arms['double']
        assert l['postroot_private_shared_sha']==r['postroot_private_shared_sha']
        assert l['actual_joint_opening'][0]==r['actual_joint_opening'][0]==t['historical_joint_opening'][0]
        assert sorted(l['actual_joint_opening'][1].values())==['RECCOLLECTOR','RECPLANTER']
        assert sorted(r['actual_joint_opening'][1].values())==['RECCOLLECTOR','RECCOLLECTOR']
        assert sum(l['actual_joint_opening'][1][k]!=v for k,v in r['actual_joint_opening'][1].items())==1
        info={}
        for k,y in arms.items():
            last=y['result']['phases'][-1];by_side=last['cumulative_receipts']
            econ={field:by_side[1].get(field,0.)-by_side[0].get(field,0.) for field in sorted(set(by_side[0])|set(by_side[1]))}
            assert abs(sum(econ.values())-last['margin'])<1e-5
            clocks=[]
            for side in (0,1):
                walls=sorted(row['clock'][side]['wall'] for row in y['rows'])
                q=lambda a:walls[ceil(a*len(walls))-1]
                clocks.append(dict(y['result']['calls_per_seat'][side],p95=q(.95),p99=q(.99),
                    cpu_seconds=sum(row['clock'][side]['cpu'] for row in y['rows']),
                    searched=sum(row['clock'][side]['searched'] for row in y['rows']),
                    zero_depth_search=sum(row['clock'][side]['searched'] and row['clock'][side]['H']==0 for row in y['rows'])))
            info[k]=dict(scores=y['result']['official_scores'],margin=y['result']['seat1_margin'],win=y['result']['seat1_win'],
                         steps=y['result']['steps'],final_flow_margin=econ,elapsed=y['result']['whole_wall'],timing=clocks,
                         native_opening_equal_to_record=[call[0]==order for call,order in zip(y['root_original_replies'], t['historical_joint_opening'])],
                         evidence=f[k]['evidence_sha256'])
        deltas={name:info['double']['final_flow_margin'].get(name,0)-info['mixed']['final_flow_margin'].get(name,0)
                   for name in sorted(set(info['double']['final_flow_margin'])|set(info['mixed']['final_flow_margin']))}
        bystep={k:{p['step']:p for p in y['result']['phases']} for k,y in arms.items()}
        phase=[]
        for s in sorted(set(bystep['mixed'])&set(bystep['double'])):
            phase.append(dict(step=s,mixed=bystep['mixed'][s]['margin'],double=bystep['double'][s]['margin'],
                              margin_change=bystep['double'][s]['margin']-bystep['mixed'][s]['margin']))
        res.append(dict(seed=n,seat=1,arms=info,phase=phase,
            official_margin_change=info['double']['margin']-info['mixed']['margin'],cash_margin_components_change=deltas))
    out=dict(orientation='seat 1 subject minus seat 0 original, change is double minus mixed',maps=res,
             mean_official_margin_change=sum(r['official_margin_change'] for r in res)/len(res),
             wins={k:sum(r['arms'][k]['win'] for r in res) for k in ('mixed','double')})
    p=HERE/'paired_summary.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n');print(out['mean_official_margin_change'],out['wins'])

if __name__=='__main__':summary()
