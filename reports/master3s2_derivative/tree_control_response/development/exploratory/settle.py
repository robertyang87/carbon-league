"""Verify each paired original response, then average repeats within the two physical maps."""
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent


def digest(b):return hashlib.sha256(b).hexdigest()
def get(path):
    b=path.read_bytes()
    return json.loads(gzip.decompress(b) if path.suffix=='.gz' else b)


def settle():
    reg=get(HERE/'protocol.json');rep=get(HERE/'repeat/protocol.json')
    assert rep['parent_protocol_sha256']==digest((HERE/'protocol.json').read_bytes())
    by_map=[];max_wall=0.;slow=0;total=0
    for i,x in enumerate(reg['games']):
        seed,seat=x['seed'],x['seat'];per_map=[]
        for name,folder in [('first',HERE),('repeat',HERE/'repeat')]:
            path=folder/f'{seed}_{seat}_result.json';j=get(path)
            assert j['protocol_sha256']==digest((folder/'protocol.json').read_bytes())
            assert (seed,seat,x['step'])==(j['seed'],j['seat'],j['starting_step'])
            row={'stream':name,'root_ordinal':x['ordinal'],'settlement_sha256':digest(path.read_bytes())};opponent=[];physical=[];nsteps=[];per_arm={}
            for arm,tag in [('parent','P'),('new','E')]:
                link=j['responses'][arm];f=folder/link['tape'];assert digest(f.read_bytes())==link['sha256']
                v=get(f);h=v['rows'];assert h[0]['step']==x['step'] and h[-1]['step']==298
                assert [k['step'] for k in h]==list(range(x['step'],299))
                assert v['official_scores']==j['tree_economy'][arm]['scores'] and v['official_margin']==v['official_scores'][seat]-v['official_scores'][1-seat]
                assert h[0]['orders'][seat]==x['arms'][tag]['order']
                opponent.append(h[0]['orders'][1-seat]);physical.append(h[0]['public_sha256']);nsteps.append(len(h))
                clocks=[c['wall'] for item in h for c in item['clocks']];max_wall=max(max_wall,max(clocks));slow+=sum(t>1 for t in clocks);total+=len(clocks)
                econ=j['tree_economy'][arm];ledger=econ['cash_sources']
                net=lambda k:ledger[seat].get(k,0)-ledger[1-seat].get(k,0)
                per_arm[arm]=dict(official_ours=v['official_scores'][seat],official_opponent=v['official_scores'][1-seat],margin=v['official_margin'],
                                   future_net_own_delivery=net('own_delivery'),future_net_ground_rent=net('ground_rent'),target=econ['target'],
                                   target_owned_steps=econ['target_owner_steps'][seat],target_own_income=econ['target_gross_tree_income_by_seat'][seat],
                                   branches_wall_record_max_s=max(clocks),root_command=x['arms'][tag]['order'],tape=link['tape'],sha256=link['sha256'])
            assert opponent[0]==opponent[1] and physical[0]==physical[1] and nsteps[0]==nsteps[1]
            row.update(arms=per_arm,terminal_margin_difference=per_arm['new']['margin']-per_arm['parent']['margin'],
                       opposing_root_order=opponent[0],steps_per_arm=nsteps[0],physical_preaction_sha256=physical[0])
            per_map.append(row)
        assert per_map[0]['physical_preaction_sha256']==per_map[1]['physical_preaction_sha256']
        by_map.append(dict(seed=seed,seat=seat,step=x['step'],map_sha256=x['terrain'],root_order_target='tree model changes an existing native menu choice',
                           mean_paired_difference=mean(s['terminal_margin_difference'] for s in per_map),streams=per_map))
    return dict(independent_maps=2,root_interventions=2,paired_streams=4,completed_suffixes=8,
                mean_over_physical_maps=mean(x['mean_paired_difference'] for x in by_map),
                slower_than_1s_records=slow,max_wall_record_s=max_wall,clock_records=total,
                timing_scope='Injected root orders reuse recorded source-search clocks, simultaneous root original reply is shared within each pair. All later clocks come from individual new native public calls on one CPU, not platform cgroups.',
                games=by_map)


if __name__=='__main__':
    result=settle();dest=HERE/'settlement.json'
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if dest.exists():assert raw==dest.read_bytes()
    else:dest.write_bytes(raw)
    print(digest(raw),result['mean_over_physical_maps'],result['max_wall_record_s'])
