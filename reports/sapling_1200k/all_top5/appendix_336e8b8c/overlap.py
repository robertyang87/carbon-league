"""Compare both symmetry development maps with the frozen ladder registry and raw new boards."""
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP.parent))
import map_identity
from collect import BATCHES, REV, fetch
from contrasts import inputs


def main():
    dev=TOP.parent/'master6_actor_dev/symmetry_dev'
    clearance=dev/'clearance.json'
    inp=json.loads(clearance.read_bytes())
    dev_seeds={int(k) for k in inp['boards']}
    dev_terrains=set(inp['boards'].values())
    if inp['source_revision']!=REV or len(dev_seeds)!=2 or len(dev_terrains)!=2:
        raise ValueError('wrong development declaration')
    sources=list(map_identity.SOURCES)
    delta=json.loads((HERE/'map_registry_candidate.json').read_bytes())['source_for_map_identity']
    # The archive can also be checked after a later admission of this source.
    sources=[s for s in sources if (TOP.parent/s[0]).parent in {TOP/b for b in BATCHES}] +[delta]
    who={};seeds=set();terrains=set(); groups=[]
    for name,n,plain_w,id_w in sources:
        src=TOP.parent/name
        packed=src.read_bytes();plain=gzip.decompress(packed);rows=[json.loads(raw) for raw in plain.splitlines()]
        if fetch.sha(plain)!=plain_w or fetch.sha('\n'.join(sorted(r['id'] for r in rows)).encode())!=id_w or n!=len(rows):
            raise ValueError('changed source registry')
        group_s={r['seed'] for r in rows};group_t={r['initial_terrain_u8_sha256'] for r in rows}
        if seeds&group_s or terrains&group_t or len(group_s)!=n or len(group_t)!=n:raise ValueError('reused seed or terrain')
        seeds|=group_s;terrains|=group_t;who.update({r['id']:r for r in rows})
        groups.append(dict(source=name,packed_sha256=fetch.sha(packed),games=n,dev_seed_overlap=len(group_s&dev_seeds),
                           dev_terrain_overlap=len(group_t&dev_terrains)))
    if len(who)!=1475 or len(groups)!=len(BATCHES)+1 or dev_seeds&seeds or dev_terrains&terrains:raise ValueError('overlap')
    infos,matches=inputs();comparisons=[]
    for x in matches:
        if not x.get('prior'):continue
        a,b=[who[x[k]['id']] for k in ('new','prior')]
        verified=[]
        for row in (a,b):
            info=infos[row['id']][0];payload=(fetch.CACHE/info['path']).read_bytes()
            if fetch.sha(payload)!=info['sha256']:raise ValueError('replay digest')
            tape=json.loads(gzip.decompress(payload));terrain=map_identity.terrain_sha256(tape['steps'][0][0]['observation']['full_carbon'])
            if terrain!=row['initial_terrain_u8_sha256']:raise ValueError('raw opening')
            verified.append(terrain)
        if verified[0]==verified[1] or a['seed']==b['seed']:raise ValueError('same map pair')
        comparisons.append(dict(new=a['id'],previous=b['id'],seats_match=x['new']['seat']==x['prior']['seat'],
                         physical_root_terrain_sha256=verified,different_nominal_seeds=True))
    result=dict(revision=REV,symmetry_clearance_sha256=fetch.sha(clearance.read_bytes()),
                development_maps=len(dev_seeds),all_ladder_maps=len(who),new_ladder_maps=groups[-1]['games'],
                seed_overlaps=0,terrain_overlaps=0,new_seed_overlaps=groups[-1]['dev_seed_overlap'],
                new_terrain_overlaps=groups[-1]['dev_terrain_overlap'],sources=groups,
                same_opponent_repeats=len(comparisons),all_distinct_replay_terrain_pairs=len(comparisons),
                comparisons=comparisons,method='For every registered source verify decompressed SHA, sorted-ID SHA and count. Compare physical initial-terrain SHA and nominal seeds independently; additionally recompute both terrain hashes from raw verified first frames for each historical named-opponent pair. Development seeds and terrain identities are in the signed input only.')
    (dev/'ladder_overlap_336e8b8c.json').write_text(json.dumps(result,ensure_ascii=False,indent=1)+'\n')


if __name__=='__main__':main()
