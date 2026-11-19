"""Count scored and unscored physical boards separately and recover the fault game's actual opening."""
import gzip
import json
import sys
from shared import BASE,HERE,PROTO,ROOT,read,require,save,sha
sys.path.insert(0,str(BASE))
from map_identity import local_board,terrain_sha256


def check():
    p=read(PROTO);seeds=set(p['new_dev']['seeds']);boards={local_board(s) for s in seeds}
    root=BASE/'all_top5';signed={};exclude=[]
    for file in sorted(root.rglob('validation_maps.jsonl.gz')):
        for line in gzip.decompress(file.read_bytes()).splitlines():
            row=json.loads(line);b=row['initial_terrain_u8_sha256'];s=int(row['seed'])
            require(s not in seeds and b not in boards,'scored or fault board overlap')
            signed.setdefault(s,set()).add(b)
        if 'unscored' in file.parts:
            for r in read(file.parent/'manifest.json')['games']:
                raw=file.parent/r['path'];require(sha(raw)==r['sha256'],'raw fault identity')
                z=read(raw);obs=z['steps'][0][0]['observation'];actual=terrain_sha256(obs['full_carbon'])
                require(actual==r['initial_terrain_u8_sha256'] and actual in signed[r['seed']] and actual not in boards and r['seed'] not in seeds,'fault terrain')
                exclude.append(dict(id=r['id'],seed=r['seed'],initial_terrain_sha256=actual,raw_sha256=sha(raw),ranked_score_split=r['score_by_seat'],observed_transitions=r['verified_active_transitions']))
    out=dict(protocol_sha256=sha(PROTO),all_seen_unique_seeds=len(signed),all_seen_unique_terrains=len({t for v in signed.values() for t in v}),
        complete_unique_terrains=len(signed)-len(exclude),fault_games=exclude,seeds_and_terrain_overlap=0)
    require(exclude and out['all_seen_unique_terrains']==len(signed),'missing fault or overlapping historical aliases')
    save(HERE/'clearance_fault.json',out)
    return out

if __name__=='__main__':print(check())
