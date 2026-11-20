"""Reserve pre-action planter opportunities before revealing this policy's decisions."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


archive=module('registered_archive_roots',FAMILY/'return_rank/development/register_roots.py')
tree=module('observable_tree_geometry',FAMILY/'ladder/tree_payback/runtime.py')
econ=module('immutable_game_configuration',FAMILY/'freight_response/run.py').econ


def opportunities(observation,config):
    size,life=config['size'],config['treeLifespan']
    players=observation['players'];seat=observation['player'];step=observation['step']
    own,enemy=players[seat],players[1-seat]
    bases={v for p in players for v in p[1].values()};our_trees={v[0] for v in own[3].values()}
    if own[0]<config['seizeCost']:return []
    targets=[]
    for worker,(pos,_,kind) in own[2].items():
        if kind!='PLANTER' or pos in bases|our_trees:continue
        for tree_id,(cell,age) in enemy[3].items():
            d=tree.distance(pos,cell,size);life_left=min(life-age,config['episodeSteps']-1-step)
            if d>2 or life_left<=d+1 or (d>0 and own[0]<tree.quoted_price(players,seat,config)):continue
            targets.append(dict(worker=worker,pos=pos,enemy_tree=tree_id,target=cell,age=age,distance=d,life_left=life_left))
    return targets


def freeze():
    cfg=econ.config()
    fitted=json.loads((HERE.parent/'weights.json').read_bytes())
    used=set(fitted['train_maps']+fitted['held_maps'])
    earlier=FAMILY/'return_rank/development/root_protocol.json'
    raw=earlier.read_bytes();original=json.loads(raw)
    assert archive.sha(raw)=='3b5bba544d13cb26ae9692e2414a1d14a49277b0232864a31230f8ea3a040f4c' and len(used)==32
    placement=(FAMILY/'ladder/exposed_ladder_maps.json').read_bytes()
    placements=json.loads(placement)
    assert archive.sha(placement)==fitted['exposure_source'] and len(placements)==35
    excluded={r['opening_carbon_bytes_sha256'] for r in placements}
    tickets=[];frames={}
    for x in original['roots']:
        assert x['terrain'] not in used|excluded
        if x['archive_id'] not in frames:
            path=FAMILY/'ladder'/x['lane']/'development/hosted'/(str(x['archive_id'])+'.zip')
            assert archive.sha(path.read_bytes())==x['archive_sha256']
            with zipfile.ZipFile(path) as z:
                payload=z.read('parent/observations.jsonl.gz');assert archive.sha(payload)==x['public_stream_sha256']
            frames[x['archive_id']]=[json.loads(l) for l in gzip.decompress(payload).splitlines()]
        seat=x['seat']; start,end=x['window']; available=[]
        assigned=(x['ordinal']%3+seat+x['seed']-621758592)%3
        for f in frames[x['archive_id']][start:end]:
            obs=f['observations'][seat]; candidates=[v for v in opportunities(obs,cfg) if v['distance']==assigned]
            if candidates:
                # Prefer more legal approaches, hold a younger contested tree for longer, then act earlier.
                q=min(candidates,key=lambda a:(a['distance'],-a['life_left'],a['worker'],a['enemy_tree']))
                available.append((q['distance'],-len(candidates),-q['life_left'],obs['step'],q,f))
        row={k:x[k] for k in ('ordinal','lane','archive_id','archive_sha256','seat','seed','terrain','window','public_stream_sha256')}
        row['available_turns']=len(available);row['assigned_distance']=assigned
        if available:
            _,_,_,t,q,f=min(available,key=lambda a:a[:4])
            row.update(step=t,preaction_opportunity=q, public_observation_sha256=f['public_observation_sha256'][seat])
        tickets.append(row)
    return dict(schema='tree-control-original-response/1', method='Previous untouched-paper deck of four development maps/two seats/three windows. Use only preaction public observations. Assign distance 0, 1 or 2 by (window index + seat + map index) modulo 3 before looking at model outputs. Within that distance maximize number of affordable worker-tree pairs and remaining productive lifespan, then earliest turn. Retain empty windows. No new model, marks, enemy commands or future views in selection.',
                root_test='Native parent and trained variant twice each, shared same-root private RNG and identical prior own-seat views. Alternate P/E/E/P, E/P/P/E. An attributed change must be stable within each arm and the new winning whole order absent in the parent menu, explicitly offered by the new model; require equal width, depth and rollouts across repeats. Log every unqualified or empty row. On any qualifying change, compare two identical native searches with all learned slopes set to zero, leaving training means/intercept and rule/menus active. A choice that survives ablation cannot be credited to the learned heads. Direct seizure on an already-occupied tree has no learned score.',
                response='Up to two first qualified roots in ordinal order on different physical maps. Common original root answer then common parent and original policies with identical starting RNG and belief states, real adaptive .58/T1 replies until official terminal. Separate first 10 moves, tree cash, target control, fees, losses, scores and timing. Follow the historical record for prefix only. Local diagnostic.',
                source_streams_sha256=archive.sha(raw),parent_sha256=original['parent_sha256'],original_sha256=original['original_sha256'],
                economic_predictors_training_and_held_map_count=len(used),known_ladder_opening_count=len(excluded),ladder_placement_sha256=archive.sha(placement),prior_map_overlap=[],roots=tickets)


if __name__=='__main__':
    dest=HERE/'roots.json';b=(json.dumps(freeze(),sort_keys=True,indent=2)+'\n').encode()
    if dest.exists():assert dest.read_bytes()==b
    else:dest.write_bytes(b)
    print(hashlib.sha256(b).hexdigest(),sum('step' in x for x in json.loads(b)['roots']))
