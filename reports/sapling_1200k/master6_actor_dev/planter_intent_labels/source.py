"""Export every exposed planter turn at a purchasable kind of location, without filtering on cash, action or success."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import subprocess
import json
import os
from pathlib import Path
import tarfile

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
TENURE=HERE.parent/'tree_tenure_dev'
TOP=BASE/'all_top5'
CACHE=Path('/workspace/sapling-all-top5-20261119/replays')
WIDTH=64
MOVES=('NONE','UP','RIGHT','DOWN','LEFT')
# Coordinates in all public geometry are physical cells, independently of seat orientation.
DISTS=[[min(abs(p//15-c//15),15-abs(p//15-c//15))+min(abs(p%15-c%15),15-abs(p%15-c%15))
        for c in range(225)] for p in range(225)]
ADJ=[tuple(i for i,d in enumerate(ds) if d==1) for ds in DISTS]
NAMES=('map_index t seat worker_number cell action kind '
       'pre_cash quote affordable neighbor_carbon remaining_turns old_tree_age own_trees global_alive_trees '
       'enemy_collector_dist enemy_planter_dist own_collector_dist own_home_dist enemy_home_dist '
       'neighbor_enemy_collectors neighbor_enemy_planters neighbor_teammates adjacent_seizable_trees '
       'y_survived y_paid y_actual_price y_post_cell').split()
DTYPES=('u2 u2 u1 u2 u1 u1 u1 f8 f8 u1 f4 u2 i2 u1 u1 u1 u1 u1 u1 u1 u1 u1 u1 u1 u1 u1 f8 i2').split()
assert len(DTYPES)==len(NAMES)
TYPES=dict(zip(NAMES,DTYPES))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    b=path.read_bytes()
    return json.loads(gzip.decompress(b) if path.suffix=='.gz' else b)


def json_save(file,d):
    p=json.dumps(d,allow_nan=False,ensure_ascii=False,separators=(',',':')).encode()+b'\n'
    if file.suffix=='.gz':p=gzip.compress(p,mtime=0)
    if file.exists():
        if file.read_bytes()!=p:raise ValueError(f'frozen JSON differs: {file}')
    else:
        tmp=file.with_suffix('.partial');tmp.write_bytes(p);os.replace(tmp,file)


def metadata():
    original=read(TENURE/'maps.json.gz')
    cards={};manifests=[]
    archive_manifests={sha(p.read_bytes()):p for p in (TOP/'manifest.json',*TOP.glob('appendix_*/manifest.json'))}
    for source in original['sources']:
        p=archive_manifests[source['manifest_sha256']];manifest=read(p)
        assert manifest['revision']==source['revision']
        manifests.append({'path':str(p.relative_to(ROOT)),**source})
        for c in manifest['games']:
            assert c['id'] not in cards
            cards[c['id']]=(c,manifest['revision'])
    rows=[]
    for idx,item in enumerate(original['maps']):
        card,revision=cards[item['id']]
        assert (card['sha256'],card['seat_names'],card['seed'],card['transitions'])==(item['sha256'],item['names'],item['seed'],item['turns'])
        loc=card.get('archive_location') or card.get('frozen_archive')
        if not loc:raise ValueError("missing publicly archived raw source")
        rows.append({**item,'map_index':idx,'seat_strategy_ids':card['seat_strategy_ids'],'manifest_revision':revision,
                     'archive':loc.get('archive'),'member':loc.get('member'),'git_path':loc.get('repo_path'),'git_blob':loc.get('git_blob')})
    assert len(rows)==len(cards)==len({m['terrain'] for m in rows})==len({m['seed'] for m in rows})==1542
    return {'schema_version':1,'original_sha256':sha((TENURE/'maps.json.gz').read_bytes()),'manifests':manifests,'maps':rows}


def raw_game(item,cache):
    raw_path=cache/(item['id']+'.json.gz')
    if raw_path.is_file():data=raw_path.read_bytes()
    elif item['git_path']:
        path=ROOT/item['git_path']
        data=path.read_bytes() if path.exists() else subprocess.check_output(['git','cat-file','blob',item['git_blob']],cwd=ROOT)
    else:
        path=ROOT/item['archive']
        if path.exists():
            with tarfile.open(path) as reader:data=reader.extractfile(item['member']).read()
        else:
            blob=subprocess.check_output(['git','show','HEAD:'+item['archive']],cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(blob)) as reader:data=reader.extractfile(item['member']).read()
    if sha(data)!=item['sha256']:raise ValueError('raw bytes differ: '+item['id'])
    return json.loads(gzip.decompress(data))


def dist(v,positions):
    return min((v[p] for p in positions),default=99)


def game(item,cache,columns,counts):
    env=raw_game(item,cache);frames=env['steps'];cfg=env['configuration']
    assert len(frames)-1==item['turns'] and cfg['size']==15 and cfg['randomSeed']==item['seed']
    assert sha(bytes(frames[0][0]['observation']['full_carbon']))==item['terrain']
    for t in range(len(frames)-1):
        before=frames[t];after=frames[t+1]
        players=before[0]['observation']['players'];post=after[0]['observation']['players']
        bases={c for p in players for c in p[1].values()}
        trees={cell:(s,age) for s,p in enumerate(players) for cell,age in p[3].values()}
        post_trees=[{c for c,a in q[3].values()} for q in post]
        life=cfg['treeLifespan'];alive=sum(a<life for _,a in trees.values())
        market=cfg['plantCost']+cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**alive
        for seat in (0,1):
            own,opp=players[seat],players[1-seat]
            own_homes,opp_homes=list(own[1].values()),list(opp[1].values())
            own_collectors=[w[0] for w in own[2].values() if w[2]=='COLLECTOR']
            enemy_collectors=[w[0] for w in opp[2].values() if w[2]=='COLLECTOR']
            enemy_planters=[w[0] for w in opp[2].values() if w[2]=='PLANTER']
            price=cfg['plantCost'] if len(own[3])<=cfg['playerTreeProtectiveNumber'] else market
            command=after[seat]['action'] or {};carbon=before[seat]['observation']['carbon']
            for uid,(cell,_,role) in own[2].items():
                tree=trees.get(cell)
                if role!='PLANTER' or cell in bases or (tree and tree[0]==seat and tree[1]<life):
                    continue
                action=MOVES.index(command.get(uid) or 'NONE')
                if action==0:counts['stop_'+('missing_command' if uid not in command else 'explicit_command')]+=1
                kind=int(bool(tree and tree[1]<life))
                quote=cfg['seizeCost'] if kind else price;affordable=own[0]>=quote
                around=ADJ[cell];d=DISTS[cell]
                seen=sum(float(carbon[p]) for p in around if p not in trees)
                near_c=sum(d[p]<=1 for p in enemy_collectors);near_p=sum(d[p]<=1 for p in enemy_planters)
                teammates=sum(d[w[0]]<=1 for k,w in own[2].items() if k!=uid)
                adjacent_enemy=sum(p in trees and trees[p][0]!=seat and trees[p][1]<life-1 for p in around)
                after_worker=post[seat][2].get(uid)
                survived=after_worker is not None
                paid=action==0 and survived and cell in post_trees[seat]
                when=uid.rsplit('-',1)[1];assert uid==f'player-{seat}-worker-{when}'
                if paid:assert after_worker[0]==cell
                row=(item['map_index'],t,seat,int(when),cell,action,kind,own[0],quote,int(affordable),seen,
                     cfg['episodeSteps']-1-t,tree[1] if tree else -1,len(own[3]),alive,
                     dist(d,enemy_collectors),dist(d,enemy_planters),dist(d,own_collectors),dist(d,own_homes),dist(d,opp_homes),
                     near_c,near_p,teammates,adjacent_enemy,int(survived),int(paid),quote if paid else 0.,
                     after_worker[0] if survived else -1)
                for col,val in zip(columns,row):col.append(val)
                counts['rows']+=1;counts['stop' if action==0 else 'move']+=1;counts['paid']+=int(paid)
                if affordable or paid:counts['previous_choices_rows']+=1
                zone='near' if near_c or near_p else 'far'
                fin='funded' if affordable else 'unfunded'
                state='paid' if paid else ('unpaid_alive' if survived else 'unpaid_dead')
                if action==0:counts['stop/'+fin+'/'+zone+'/'+state]+=1


def export(cache):
    meta=metadata();json_save(HERE/'maps.json.gz',meta)
    schema=dict(schema_version=1,fields=TYPES,identity=['map_index','t','seat','worker_number'],
                worker_uid_template='player-{seat}-worker-{worker_number}',action_codes=MOVES,kind_codes=['plant','seize'],
                scope='Each physically live planter on a location where a new tree can be planted or a live enemy tree can be seized. Includes every actual order regardless of pre-turn affordability. Excludes bases and friendly live trees. Complete numeric-score games only.',
                availability=dict(pre_action=NAMES[6:24]+['cell'], observed_action='action', identity=NAMES[:4],
                                  after_transition=NAMES[24:],label_key_prefix='y_'),
                missing_distance=99,after_death_cell=-1,unpaid_debit=0,remaining_turns='episodeSteps-1-t',
                nearest_distances='toroidal Manhattan; neighbor counts mean distance <=1. Own planter itself excluded from neighbor_teammates.',
                quote='Full-precision start-of-turn official price. y_actual_price set to this price exactly when a stationary surviving planter acquires its tile. Independent verification joins each such identity to the published per-worker actual debit.',
                neighbor_carbon='Own-seat public four-neighbor carbon, excluding currently occupied tree cells; no full-map carbon used by any input.',
                availability_note='The focal chosen action becomes available when the actor decides; recorded opponent orders and following public states are never input features. Maps and registration names are grouping identifiers. Input labels for counterfactual actions are unavailable.',
                previous_paid_ledger_filter='affordable OR y_paid, a retrospective filter. Clean pre-action financially eligible population uses only affordable.')
    json_save(HERE/'schema.json',schema)
    summaries=[];overall=Counter();files=[]
    for start in range(0,len(meta['maps']),WIDTH):
        idx=start//WIDTH;output=HERE/f'labels_{idx:03}.npz';columns=[[] for name in NAMES];batch_counts=Counter()
        for item in meta['maps'][start:start+WIDTH]:
            c=Counter();game(item,cache,columns,c)
            batch_counts.update(c); summaries.append({'map_index':item['map_index'],'counts':dict(c)})
        arrays={k:np.asarray(column,dtype=TYPES[k]) for k,column in zip(NAMES,columns)}
        if output.is_file():
            with np.load(output,allow_pickle=False) as saved:
                if set(saved.files)!=set(NAMES) or any(not np.array_equal(saved[k],arr) or saved[k].dtype!=arr.dtype for k,arr in arrays.items()):
                    raise ValueError(f'frozen part differs: {output.name}')
        else:
            tmp=output.with_suffix('.partial.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,output)
        files.append({'path':output.name,'sha256':sha(output.read_bytes()),'bytes':output.stat().st_size,'first_map_index':start,
                      'last_map_index':min(start+WIDTH,len(meta['maps']))-1,'rows':batch_counts['rows']})
        overall.update(batch_counts);print(output.name,batch_counts['rows'],overall['paid'],flush=True)
    assert overall['previous_choices_rows']==1359485 and overall['paid']==145177 and overall['stop']==196707
    json_save(HERE/'coverage.json',dict(schema_sha256=sha((HERE/'schema.json').read_bytes()),maps_sha256=sha((HERE/'maps.json.gz').read_bytes()),
              extractor_sha256=sha(Path(__file__).read_bytes()),files=files,counts=dict(overall),per_map=summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=CACHE)
    export(p.parse_args().cache)
