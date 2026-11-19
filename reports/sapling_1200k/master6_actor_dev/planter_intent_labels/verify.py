"""Independently check the published arrays against frozen cash ledgers, source maps and original frame rosters."""
import argparse
from collections import Counter
import gzip
import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TOP = HERE.parents[1] / 'all_top5'
sys.path.insert(0, str(HERE.parent / 'tree_tenure_dev'))
import dataset

BASE_FIELDS = ('map_index', 't', 'seat', 'worker_number', 'cell', 'action', 'kind',
               'pre_cash', 'quote', 'affordable', 'y_survived', 'y_paid', 'y_actual_price', 'y_post_cell')
MOVES = ('NONE', 'UP', 'RIGHT', 'DOWN', 'LEFT')
COUNTS = ('rows', 'stop', 'paid', 'prefunded', 'financed', 'stop_unpaid_alive', 'stop_unpaid_dead',
          'moving_alive', 'moving_dead')


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())


def old_choice_code(t,seat,number,cell,kind,action,funded,paid):
    # Bound by documented uint16 worker, 15x15 cell, 299-step game, two roles, five actions.
    return ((((t*2+seat)*65536+number)*225+cell)*2+kind)*20+action*4+funded*2+paid


def sorted_code_digest(codes):
    return digest(np.sort(codes).astype('<u8',copy=False).tobytes())


def evidence(maps):
    cards, cash, past, files = {}, {}, {}, {}
    for path, m in dataset.identities():
        reg = path.parent/'validation_maps.jsonl.gz'
        boards = {r['id']:r['initial_terrain_u8_sha256'] for r in map(json.loads,gzip.open(reg,'rt'))}
        for g in m['games']:
            gid = g['id']
            loc=g.get('archive_location') or g.get('frozen_archive')
            check(g['frame_verified'] and gid not in cards and bool(loc) and (bool(loc.get('repo_path')) or loc.get('member')==g['path']), f'bad source {gid}')
            cards[gid] = (g, m['revision'], boards[gid])
        files[str(path.relative_to(ROOT))] = digest(path.read_bytes())
        for gid, sha, ledger in dataset.source_records(path):
            card, _, _ = cards[gid]
            check(card['sha256']==sha and ledger['seed']==card['seed'] and ledger['rewards']==card['score_by_seat']
                  and ledger['turns']==card['transitions'] and gid not in cash, f'cash source differs {gid}')
            purchases = {}
            for p in ledger['purchases']:
                t,s,uid,_,cell,kind,price=p[:7]
                number=int(uid.rsplit('-',1)[1])
                key=(int(t),int(s),number)
                check(key not in purchases and uid==f'player-{s}-worker-{number}', 'duplicate or malformed paid-worker identity')
                purchases[key]=(cell,int(kind=='seize'),price)
            cash[gid]=purchases
            # Older timing archives retained every action where money was already available, plus later-financed purchases.
            rows=ledger['choices']
            codes=np.fromiter((old_choice_code(r[0],r[1],int(r[2].rsplit('-',1)[1]),r[3],int(r[4]=='seize'),r[14],int(not r[17]),int(r[15] is not None)) for r in rows),dtype=np.uint64,count=len(rows))
            past[gid] = dict(choices_sha256=sorted_code_digest(codes),choices=len(rows), stopped=sum(c[14]==0 for c in rows), paid=len(purchases),
                             funded=sum(not c[17] for c in rows), funded_stops=sum(c[14]==0 and not c[17] for c in rows),
                             financed=sum(bool(c[17]) for c in rows))
    check(len(cards)==1542 and cards.keys()==cash.keys()==past.keys()=={m['id'] for m in maps},'frozen cohort differs')
    return cards,cash,past,files


def board_bytes(card, cache):
    path=cache/card['path']
    if path.is_file():
        data=path.read_bytes()
    else:
        location=card.get('archive_location') or card['frozen_archive']
        if location.get('repo_path'):
            p=ROOT/location['repo_path']
            data=p.read_bytes() if p.is_file() else subprocess.check_output(['git','cat-file','blob',location['git_blob']],cwd=ROOT)
        else:
            archive=ROOT/location['archive']
            handle=tarfile.open(archive) if archive.is_file() else tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','show','HEAD:'+location['archive']],cwd=ROOT)))
            with handle as tar:
                data=tar.extractfile(location['member']).read()
    check(digest(data)==card['sha256'],f'raw source changed: {card["id"]}')
    return data


def numeric(d):
    required=set(BASE_FIELDS)
    check(required<=d.keys(),f'missing required array fields {required-d.keys()}')
    lengths={len(v) for v in d.values()}
    check(len(lengths)==1 and all(v.ndim==1 and v.dtype.kind in 'ifub' for v in d.values()),'invalid array type or dimensions')
    for name in ('map_index','t','seat','worker_number','cell','action','kind','y_post_cell'):
        check(d[name].dtype.kind in 'iu',f'{name} not integral')
    for name in ('affordable','y_survived','y_paid'):
        check(d[name].dtype.kind in 'bui' and np.all((d[name]==0)|(d[name]==1)),f'invalid flag {name}')
    check(d['y_post_cell'].dtype==np.dtype('int16') and d['y_actual_price'].dtype==np.dtype('float64'),'unexpected death or price field type')
    for k in ('pre_cash','quote','y_actual_price'):
        check(np.isfinite(d[k]).all(),f'nonfinite economic field {k}')
    live=d['y_survived'].astype(bool);paid=d['y_paid'].astype(bool)
    check(np.all(d['quote']>0) and np.all(d['pre_cash']>=0) and np.all(d['action']<5)
          and np.all((d['action']>=0)&(d['cell']>=0)&(d['cell']<225)&(d['seat']>=0)&(d['seat']<2))
          and np.all((d['kind']==0)|(d['kind']==1)) and np.all(d['worker_number']>=0),'invalid state or action field')
    check(np.all(d['y_post_cell'][~live]==-1) and np.all((d['y_post_cell'][live]>=0)&(d['y_post_cell'][live]<225))
          and np.all((~paid)|(live&(d['action']==0))) and np.all(d['y_actual_price'][~paid]==0),'inconsistent survival or purchase')
    offset=np.array(((0,0),(-1,0),(0,1),(1,0),(0,-1)),dtype=np.int16)
    y,x=np.divmod(d['cell'].astype(np.int16),15)
    destinations=((y+offset[d['action'],0])%15)*15+(x+offset[d['action'],1])%15
    check(np.all(d['y_post_cell'][live]==destinations[live]) and np.array_equal(d['affordable'].astype(bool),d['pre_cash']>=d['quote']),
          'after-cell or pre-funding identity disagrees with selected move and quote')
    check(np.all(d['y_actual_price'][paid]>0) and np.allclose(d['quote'][paid],d['y_actual_price'][paid],rtol=1e-6,atol=1e-5),
          'published price differs from charged price')


def count_fields(one):
    live=one['y_survived'].astype(bool);buy=one['y_paid'].astype(bool)
    stop=one['action']==0;funded=one['affordable'].astype(bool)
    return dict(rows=len(stop),stop=int(stop.sum()),paid=int(buy.sum()),prefunded=int(funded.sum()),
                financed=int((~funded&buy).sum()),stop_unpaid_alive=int((stop&~buy&live).sum()),
                stop_unpaid_dead=int((stop&~buy&~live).sum()),moving_alive=int((~stop&live).sum()),
                moving_dead=int((~stop&~live).sum()))


def compare_purchases(gid,one,records,previous):
    idx=np.flatnonzero(one['y_paid'])
    check(len(idx)==len(records),f'wrong purchase count {gid}')
    for j in idx:
        k=(int(one['t'][j]),int(one['seat'][j]),int(one['worker_number'][j]))
        cell,kind,price=records[k]
        check(cell==one['cell'][j] and kind==one['kind'][j] and abs(price-one['y_actual_price'][j])<=1e-5,
              f'wrong worker, location, purchase type, or paid price: {gid}/{k}')
    paid=one['y_paid'].astype(bool);funded=one['affordable'].astype(bool);stop=one['action']==0
    active=funded|paid
    codes=old_choice_code(one['t'].astype(np.uint64),one['seat'],one['worker_number'],one['cell'],one['kind'],one['action'],funded.astype(np.uint8),paid.astype(np.uint8))
    check(previous==dict(choices_sha256=sorted_code_digest(codes[active]),choices=int((funded|paid).sum()),stopped=int((stop&(funded|paid)).sum()),paid=int(paid.sum()),
                         funded=int(funded.sum()),funded_stops=int((funded&stop).sum()),financed=int((~funded&paid).sum())),
          f'overlap with original cash-eligible choice population differs: {gid}')


def nearest(cell, workers, role):
    y,x=divmod(cell,15)
    ds=[]
    for location,_,kind in workers.values():
        if kind==role:
            cy,cx=divmod(location,15);dy=abs(cy-y);dx=abs(cx-x)
            ds.append(min(dy,15-dy)+min(dx,15-dx))
    return min(ds,default=99)


def raw_check(meta,card,one,cache,example_keys):
    game=json.loads(gzip.decompress(board_bytes(card,cache)))
    frames=game['steps'];cfg=game['configuration'];gid=meta['id']
    check(len(frames)==meta['turns']+1 and game['rewards']==card['score_by_seat'] and
          digest(bytes(frames[0][0]['observation']['full_carbon']))==meta['terrain'],f'raw official terrain or score differs: {gid}')
    ix={(int(t),int(s),int(w)):i for i,(t,s,w) in enumerate(zip(one['t'],one['seat'],one['worker_number']))}
    counted=0;examples={}
    missing_stop=explicit_stop=0
    for t in range(len(frames)-1):
        players=frames[t][0]['observation']['players'];end=frames[t+1][0]['observation']['players']
        bases={pos for pl in players for pos in pl[1].values()}
        trees={p:(s,age) for s,pl in enumerate(players) for p,age in pl[3].values()}
        live_count=sum(age<cfg['treeLifespan'] for _,age in trees.values())
        for s,(friendly,enemy) in enumerate(((players[0],players[1]),(players[1],players[0]))):
            orders=frames[t+1][s]['action'] or {}; end_trees={p for p,_ in end[s][3].values()}
            mine=friendly[0]
            market=cfg['plantCost']
            if len(friendly[3])>cfg['playerTreeProtectiveNumber']:
                market+=cfg['plantCostInflationRatio']*cfg['plantCostInflationBase']**live_count
            for uid,(cell,cargo,role) in friendly[2].items():
                tr=trees.get(cell)
                if role!='PLANTER' or cell in bases or tr and tr[0]==s and tr[1]<cfg['treeLifespan']:continue
                number=int(uid.rsplit('-',1)[1]);key=(t,s,number)
                j=ix.get(key)
                check(j is not None and uid==f'player-{s}-worker-{number}',f'missing raw decision {gid}/{key}')
                counted+=1;kind=int(bool(tr and tr[1]<cfg['treeLifespan']))
                price=cfg['seizeCost'] if kind else market
                order=orders.get(uid) or 'NONE';a=MOVES.index(order);alive=uid in end[s][2]
                paid=alive and a==0 and cell in end_trees
                if a==0:
                    if uid in orders:explicit_stop+=1
                    else:missing_stop+=1
                check(one['cell'][j]==cell and one['kind'][j]==kind and one['action'][j]==a and
                      abs(one['pre_cash'][j]-mine)<.01 and abs(one['quote'][j]-price)<.01 and
                      bool(one['affordable'][j])==(mine>=price) and bool(one['y_survived'][j])==alive and
                      bool(one['y_paid'][j])==paid and one['y_post_cell'][j]==(end[s][2][uid][0] if alive else -1),
                      f'raw action, price, funding, ownership, or death differs: {gid}/{uid}/{t}')
                # For sampled classes, independently read the two seats' own visibility and geometrics.
                outcome=('paid_by_intra_turn_cash' if paid and mine<price else
                         'paid_under_pre_cash' if paid else 'stop_no_pay_survived' if a==0 and alive else
                         'stop_no_pay_died' if a==0 else 'moving_died' if not alive else 'moving_survived')
                if outcome in example_keys and outcome not in examples:
                    y,x=divmod(cell,15);nbr=[((y-1)%15)*15+x,y*15+(x+1)%15,((y+1)%15)*15+x,y*15+(x-1)%15]
                    vis=frames[t][s]['observation']['carbon']
                    example=dict(map=gid,seat=s,root=t,worker=uid,cell=cell,command=order,order_key_present=uid in orders,
                                 quote=price,cash=mine,raw_next_alive=alive,raw_next_owns_tree=cell in end_trees,
                                 visible_neighbor_carbon=sum(vis[q] for q in nbr if q not in trees),
                                 opponent_collector_distance=nearest(cell,enemy[2],'COLLECTOR'),
                                 opponent_planter_distance=nearest(cell,enemy[2],'PLANTER'))
                    def distance(loc):
                        cy,cx=divmod(loc,15); dy=abs(cy-y);dx=abs(cx-x)
                        return min(dy,15-dy)+min(dx,15-dx)
                    pre=dict(remaining_turns=cfg['episodeSteps']-1-t,old_tree_age=tr[1] if tr else -1,
                             own_trees=len(friendly[3]),global_alive_trees=live_count,
                             enemy_collector_dist=example['opponent_collector_distance'],
                             enemy_planter_dist=example['opponent_planter_distance'],
                             own_collector_dist=nearest(cell,friendly[2],'COLLECTOR'),
                             own_home_dist=min(map(distance,friendly[1].values())),
                             enemy_home_dist=min(map(distance,enemy[1].values())),
                             neighbor_enemy_collectors=sum(w[2]=='COLLECTOR' and distance(w[0])<=1 for w in enemy[2].values()),
                             neighbor_enemy_planters=sum(w[2]=='PLANTER' and distance(w[0])<=1 for w in enemy[2].values()),
                             neighbor_teammates=sum(who!=uid and distance(w[0])<=1 for who,w in friendly[2].items()),
                             adjacent_seizable_trees=sum(q in trees and trees[q][0]!=s and trees[q][1]<cfg['treeLifespan']-1 for q in nbr))
                    check(all(one[k][j]==v for k,v in pre.items()) and abs(one['neighbor_carbon'][j]-example['visible_neighbor_carbon'])<.001,
                          f'public geometric input mismatch at {gid}/{uid}/{t}')
                    examples[outcome]=example
    check(counted==len(ix)==len(one['t']),f'unexpected rows outside the original map cohort: {gid}')
    return examples,missing_stop,explicit_stop


def map_meta(maps,cards):
    original=load(dataset.HERE/'maps.json.gz')['maps']
    check(len(maps)==len(original)==len(cards) and len({m['id'] for m in maps})==len(maps)
          and len({m['terrain'] for m in maps})==len(maps),'duplicate map in index')
    for i,(meta,old) in enumerate(zip(maps,original)):
        card,revision,board=cards[meta['id']]
        loc=card.get('archive_location') or card['frozen_archive']
        check(all(meta[k]==v for k,v in old.items()) and meta['map_index']==i and meta['sha256']==card['sha256']
              and meta['terrain']==board and meta['seat_strategy_ids']==card['seat_strategy_ids'] and meta['manifest_revision']==revision
              and meta['archive']==loc.get('archive') and meta['member']==loc.get('member') and meta['git_path']==loc.get('repo_path') and meta['git_blob']==loc.get('git_blob'),
              f'wrong map binding {i} / {meta["id"]}')


def main(cache):
    maps=load(HERE/'maps.json.gz')['maps'];schema=load(HERE/'schema.json');coverage=load(HERE/'coverage.json')
    cards,cash,past,source_hashes=evidence(maps);map_meta(maps,cards)
    required_examples={'paid_by_intra_turn_cash','paid_under_pre_cash','stop_no_pay_survived','stop_no_pay_died','moving_died','moving_survived'}
    # Fixed sample from each split and both base and appendix maps; pay events are reconciled on every game.
    raw_ids={maps[i]['id'] for i in range(0,len(maps),32)}|{maps[-1]['id']}
    for split in {m['split'] for m in maps}:
        raw_ids.add(next(m['id'] for m in maps if m['split']==split))
    check(coverage['maps_sha256']==digest((HERE/'maps.json.gz').read_bytes()) and coverage['schema_sha256']==digest((HERE/'schema.json').read_bytes())
          and coverage['extractor_sha256']==digest((HERE/'source.py').read_bytes()),'release root identity mismatch')
    releases={x['path']:x for x in coverage['files']}
    per_map={x['map_index']:x['counts'] for x in coverage['per_map']}
    check(set(releases)=={p.name for p in HERE.glob('labels_*.npz')} and set(per_map)==set(range(len(maps))), 'release membership mismatch')
    complete=set();summary=Counter();by_map=[];examples={};provenance={}
    sampled_missing=sampled_explicit=0
    for p in sorted(HERE.glob('labels_*.npz')):
        provenance[p.name]=digest(p.read_bytes())
        with np.load(p,allow_pickle=False) as f:
            cols={k:f[k] for k in f.files};numeric(cols)
            check(set(cols)==set(schema['fields']) and all(cols[k].dtype==np.dtype(v) for k,v in schema['fields'].items()),'array does not match public schema')
        idx=cols['map_index'];check(np.all((idx>=0)&(idx<len(maps))),'invalid map index')
        release=releases[p.name]
        check(provenance[p.name]==release['sha256'] and p.stat().st_size==release['bytes'] and len(idx)==release['rows']
              and np.unique(idx).tolist()==list(range(release['first_map_index'],release['last_map_index']+1)),'corrupt or incomplete shard')
        for item in np.unique(idx):
            n=int(item);meta=maps[n];gid=meta['id'];check(n not in complete,'map appears across shards')
            complete.add(n);mask=idx==n;d={k:v[mask] for k,v in cols.items()}
            sorted_i=np.lexsort((d['worker_number'],d['seat'],d['t']))
            same=np.diff(d['t'][sorted_i]).astype(bool)|np.diff(d['seat'][sorted_i]).astype(bool)|np.diff(d['worker_number'][sorted_i]).astype(bool)
            check(same.all() and np.all(d['t']<meta['turns']) and np.all(d['t']>=0),f'nonunique worker-time or invalid turn {gid}')
            compare_purchases(gid,d,cash[gid],past[gid])
            counts=count_fields(d);summary.update(counts);by_map.append([n]+[counts[k] for k in COUNTS])
            stats=per_map[n]
            check(all(counts[k]==stats.get(k,0) for k in ('rows','stop','paid')) and counts['prefunded']+counts['financed']==stats.get('previous_choices_rows',0)
                  and counts['rows']-counts['stop']==stats.get('move',0) and counts['stop']==stats.get('stop_missing_command',0)+stats.get('stop_explicit_command',0),'published map denominator differs')
            for funded,value in ((True,'funded'),(False,'unfunded')):
                for near,zone in ((True,'near'),(False,'far')):
                    filt=(d['action']==0)&(d['affordable']==funded)&((np.minimum(d['enemy_planter_dist'],d['enemy_collector_dist'])<=1)==near)
                    for state,mask in [('paid',d['y_paid']==1),('unpaid_alive',(d['y_paid']==0)&(d['y_survived']==1)),('unpaid_dead',(d['y_paid']==0)&(d['y_survived']==0))]:
                        check(np.count_nonzero(filt&mask)==stats.get(f'stop/{value}/{zone}/{state}',0),f'wrong published risk subpopulation {gid}')
            if gid in raw_ids:
                samples,miss,explicit=raw_check(meta,cards[gid][0],d,cache,required_examples)
                check(miss==stats.get('stop_missing_command',0) and explicit==stats.get('stop_explicit_command',0),'raw source command distinction differs')
                for label,example in samples.items():examples.setdefault(label,example)
                sampled_missing+=miss;sampled_explicit+=explicit
    check(complete==set(range(len(maps))) and set(examples)==required_examples and summary['paid']==145177,'incomplete census')
    accumulated=Counter()
    for cnt in per_map.values():accumulated.update(cnt)
    check(dict(accumulated)==coverage['counts'] and summary['stop']==196707,'release or historical all-map stop count differs')
    result=dict(schema_version=1,study='Independent official cash/worker assertions on published label shards',
                source_sha256={k:digest((HERE/k).read_bytes()) for k in ('verify.py','source.py','schema.json','coverage.json','maps.json.gz')},
                earlier_cash_manifests_sha256=source_hashes,shard_sha256=provenance,coverage=summary,
                maps=len(maps),all_earlier_cash_eligible_choices=sum(x['choices'] for x in past.values()),
                raw_fully_checked_maps=sorted(raw_ids),raw_checked_default_stops=sampled_missing,raw_checked_explicit_stops=sampled_explicit,raw_spot_examples=examples,map_count_fields=['map_index',*COUNTS],
                independent_per_map_counts=sorted(by_map))
    target=HERE/'verification.json'
    payload=(json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
    if target.exists():
        check(target.read_bytes()==payload,'existing independent verification differs')
    else:
        tmp=target.with_suffix('.part');tmp.write_bytes(payload);tmp.replace(target)
    print('verified',len(maps),dict(summary),digest(payload),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-cache',type=Path,default=Path('/workspace/sapling-all-top5-20261119'))
    main(parser.parse_args().replay_cache)
