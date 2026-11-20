"""Publish missing whole-team action windows on the fixed 1,646-scorecard Top Five cohort."""
import argparse
from collections import Counter
import gzip
from hashlib import sha1
import io
import json
import os
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from continue_appendix import previous_manifests

REFERENCE='ea2249324c75a820c983b56a2557278ac703364d'
CUTOFF='e0916a448e36b37e87537bebd5b67ca026a548f0'
HEX='0123456789abcdef'


def require(ok,msg):
    if not ok:
        raise ValueError(msg)


def packed(obj):
    return (json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()


def source_inventory():
    manifests=previous_manifests(TOP/'appendix_ea224932')
    games={};existing={};windows={};source_hashes={}
    for path in manifests:
        d=path.parent
        manifest=fetch.load(path)
        records={g['id']:g for g in manifest['games']}
        require(not (games.keys()&records.keys()),'Overlapping original manifests')
        games.update(records)
        inputs=[path]
        if (d/'turn_actions.json.gz').exists():
            src=d/'turn_actions.json.gz'
            old=json.loads(gzip.decompress(src.read_bytes()))['games']
            ids=[g['id'] for g in old]
            require(set(ids)==records.keys() and len(ids)==len(records),'Existing action table is incomplete')
            for g in old:
                existing[g['id']]=str(src.relative_to(TOP))
            inputs.append(src)
        else:
            if d==TOP:
                rows={}
                for file in sorted((TOP/'economy_data').glob('*.jsonl.gz')):
                    inputs.append(file)
                    for line in gzip.open(file):
                        r=json.loads(line)
                        require(r['id'] not in rows and r['replay_sha256']==records[r['id']]['sha256'],'Different base economic row')
                        rows[r['id']]=r['record']
            else:
                src=d/'cases.json.gz'
                inputs.append(src)
                data=json.loads(gzip.decompress(src.read_bytes()))['rows']
                rows={r['id']:r['economy'] for r in data}
                require(len(rows)==len(data),'Duplicate appendix economic root')
                require(all(r['source_sha256']==records[r['id']]['sha256'] for r in data),'Different appendix economic source')
            require(rows.keys()==records.keys(),'Missing original economic windows')
            for gid,r in rows.items():
                windows[gid]={name:{'start':r[name]['a'],'end':r[name]['b'],
                    'lookback':{str(k):max(0,r[name]['a']-k) for k in (3,5,10)}} for name in ('swing','decisive') if r[name] is not None}
                for name,w in windows[gid].items():
                    require(set(w['lookback'].values())<={pos['t'] for pos in r[name]['positions'] if len(pos['seats'])==2},
                            'Published case has different physical lookback for '+gid)
        for f in inputs:
            name=str(f.relative_to(fetch.ROOT))
            # Validate the compact source ledger against its existing Git blob without re-downloading data.
            oid=fetch.git('rev-parse',CUTOFF+':'+name).decode().strip()
            content=f.read_bytes()
            git_oid=sha1(('blob '+str(len(content))+'\0').encode()+content).hexdigest()
            require(git_oid==oid,'Different source ledger in local working tree '+name)
            source_hashes[str(f.relative_to(TOP))]=oid
    require(len(games)==1646 and len(existing)==144 and set(games)-set(existing)==set(windows),'Frozen cohort differs')
    return games,existing,windows,source_hashes


def source_bytes(g,cache):
    path=cache/g['path']
    if path.is_file():
        raw=path.read_bytes()
    else:
        loc=g['archive_location']
        f=fetch.ROOT/loc['archive']
        with (f.open('rb') if f.is_file() else io.BytesIO(fetch.git('show',CUTOFF+':'+loc['archive']))) as af:
            with tarfile.open(fileobj=af,mode='r:') as ar:
                raw=ar.extractfile(loc['member']).read()
    require(fetch.sha(raw)==g['sha256'],'Raw replay hash changed '+g['id'])
    return raw


def present(players):
    return [set(p[1])|set(p[2]) for p in players]


def extract(g,windows,cache):
    replay=json.loads(gzip.decompress(source_bytes(g,cache)))
    specs=replay['specification']['observation']
    require(all(specs[k]['shared'] and not specs[k].get('hidden',False) for k in ('players','trees')),
            'Seat visibility definition changed')
    steps=replay['steps']
    require(len(steps)-1==g['transitions'] and g['frame_verified'] and list(replay['rewards'])==g['score_by_seat'],
            'Source terminal identity differs for '+g['id'])
    ticks=sorted({t for v in windows.values() for t in range(v['lookback']['10'],v['end'])})
    data=[]
    for t in ticks:
        pre=steps[t][0]['observation'];nxt=steps[t+1]
        require(t==pre['step']==nxt[0]['observation']['step']-1,'Wrong replay frame alignment')
        players=pre['players']
        ids=present(players)
        previous=present(steps[t-1][0]['observation']['players']) if t else [set(),set()]
        explicit=[agent.get('action') or {} for agent in nxt]
        dense=[]
        for s in (0,1):
            require(set(explicit[s])<=ids[s] and set(explicit[s].values())<=fetch.ALLOWED,'Action or actor invalid')
            dense.append({u:explicit[s].get(u) or 'NONE' for u in sorted(ids[s])})
        data.append(dict(t=t,public_before=dict(players=players,trees=pre['trees']),
                     departed_since_previous_by_seat=[sorted(p-c) for p,c in zip(previous,ids)],
                     orders_as_recorded=explicit,orders_all_present=dense,
                     labels_after=dict(cash_by_seat=[p[0] for p in nxt[0]['observation']['players']])))
    return dict(id=g['id'],seed=g['seed'],physical_seats=g['seat_names'],replay_sha256=g['sha256'],
                windows=windows,transitions=data)


def run(cache):
    games,existing,ws,bindings=source_inventory()
    coverage={};counts=Counter()
    directory=HERE/'shards'
    directory.mkdir(exist_ok=True)
    for char in HEX:
        name=char+'.jsonl.gz'
        pending=directory/(name+'.tmp')
        ids=sorted(k for k in ws if k.startswith(char))
        per=Counter()
        with pending.open('wb') as fp:
            with gzip.GzipFile(filename='',fileobj=fp,mode='wb',compresslevel=4,mtime=0) as out:
                for gid in ids:
                    doc=extract(games[gid],ws[gid],cache)
                    out.write(packed(doc));per['games']+=1
                    per['unique_joint_turns']+=len(doc['transitions'])
                    per['window_gates']+=len(doc['windows'])
                    for row in doc['transitions']:
                        for seat in (0,1):
                            per['base_and_worker_action_slots']+=len(row['orders_all_present'][seat])
                            per['slots_with_omitted_orders']+=len(row['orders_all_present'][seat])-len(row['orders_as_recorded'][seat])
                            per['removed_entities_recorded']+=len(row['departed_since_previous_by_seat'][seat])
        fp=directory/name
        os.replace(pending,fp)
        coverage[name]=dict(**per,bytes=fp.stat().st_size,sha256=fetch.sha(fp.read_bytes()))
        counts.update(per)
        print(name,dict(per),flush=True)
    stamp=dict(schema='sapling-full-team-economic-action-windows/1',source_revision=REFERENCE,index_cutoff=CUTOFF,
        archived_scored_games=len(games),existing_window_games=len(existing),supplement_games=len(ws),
        earlier_coverage_by_file=dict(sorted(Counter(existing.values()).items())),
        previous_scored_game_ids_sha256=fetch.sha('\n'.join(sorted(games)).encode()),
        supplement_game_ids_sha256=fetch.sha('\n'.join(sorted(ws)).encode()),
        existing_game_ids_sha256=fetch.sha('\n'.join(sorted(existing)).encode()),
        shard_totals=dict(counts),shards=coverage,source_ledger_git_blobs=bindings,
        extractor_sha256=fetch.sha(Path(__file__).read_bytes()))
    fetch.atomic(HERE/'coverage.json',packed(stamp))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--cache',type=Path,default=fetch.CACHE)
    run(cli.parse_args().cache)
