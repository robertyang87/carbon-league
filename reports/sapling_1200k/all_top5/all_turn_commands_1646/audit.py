"""Independently align all retrospective two-seat command windows against their archived public tapes."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from continue_appendix import previous_manifests

CUTOFF='ea2249324c75a820c983b56a2557278ac703364d'
NEW_GAME='caf9ede770dd206b72d0c8f414cb79a3-1'


def need(b,m):
    if not b:raise ValueError(m)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def economic(rows):
    return {k:(rows[k]['a'],rows[k]['b']) for k in ('swing','decisive') if rows.get(k) is not None}


def ledgers():
    games={};windows={};old={}
    for p in previous_manifests(TOP/'appendix_ea224932'):
        manifest=json.loads(p.read_bytes())
        if p.parent==TOP:
            for source in sorted((TOP/'economy_data').glob('*.jsonl.gz')):
                for line in gzip.open(source):
                    r=json.loads(line);windows[r['id']]=economic(r['record'])
        else:
            for r in json.load(gzip.open(p.parent/'cases.json.gz'))['rows']:
                windows[r['id']]=economic(r['economy'])
        for g in manifest['games']:
            need(g['id'] not in games,'Duplicate source identity');games[g['id']]=g
        legacy=p.parent/'turn_actions.json.gz'
        if legacy.is_file():
            for g in json.load(gzip.open(legacy))['games']:
                need(g['id'] not in old,'Duplicate existing window');old[g['id']]=g
    need(len(games)==1646 and len(old)==144 and NEW_GAME not in games and games.keys()==windows.keys(),'Wrong frozen cohort')
    return games,windows,old


def replay(g):
    f=fetch.CACHE/g['path']
    if f.is_file():
        b=f.read_bytes()
    else:
        loc=g['archive_location'];archive=fetch.ROOT/loc['archive']
        with tarfile.open(archive) as z:b=z.extractfile(loc['member']).read()
    need(hashlib.sha256(b).hexdigest()==g['sha256'],'Unbound physical replay: '+g['id'])
    return json.loads(gzip.decompress(b))


def review(g,r,w,legacy,co):
    source=replay(g);frames=source['steps'];gid=g['id']
    need(r['physical_seats']==g['seat_names'] and r['replay_sha256']==g['sha256'] and source['rewards']==g['score_by_seat'] and
         source['configuration']['randomSeed']==g['seed'] and len(frames)==g['transitions']+1,'Identity, terminal or source bytes '+gid)
    obsdef=source['specification']['observation']
    need(all(obsdef[k].get('shared') and not obsdef[k].get('hidden',False) for k in ('players','trees')),'Nonpublic field in actor context')
    named=r['windows'];wanted=set()
    if not legacy:need(set(named)==w.keys() and r['seed']==g['seed'],'Missing or fabricated window')
    for label,(start,stop) in w.items():
        co[label+'_windows']+=1
        expected={str(k):max(0,start-k) for k in (3,5,10)}
        if legacy:need(named[label]==[expected['10'],start,stop],'Legacy window alignment '+gid)
        else:need(named[label]=={'start':start,'end':stop,'lookback':expected},'Economic window alignment '+gid)
        wanted.update(range(expected['10'],stop))
    if legacy:
        for label,span in named.items():
            if label not in w:
                need(label.startswith('final_collector_loss_seat'),'Unexpected old window name')
                wanted.update(range(span[0],span[2]))
    ticks=[z['t_before' if legacy else 't'] for z in r['transitions']]
    need(ticks==sorted(wanted),'Missing, duplicate or unordered window step '+gid)
    for t,row in zip(ticks,r['transitions']):
        pub=frames[t][0]['observation'];after=frames[t+1];players=pub['players']
        commands=[x.get('action') or {} for x in after]
        cash=[p[0] for p in after[0]['observation']['players']]
        need(pub['step']==t and after[0]['observation']['step']==t+1,'Wrong transition')
        if legacy:
            need(row['public_players_before']==players and row['cash_after']==cash and row['commands_as_recorded']==commands,
                 'Wrong old public state/cash/order '+gid)
        else:
            need(set(row)=={'t','public_before','departed_since_previous_by_seat','orders_as_recorded','orders_all_present','labels_after'} and
                 row['public_before']=={'players':players,'trees':pub['trees']} and row['labels_after']=={'cash_by_seat':cash} and row['orders_as_recorded']==commands,
                 'Future/private fields or wrong public state/cash/order '+gid)
        for s in (0,1):
            roster=players[s][1].keys()|players[s][2].keys()
            order=commands[s]
            need(order.keys()<=roster,'Action references a departed or unknown UID')
            for uid,action in order.items():
                legal={'NONE',None,'RECCOLLECTOR','RECPLANTER'} if uid in players[s][1] else {'NONE',None,'UP','RIGHT','DOWN','LEFT'}
                need(action in legal,'Movement/role mismatch '+gid)
            if not legacy:
                before=frames[t-1][0]['observation']['players'][s] if t else [0,{},{}]
                departed=(before[1].keys()|before[2].keys())-roster
                need(row['departed_since_previous_by_seat'][s]==sorted(departed) and
                     row['orders_all_present'][s]=={uid:order.get(uid) or 'NONE' for uid in roster},'Dense order/roster mismatch '+gid)
                co['departed_actor_ids']+=len(departed)
            co['physical_actor_slots']+=len(roster);co['explicit_commands']+=len(order);co['omitted_none']+=len(roster)-len(order)
        co['unique_joint_steps']+=1
    co['games']+=1;co['no_decisive_window']+=int('decisive' not in w)


def main():
    a=argparse.ArgumentParser();a.add_argument('--parts',required=True,type=Path);a.add_argument('--target',required=True,choices=['legacy','shards','merge'])
    args=a.parse_args();args.parts.mkdir(parents=True,exist_ok=True)
    games,econ,legacy=ledgers()
    tags=['old']+list('0123456789abcdef')
    def path(tag):return args.parts/(tag+'.json')
    def scan(tag,entries):
        counters=Counter();seen=set()
        for r in entries:
            gid=r['id'];need(gid not in seen,'Duplicate output row')
            review(games[gid],r,econ[gid],tag=='old',counters);seen.add(gid)
        want=set(legacy) if tag=='old' else {gid for gid in games if gid.startswith(tag) and gid not in legacy}
        need(seen==want,'Incomplete window shard '+tag)
        source_sha=sha(HERE/'shards'/(tag+'.jsonl.gz')) if tag!='old' else sorted({sha(p.parent/'turn_actions.json.gz') for p in previous_manifests(TOP/'appendix_ea224932') if (p.parent/'turn_actions.json.gz').is_file()})
        d=dict(tag=tag,ids=sorted(seen),counts=counters,source_hash=source_sha,program_hash=sha(Path(__file__)))
        fetch.dump(path(tag),d);print(tag,dict(counters),flush=True)
    if args.target=='legacy':
        scan('old',legacy.values());return
    if args.target=='shards':
        for tag in tags[1:]:
            shard=HERE/'shards'/(tag+'.jsonl.gz')
            while not shard.is_file():time.sleep(1)
            with gzip.open(shard) as f:scan(tag,(json.loads(line) for line in f))
        return
    rows=[json.loads(path(k).read_bytes()) for k in tags];meta=json.loads((HERE/'coverage.json').read_bytes())
    ids=[i for r in rows for i in r['ids']]
    need(len(ids)==len(set(ids))==1646 and set(ids)==games.keys() and all(r['program_hash']==sha(Path(__file__)) for r in rows),'Incomplete or stale audit')
    for r in rows[1:]:
        m=meta['shards'][r['tag']+'.jsonl.gz'];c=r['counts']
        need(r['source_hash']==m['sha256']==sha(HERE/'shards'/(r['tag']+'.jsonl.gz')) and
             c['physical_actor_slots']==m['base_and_worker_action_slots'] and c['omitted_none']==m['slots_with_omitted_orders'] and
             c['unique_joint_steps']==m['unique_joint_turns'] and c['departed_actor_ids']==m['removed_entities_recorded'],'Receipt total mismatch')
    total=lambda rr:dict(sum((Counter(x['counts']) for x in rr),Counter()))
    out=dict(scope='All existing 144 and supplementary 1502 scored games through ea224932; later game caf9ede7 separately archived. Each source independently compared, including 3/5/10 cutoffs, all physical commands and visible rosters. Cash and issued simultaneous orders used after the recorded action only.',
             source_revision=CUTOFF,program_sha256=sha(Path(__file__)),archive_index_sha256=sha(HERE/'coverage.json'),
             existing=total(rows[:1]),supplement=total(rows[1:]),all=total(rows),parts=[{k:v for k,v in r.items() if k!='ids'} for r in rows])
    fetch.dump(HERE/'validation.json',out);print(out['all'])


if __name__=='__main__':main()
