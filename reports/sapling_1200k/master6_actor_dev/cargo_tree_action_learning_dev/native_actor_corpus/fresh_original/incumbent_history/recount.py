"""Recount original-Master3s2 complete games of the exact first-win package."""
from collections import Counter
import csv
import gzip
from itertools import combinations_with_replacement
from math import factorial, prod
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess
import zipfile

from discover import ROOT, SNAPSHOT, PACKAGE

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2]
OPP = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
MD5 = '57f9a594cce4f2ec9e3573d8d9d26ca1'
REG = [('direct_original','57445f858'),('replicate_original','a47eeb926'),
       ('territory_sequence/fresh_original','f82b98f91'), ('territory_sequence/windowed/fresh_original','bc379ddee'),
       ('native_actor_corpus/fresh_original','5567aadda'),('ladder_live','90056a9eb')]


def sha(b): return hashlib.sha256(b).hexdigest()
def rel(p): return str(p.relative_to(ROOT))
def sig(obj): return sha(json.dumps(obj,sort_keys=True,separators=(',',':')).encode())
def load(p): return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())


def artifact(p):
    data=p.read_bytes()
    tracked=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',f'{SNAPSHOT}:{rel(p)}']).decode().strip()
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==tracked
    return {'path':rel(p),'sha256':sha(data)}


def standard(terrain,seed,seat,scores,actions):
    assert len(actions)==299 and all(len(joint)==2 and all(isinstance(a,dict) for a in joint) for joint in actions)
    delta=round(scores[seat]-scores[1-seat],2)
    return dict(terrain=terrain,seed=seed,seat=seat,official_scores_by_physical_seat=scores,margin=delta,
                result='win' if delta>0 else 'loss' if delta<0 else 'draw',
                actions_sha256=sig(actions), trajectory_sha256=sig([terrain,actions]))


def native(folder,seed,seat,protocol_sha):
    files={key:folder/f'games/candidate/{key}_{seed}_{seat}.{ext}' for key,ext in [('receipt','json'),('ticket','json'),('raw','json.gz'),('wall','json.gz')]}
    provenance={k:artifact(p) for k,p in files.items()}
    receipt,ticket,raw,wall=(load(files[k]) for k in ('receipt','ticket','raw','wall'))
    binding=receipt['binding'];game=receipt['game'];official=receipt['official'];calls=wall['calls']
    assert ticket['binding']==binding and ticket['arm']=='candidate' and binding['candidate']==raw['pack_sha256']==PACKAGE
    assert binding['original']==OPP and binding['protocol']==protocol_sha and raw['examiner_md5']==MD5
    assert raw['configuration']['_S_ABORT']==.58 and raw['configuration']['_SEED'] is None and raw['configuration']['_TEMP']==1
    assert game['raw_sha256']==wall['raw_sha256']==official['transcript_sha256']==provenance['raw']['sha256']
    assert game['valid'] and not raw['errors'] and game['agent_errors_by_seat']==[0,0] and official['statuses']==['DONE','DONE']
    assert (raw['seed'],raw['candidate_seat'],raw['steps'])==(official['seed'],official['seat'],official['steps'])==(seed,seat,299)
    terrain=sha(bytes(map(int,raw['initial_full_carbon'])));assert terrain==ticket['terrain']==official['map_sha256']==raw['initial_terrain_u8_sha256']
    row=standard(terrain,seed,seat,official['official_reward_by_seat'],raw['actions'])
    assert row['actions_sha256']==game['actions_sha256'] and raw['official_score_by_seat']==row['official_scores_by_physical_seat']
    assert [c['step'] for c in calls]==list(range(299)) and [[c['step'],c['kind']] for c in calls]==raw['search_calls']
    kinds=Counter(c['kind'] for c in calls); assert dict(kinds)==game['search_return_types']
    row.update(artifacts=provenance, clock=dict(source='local native-clock wrapper',calls=len(calls),positive_depth=kinds['searched'],
       single=kinds['single'],multi_zero=kinds['depth_zero_return_base'],depth8=sum(c['max_depth']==8 for c in calls),
       public_seconds=sum(c['public_wall_seconds'] for c in calls),cpu_seconds=sum(c['process_cpu_seconds'] for c in calls),
       external_request_seconds=None,affinities=[s['host']['affinity'] for s in wall['segments']]))
    return row


def hosted(folder,seed,seat,protocol_sha,repeat):
    arc=folder/'hosted_attempt/artifact.zip' if repeat else folder/f'hosted/{seed}-{seat}.zip'
    proof={'archive':artifact(arc)}
    prefix='game' if repeat else 'parent'
    with zipfile.ZipFile(arc) as z:
        def read(name):
            b=z.read(name); proof[name]=dict(member=name,sha256=sha(b));return b
        r=json.loads(read(f'{prefix}/receipt.json')); data=read(f'{prefix}/turns.jsonl'); turns=[json.loads(line) for line in data.splitlines()]
        assert sha(data)==r['turns_sha256'] and r['status']=='complete' and r['counted'] and r['official_transitions']==299
        assert (r['seed'],r['candidate_seat'])==(seed,seat) and [t['step'] for t in turns]==list(range(299))
        if repeat:
            vpath=folder/'hosted_attempt/verification.json';proof['verification']=artifact(vpath);v=load(vpath)
            files=v['frozen_files'];terrain=v['physical_map_sha256'];run=v['run']['id']
            assert r['protocol_sha256']==protocol_sha and v['independently_replayed_official_scores_by_seat']==r['official_scores_by_seat']
            assert v['artifact']['digest'].split(':')[-1]==proof['archive']['sha256']
        else:
            reg=json.loads(read('parent/registration.json'));v=json.loads(read('verification.json'))['games']['parent']
            files=reg['files'];terrain=reg['physical_map_sha256'];run=reg['workflow_run']
            assert reg['matrix_sha256']==protocol_sha and (reg['seed'],reg['candidate_seat'])==(seed,seat)
            assert r['official_scores_by_seat']==v['official_scores_by_seat'] and v['replayed_public_observations']==598 and v['receipt_sha256']==proof['parent/receipt.json']['sha256']
    assert files['candidate']['sha256']==PACKAGE and files['opponent']['sha256']==OPP and not r['blockers']
    assert all(c['network']=='none' and c['nano_cpus']==1500000000 and c['memory']==c['memory_swap']==536870912 for c in r['containers'])
    calls=[t['calls'][1-seat] for t in turns];assert all(c['status']=='ok' for t in turns for c in t['calls'])
    prev=Counter();kinds=Counter();depths=Counter();wall=[];cpu=[]
    for c in calls:
        resp=c['response'];counts=Counter(resp['search_counts']);delta=counts-prev
        kind='searched' if delta['searched'] else 'single' if delta['single'] else 'zero' if delta['aborted'] else None
        assert kind is not None and all(prev[k]<=counts[k]<=prev[k]+1 for k in counts);kinds[kind]+=1;prev=counts
        if 'search_last' in resp:depths[resp['search_last']['depth']]+=1
        if 'public_wall_seconds' in resp:wall.append(resp['public_wall_seconds']);cpu.append(resp['public_cpu_seconds'])
    assert repeat or len(wall)==299
    if not repeat:
        old=v['examiner_search'];assert abs(old['public_wall_seconds']-sum(wall))<1e-7 and old['depth_histogram'].get('8',0)==depths[8]
    row=standard(terrain,seed,seat,r['official_scores_by_seat'],[[c['response']['action'] for c in t['calls']] for t in turns])
    row.update(artifacts=proof,workflow_run=str(run),clock=dict(source='hosted quota containers',calls=len(calls),positive_depth=kinds['searched'],single=kinds['single'],
        multi_zero=kinds['zero'],depth8=depths[8] if depths else None,public_seconds=sum(wall) if wall else None,cpu_seconds=sum(cpu) if cpu else None,
        external_request_seconds=sum(c['elapsed'] for c in calls),candidate_slow_steps=r['slow_steps'][seat],opponent_slow_steps=r['slow_steps'][1-seat]))
    return row


def stats(rows):
    return dict(n=len(rows),wins=sum(r['margin']>0 for r in rows),draws=sum(r['margin']==0 for r in rows),
                losses=sum(r['margin']<0 for r in rows),mean_margin=mean(r['margin'] for r in rows))


def cluster_interval(maps):
    # Exact percentile distribution of sampling ten whole maps, with replacement.
    values=list(maps.values()); n=len(values); assert n==10
    margin=Counter(); rate=Counter(); fact=factorial(n)
    for sequence in combinations_with_replacement(range(n),n):
        weight=fact//prod(factorial(k) for k in Counter(sequence).values())
        rate[sum(values[i]['wins'] for i in sequence)/(2*n)]+=weight
        margin[sum(round(values[i]['mean_margin']*200) for i in sequence)/(200*n)]+=weight
    def quantiles(table):
        assert sum(table.values())==n**n
        q=[]; partial=0
        for value,weight in sorted(table.items()):
            partial+=weight
            if not q and partial*40>=n**n:q.append(value)
            if partial*40>=39*n**n:return q+[value]
    return dict(method='exact whole-map percentile bootstrap, ten draws with replacement, both seats retained',
                nominal_fraction=.95,win_fraction=quantiles(rate),mean_margin=quantiles(margin))


def recount():
    assert __debug__ and artifact(EXP/'candidate.py')['sha256']==PACKAGE and sha(subprocess.check_output(['git','-C',str(ROOT),'show',SNAPSHOT+':sources/master3s2/master3s2.py']))==OPP
    rows=[];batches=[]
    for idx,(name,commit) in enumerate(REG):
        folder=EXP/name;p=folder/'protocol.json';reg=artifact(p);plan=load(p);repeat=name=='ladder_live'
        committed=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{rel(p)}']);assert sha(committed)==reg['sha256']
        pairs=[(plan['seed'],plan['candidate_seat'])] if repeat else [(s,k) for s in plan['seeds'] for k in (0,1)]
        for seed,seat in pairs:
            row=native(folder,seed,seat,reg['sha256']) if idx<2 else hosted(folder,seed,seat,reg['sha256'],repeat)
            row.update(batch=name,category='exposed_retest' if repeat else 'first_registered_pair');rows.append(row)
        batches.append(dict(name=name,preregistration_commit=commit,protocol=reg,scheduled_pairs=pairs))
    inventory=load(HERE/'discovery.json'); assert inventory['snapshot']==SNAPSHOT
    raw_paths={r['artifacts']['raw']['path'] for r in rows if 'raw' in r['artifacts']}
    hosted_tickets={r['artifacts']['archive']['path']+'!parent/registration.json' for r in rows if 'parent/registration.json' in r['artifacts']}
    repeated=[r['artifacts']['verification']['path'] for r in rows if 'verification' in r['artifacts']]
    assert raw_paths==set(inventory['exact_package_raw_games'])
    assert hosted_tickets|{rel(EXP/'ladder_live/protocol.json')}==set(inventory['exact_package_registrations'])
    assert repeated==inventory['directly_bound_hosted_verifications']
    tracks=[r['trajectory_sha256'] for r in rows];assert len(tracks)==len(set(tracks)) and len({r['actions_sha256'] for r in rows})==len(rows)
    fresh=[r for r in rows if r['category']=='first_registered_pair'];seeds=sorted({r['seed'] for r in fresh})
    maps={str(s):stats([r for r in fresh if r['seed']==s]) for s in seeds}
    assert len({r['terrain'] for r in fresh})==len(seeds) and all({r['seat'] for r in fresh if r['seed']==s}=={0,1} for s in seeds)
    return dict(snapshot=SNAPSHOT,package_sha256=PACKAGE,examiner_sha256=OPP,registrations=batches,games=rows,
        aggregate=stats(fresh),by_seat={str(k):stats([r for r in fresh if r['seat']==k]) for k in (0,1)},by_map=maps,
        map_cluster_interval=cluster_interval(maps),map_equal_mean_margin=mean(r['mean_margin'] for r in maps.values()),maps_with_any_win=sum(r['wins']>0 for r in maps.values()),
        maps_with_two_wins=sum(r['wins']==2 for r in maps.values()),maps_with_positive_mean_margin=sum(r['mean_margin']>0 for r in maps.values()),
        retest=stats([r for r in rows if r['category']=='exposed_retest']))


if __name__=='__main__':
    result=recount()
    (HERE/'history.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    with (HERE/'games.tsv').open('w', newline='') as handle:
        table=csv.writer(handle, delimiter='\t', lineterminator='\n')
        table.writerow(['category','batch','seed','seat','own_score','examiner_score','margin','original_calls','depth8','multi_zero','public_wall_s','external_request_s','cpu_s','terrain_sha256','actions_sha256','terrain_plus_actions_sha256','receipt_sha256','source_sha256','source_path'])
        for g in result['games']:
            c=g['clock']; a=g['artifacts']; src=a['raw'] if 'raw' in a else a['archive']; receipt=next(v for k,v in a.items() if k=='receipt' or k.endswith('/receipt.json')); s=g['seat'];score=g['official_scores_by_physical_seat']
            table.writerow([g['category'],g['batch'],g['seed'],s,score[s],score[1-s],g['margin'],c['calls'],c['depth8'],c['multi_zero'],c['public_seconds'],c['external_request_seconds'],c['cpu_seconds'],g['terrain'],g['actions_sha256'],g['trajectory_sha256'],receipt['sha256'],src['sha256'],src['path']])
