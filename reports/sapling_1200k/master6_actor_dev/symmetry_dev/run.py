"""Run the preregistered four-view development comparison on an immutable native arena."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
T=HERE.parent/'tree_response_training'
sys.path[:0]=[str(T/'supplement_01'),str(T)]
import priority_slot as slot
from source_games import BASE,ROOT,ENV,gate,trial,sha
from select_roots import load,retain

PLAN='55ae16a63cf167aef015bc2f10a8073ef26c13287ad18480c6402b86b235aaef'
ARMS=('parent','candidate')


def check(ok,message):
    if not ok:raise ValueError(message)


def inputs():
    plan=load(HERE/'protocol.json');rel=load(HERE/'release.json');clear=load(HERE/'clearance.json')
    check(sha(HERE/'protocol.json')==PLAN==rel['protocol_sha256']==clear['declaration_sha256'],'changed study registration')
    check(clear['verdict']=='eligible_in_checked_scope' and set(clear['boards'])==set(map(str,plan['maps'])) and len(set(clear['boards'].values()))==2,'incomplete map clearance')
    for a in clear['full_audits'].values():
        check(sha(HERE/a['archive'])==a['sha256'],'map audit file changed')
    check(rel['candidate_pack_sha256']==sha(HERE/'candidate.py') and rel['parent_pack_sha256']==plan['parent_pack_sha256']
          and rel['student_npz_sha256']==plan['student_npz_sha256']==sha(ROOT/'data/weights/sapling_1200k/master6_actor_v1.npz')
          and rel['parent_weight_manifest_sha256']==plan['parent_weight_manifest_sha256']
          and rel['runtime_sha256']==sha(HERE/'runtime.py')
          and rel['views']==[0,1,2,3] and (rel['expert_weight'],rel['student_weight'])==(.75,.25)
          and sha(ROOT/'data/master3s2_source.py')==plan['examiner_sha256'],'different model, native examiner or algorithm')
    return plan,rel,clear


def approved():
    plan,rel,clear=inputs()
    m=load(HERE/'mechanism.json')
    check(m.get('passed') is True and m['protocol_sha256']==PLAN and m['candidate_pack_sha256']==rel['candidate_pack_sha256'],'four-view mechanism gate did not pass')
    return plan,rel,clear


@contextmanager
def arena_snapshot(plan):
    rev=plan['source_revision']
    listed=subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-r','--name-only',rev],text=True).splitlines()
    keys=set(gate.protocol_hashes())
    def needed(p):
        if p in keys:return True
        if p.startswith('code/env_src/carbon_challenge-master/zerosum_env/'):return True
        if p.startswith(('code/','reports/sapling_1200k/','reports/sapling_bootstrap/')):
            return p.endswith(('.py','validation_maps.jsonl.gz')) or Path(p).name=='learned_checkpoint.json'
        return p=='data/master3s2_source.py'
    files=[s for s in listed if needed(s)]
    with tempfile.TemporaryDirectory(prefix='four-view-arena-') as tmp:
        root=Path(tmp)
        producer=subprocess.Popen(['git','-C',str(ROOT),'archive',rev,'--',*files],stdout=subprocess.PIPE)
        try:
            subprocess.run(['tar','-x','-C',str(root)],stdin=producer.stdout,check=True)
        finally:
            producer.stdout.close();check(producer.wait()==0,'failed to extract the frozen arena')
        base=root/BASE.relative_to(ROOT)
        sources=json.loads(subprocess.check_output([sys.executable,'-c','import json,tree_win_gate as g;print(json.dumps(g.protocol_hashes()))'],cwd=base,env=ENV))
        check(sources['data/master3s2_source.py']==plan['examiner_sha256'] and all(sha(root/k)==v for k,v in sources.items()),'immutable arena source bind failed')
        yield base,sources


def arm_records(arm,certificates,*,official_again=False):
    d=HERE/arm;game=load(d/'results.json');cert=load(d/'official_replay.json');plan,rel,clear=approved()
    native=cert['runs'][arm];head=game['header'];expected={(s,p) for s in plan['maps'] for p in (0,1)}
    pkg=rel[f'{arm}_pack_sha256']
    weights=plan['parent_weight_manifest_sha256'] if arm=='parent' else sha(HERE/'release.json')
    check(game['status']=='complete' and head['seeds']==plan['maps'] and head['pack_sha256']==pkg==native['pack_sha256']
          and head['weight_sha256']==weights and head['examiner_md5']==plan['examiner_md5']
          and head['cpus']==[0,1] and head['source_sha256']==certificates['arena_sources']
          and cert['engine_source_sha256']==certificates['official_engine_sources'],'different native source, participant or resource')
    by_key={(r['seed'],r['candidate_seat']):r for r in game['games']};off={(r['seed'],r['seat']):r for r in native['games']}
    check(set(by_key)==set(off)==expected and len(game['games'])==len(native['games'])==4,'incomplete or duplicate map seat')
    if official_again:
        import official_game_replay
        check(official_game_replay.audit(d)==native,'independent original-engine terminal replay differs')
    results={}
    for s,p in sorted(expected):
        row,o=by_key[s,p],off[s,p];raw=d/row['raw_file']
        gate.validate_examiner_configuration(load(raw)['configuration']);gate.validate_game(row)
        gate.verify_raw(d,row,pkg,weights,expected_terrain=clear['boards'][str(s)])
        scored=o['official_reward_by_seat'];margin=scored[p]-scored[1-p]
        check(o['statuses']==['DONE','DONE'] and scored==row['official_score_by_seat'] and row['margin']==margin
              and o['transcript_sha256']==row['raw_sha256']==sha(raw) and o['steps']==row['actual_transitions']
              and o['map_sha256']==clear['boards'][str(s)],'wrong map, game actions, terminal score or termination')
        wdl=1 if margin>0 else .5 if margin==0 else 0
        u=.5*wdl+.5/(1+math.exp(-max(-40,min(40,margin/1000))))
        results[s,p]={'seed':s,'seat':p,'official_scores':scored,'margin':margin,'U':u,'outcome':row['result'],'transitions':o['steps'],
                      'search':row['search_return_types'],'raw_sha256':sha(raw),'raw_file':row['raw_file'],'wall_seconds':row['wall_seconds']}
    return results


def compare(pair):
    plan=load(HERE/'protocol.json');maps=plan['maps'];rows=[]
    for s in maps:
        for p in (0,1):
            b,c=pair['parent'][s,p],pair['candidate'][s,p]
            rows.append({'seed':s,'seat':p,'parent':b,'candidate':c,'delta_margin':c['margin']-b['margin'],'delta_U':c['U']-b['U']})
    across=[statistics.mean(r['delta_margin'] for r in rows if r['seed']==s) for s in maps]
    wins={a:sum(r[a]['outcome']=='win' for r in rows) for a in ARMS}
    wdl={a:{k:sum(r[a]['outcome']==k for r in rows) for k in ('win','draw','loss')} for a in ARMS}
    ok={'positive_each_map':all(v>0 for v in across),'at_least_three_seat_U_gains':sum(r['delta_U']>0 for r in rows)>=3,
        'no_fewer_wins':wins['candidate']>=wins['parent']}
    se=statistics.stdev(across)/math.sqrt(len(maps));avg=statistics.mean(across)
    return {'schema':'four-view-development-comparison-v1','protocol_sha256':PLAN,'release_sha256':sha(HERE/'release.json'),
            'mechanism_sha256':sha(HERE/'mechanism.json'),'arena_binding_sha256':sha(HERE/'arena_binding.json'),
            'competitors':{a:{'results_sha256':sha(HERE/a/'results.json'),'official_sha256':sha(HERE/a/'official_replay.json')} for a in ARMS},
            'paired_map_seats':rows,'per_map_mean_margin_change':dict(zip(maps,across)),'average_margin_change':avg,
            'illustrative_map_cluster_t95':[avg-12.706204736*se,avg+12.706204736*se],'maps':len(maps),
            'rollup':wdl,'seat_average_margin_change':{p:statistics.mean(z['delta_margin'] for z in rows if z['seat']==p) for p in (0,1)},
            'positive_U_pairs':sum(r['delta_U']>0 for r in rows),'criteria':ok,
            'verdict':'exploratory_followup_signal' if all(ok.values()) else 'retain_parent','formal':False}


def certified(official_again=False):
    bound=load(HERE/'arena_binding.json')
    groups={a:arm_records(a,bound,official_again=official_again) for a in ARMS}
    return compare(groups)


def compete():
    plan,rel,_=approved()
    with slot.native_session(), arena_snapshot(plan) as (base,sources),tempfile.TemporaryDirectory() as tmp:
        parent,weights,digest=trial.recipe('learned',Path(tmp))
        check(digest==sha(parent)==plan['parent_pack_sha256'] and sha(weights)==plan['parent_weight_manifest_sha256'],'frozen v1 differs')
        reference=load(T/'source/official_replay.json')['engine_source_sha256']
        native_engine=base.parents[1]/'code/env_src/carbon_challenge-master'
        check(all(sha(native_engine/k)==v for k,v in reference.items()),'different official interpreter')
        binding={'revision':plan['source_revision'],'protocol_sha256':PLAN,'release_sha256':sha(HERE/'release.json'),
                 'arena_sources':sources,'official_engine_sources':reference,'map_clearance_sha256':sha(HERE/'clearance.json')}
        retain(HERE/'arena_binding.json',binding)
        files={'parent':(parent,weights),'candidate':(HERE/'candidate.py',HERE/'release.json')}
        for arm in ARMS:
            pack,identity=files[arm];d=HERE/arm
            subprocess.run([sys.executable,'-u',str(base/'tree_pack_arena.py'),'--pack',str(pack),'--weights',str(identity),
                            '--pack-sha256',rel[f'{arm}_pack_sha256'],'--run',str(d),'--cpus','0','1','--maps',*map(str,plan['maps'])],check=True,env=ENV)
            output=Path(tmp)/(arm+'.json')
            subprocess.run([sys.executable,str(base/'official_game_replay.py'),str(d),'--out',str(output)],check=True,env=ENV)
            retain(d/'official_replay.json',load(output))
            arm_records(arm,binding)
        result=certified()
        retain(HERE/'comparison.json',result)
    from priority_verify import release
    release();print(json.dumps({'verdict':result['verdict'],'margin':result['average_margin_change'],'wins':result['rollup']}),flush=True)


def oversee():
    with (HERE/'.run-owner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('four-view comparison already supervised');return
        while not all((HERE/f).exists() for f in ('release.json','clearance.json','mechanism.json')):
            if (HERE/'mechanism_skip.json').exists():
                from priority_verify import skip
                skip();return
            time.sleep(4)
        if not load(HERE/'mechanism.json').get('passed'):
            from priority_verify import skip
            skip();return
        if (HERE/'comparison.json').exists():
            check(load(HERE/'comparison.json')==json.loads(json.dumps(certified(official_again=True))),'comparison seal differs')
            from priority_verify import release
            release();return
        compete()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','check','capsule'))
    command=parser.parse_args().mode
    if command=='run':oversee()
    elif command=='check':print(json.dumps(certified(official_again=True)))
    else:
        with arena_snapshot(load(HERE/'protocol.json')) as (_,sources):print('verified frozen arena',len(sources))
