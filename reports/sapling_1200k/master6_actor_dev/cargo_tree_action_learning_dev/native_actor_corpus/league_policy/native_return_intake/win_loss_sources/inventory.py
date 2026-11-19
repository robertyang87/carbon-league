"""Recount official winner from both-seat terminal scores and preserve original registration scope."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent
LEAGUE = INTAKE.parent
NATIVE = LEAGUE.parent
FAMILY = NATIVE.parent
M6 = FAMILY.parent
ROOT = M6.parents[2]
REV = '8ddc6e087a23fa5307c1c4faaeabd02553d1b94c'
NATIVE_SHA = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
PARENT = '851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee'
BASE = '080b235b42aa2aca86a8352e068d25a48576e2c8dbdce53ca3b14d598017d912'
PPO_ONLY = '57e25576'
inputs={}
def rel(p):return p.relative_to(ROOT).as_posix()
def sha(b):return hashlib.sha256(b).hexdigest()
def require(check,message):
    if not check: raise ValueError(message)
def get(path,record=True):
    b=path.read_bytes(); d=sha(b)
    require(hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==subprocess.check_output(
        ['git','-C',str(ROOT),'rev-parse',REV+':'+rel(path)],text=True).strip(),str(path))
    if record:inputs[rel(path)]=d
    return b
def js(p):return json.loads(get(p))
def enc(x):return json.dumps(x,sort_keys=True,allow_nan=False,ensure_ascii=False,separators=(',',':')).encode()
def new_archive_rows(directory,training):
    report=js(directory/'hosted/settlement.json');plan=js(directory/'protocol.json')
    require(report['records_reconciled'] and not report['verification_errors'] and not report['missing_tickets'] and
            plan['files']['opponent']['sha256']==NATIVE_SHA,'Unsettled or incorrect native source')
    expected={(r['seed'],r['seat'],r['arm']):r for r in report['games']}
    if training:
        m=js(directory/'hosted/training_manifest.json')
        require(m['training_eligible'] and m['settlement_sha256']==inputs[rel(directory/'hosted/settlement.json')] and
                m['policy_sha256']==PARENT and m['examiner_sha256']==NATIVE_SHA,'Detached true native training manifest')
        episodes={(r['seed'],r['seat'],r['arm']):r for r in m['episodes']}
        require(set(episodes)==set(expected),'Unmatched real training trajectories')
    for idx in report['archives']:
        path=directory/'hosted'/(str(idx['id'])+'.zip')
        blob=get(path);require(sha(blob)==idx['sha256'],'Source archive changed')
        with zipfile.ZipFile(path) as z:
            for arm in plan['arms']:
                key=(idx['seed'],idx['seat'],arm);g=expected.pop(key)
                t=z.read(arm+'/turns.jsonl');rx=z.read(arm+'/receipt.json');reg=z.read(arm+'/registration.json')
                rec=json.loads(rx);pr=json.loads(reg);ps=idx['seat'];r0=rec['initialization'][ps]['response'];nr=rec['initialization'][1-ps]['response']
                proof=g['action_support']
                require(pr['matrix_sha256']==report['plan_sha256'] and pr['seed']==g['seed'] and pr['candidate_seat']==ps and pr['physical_map_sha256']==g['terrain'] and
                        sha(rx)==g['receipt_sha256'] and rec['protocol_sha256']==sha(reg) and sha(t)==rec['turns_sha256']==proof['tape_sha256'] and
                        r0['sha256']==plan['files'][arm]['sha256']==proof['deployment_sha256'] and nr['sha256']==NATIVE_SHA and
                        nr['original']==dict(_SEED=None,_SEARCH_OK=True,_TEMP=1,_S_ABORT=.58),'Wrong receipt, terrain or seat')
                require(g['status']=='complete' and g['counted'] and proof['matched_all_steps'] and
                        g['official_scores_by_seat']==rec['official_scores_by_seat'] and g['replayed_public_observations']==2*g['official_transitions'],'Incomplete official replay')
                commands=[]
                for i,line in enumerate(t.splitlines()):
                    q=json.loads(line);require(q['step']==i and len(q['calls'])==2 and all(x['status']=='ok' for x in q['calls']),'Missing commands')
                    commands.append([x['response']['action'] for x in q['calls']])
                require(len(commands)==g['official_transitions'],'Incomplete tape')
                if training:
                    e=episodes[key];require(e['episode_verified'] and e['policy_sha256']==r0['sha256'] and e['official_scores_by_seat']==g['official_scores_by_seat'] and e['tape_sha256']==sha(t),'Training join changed')
                yield dict(source=rel(path),arm=arm,candidate_sha256=r0['sha256'],seed=g['seed'],seat=ps,terrain=g['terrain'],kind='hosted',
                           n=len(commands),official_scores=g['official_scores_by_seat'],engine_receipt=sha(rx),actions_sha256=sha(enc(commands)))
    require(not expected,'Missing archived official games')

def main():
    supply=LEAGUE/'opponent_deposit_horizon/sources.json';d=js(supply)
    require(inputs[rel(supply)]==BASE and d['unique_complete_actions']==361,'Different historical census')
    old=[r for r in d['rows']+d['excluded'] if 'official_scores' in r]
    missing=[r for r in d['excluded'] if 'official_scores' not in r]
    allrows=old+list(new_archive_rows(INTAKE,True))+list(new_archive_rows(INTAKE/'validation',False))
    # Keep the original registration and the actual later role separately, especially for the positive score used as a baseline.
    raws={r['path']:r for r in (json.loads(i) for i in gzip.decompress(get(NATIVE/'expansion/inventory.jsonl.gz')).splitlines())}
    raw_packs=js(NATIVE/'expansion/lineups.json')
    oni=FAMILY/'onpolicy_native/protocol.json'; training_plan=js(oni)
    controls={r['raw'] for r in training_plan['historical_controls']}
    teacher_file=NATIVE/'opponent_rollout/sources.json.gz'; teacher=json.loads(gzip.decompress(get(teacher_file)))
    practice={g['path']:part for part,group in teacher.items() for g in group}
    nativefit=M6/'deployed_mix_policy_gradient_dev/full_episode/direct_native_fit'
    exact_rel=js(nativefit/'maps.json')
    require('previously exposed' in exact_rel['use'].lower() and 'used for optimization' in exact_rel['use'] and 'future assessment' in exact_rel['use'],'Missing retirement')
    retired1={int(s):h for s,h in exact_rel['boards'].items()}
    bs=LEAGUE/'native_gradient/balanced_step'
    require('four retired original-opponent training maps' in js(bs/'protocol.json')['cohorts'],'No four-board retirement')
    retired2={int(s):h for s,h in js(bs/'original_response/protocol.json')['training_physical_maps']['original'].items()}
    require(set(retired2)==set(training_plan['seeds']) and not set(retired1)&set(retired2),'Retirement group changed')
    fitp=M6/'terminal_return_fit'; p0=js(fitp/'protocol.json');p1=js(fitp/'multimap/protocol.json')
    allow_training_sources=js(nativefit/'protocol.json');original_packs={}
    for k,v in raw_packs.items():original_packs[k]={'paths':v.get('paths',[]),'builder':v.get('builder')}
    reserved={}
    for s in [LEAGUE/'natural_original/fresh_repeat/protocol.json',INTAKE/'validation/protocol.json']:
        j=js(s);reserved.update(j['physical_boards'])
    mixed=[]
    for r in allrows:
        source=r['source'];side=r['seat'];seed=r['seed'];h=r['terrain'];pack=r['candidate_sha256']
        p=ROOT/source; arm=r.get('arm'); raw=raws.get(source)
        scores=r['official_scores'];m=round(scores[side]-scores[1-side],2)
        chapter=next((q for q in p.parents if (q/'protocol.json').is_file() and (q==M6.parent or M6.parent in q.parents)),None)
        reg=rel(chapter/'protocol.json') if chapter else None
        if pack.startswith(PPO_ONLY):
            family='self_owned_ppo_it50_no_sapling'
        else:
            family='registered_sapling_lineage'
        if r['kind']=='raw':
            evidence=raw_packs[pack]
            require(raw and raw['native_action_supervision'] and raw['full_official_terminal'] and raw['raw_sha256']==r['source_sha256'] and
                    (evidence.get('paths') or evidence.get('builder')) and not raw['duplicate_trajectory_of'],'No verified expert/actor identity')
        elif r in old:
            with zipfile.ZipFile(p) as z:
                pr=arm+'/registration.json'
                t=json.loads(z.read(pr)) if pr in z.namelist() else js(FAMILY/'ladder_live/protocol.json')
                actor=t['files']['candidate'];require(actor['sha256']==pack,'Wrong originally registered actor')
                original_packs.setdefault(pack,{'paths':[actor['path']],'builder':None})
        else:
            group=INTAKE if p.parent.parent==INTAKE else INTAKE/'validation'
            original_packs.setdefault(pack,{'paths':[json.loads((group/'protocol.json').read_bytes())['files'][arm]['path']],'builder':None})
        if r['kind']=='hosted' and arm=='candidate' and 'hosted_attempt/' in source: reg=rel(FAMILY/'ladder_live/protocol.json')
        purpose='archived_development_or_diagnostic'; use='no_registered_own_terminal_training_use_found'
        if r['kind']=='raw' and raw['geography'].get('in_original_29'):purpose='archived_native_source_'+raw['geography']['split']
        if source.startswith(rel(M6/'actor_partner_dev/native')+'/'):purpose='historical_exposed_partner_development'
        if source.startswith(rel(FAMILY/'direct_original/games')+'/'):purpose='historical_development'
        if source.startswith(rel(INTAKE/'hosted')+'/'):
            purpose='registered_native_training';use='851c_natural_behavior_gradient';reg=rel(INTAKE/'protocol.json')
        elif str(seed) in reserved or h in reserved.values():
            purpose='protected_validation';use='excluded_validation'
        elif source in controls:
            purpose='historical_development';use='explicit_signed_baseline_only'
        elif source.startswith(rel(oni.parent/'games')+'/'):
            require(seed in training_plan['seeds'] and pack==training_plan['training_pack'],'Different old sampling policy')
            purpose='registered_native_training';use='previous_sampled_policy_gradient'
        elif ((source.startswith(rel(fitp/'train_')) and seed==p0['training_seed']) or
              (source.startswith(rel(fitp/'multimap/train')+'/') and seed in p1['train_maps'])):
            purpose='registered_native_training';use='previous_paired_perturbation_terminal_fit'
        elif source.startswith(rel(nativefit/'games')+'/') and seed in allow_training_sources['seeds']:
            purpose='registered_native_training';use='previous_paired_perturbation_terminal_fit'
        elif source.startswith(rel(LEAGUE/'natural_original/hosted')+'/'):
            purpose='exposed_development_comparison'
        elif source.startswith(rel(NATIVE/'fresh_original/hosted')+'/'):
            purpose='prospective_held_development'
        if 1887000000<=seed<1888000000 or 20291101<=seed<=20292100:purpose=use='excluded_exam'
        retirement=None
        if seed in retired1 or seed in retired2:
            require(h==(retired1|retired2)[seed],'Different physical retired terrain')
            retirement=rel(nativefit/'maps.json') if seed in retired1 else rel(bs/'protocol.json')
        labels=dict(seed=seed,seat=side,terrain=h,arm=arm,source=source,kind=r['kind'],candidate_sha256=pack,
                    actions_sha256=r['actions_sha256'],official_replay_receipt=r['engine_receipt'],transitions=r['n'],official_scores_by_seat=scores,
                    official_margin=m,wdl='W' if m>0 else 'L' if m<0 else 'D',from_t0=True,actor_family=family,
                    registration=reg,original_role=purpose,candidate_return_use=use,retired_terrain_evidence=retirement,
                    opponent_move_head_historical_split=practice.get(source))
        mixed.append(labels)
    require(len({(x['terrain'],x['seat'],x['actions_sha256']) for x in mixed})==len(mixed),'Duplicated played trajectory')
    winners=[x for x in mixed if x['wdl']=='W' and x['actor_family']=='registered_sapling_lineage']
    require(len(allrows)==393 and len(winners)==5,'Review changed cohort')
    accepted=[x for x in mixed if x['candidate_return_use']=='851c_natural_behavior_gradient']
    require(len(accepted)==24 and all(r['candidate_sha256']==PARENT for r in accepted),'Not 24 natural on-policy training records')
    scopes={};pack_scopes={}
    for r in mixed:
        scopes.setdefault(r['candidate_return_use'],collections.Counter())[r['wdl']]+=1
        pack_scopes.setdefault(r['candidate_sha256'],collections.Counter())[r['wdl']]+=1
    positives=[dict(seed=r['seed'],seat=r['seat'],source=r['source'],arm=r['arm'],package=r['candidate_sha256'],official_margin=r['official_margin'],
                    candidate_return_use=r['candidate_return_use'],original_role=r['original_role'],retired_terrain_evidence=r['retired_terrain_evidence'],
                    opponent_move_head_historical_split=r['opponent_move_head_historical_split']) for r in winners]
    # Exclusions other than failed raw candidates, never placed in the population denominator.
    excluded=[dict(source=rel(LEAGUE/'team_teacher/settled_windows.json'),role='Apex teacher control for a window, not an all-game Sapling policy'),
              dict(source=rel(INTAKE/'live_curriculum/verification.json'),role='Apex roll-in until t240; no Sapling policy from t0'),
              dict(source='reports/sapling_1200k/MASTER3S2_90.md',role='The reported Apex 28/40 result belongs to another actor'),
              dict(source=rel(LEAGUE/'collection.json'),role='Self-owned league opponents rather than native Master3s2'),
              dict(source=rel(NATIVE/'terminal_direction/result.json'),role='Conditional native response suffixes, not fresh complete policy trajectories')]
    info=dict(snapshot=REV,examiner_sha256=NATIVE_SHA,official_count=len(mixed),unique_terrains=len({r['terrain'] for r in mixed}),
         confirmed_positive_games=positives,counts_by_registered_return_use=scopes,counts_by_package=pack_scopes,
         package_identity=original_packs,incomplete_native_or_failed=missing,
         outside_full_game_population=excluded,research_scope='The older 361 verified raw/hosted trajectories plus all 24 native training and 8 paired holdout games. No new play. No training-use authorization inferred from filename, earlier map exposure, or membership in an opponent action-head corpus.',
         inputs=inputs,rows=sorted(mixed,key=lambda r:(r['seed'],r['seat'],r['source'],r['arm'] or '')))
    compressed=gzip.compress(enc(info),mtime=0)
    target=HERE/'inventory.json.gz';tmp=HERE/'inventory.tmp';tmp.write_bytes(compressed);tmp.replace(target)
    print(len(mixed),len(winners),sha(compressed),scopes)

if __name__=='__main__':main()
