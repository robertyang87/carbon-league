"""Reconcile the two independent 851c trajectories per training map/seat against native Master3s2."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / '.git').exists())
NATURAL = HERE.parent/'natural_original'
POLICY = '851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee'
EXAMINER = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
ARMS = ('draw_a', 'draw_b')
INPUTS = ('registration.json','receipt.json','attempt.json','turns.jsonl')
TICKET_FILES = ('pair_status.json','execution.jsonl','exposure.json','verification.json')
sys.path.insert(0,str(NATURAL))
import audit_menu
import settle
sys.path.insert(0,str(HERE))
import matrix


def need(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, doc):
    raw=(json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(raw)
    os.replace(tmp,path)


def payoff(margin):
    w=float(margin>0)+.5*float(margin==0)
    return w,.5*w+.25*(1+math.tanh(margin/2000.))


def menu_for_training(directory,checked):
    # Capture the *same* public pre-action frame used by the existing package-level
    # support replayer. Reconstructing a previously unobserved RNG uniform is unnecessary.
    reader=audit_menu.per_order
    def capture(actor,arm,observation,configuration,issued):
        event=reader(actor,arm,observation,configuration,issued)
        if event is None:
            return None
        seat=checked['seat']
        need(set(observation)=={'step','player','players','carbon','trees','remainingOverageTime'}
             and observation['player']==seat and observation['step']==event['step'],'Wrong information set at root')
        event['public_observation']=deepcopy(observation)
        event['seat']=seat
        if event['supported']:
            uid=str(actor.mod._FEAT.own_uid[event['slot']])
            pos,load,role=observation['players'][seat][2][uid]
            need(role in ('COLLECTOR','PLANTER') and 0<=event['slot']<10,'Unbound focal worker')
            p=event['probabilities']
            gap=event['native_logits'][0]-event['native_logits'][1]
            need(0<=gap<=1.000001 and abs(p[0]-1/(1+math.exp(-gap)))<1.e-12
                 and abs(event['logp']-math.log(p[event['choice']]))<1.e-12,'Invalid conditional action propensity')
            need(event['options'][event['choice']]==issued and event['options'][0]!=event['options'][1],
                 'Physical sampled action not identifiable')
            event.update(uid=uid,role=int(role=='PLANTER'),position=pos,cargo=load)
        return event
    with patch.object(audit_menu,'per_order',capture):
        report=audit_menu.verify_menu(directory,'candidate',persist=False)
    need(report['deployment_sha256']==POLICY and report['official_transitions']==checked['official_transitions'],
         'Different policy or trajectory in independent support replay')
    if checked['status']=='complete':
        need(report['matched_all_steps'] and report['official_scores_by_seat']==checked['official_scores_by_seat'],
             'Complete trajectory lacks every issued action and exact score')
    return report


def verify_archive(args):
    archive,seed,side,run_id,commit,protocol_sha = args
    archive=Path(archive)
    complete={}
    issues=[]
    timeline=[]
    try:
        with zipfile.ZipFile(archive) as z, TemporaryDirectory() as folder:
            members=z.namelist()
            need(len(members)==len(set(members)) and z.testzip() is None,'Archive member integrity')
            root=Path(folder)
            for name in [*TICKET_FILES,*(arm+'/'+name for arm in ARMS for name in INPUTS)]:
                if name in members:
                    p=root/name
                    p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_bytes(z.read(name))
            ticket=read(root/'pair_status.json')
            need((ticket['seed'],ticket['seat'],str(ticket['run_id']),ticket['commit'],ticket['protocol_sha256'])==
                 (seed,side,str(run_id),commit,protocol_sha),'Wrong ticket, run, protocol or commit')
            checked=matrix.verify_block(root,seed,side)
            timeline=checked.get('host_sequence',[])
            if 'verification.json' not in members or read(root/'verification.json')!=json.loads(json.dumps(checked)):
                issues.append('Hosted and local official verification differ or hosted receipt missing')
            issues.extend(f'{k}: {v}' for k,v in checked['verification_errors'].items())
            for arm,g in checked['games'].items():
                complete[arm]=g
                reg=read(root/arm/'registration.json')
                receipt=read(root/arm/'receipt.json')
                need(reg['files']['candidate']['sha256']==POLICY and reg['files']['opponent']['sha256']==EXAMINER
                     and str(g['workflow_run'])==str(run_id) and g['receipt_sha256']==sha(root/arm/'receipt.json'),
                     f'{arm} actor/receipt binding')
                g.update(registration_sha256=sha(root/arm/'registration.json'),tape_sha256=receipt.get('turns_sha256'))
                try:
                    g['action_support']=menu_for_training(root/arm,g)
                except (ValueError,OSError,KeyError,IndexError,AssertionError,TypeError) as e:
                    issues.append(f'{arm} support: {type(e).__name__}: {e}')
    except (ValueError,OSError,KeyError,IndexError,AssertionError,TypeError,zipfile.BadZipFile) as e:
        issues.append(f'{type(e).__name__}: {e}')
    return dict(seed=seed,seat=side,games=complete,errors=issues,host_sequence=timeline)


def merge(run_id, commit, protocol_sha, sources, dest, workers, auditor_commit):
    matrix.check(protocol_sha)
    plan=read(HERE/'protocol.json')
    need(tuple(plan['arms'])==ARMS and len(set(plan['seeds']))==6 and plan['expected_games']==24,
         'Wrong training matrix')
    # The post-game converter is released separately; the game and learning plans
    # were already frozen before this converter was implemented.
    src=Path(__file__).resolve()
    oid=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',auditor_commit+'^{commit}'],text=True).strip()
    need(oid==auditor_commit and len(oid)==40,'Specify a full immutable review commit')
    signed_source=subprocess.check_output(['git','-C',str(ROOT),'show',oid+':'+str(src.relative_to(ROOT))])
    need(signed_source==src.read_bytes(),'Converter source differs from its release')
    origin=json.loads(subprocess.check_output(['gh','api','repos/robertyang87/carbon-league/compare/'+oid+'...main']))
    need(origin.get('merge_base_commit',{}).get('sha')==oid,'Converter release is outside main history')
    converter=dict(commit=oid,path=str(src.relative_to(ROOT)),sha256=sha(src))
    frozen=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+str((HERE/'protocol.json').relative_to(ROOT))])
    need(frozen==(HERE/'protocol.json').read_bytes(),'Different dispatched matrix')
    job=settle.gh(f'runs/{run_id}')
    need(job['id']==run_id and job['head_sha']==commit and job['head_branch']=='main' and job['run_attempt']==1
         and job['event']=='workflow_dispatch' and job['status']=='completed'
         and Path(job['path']).name=='sapling-native-return-intake.yml','Wrong training run')
    inventory=settle.gh(f'runs/{run_id}/artifacts?per_page=100')
    remote={r['name']:r for r in inventory['artifacts']}
    need(len(remote)==inventory['total_count'],'Incomplete or duplicate GitHub artifact names')
    expect={(r['seed'],r['seat']):f"native-intake-{r['seed']}-{r['seat']}-{run_id}" for r in plan['schedule']}
    need(set(expect)=={(s,p) for s in plan['seeds'] for p in (0,1)},'Duplicate or incomplete map/seat schedule')
    signed={remote[n]['digest']:(k,remote[n]) for k,n in expect.items() if n in remote}
    seen={}
    dest=Path(dest).resolve()
    dest.mkdir(exist_ok=True,parents=True)
    tasks=[]
    for file in sources:
        key='sha256:'+sha(file)
        need(key in signed,'Unexpected or duplicate GitHub ZIP')
        (seed,side),item=signed.pop(key)
        need(item['workflow_run']['id']==run_id and item['workflow_run']['head_sha']==commit
             and file.stat().st_size==item['size_in_bytes'],'Artifact size/run mismatch')
        path=dest/(str(item['id'])+'.zip')
        if path.exists():
            need(sha(path)==key[7:],'An archived file was replaced')
        else:
            shutil.copyfile(file,path)
        seen[(seed,side)]=dict(id=item['id'],seed=seed,seat=side,name=item['name'],sha256=key[7:],
                               bytes=item['size_in_bytes'],path=str(path.relative_to(ROOT)))
        tasks.append((str(path),seed,side,run_id,commit,protocol_sha))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        checked=list(pool.map(verify_archive,tasks))
    blocks={(b['seed'],b['seat']):b for b in checked}
    games=[]
    issues=[]
    training=[]
    for (seed,side),label in sorted(expect.items()):
        ref=seen.get((seed,side),{})
        block=blocks.get((seed,side),dict(games={},errors=['Missing GitHub ticket'],host_sequence=[]))
        if block['errors']:
            issues.append(dict(seed=seed,seat=side,errors=block['errors']))
        if ref:
            ref['host_sequence']=block['host_sequence']
        siblings=[]
        for arm,partner in zip(ARMS,reversed(ARMS),strict=True):
            g=block['games'].get(arm)
            support=(g or {}).get('action_support',{})
            scores=(g or {}).get('official_scores_by_seat')
            margin=wdl=u=None
            if g and g['status']=='complete' and scores is not None:
                margin=float(scores[side]-scores[1-side])
                need(math.isfinite(margin),'Nonfinite official score')
                wdl,u=payoff(margin)
            valid=bool(g and g['status']=='complete' and g['counted'] and
                       support.get('matched_all_steps') and u is not None and not block['errors'])
            events=support.get('per_event',[])
            if valid:
                target=[t for t in range(12,277,24) if t<g['official_transitions']]
                need([e['step'] for e in events]==target and len(events)==support['scheduled'],
                     'Missing scheduled choices/absences on a completed tape')
            siblings.append(dict(seed=seed,seat=side,arm=arm,partner_arm=partner,paired_seed_seat=[seed,side],
                terrain=plan['physical_boards'][str(seed)],archive_path=ref.get('path'),archive_sha256=ref.get('sha256'),
                registration_member=arm+'/registration.json',receipt_member=arm+'/receipt.json',turns_member=arm+'/turns.jsonl',
                policy_sha256=POLICY,examiner_sha256=EXAMINER,status=(g or {}).get('status','missing'),
                registration_sha256=(g or {}).get('registration_sha256'),receipt_sha256=(g or {}).get('receipt_sha256'),
                tape_sha256=(g or {}).get('tape_sha256'),official_transitions=(g or {}).get('official_transitions'),
                official_scores_by_seat=scores,margin=margin,wdl=wdl,u=u,episode_verified=valid,events=events))
            if g:
                # The full pre-action input and probabilities live once in the training manifest.
                retained={k:v for k,v in g.items() if k!='action_support'}
                retained['action_support']={k:v for k,v in support.items() if k!='per_event'}
                games.append(retained)
        for r,partner in zip(siblings,reversed(siblings),strict=True):
            pair_ok=r['episode_verified'] and partner['episode_verified']
            r['baseline_u']=partner['u'] if pair_ok else None
            r['advantage']=r['u']-partner['u'] if pair_ok else None
        training.extend(siblings)
    eligible=len(training)==24 and all(r['episode_verified'] and r['advantage'] is not None for r in training)
    report=dict(schema='sapling-native-learn-intake-settlement/1',workflow_run=run_id,commit=commit,plan_sha256=protocol_sha,
                registered_games=24,converter=converter,archives=[v for _,v in sorted(seen.items())],games=games,verification_errors=issues,
                missing_tickets=[list(t) for t in sorted(set(expect)-set(seen))],records_reconciled=len(games)==24 and not issues,
                training_eligible=eligible,formal_exam_games=0)
    save(dest/'settlement.json',report)
    training_manifest=dict(schema='sapling-native-learn-trajectories/1',training_eligible=eligible,run_id=run_id,commit=commit,
        protocol_sha256=protocol_sha,learning_sha256=sha(HERE/'learning.json'),settlement_sha256=sha(dest/'settlement.json'),
        policy_sha256=POLICY,examiner_sha256=EXAMINER,converter=converter,expected_episodes=24,matched_map_seat_pairs=12,
        terrain_count=6,normalization=24*12,episodes=training,
        scope='Full natural return training episodes. A sampled RNG uniform is unobserved. Wins in this corpus are training outcomes.')
    save(dest/'training_manifest.json',training_manifest)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=int)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--matrix-sha',required=True)
    parser.add_argument('--out',type=Path,default=HERE/'hosted')
    parser.add_argument('--auditor-commit',help='Full commit on main containing this exact postgame converter; defaults to dispatched commit')
    parser.add_argument('--workers',type=int,choices=range(1,5),default=2)
    parser.add_argument('archives',nargs='*',type=Path)
    a=parser.parse_args()
    result=merge(a.run,a.commit,a.matrix_sha,a.archives,a.out,a.workers,a.auditor_commit or a.commit)
    print(json.dumps({k:result[k] for k in ('records_reconciled','training_eligible','missing_tickets','verification_errors')}))
    sys.exit(0 if result['training_eligible'] else 1)
