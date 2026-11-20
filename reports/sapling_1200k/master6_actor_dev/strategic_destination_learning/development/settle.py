"""Reconcile twelve hosted Sapling development attempts against the frozen original examiner."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import subprocess
from tempfile import TemporaryDirectory
import zipfile

import matrix as m

CONTRASTS=(('bc','parent'),('joint','bc'),('joint','parent'))
ERRORS=(OSError,ValueError,KeyError,TypeError,IndexError,AssertionError,zipfile.BadZipFile)
FILES=('registration.json','receipt.json','turns.jsonl','attempt.json')


def outcomes(rows):
    wins=draws=losses=0
    margins=[]
    for g in rows:
        if g['counted']:
            winner=g['winner_seat']
            if winner is None:draws+=1
            elif winner==g['seat']:wins+=1
            else:losses+=1
        if g['official_scores_by_seat'] is not None:
            s=g['seat'];p=g['official_scores_by_seat'];margins.append(p[s]-p[1-s])
    n=len(rows)
    return dict(registered=n,wins=wins,draws=draws,losses=losses,unresolved=n-wins-draws-losses,
                observed_wins_over_registered=wins/n if n else None,
                observed_win_fraction_when_all_decided=wins/n if n and wins+draws+losses==n else None,
                numeric_scores=len(margins),mean_official_margin_all=mean(margins) if n and len(margins)==n else None,
                mean_margin_of_available_official_scores=mean(margins) if margins else None,
                status_counts=dict(sorted(Counter(g['status'] for g in rows).items())))


def aggregate(games,seeds):
    indexed={(g['seed'],g['seat'],g['arm']):g for g in games}
    expected={(s,p,a) for s in seeds for p in (0,1) for a in m.ARMS}
    m.require(len(indexed)==len(games) and set(indexed)==expected,'Different registered denominator')
    by_arm={}
    for arm in m.ARMS:
        rows=[g for g in games if g['arm']==arm]
        by_arm[arm]={'all':outcomes(rows),**{'seat'+str(p):outcomes([g for g in rows if g['seat']==p]) for p in (0,1)}}
    comparisons={}
    for trial,control in CONTRASTS:
        tickets=[];maps=[]
        for seed in seeds:
            ds=[]
            for seat in (0,1):
                scores=[];wl=[]
                for arm in (trial,control):
                    g=indexed[seed,seat,arm]
                    v=g['official_scores_by_seat']
                    scores.append(v[seat]-v[1-seat] if v is not None else None)
                    wl.append(int(g['winner_seat']==seat) if g['counted'] else None)
                diff=scores[0]-scores[1] if None not in scores else None
                wd=wl[0]-wl[1] if None not in wl else None
                tickets.append(dict(seed=seed,seat=seat,trial_margin=scores[0],control_margin=scores[1],
                                    paired_margin_change=diff,paired_win_indicator_change=wd))
                ds.append((diff,wd))
            summary=dict(seed=seed)
            for f,idx in [('margin_change',0),('win_fraction_change',1)]:
                v=[p[idx] for p in ds];summary[f]=mean(v) if None not in v else None
            maps.append(summary)
        leave_one_out=[dict(omitted=seed,retained=o['seed'],margin_change=o['margin_change'],
                            win_fraction_change=o['win_fraction_change']) for seed in seeds for o in maps if o['seed']!=seed]
        comparisons[trial+'_minus_'+control]=dict(tickets=tickets,by_physical_map=maps,leave_one_map_out=leave_one_out,
            average_map_margin_change=mean(z['margin_change'] for z in maps) if all(z['margin_change'] is not None for z in maps) else None,
            average_win_fraction_change=mean(z['win_fraction_change'] for z in maps) if all(z['win_fraction_change'] is not None for z in maps) else None)
    return dict(arms=by_arm,paired=comparisons,cluster_unit='Two physical maps with switched seats; individual map estimates and leave-one-map summaries are descriptive.')


def missing(seed,seat,arm,reason):
    return dict(seed=seed,seat=seat,arm=arm,status='missing_or_unverified',reason=reason,counted=False,winner_seat=None,
                official_scores_by_seat=None,official_transitions=None)


def unpack(archive,dest):
    names=['pair_status.json','exposure.json','execution.jsonl','verification.json']
    names += [arm+'/'+f for arm in m.ARMS for f in FILES]
    with zipfile.ZipFile(archive) as z:
        paths=z.namelist()
        m.require(len(paths)==len(set(paths)) and z.testzip() is None,'Damaged or duplicate ZIP members')
        for name in names:
            if name in paths:
                out=dest/name
                out.parent.mkdir(parents=True,exist_ok=True)
                out.write_bytes(z.read(name))


def block(archive,seed,seat,run_id,commit,protocol):
    errors={};games={}
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)
        unpack(archive,folder)
        status=m.read(folder/'pair_status.json')
        m.require((status['seed'],status['seat'])==(seed,seat) and str(status['run_id'])==str(run_id) and
                  str(status['attempt'])=='1' and status['commit']==commit and status['protocol_sha256']==protocol,
                  'Artifact ticket identity does not match its hosted run')
        check=m.verify_block(folder,seed,seat)
        errors.update(check['verification_errors'])
        sequence=check['host_sequence']
        m.require(all(str(n['workflow_run'])==str(run_id) for n in sequence),'Different workflow in hosted sequence')
        try:
            hosted=m.read(folder/'verification.json')
            m.require(hosted==json.loads(json.dumps(check)),'Hosted and downloaded-archive replay differ')
        except ERRORS as exc:errors['hosted_verification']=str(exc)
        try:
            proof=m.read(folder/'exposure.json')
            m.require((proof['seed'],proof['seat'])==(seed,seat) and str(proof['run_id'])==str(run_id) and
                      str(proof['attempt'])=='1' and proof['protocol_sha256']==protocol and set(proof['arms'])==set(m.ARMS),
                      'Wrong exposure ticket')
            no_exposure=True
            for arm,card in proof['arms'].items():
                path=folder/arm
                receipt=m.read(path/'receipt.json') if (path/'receipt.json').exists() else {}
                claim=(path/'attempt.json').exists()
                m.require(card==dict(attempt=claim,initializations=receipt.get('strategy_initializations'),
                                     action_calls=receipt.get('action_calls')),'Unbound exposure for '+arm)
                no_exposure &= not claim and type(card['initializations']) is int and card['initializations']==0 and card['action_calls']==[0,0]
            m.require(proof['confirmed_no_exposure'] is bool(no_exposure),'Exposure conclusion differs from attempts')
        except ERRORS as exc:
            errors['exposure']=str(exc)
            proof=None
        for arm,g in check['games'].items():
            m.require(arm in m.ARMS and (g['seed'],g['seat'],g['arm'])==(seed,seat,arm) and str(g['workflow_run'])==str(run_id),
                      'Different game in archived receipt')
            games[arm]=g
    return games,errors,sequence,proof


def merge(run_id,commit,archives,dest,protocol):
    plan=m.check(protocol)
    run=m.timing.gh(f'runs/{run_id}')
    workflow='.github/workflows/'+m.WORKFLOW
    m.require(run['id']==run_id and run['head_sha']==commit and run['head_branch']=='main' and
              run['run_attempt']==1 and run['event']=='workflow_dispatch' and run['status']=='completed' and
              run['path']==workflow and run['head_repository']['full_name']=='robertyang87/carbon-league','Unexpected GitHub workflow or source')
    signed=subprocess.check_output(['git','-C',str(m.ROOT),'show',commit+':'+m.PLAN.relative_to(m.ROOT).as_posix()])
    m.require(signed==m.PLAN.read_bytes(),'Signed protocol differs at hosted commit')
    inventory=m.timing.gh(f'runs/{run_id}/artifacts?per_page=100')
    listed={r['name']:r for r in inventory['artifacts']}
    m.require(len(listed)==len(inventory['artifacts'])==inventory['total_count'],'Duplicate or truncated artifact list')
    expected={(s,p):f'strategic-dev-{s}-{p}-{run_id}' for s in plan['seeds'] for p in (0,1)}
    by_hash={listed[n]['digest']:k for k,n in expected.items() if n in listed}
    m.require(len(by_hash)==sum(n in listed for n in expected.values()),'Shared archive digest')
    seen={};rows=[];errors=[]
    for archive in archives:
        digest=m.sha(archive);key=by_hash.get('sha256:'+digest)
        m.require(key is not None and key not in seen,'Unknown or duplicate supplied archive')
        seed,seat=key;label=expected[key];item=listed[label]
        m.require(item['workflow_run']['id']==run_id and item['workflow_run']['head_sha']==commit and
                  item['size_in_bytes']==archive.stat().st_size,'Archive size or source mismatch')
        card=dict(seed=seed,seat=seat,id=item['id'],name=label,sha256=digest,bytes=archive.stat().st_size)
        seen[key]=card
        try:
            gs,err,sequence,exposure=block(archive,seed,seat,run_id,commit,protocol)
            errors.extend(dict(seed=seed,seat=seat,role=a,reason=e) for a,e in err.items())
            card.update(host_sequence=sequence,exposure=exposure)
        except ERRORS as exc:
            err={'block':str(exc)};gs={}
            errors.append(dict(seed=seed,seat=seat,role='block',reason=str(exc)))
        for arm in m.ARMS:
            rows.append(gs[arm] if arm in gs else missing(seed,seat,arm,err.get(arm,err.get('block','Missing official verification'))))
    absent=[]
    for key,label in expected.items():
        if key in seen:continue
        why='Signed archive absent locally' if label in listed else 'No matching GitHub artifact'
        absent.append(dict(seed=key[0],seat=key[1],expected=label,reason=why))
        rows.extend(missing(*key,arm,why) for arm in m.ARMS)
    rows.sort(key=lambda g:(g['seed'],g['seat'],m.ARMS.index(g['arm'])))
    result=dict(workflow_run=run_id,commit=commit,protocol_sha256=protocol,workflow_path=workflow,
                run_url=run['html_url'],workflow_conclusion=run['conclusion'],registered_games=len(rows),
                archives=[seen[k] for k in sorted(seen)],missing_tickets=absent,verification_errors=errors,
                games=rows,results=aggregate(rows,plan['seeds']),records_reconciled=not errors and not absent and
                all(g['status']!='missing_or_unverified' for g in rows),formal_exam_games=0)
    dest.parent.mkdir(parents=True,exist_ok=True)
    m.play.live.save(dest,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=int,required=True);parser.add_argument('--commit',required=True)
    parser.add_argument('--matrix-sha',required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('archives',type=Path,nargs='*');args=parser.parse_args()
    result=merge(args.run,args.commit,args.archives,args.out,args.matrix_sha)
    print(json.dumps(result['results'],ensure_ascii=False,indent=2))
    raise SystemExit(int(not result['records_reconciled']))
