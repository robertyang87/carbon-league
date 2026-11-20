"""Audit only the sixteen registered fresh-map Sapling BC replication attempts."""
import argparse
from hashlib import sha256
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'development/settle.py'
PREVIOUS_SHA='72fb16f59aafd2da78c95e0133e3a73d9d9a8fc27333a99dd09cf60f2cce6956'
SEEDS=(621758510,621758511,621758512,621758513)
ARMS=('parent','bc')
WORKFLOW='.github/workflows/sapling-strategic-bc-replication.yml'
T95_DF3=3.182446305284263


def need(ok,message):
    if not ok:
        raise ValueError(message)


def previous(matrix):
    """Use the published physical, cgroup, timing, terminal and receipt audits with this registration."""
    need(sha256(PRIOR.read_bytes()).hexdigest()==PREVIOUS_SHA,'Previous independent auditor bytes changed')
    spec=importlib.util.spec_from_file_location('signed_strategic_development_auditor',PRIOR)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.m=matrix
    return module


def percentile(values,p):
    h=(len(values)-1)*p
    left=int(h);right=math.ceil(h)
    return values[left]+(values[right]-values[left])*(h-left)


def uncertainty(xs):
    result=dict(map_count=len(xs),required_map_count=4,complete_map_effects=xs,
                mean=None,sample_sd=None,standard_error=None,t95_df3=None,
                bootstrap_enumeration=0,bootstrap95_type7=None)
    if len(xs)!=4:
        return result
    avg=mean(xs);sd=stdev(xs);se=sd/2
    draws=sorted(mean(sample) for sample in product(xs,repeat=4))
    result.update(mean=avg,sample_sd=sd,standard_error=se,t95_df3=[avg-T95_DF3*se,avg+T95_DF3*se],
                  bootstrap_enumeration=len(draws),bootstrap95_type7=[percentile(draws,.025),percentile(draws,.975)])
    return result


def aggregate(rows,common):
    indexed={(g['seed'],g['seat'],g['arm']):g for g in rows}
    expected={(s,seat,a) for s in SEEDS for seat in (0,1) for a in ARMS}
    need(len(indexed)==len(rows)==16 and set(indexed)==expected,'Missing registered replication denominator')
    arms={}
    for arm in ARMS:
        group=[g for g in rows if g['arm']==arm]
        arms[arm]={'all':common.outcomes(group),**{'seat'+str(p):common.outcomes([g for g in group if g['seat']==p]) for p in (0,1)}}
    paired=[];maps=[]
    for seed in SEEDS:
        gm=[];gw=[]
        for seat in (0,1):
            g={a:indexed[seed,seat,a] for a in ARMS}
            margins={a:x['official_scores_by_seat'][seat]-x['official_scores_by_seat'][1-seat]
                     if x['official_scores_by_seat'] is not None else None for a,x in g.items()}
            wins={a:int(x['winner_seat']==seat) if x['counted'] else None for a,x in g.items()}
            dm=margins['bc']-margins['parent'] if None not in margins.values() else None
            dw=wins['bc']-wins['parent'] if None not in wins.values() else None
            gm.append(dm);gw.append(dw)
            paired.append(dict(seed=seed,seat=seat,official_margins=margins,observed_wins=wins,
                            bc_minus_parent_margin=dm,bc_minus_parent_win=dw))
        maps.append(dict(seed=seed,complete_paired_scores=sum(x is not None for x in gm),
                         bc_minus_parent_mean_margin=mean(gm) if None not in gm else None,
                         bc_minus_parent_win_fraction=mean(gw) if None not in gw else None))
    effects=[r['bc_minus_parent_mean_margin'] for r in maps if r['bc_minus_parent_mean_margin'] is not None]
    rates=[r['bc_minus_parent_win_fraction'] for r in maps if r['bc_minus_parent_win_fraction'] is not None]
    return dict(arms=arms,paired_map_seats=paired,map_clusters=maps,
                bc_minus_parent_official_margin=uncertainty(effects),bc_minus_parent_win_fraction=uncertainty(rates),
                inference='One paired effect per physical map, equally averaging both seats. Two-sided Student t 95% interval has 3 df. Bootstrap sensitivity enumerates all 256 ordered four-map bootstrap resamples with equal mass; 2.5% and 97.5% quantiles use type-7 linear interpolation. Both intervals require all four complete map effects for the relevant endpoint. Development-only uncertainty is sensitive to four maps.')


def merge(run_id,commit,paths,dest,protocol):
    import matrix as m
    old=previous(m)
    plan=m.check(protocol)
    need(tuple(plan['seeds'])==SEEDS and tuple(plan['arms'])==ARMS and plan['expected_games']==16,
         'Wrong replication design')
    run=m.timing.gh(f'runs/{run_id}')
    need(run['id']==run_id and run['head_sha']==commit and run['head_branch']=='main' and
         run['run_attempt']==1 and run['event']=='workflow_dispatch' and run['status']=='completed' and
         run['path']==WORKFLOW and run['head_repository']['full_name']=='robertyang87/carbon-league','Unexpected hosted run')
    signed=subprocess.check_output(['git','-C',str(m.ROOT),'show',commit+':'+m.PLAN.relative_to(m.ROOT).as_posix()])
    need(signed==m.PLAN.read_bytes(),'Different registration at hosted commit')
    inventory=m.timing.gh(f'runs/{run_id}/artifacts?per_page=100')
    listed={r['name']:r for r in inventory['artifacts']}
    need(len(listed)==len(inventory['artifacts'])==inventory['total_count'],'Non-unique or partial archive list')
    expected={(s,seat):f'bc-repl-{s}-{seat}-{run_id}' for s in SEEDS for seat in (0,1)}
    hashed={listed[name]['digest']:key for key,name in expected.items() if name in listed}
    need(len(hashed)==sum(name in listed for name in expected.values()),'Repeated hosted archive digest')
    seen={};rows=[];errors=[]
    for path in paths:
        digest=m.sha(path);key=hashed.get('sha256:'+digest)
        need(key is not None and key not in seen,'Unexpected or duplicate archive')
        seed,seat=key;label=expected[key];item=listed[label]
        need(item['workflow_run']['id']==run_id and item['workflow_run']['head_sha']==commit and item['size_in_bytes']==path.stat().st_size,
             'Different archive identity')
        proof=dict(seed=seed,seat=seat,name=label,id=item['id'],sha256=digest,bytes=path.stat().st_size)
        seen[key]=proof
        try:
            games,issues,sequence,exposure=old.block(path,seed,seat,run_id,commit,protocol)
            proof.update(host_sequence=sequence,exposure=exposure)
            errors.extend(dict(seed=seed,seat=seat,role=k,reason=v) for k,v in issues.items())
        except old.ERRORS as exc:
            games={};issues={'block':str(exc)}
            errors.append(dict(seed=seed,seat=seat,role='block',reason=str(exc)))
        for arm in ARMS:
            rows.append(games[arm] if arm in games else old.missing(seed,seat,arm,issues.get(arm,issues.get('block','Missing official verification'))))
    absent=[]
    for key,label in expected.items():
        if key not in seen:
            why='Signed archive absent locally' if label in listed else 'No matching GitHub artifact'
            absent.append(dict(seed=key[0],seat=key[1],expected=label,reason=why))
            rows.extend(old.missing(*key,arm,why) for arm in ARMS)
    rows.sort(key=lambda g:(g['seed'],g['seat'],ARMS.index(g['arm'])))
    result=dict(workflow_run=run_id,commit=commit,protocol_sha256=protocol,workflow_path=WORKFLOW,run_url=run['html_url'],
                workflow_conclusion=run['conclusion'],archives=[seen[k] for k in sorted(seen)],missing_tickets=absent,
                verification_errors=errors,registered_games=16,games=rows,results=aggregate(rows,old),
                records_reconciled=not absent and not errors and all(g['status']!='missing_or_unverified' for g in rows),
                formal_exam_games=0,previous_development_games_included=0)
    dest.parent.mkdir(parents=True,exist_ok=True)
    m.play.live.save(dest,result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=int,required=True);ap.add_argument('--commit',required=True)
    ap.add_argument('--matrix-sha',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('archives',nargs='*',type=Path)
    args=ap.parse_args()
    result=merge(args.run,args.commit,args.archives,args.out,args.matrix_sha)
    print(json.dumps(result['results'],ensure_ascii=False,indent=2))
    sys.exit(int(not result['records_reconciled']))
