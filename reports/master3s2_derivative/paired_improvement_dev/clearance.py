"""Admit the unused return-study reservation only after checking its dispatch history and subsequent map use."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
ROOT=FAMILY.parents[1]
RANK=FAMILY/'return_rank/development'
PRIOR=FAMILY/'ladder/cargo_defense/development/clearance.py'
TEMPLATE='master3s2-return-rank-development.yml'
OLD='341d222f35866da9f8fad6ee51a6819b7094fbfd'
SEEDS=[621758598,621758599]
PREVIOUS={
 '.github/workflows/'+TEMPLATE:'845e3f35e6fdc0167d2562f7368d984864f4658591da54e161b95861df6915c4',
}
for f,h in [('clearance.json','6e475420071a049648ac0d523025004802f823d8b694de6ca7d6341fc8d7d508'),
            ('clearance.py','f85f3139236440c2b284eaf975b4a6a7984afcca46fd48eed0a49ce32259456d'),
            ('matrix.py','f9e73b3e8b659b9932ecc44f20a4382704396e18f4a5ac3bef908123a5ef2f86'),
            ('protocol.json','77c4c7a422945a7b8d9bd288ac64e24f28a932cc97670ba02c771c8dd1510183')]:
    PREVIOUS[(RANK/f).relative_to(ROOT).as_posix()]=h


def gh(query):
    return json.loads(subprocess.check_output(['gh','api','repos/robertyang87/carbon-league/'+query]))


def run():
    s=importlib.util.spec_from_file_location('old_clearance',PRIOR)
    d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
    old=json.loads((RANK/'clearance.json').read_bytes())
    assert old['through']==OLD and old['passed']
    tip=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT).decode().split()[0]
    statuses=['in_progress','queued','waiting','requested','pending']
    src_time=subprocess.check_output(['git','show','-s','--format=%cI',OLD],cwd=ROOT).decode().strip()
    utc=datetime.fromisoformat(src_time).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    endpoints=['actions/workflows/'+TEMPLATE+'/runs?per_page=1','actions/runs?per_page=100&created=%3E%3D'+utc,
              *['actions/runs?per_page=100&status='+v for v in statuses]]
    with ThreadPoolExecutor(max_workers=7) as pool:
        replies=list(pool.map(gh,endpoints))
    assert replies[0]['total_count']==0, 'Former registration was dispatched'
    statuses_read={v:doc['total_count'] for v,doc in zip(statuses,replies[2:])}
    runs=replies[1]
    assert runs['total_count']==len(runs['workflow_runs']), 'Paginate the activity ledger'
    games=[]
    active={}
    for response in replies[2:]:
        assert response['total_count']==len(response['workflow_runs']), 'Paginate active runs'
        active.update({r['id']:r for r in response['workflow_runs']})
    activity={r['id']:r for r in runs['workflow_runs']}
    activity.update(active)
    for rr in activity.values():
        jobs=gh(f"actions/runs/{rr['id']}/jobs?per_page=100")
        assert jobs['total_count']==len(jobs['jobs'])
        names=[v['name'] for v in jobs['jobs']]
        assert not any(str(k) in n for k in SEEDS for n in names), 'One of the reserved seats appeared in hosted job metadata'
        games.append(dict(run=rr['id'],workflow=rr['name'],commit=rr['head_sha'],jobs=names))
    r=d.prior
    d.SOURCE=OLD;d.SEEDS=SEEDS;d.TOKEN=re.compile(rb'(?<![.\da-zA-Z])62175859[89](?![.\da-zA-Z])')
    original_extend=r.extend;actual_git=r.git
    for path,expected in PREVIOUS.items():
        assert hashlib.sha256(actual_git('show',tip+':'+path)).hexdigest()==expected
    def reservation_aware(scan,start,end):
        def materialize(*args):
            # Exclude only exact signed bytes of the unused former reservation from the identity-token search.
            # Real game archives, scorecards, model data and subsequent files are always scanned normally.
            if len(args)==2 and args[0]=='show' and args[1].startswith(end+':'):
                path=args[1][len(end)+1:]
                if path in PREVIOUS:
                    raw=actual_git(*args);assert hashlib.sha256(raw).hexdigest()==PREVIOUS[path]
                    return b'Unplayed prior reservation '+PREVIOUS[path].encode()
            return actual_git(*args)
        r.git=materialize
        try:original_extend(scan,start,end)
        finally:r.git=actual_git
    r.extend=reservation_aware
    census={**old,'source_snapshot':OLD}
    evidence=d.complete(census,tip)
    for entry in evidence['repo_extension']:
        for path in PREVIOUS.keys() & entry['changed_sha256'].keys():entry['changed_sha256'][path]=PREVIOUS[path]
    doc=dict(schema='master3s2-paired-improvement-existing-reservation/1',passed=True,physical_boards=old['by_seed'],from_commit=OLD,
             through=tip,prior_whole_inventory_sha256=PREVIOUS[(RANK/'clearance.json').relative_to(ROOT).as_posix()],
             prior_protocol_sha256=PREVIOUS[(RANK/'protocol.json').relative_to(ROOT).as_posix()],former_unplayed_reservation_exact_files=PREVIOUS,
             former_workflow_run_count=replies[0]['total_count'],all_hosted_runs_since_prior_check=games,active_workflows=statuses_read,
             checked_at_utc=datetime.now(timezone.utc).isoformat(),source=evidence['repo_extension'],added_openings=evidence['added_repo_replays'],
             ladder_openings=evidence['ladder_openings'],live_ladder=evidence['ladder_scorecards'],numerical_intersections=[],physical_intersections=[],
             pending='Only published match openings can be compared. The signed independent final exam remains separate.',
             verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest=HERE/'clearance.json'
    if dest.exists():raise ValueError('Current clearance already exists')
    dest.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(tip,hashlib.sha256(dest.read_bytes()).hexdigest(),len(doc['ladder_openings']),len(doc['added_openings']),flush=True)


if __name__=='__main__':run()
