"""Validate each of four fresh development maps against numeric seeds and actual published openings."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import types

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
ROOT=FAMILY.parents[1]
SEEDS=list(range(621758602,621758606))
TOKEN=re.compile(rb'(?<![.\da-zA-Z])62175860[2-5](?![.\da-zA-Z])')
SOURCE='91d1e13719060a50efd36cfc443510f0456c2b0c'
baseline=FAMILY/'ladder/cargo_defense/development/clearance.py'
spec=importlib.util.spec_from_file_location('earlier_census',baseline)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def run():
    endpoint=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT).decode().split()[0]
    # Start from the previously verified whole tracked tree. Only the literal prefilter
    # in that old scanner changes; both numerical seeds and generated opening digests are still tested.
    module=types.ModuleType('updated_filter');module.__file__=d.old.__file__
    prior=Path(d.old.__file__).read_text()
    needle="if b'62175859' in b and CANDIDATE.search(b)"
    assert prior.count(needle)==1
    exec(compile(prior.replace(needle,"if b'62175860' in b and CANDIDATE.search(b)"),module.__file__,'exec'),module.__dict__)
    module.SEEDS,module.TOKEN,module.BASE=SEEDS,TOKEN,SOURCE
    primary=module.generate()
    d.SEEDS,d.SOURCE,d.TOKEN=SEEDS,SOURCE,TOKEN
    result=d.complete(primary,endpoint)
    original=FAMILY/'paired_improvement_dev/protocol.json'
    old=json.loads(d.prior.frozen(original,endpoint))
    assert not set(old['seeds'])&set(SEEDS) and not set(old['physical_boards'].values())&set(result['by_seed'].values())
    def api(query):
        return json.loads(subprocess.check_output(['gh','api','repos/robertyang87/carbon-league/'+query]))
    jobs={}
    for status in ('in_progress','queued','waiting','requested','pending'):
        reply=api('actions/runs?per_page=100&status='+status)
        assert len(reply['workflow_runs'])==reply['total_count']
        for item in reply['workflow_runs']:
            if item['id'] in jobs:continue
            detail=api(f"actions/runs/{item['id']}/jobs?per_page=100")
            assert len(detail['jobs'])==detail['total_count']
            names=[r['name'] for r in detail['jobs']]
            if any(str(seed) in n for seed in SEEDS for n in names):raise ValueError('Hosted physical reservation already active')
            jobs[item['id']]=dict(head=item['head_sha'],workflow=item['name'],job_names=names)
    result.update(schema='identity-crn-replication-four-map-clearance/1',groups={'independent_development_replication':SEEDS},
                  purpose='Four new development boards to replicate frozen CRN intervention. All physical boards excluded from formal examination and training.',
                  verifier_sha256=d.prior.sha(Path(__file__).read_bytes()),source_census_runner_sha256=d.prior.sha(baseline.read_bytes()),
                  previous_two_map_sha256=d.prior.sha(d.prior.frozen(original,endpoint)),previous_two_map_run=44692146016,
                  prior_comparison_boards=old['physical_boards'],active_github_actions=jobs)
    assert set(map(int,result['by_seed']))==set(SEEDS) and len(set(result['by_seed'].values()))==4
    print(d.old.shared.save(HERE/'clearance.json',result),result['by_seed'],endpoint,flush=True)


if __name__=='__main__':run()
