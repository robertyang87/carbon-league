"""Clear two v3 development maps against repository, platform and other development panels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import types

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parents[1]
SCANNER=FAMILY/'ladder/cargo_defense/development/clearance.py'
SOURCE='91d1e13719060a50efd36cfc443510f0456c2b0c'
SEEDS=[621758606,621758607]
TOKEN=re.compile(rb'(?<![.\da-zA-Z])62175860[67](?![.\da-zA-Z])')

# A literal 606 appears as the *exclusive* bound of the previously run 602–605
# schedule in precisely these two frozen files. Reject every other occurrence.
UPPER_BOUND_FILES={'d98db9d48b63b4d37c36baf939b61b8d62f75d0eb6f30a5e32b9d4387f5a471d',
                   'eab59c789f91de8c446d071e920e2ab7d11fef85396611ae587311bc29c97c8f'}

class RegistrationToken:
    def search(self,raw):
        hit=TOKEN.search(raw)
        if not hit or hashlib.sha256(raw).hexdigest() not in UPPER_BOUND_FILES:
            return hit
        bound=b'range(621758602,621758606)'
        assert raw.count(bound)==1 and 621758606 not in range(621758602,621758606)
        return TOKEN.search(raw.replace(bound,b'range(602,606)'))


def main():
    spec=importlib.util.spec_from_file_location('official_and_published_maps',SCANNER)
    shared=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    tip=shared.prior.checked_remote()
    old=types.ModuleType('numeric_prefilter')
    old.__file__=shared.old.__file__
    source=Path(old.__file__).read_text()
    test="if b'62175859' in b and CANDIDATE.search(b)"
    assert source.count(test)==1
    exec(compile(source.replace(test,"if b'62175860' in b and CANDIDATE.search(b)"),old.__file__,'exec'),old.__dict__)
    old.SEEDS,old.TOKEN,old.BASE=SEEDS,TOKEN,SOURCE
    shared.SEEDS,shared.TOKEN,shared.SOURCE=SEEDS,RegistrationToken(),SOURCE
    census=shared.complete(old.generate(),tip)
    boards=census['by_seed']
    panels={}
    for folder in ('paired_improvement_dev','calibration_improvement_dev','identity_replication_dev'):
        record=json.loads(shared.prior.frozen(FAMILY/folder/'protocol.json',tip))
        historical=record['physical_boards']
        assert not set(historical)&set(boards) and not set(historical.values())&set(boards.values())
        panels[folder]=historical
    def api(path):
        return json.loads(subprocess.check_output(['gh','api','repos/robertyang87/carbon-league/'+path]))
    active={}
    for status in ('in_progress','queued','waiting','requested','pending'):
        runs=api('actions/runs?per_page=100&status='+status)
        assert runs['total_count']==len(runs['workflow_runs'])
        for r in runs['workflow_runs']:
            if r['id'] in active:continue
            jobs=api(f"actions/runs/{r['id']}/jobs?per_page=100")
            assert jobs['total_count']==len(jobs['jobs'])
            names=[j['name'] for j in jobs['jobs']]
            assert not any(TOKEN.search(n.encode()) for n in names)
            active[r['id']]={'head':r['head_sha'],'jobs':names}
    assert set(map(int,boards))==set(SEEDS) and len(set(boards.values()))==2
    census.update(schema='v3-development-clearance/1',purpose='Two development boards only, separate from formal and training.',
                  groups={'v3_development':SEEDS},reviewed_exclusive_range_endpoint_files=sorted(UPPER_BOUND_FILES),older_comparison_boards=panels,active_hosted_jobs=active,
                  verifier_sha256=shared.prior.sha(Path(__file__).read_bytes()),legacy_scanner_sha256=shared.prior.sha(SCANNER.read_bytes()))
    print(old.shared.save(HERE/'clearance.json',census),boards,tip,flush=True)


if __name__=='__main__':main()
