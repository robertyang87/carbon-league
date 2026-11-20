"""Clear two official opening identities against frozen repository, reserved panels and current ladder."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import types

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
SOURCE="91d1e13719060a50efd36cfc443510f0456c2b0c"
SEEDS=(621758600,621758601)
TOKEN=re.compile(rb"(?<![.a-zA-Z0-9])62175860[01](?![.a-zA-Z0-9])")

def verify():
    helper=FAMILY/"ladder/cargo_defense/development/clearance.py"
    spec=importlib.util.spec_from_file_location("registered_census",helper)
    d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
    prior=d.prior
    d.SOURCE=prior.old.SOURCE=SOURCE;d.SEEDS=prior.old.SEEDS=list(SEEDS);d.TOKEN=TOKEN
    names=prior.git("ls-tree","-r","--name-only",SOURCE).decode().splitlines()
    top,receipts,errors=prior.old.top_ledgers(names)
    original=prior.frozen(prior.TEMPLATE,SOURCE)
    script=original.decode()
    for old,new in (("len(top5)==1637",f"len(top5)=={len(top)}"),("if b'6217584' in b and CANDIDATE.search(b)","if b'62175860' in b and CANDIDATE.search(b)")):
        assert script.count(old)==1
        script=script.replace(old,new)
    scanner=types.ModuleType("exact_archive_census")
    scanner.__file__=str(prior.TEMPLATE)
    exec(compile(script,scanner.__file__,"exec"),scanner.__dict__)
    scanner.SNAPSHOT=SOURCE;scanner.SEEDS=list(SEEDS);scanner.CANDIDATE=TOKEN
    scanner.GROUPS={"development_only":list(SEEDS)}
    result=scanner.scan()
    assert result["by_seed"]==prior.old.boards()
    tip=prior.checked_remote()
    data=d.complete(result,tip)
    previous=json.loads((FAMILY/"paired_improvement_dev/protocol.json").read_bytes())
    assert not set(SEEDS)&set(previous["seeds"]) and not set(result["by_seed"].values())&set(previous["physical_boards"].values())
    def get(q):return json.loads(subprocess.check_output(["gh","api","repos/robertyang87/carbon-league/"+q]))
    run_states=["queued","in_progress","waiting","requested","pending"]
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        active=list(pool.map(lambda s:get("actions/runs?per_page=100&status="+s),run_states))
    in_flight={}
    for r in active:
        assert r["total_count"]==len(r["workflow_runs"])
        for run in r["workflow_runs"]:
            if run["id"] in in_flight:continue
            jobs=get(f"actions/runs/{run['id']}/jobs?per_page=100")
            assert jobs["total_count"]==len(jobs["jobs"])
            names=[j["name"] for j in jobs["jobs"]]
            assert not any(TOKEN.search(k.encode()) for k in names),"seed currently running"
            in_flight[run["id"]]=dict(workflow=run["name"],jobs=names)
    data.update(schema="master3s2-rollout-calibration-clearance/1",physical_boards=data["by_seed"],active_runs=in_flight,
                distinct_from_prior_workflow=previous["physical_boards"],top5_receipt_digests=receipts,
                top5_faults=errors,archive_scanner_sha256=prior.sha(original),
                verifier_sha256=prior.sha(Path(__file__).read_bytes()))
    print(d.old.shared.save(HERE/"clearance.json",data), tip, data["by_seed"],flush=True)
if __name__=="__main__":verify()
