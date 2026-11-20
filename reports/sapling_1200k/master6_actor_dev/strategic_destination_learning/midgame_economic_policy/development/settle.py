"""Verify all twelve original matches including the same-host sequence and ladder timing."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import matrix
from matrix import ARMS, require as check
WORKFLOW=".github/workflows/"+matrix.WORKFLOW

def previous(matrix):
    spec=importlib.util.spec_from_file_location("registered_three_arm_audit",matrix.BASE/"settle.py")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.m=matrix
    m.CONTRASTS=(("weighted","uniform"),("weighted","parent"),("uniform","parent"))
    return m

def merge(run_id, commit, paths, dest, protocol):
    import matrix
    auditor = previous(matrix)
    plan = matrix.check(protocol)
    seeds = tuple(plan["seeds"])
    check(tuple(matrix.SEEDS)==seeds and tuple(plan["arms"])==tuple(matrix.ARMS)==ARMS and
          plan["expected_games"]==12 and len(set(plan["physical_boards"].values()))==2,
          "Wrong model order, physical maps or denominator")
    packages={k:plan["files"][k]["sha256"] for k in (*ARMS,"opponent")}
    signed_protocol = subprocess.check_output(["git","-C",str(matrix.ROOT),"show",
                                               commit+":"+matrix.PLAN.relative_to(matrix.ROOT).as_posix()])
    check(signed_protocol==matrix.PLAN.read_bytes(),"Registration differs at hosted commit")
    run = matrix.timing.gh(f"runs/{run_id}")
    check(run["id"]==run_id and run["head_sha"]==commit and run["head_branch"]=="main" and
          run["head_repository"]["full_name"]=="robertyang87/carbon-league" and run["run_attempt"]==1 and
          run["path"]==WORKFLOW and run["event"]=="workflow_dispatch" and run["status"]=="completed",
          "Unexpected GitHub workflow run identity")
    inventory = matrix.timing.gh(f"runs/{run_id}/artifacts?per_page=100")
    listed = {item["name"]:item for item in inventory["artifacts"]}
    check(len(listed)==len(inventory["artifacts"])==inventory["total_count"],"Incomplete or duplicate artifact metadata")
    expected = {f"economic-dev-{seed}-{seat}-{run_id}":(seed,seat) for seed in seeds for seat in (0,1)}
    signed_names = {listed[name]["digest"]:name for name in expected.keys() & listed.keys()}
    check(len(signed_names)==len(expected.keys() & listed.keys()),"Repeated archive digests")
    games, checked, missing, errors = [], {}, [], []
    for path in paths:
        h=matrix.sha(path)
        name=signed_names.get("sha256:"+h)
        check(name in expected and name not in checked,"Unregistered or duplicate local archive")
        seed,seat=expected[name]; item=listed[name]
        check(item["workflow_run"]["id"]==run_id and item["workflow_run"]["head_sha"]==commit and
              item["size_in_bytes"]==path.stat().st_size,"Hosted artifact ID or byte count mismatch")
        proof=dict(seed=seed,seat=seat,name=name,id=item["id"],sha256=h,bytes=path.stat().st_size)
        checked[name]=proof
        try:
            rows,issues,sequence,exposure=auditor.block(path,seed,seat,run_id,commit,protocol)
            proof.update(host_sequence=sequence,exposure=exposure)
        except auditor.ERRORS as exc:
            rows,issues={},{"block":str(exc)}
        errors.extend(dict(seed=seed,seat=seat,role=k,reason=v) for k,v in issues.items())
        for arm in ARMS:
            games.append(rows[arm] if arm in rows else auditor.missing(seed,seat,arm,issues.get(arm,issues.get("block","Missing official game"))))
    for name,(seed,seat) in expected.items():
        if name in checked:
            continue
        why="Signed artifact absent locally" if name in listed else "Missing GitHub artifact"
        missing.append(dict(seed=seed,seat=seat,expected=name,reason=why))
        games.extend(auditor.missing(seed,seat,arm,why) for arm in ARMS)
    games.sort(key=lambda g:(g["seed"],g["seat"],ARMS.index(g["arm"])))
    result=dict(workflow_run=run_id,commit=commit,protocol_sha256=protocol,workflow_path=WORKFLOW,
                workflow_conclusion=run["conclusion"],run_url=run["html_url"],packages=packages,
                archives=[checked[k] for k in sorted(checked)],missing_tickets=missing,verification_errors=errors,
                registered_games=12,games=games,results=auditor.aggregate(games,seeds),
                records_reconciled=not missing and not errors and all(g["status"]!="missing_or_unverified" for g in games),
                formal_exam_games=0,previous_development_games_included=0)
    dest.parent.mkdir(parents=True,exist_ok=True)
    matrix.play.live.save(dest,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=int,required=True)
    parser.add_argument("--commit",required=True)
    parser.add_argument("--matrix-sha",required=True)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("archives",nargs="*",type=Path)
    args=parser.parse_args()
    result=merge(args.run,args.commit,args.archives,args.out,args.matrix_sha)
    print(json.dumps(result["results"],ensure_ascii=False,indent=2,allow_nan=False))
    raise SystemExit(int(not result["records_reconciled"]))
