"""Settle all scheduled held-map attempts from the hosted run and independent action/score replays."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

from menu import ARMS, NATURAL, verifier

sys.path.insert(0, str(NATURAL))
import settle

WORKFLOW = "sapling-native-return-validation.yml"
PREFIX = "native-validation"
SEEDS = (621758470,621758471)
ELIGIBLE = ("complete","forfeit")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def need(condition,message):
    if not condition:
        raise ValueError(message)


def aggregate(games):
    """One row per physical map; no imputed margin for a time loss with no official numeric result."""
    summary={k:v for k,v in settle.summarize(games).items() if k in ARMS}
    pairs={(g["seed"],g["seat"],g["arm"]):g for g in games}
    need(len(pairs)==len(games),"Repeated verified map/seat/arm")
    map_rows=[]
    for seed in SEEDS:
        seats=[]
        for seat in (0,1):
            delta=None
            def stats(arm):
                g=pairs.get((seed,seat,arm))
                if g is None: return dict(status="unverified",win=None,margin=None)
                scores=g.get("official_scores_by_seat")
                return dict(status=g["status"],win=int(g["winner_seat"]==seat) if g["counted"] and g["status"] in ELIGIBLE else None,
                            margin=scores[seat]-scores[1-seat] if scores is not None and g["status"]=="complete" else None)
            a,b=[stats(k) for k in ARMS]
            if a["margin"] is not None and b["margin"] is not None:
                delta=b["margin"]-a["margin"]
            seats.append(dict(seat=seat,parent=a,child=b,child_minus_parent_margin=delta))
        diff=[s["child_minus_parent_margin"] for s in seats]
        map_rows.append(dict(seed=seed,seats=seats,paired_map_margin_change=sum(diff)/2 if None not in diff else None))
    ds=[r["paired_map_margin_change"] for r in map_rows]
    # With only two maps, leaving one out yields the other physical map's two-seat mean.
    sensitivity={} if None in ds else {str(s):ds[1-i] for i,s in enumerate(SEEDS)}
    return dict(arms=summary,map_clusters=map_rows,complete_paired_maps=sum(x is not None for x in ds),
                leave_one_map_out_mean_margin_change=sensitivity,
                child_minus_parent_mean_margin=sum(ds)/2 if None not in ds else None,
                scope="Two physical validation maps with both seats. Each map's paired difference is shown separately. Formal exam games: zero.")


def inspect_archive(path, directory, request):
    seed,seat=request["key"]
    with zipfile.ZipFile(path) as z:
        members=z.namelist()
        need(len(members)==len(set(members)) and z.testzip() is None,"Duplicate or damaged ZIP members")
        files=["execution.jsonl","verification.json","pair_status.json","exposure.json"]
        files.extend(arm+"/"+x for arm in ARMS for x in ("registration.json","receipt.json","turns.jsonl","attempt.json"))
        for name in files:
            if name in members:
                target=directory/name
                target.parent.mkdir(exist_ok=True,parents=True)
                target.write_bytes(z.read(name))
    status=read(directory/"pair_status.json")
    need((status["seed"],status["seat"])==(seed,seat) and str(status["run_id"])==str(request["run_id"]) and
         status["commit"]==request["commit"] and status["protocol_sha256"]==request["protocol_sha"] and str(status["attempt"])=="1",
         "Hosted ticket differs from declared slot")


def merge(run_id, commit, protocol_sha, archives, out):
    # The driver owns all pre-match frozen provenance, candidate-selection and resource checks.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import matrix as m
    m.check(protocol_sha)
    plan=read(m.PLAN)
    need(tuple(plan["seeds"])==SEEDS and tuple(plan["arms"])==ARMS and plan["expected_games"]==8,
         "Different trial registration")
    run=settle.gh(f"runs/{run_id}")
    need(run["id"]==run_id and run["head_sha"]==commit and run["head_branch"]=="main" and run["run_attempt"]==1 and
         run["event"]=="workflow_dispatch" and run["status"]=="completed" and Path(run["path"]).name==WORKFLOW,
         "Host run or output commit differs")
    git_plan=subprocess.check_output(["git","-C",str(m.ROOT),"show",commit+":"+m.PLAN.relative_to(m.ROOT).as_posix()])
    need(git_plan==m.PLAN.read_bytes(),"Hosted release came from a different Git commit")
    checked_menu=verifier(m.PLAN)
    expected={(s,p):f"{PREFIX}-{s}-{p}-{run_id}" for s in SEEDS for p in (0,1)}
    listing=settle.gh(f"runs/{run_id}/artifacts?per_page=100")
    named={r["name"]:r for r in listing["artifacts"]}
    need(len(named)==listing["total_count"],"Incomplete or nonunique GitHub artifact inventory")
    bound={named[label]["digest"]:key for key,label in expected.items() if label in named}
    seen=set();games=[];proofs=[];errors=[]
    for path in archives:
        path=Path(path)
        digest=sha(path); key=bound.get("sha256:"+digest)
        need(key is not None and key not in seen,"Unexpected or duplicate authentic match ZIP")
        label=expected[key]; remote=named[label]
        need(remote["size_in_bytes"]==path.stat().st_size and remote["workflow_run"]["id"]==run_id and
             remote["workflow_run"]["head_sha"]==commit,"Remote archive size or source differs")
        seen.add(key)
        proof=dict(seed=key[0],seat=key[1],name=label,id=remote["id"],bytes=path.stat().st_size,sha256=digest)
        proofs.append(proof)
        try:
            with TemporaryDirectory() as temp:
                directory=Path(temp)
                inspect_archive(path,directory,dict(key=key,run_id=run_id,commit=commit,protocol_sha=protocol_sha))
                replay=m.verify_block(directory,*key)
                need(read(directory/"verification.json")==json.loads(json.dumps(replay)),"Hosted and local independent referee replays differ")
                errors.extend(dict(seed=key[0],seat=key[1],arm=k,reason=v) for k,v in replay["verification_errors"].items())
                proof["host_sequence"]=replay["host_sequence"]
                for arm,g in replay["games"].items():
                    games.append(g)
                    try:
                        need((g["seed"],g["seat"],g["arm"])==(*key,arm) and str(g["workflow_run"])==str(run_id),
                             "Unexpected game in this artifact")
                        receipt=checked_menu(directory/arm,arm)
                        if g["status"]=="complete":
                            need(receipt["matched_all_steps"] and receipt["official_scores_by_seat"]==g["official_scores_by_seat"] and
                                 receipt["official_transitions"]==g["official_transitions"],"Final official or menu replay differs")
                        g["action_support"]=receipt
                    except (ValueError,KeyError,OSError,TypeError,IndexError,AssertionError) as exc:
                        errors.append(dict(seed=key[0],seat=key[1],arm=arm,reason=str(exc)))
        except (ValueError,KeyError,OSError,TypeError,IndexError,zipfile.BadZipFile,AssertionError) as exc:
            errors.append(dict(seed=key[0],seat=key[1],reason=str(exc)))
    missing=sorted(set(expected)-seen)
    report=dict(workflow_run=run_id,commit=commit,plan_sha256=protocol_sha,
                package_sha256={arm:plan["files"][arm]["sha256"] for arm in ARMS},
                eligible_attempts=sum(g["status"] in ELIGIBLE for g in games),
                registered_games=8,formal_exam_games=0,archives=proofs,games=games,
                adjudication_status_counts=dict(Counter(g["status"] for g in games)),
                missing_tickets=missing,verification_errors=errors,
                records_reconciled=(len(games)==8 and not missing and not errors),results=aggregate(games))
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    settle.live.save(out,report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=int,required=True);p.add_argument("--commit",required=True)
    p.add_argument("--matrix-sha",required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("archives",nargs="+",type=Path)
    a=p.parse_args();r=merge(a.run,a.commit,a.matrix_sha,a.archives,a.out)
    print(json.dumps(r["results"],indent=2));sys.exit(not r["records_reconciled"])
