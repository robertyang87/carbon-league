"""Verify and settle the eight registered fresh-development attempts from signed GitHub artifacts."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

import matrix as m
sys.path.insert(0,str(m.PREVIOUS))
import settle
from audit_menu import verify_menu


def aggregate(games):
    cohorts={k:v for k,v in settle.summarize(games).items() if k in m.ARMS}
    complete={(g["seed"],g["seat"],g["arm"]):g for g in games if g["status"]=="complete"}
    tickets=[];maps=[]
    for seed in m.SEEDS:
        differences=[]
        for seat in (0,1):
            a,b=[complete.get((seed,seat,k)) for k in m.ARMS]
            if a is None or b is None:continue
            def margin(g):
                scores=g["official_scores_by_seat"]
                return scores[seat]-scores[1-seat]
            d=margin(b)-margin(a)
            differences.append(d)
            tickets.append(dict(seed=seed,seat=seat,parent_margin=margin(a),candidate_margin=margin(b),change=d))
        if len(differences)==2:maps.append(dict(seed=seed,both_seats_mean_change=mean(differences)))
    return dict(arms=cohorts,paired_numeric_tickets=tickets,complete_maps=maps,
                child_minus_parent_mean_margin=mean(q["both_seats_mean_change"] for q in maps) if len(maps)==2 else None,
                cluster_unit="Two physical development maps; the paired map differences are reported individually.")


def merge(run_id,commit,archives,dest,protocol_sha):
    m.check(protocol_sha)
    run=settle.gh(f"runs/{run_id}")
    m.require(run["id"]==run_id and run["head_sha"]==commit and run["head_branch"]=="main" and
              run["run_attempt"]==1 and run["event"]=="workflow_dispatch" and run["status"]=="completed" and
              Path(run["path"]).name=="sapling-natural-fresh-repeat.yml","Different GitHub workflow or run")
    signed=subprocess.check_output(["git","-C",str(m.ROOT),"show",commit+":"+str(m.PLAN.relative_to(m.ROOT))])
    m.require(signed==m.PLAN.read_bytes(),"GitHub run commit used a different registration")
    inventory=settle.gh(f"runs/{run_id}/artifacts?per_page=100")
    expected={(s,seat):f"fresh-natural-{s}-{seat}-{run_id}" for s in m.SEEDS for seat in (0,1)}
    named={r["name"]:r for r in inventory["artifacts"]}
    m.require(len(named)==inventory["total_count"],"Duplicate or truncated artifacts")
    by_sha={named[label]["digest"]:key for key,label in expected.items() if label in named}
    seen=set();games=[];proofs=[];errors=[]
    for archive in archives:
        digest=m.sha(archive);key=by_sha.get("sha256:"+digest)
        m.require(key is not None and key not in seen,"ZIP is not a unique registered GitHub match artifact")
        label=expected[key];item=named[label]
        m.require(item["workflow_run"]["id"]==run_id and item["workflow_run"]["head_sha"]==commit and
                  item["size_in_bytes"]==archive.stat().st_size,"Artifact identity mismatch")
        seen.add(key)
        proof=dict(seed=key[0],seat=key[1],name=label,id=item["id"],sha256=digest,bytes=archive.stat().st_size)
        proofs.append(proof)
        try:
            with TemporaryDirectory() as temp, zipfile.ZipFile(archive) as z:
                names=z.namelist();m.require(z.testzip() is None and len(names)==len(set(names)),"Invalid artifact members")
                root=Path(temp)
                for name in ["execution.jsonl","verification.json","pair_status.json","exposure.json"]+[
                        arm+"/"+n for arm in m.ARMS for n in ("registration.json","receipt.json","turns.jsonl","attempt.json")]:
                    if name in names:
                        path=root/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(z.read(name))
                status=m.read(root/"pair_status.json")
                m.require((status["seed"],status["seat"])==key and str(status["run_id"])==str(run_id) and
                          status["commit"]==commit and status["protocol_sha256"]==protocol_sha,"Wrong ticket or commit")
                checked=m.verify_block(root,*key)
                if m.read(root/"verification.json")!=json.loads(json.dumps(checked)):
                    errors.append(dict(ticket=key,reason="Hosted/local official replay differs"))
                for k,v in checked["verification_errors"].items():
                    errors.append(dict(ticket=key,arm=k,reason=v))
                proof["host_sequence"]=checked["host_sequence"]
                for arm,g in checked["games"].items():
                    games.append(g)
                    try:
                        audit=verify_menu(root/arm,"parent" if arm==m.A else "candidate")
                        m.require(audit["deployment_sha256"]==m.old.PARTICIPANTS[arm][1] and
                                  (g["status"]!="complete" or audit["matched_all_steps"]), "Physical support verification failed")
                        g["action_support"]=audit
                    except (ValueError,OSError,KeyError,AssertionError,IndexError,TypeError) as exc:
                        errors.append(dict(ticket=key,arm=arm,reason=str(exc)))
        except (ValueError,OSError,KeyError,zipfile.BadZipFile,TypeError,IndexError) as exc:
            errors.append(dict(ticket=key,reason=str(exc)))
    missing=sorted(set(expected)-seen)
    result=dict(workflow_run=run_id,commit=commit,plan_sha256=protocol_sha,registered_games=8,
                archives=proofs,missing_tickets=missing,verification_errors=errors,games=games,results=aggregate(games),
                records_reconciled=not errors and not missing and len(games)==8,
                formal_exam_games=0)
    dest.parent.mkdir(exist_ok=True,parents=True)
    m.old.play.live.save(dest,result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=int,required=True);p.add_argument("--commit",required=True)
    p.add_argument("--matrix-sha",required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("archives",nargs="+",type=Path)
    a=p.parse_args()
    r=merge(a.run,a.commit,a.archives,a.out,a.matrix_sha)
    print(json.dumps(r["results"],indent=2));sys.exit(not r["records_reconciled"])
