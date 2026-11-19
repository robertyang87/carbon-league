"""Bind exactly four connector-downloaded archives to the official GitHub run and registered seed/seat."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def read_meta(tail):
    return json.loads(subprocess.check_output(["gh","api","repos/robertyang87/carbon-league/actions/"+tail]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=int,required=True)
    p.add_argument("--commit",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("zipfiles",nargs=4,type=Path)
    a=p.parse_args()
    run=read_meta(f"runs/{a.run}")
    assert run["head_sha"]==a.commit and run["run_attempt"]==1 and run["head_branch"]=="main" and run["status"]=="completed" and run["event"]=="workflow_dispatch"
    items=read_meta(f"runs/{a.run}/artifacts")
    assert items["total_count"]==len(items["artifacts"])==4
    records={r["digest"]:r for r in items["artifacts"]}
    assert len(records)==4
    a.out.mkdir(parents=True,exist_ok=True)
    pairs=set(); archives=[]
    for src in a.zipfiles:
        data=src.read_bytes()
        key="sha256:"+hashlib.sha256(data).hexdigest()
        row=records.pop(key)
        assert row["size_in_bytes"]==len(data) and row["workflow_run"]["head_sha"]==a.commit and not row["expired"]
        with zipfile.ZipFile(src) as z:
            t=json.loads(z.read("parent/registration.json"));s=json.loads(z.read("student/registration.json"))
            seed,seat=t["seed"],t["candidate_seat"]
            assert (s["seed"],s["candidate_seat"])==(seed,seat) and str(t["workflow_run"])==str(s["workflow_run"])==str(a.run)
        assert (seed,seat) not in pairs
        pairs.add((seed,seat));name=f"{seed}-{seat}.zip"
        archives.append(dict(seed=seed,seat=seat,file=name,id=row["id"],digest=key,size_in_bytes=len(data),name=row["name"]))
        target=a.out/name
        if target.exists(): assert target.read_bytes()==data
        else: target.write_bytes(data)
    assert not records
    obj=dict(run_id=a.run,head_sha=a.commit,run_attempt=run["run_attempt"],status=run["status"],conclusion=run["conclusion"],html_url=run["html_url"],archives=sorted(archives,key=lambda r:(r["seed"],r["seat"])))
    (a.out/"archives.json").write_text(json.dumps(obj,indent=2)+"\n")


if __name__=="__main__":
    main()
