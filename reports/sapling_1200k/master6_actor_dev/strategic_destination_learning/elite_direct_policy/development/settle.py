"""Independently replay the sixteen archived elite-vs-control original-examiner games."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

HERE = Path(__file__).resolve().parent
REPLICATION = HERE.parents[1] / "replication/settle.py"
REPLICATION_SHA = "7c88ace63bf0d59a326229608adde76c59fa5c1705ff14d246e8cc6cc306900e"
WORKFLOW = ".github/workflows/sapling-elite-bc-comparison.yml"
ARMS = ("control", "elite")
PACKAGES = {"control":"ede48ad9770d357cbe4a4be6c24dafebc910be1e46a95c508c706e009fd713a0",
            "elite":"5e4f5da3fd63962ed559ba9efdeb788f454e3f6fb9ce617a3b5c877080795cb2",
            "opponent":"65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"}


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def prior():
    check(hashlib.sha256(REPLICATION.read_bytes()).hexdigest() == REPLICATION_SHA,
          "Signed four-map replication audit changed")
    spec = importlib.util.spec_from_file_location("elite_previous_replication", REPLICATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outcome(game):
    if not game["counted"]:
        return "unresolved"
    if game["winner_seat"] is None:
        return "draw"
    return "win" if game["winner_seat"] == game["seat"] else "loss"


def margin(game):
    p = game["official_scores_by_seat"]
    if p is None:
        return None
    check(len(p) == 2 and all(type(v) in (int,float) and math.isfinite(v) for v in p),
          "Invalid signed official score")
    s = game["seat"]
    return p[s] - p[1-s]


def aggregate(games, seeds, audited_outcomes, intervals):
    indexed = {(r["seed"],r["seat"],r["arm"]):r for r in games}
    check(len(seeds) == len(set(seeds)) == 4 and len(games) == 16 and
          set(indexed) == {(s,p,a) for s in seeds for p in (0,1) for a in ARMS},
          "Complete registered denominator required, with explicit missing rows")
    arms = {}
    for arm in ARMS:
        group = [g for g in games if g["arm"] == arm]
        arms[arm] = dict(all=audited_outcomes(group),
                         **{"seat"+str(s):audited_outcomes([g for g in group if g["seat"] == s]) for s in (0,1)})
    paired, clusters = [], []
    for seed in seeds:
        dm, dw = [], []
        for seat in (0,1):
            g = {a:indexed[seed,seat,a] for a in ARMS}
            m = {a:margin(row) for a,row in g.items()}
            wdl = {a:outcome(row) for a,row in g.items()}
            margin_change = m["elite"] - m["control"] if None not in m.values() else None
            win_change = int(wdl["elite"]=="win") - int(wdl["control"]=="win") if "unresolved" not in wdl.values() else None
            paired.append(dict(seed=seed, seat=seat, official_margins=m, results=wdl,
                               elite_minus_control_margin=margin_change, elite_minus_control_win=win_change))
            dm.append(margin_change)
            dw.append(win_change)
        clusters.append(dict(seed=seed,
            by_arm={arm:audited_outcomes([indexed[seed,seat,arm] for seat in (0,1)]) for arm in ARMS},
            paired_scores=sum(x is not None for x in dm),
            elite_minus_control_mean_margin=mean(dm) if None not in dm else None,
            elite_minus_control_win_fraction=mean(dw) if None not in dw else None))
    differences = {}
    for key in ("elite_minus_control_mean_margin","elite_minus_control_win_fraction"):
        differences[key] = intervals([c[key] for c in clusters if c[key] is not None])
    return dict(arms=arms, paired_map_seats=paired, map_clusters=clusters, **differences,
                inference="Development only. Each map is one cluster, both seats weighted equally. Two-sided Student t (df=3); all 256 ordered four-map bootstrap resamples yield type-7 2.5/97.5 percentile sensitivity. A missing seat suppresses its map effect for that endpoint, and any missing map suppresses the batch interval. Draws count as zero wins.")


def merge(run_id, commit, paths, dest, protocol):
    import matrix
    signed = prior()
    auditor = signed.previous(matrix)
    plan = matrix.check(protocol)
    seeds = tuple(plan["seeds"])
    check(tuple(matrix.SEEDS)==seeds and tuple(plan["arms"])==tuple(matrix.ARMS)==ARMS and
          plan["expected_games"]==16 and len(set(plan["physical_boards"].values()))==4,
          "Wrong model order, physical maps or denominator")
    check(all(plan["files"][name]["sha256"]==h for name,h in PACKAGES.items()),"Model or opponent mismatch")
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
    expected = {f"elite-dev-{seed}-{seat}-{run_id}":(seed,seat) for seed in seeds for seat in (0,1)}
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
                workflow_conclusion=run["conclusion"],run_url=run["html_url"],packages=PACKAGES,
                archives=[checked[k] for k in sorted(checked)],missing_tickets=missing,verification_errors=errors,
                registered_games=16,games=games,results=aggregate(games,seeds,auditor.outcomes,signed.uncertainty),
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
