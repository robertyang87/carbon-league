"""Verify every preregistered original game and report scores by physical map."""
import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from statistics import mean
import subprocess
import tempfile
import zipfile

import matrix as m

spec = importlib.util.spec_from_file_location("shared_match_auditor", m.BASE / "settle.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.m = m
spec = importlib.util.spec_from_file_location("public_replay", m.STRATEGY / "native_action_onpolicy_intake/observations.py")
views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(views)


def clustered(rows):
    maps = []
    for seed in m.SEEDS:
        pair = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["seat"])
        m.require([r["seat"] for r in pair] == [0, 1], "Incomplete registered seat pair")
        summary = audit.outcomes(pair)
        maps.append(dict(seed=seed, wins=summary["wins"], draws=summary["draws"], losses=summary["losses"],
                         win_fraction=summary["observed_win_fraction_when_all_decided"], mean_margin=summary["mean_official_margin_all"]))
    intervals = {}
    for metric in ("win_fraction", "mean_margin"):
        values = [r[metric] for r in maps]
        if None in values:
            intervals[metric] = None
            continue
        means = sorted(mean(sample) for sample in itertools.product(values, repeat=len(values)))
        def percentile(q):
            pos = (len(means) - 1) * q
            lower = int(pos)
            return means[lower] + (pos - lower) * (means[min(lower + 1,len(means) - 1)] - means[lower])
        intervals[metric] = [percentile(.025), percentile(.975)]
    return dict(groups=maps, empirical_percentile_95=intervals, independent_map_count=len(maps), resamples=4**4)


def merge(run_id, commit, digest, archives):
    plan = m.check(digest)
    signed = subprocess.check_output(["git", "-C", str(m.ROOT), "show", commit + ":" + m.PLAN.relative_to(m.ROOT).as_posix()])
    m.require(signed == m.PLAN.read_bytes(), "Hosted registration differs from local protocol")
    run = m.timing.gh(f"runs/{run_id}")
    m.require(run["id"] == run_id and run["head_sha"] == commit and run["run_attempt"] == 1
              and run["path"] == ".github/workflows/" + m.WORKFLOW and run["head_branch"] == "main"
              and run["head_repository"]["full_name"] == "robertyang87/carbon-league"
              and run["event"] == "workflow_dispatch" and run["status"] == "completed", "Incorrect hosted run")
    raw = m.timing.gh(f"runs/{run_id}/artifacts?per_page=100")
    by_name = {a["name"]: a for a in raw["artifacts"]}
    local = {m.sha(p):p for p in archives}
    m.require(len(by_name) == len(raw["artifacts"]) == raw["total_count"] and len(local) == len(archives), "Duplicate archive")
    consumed = set()
    games, errors, sources = [], [], []
    for ticket in plan["schedule"]:
        seed, seat = ticket["seed"], ticket["seat"]
        name = f"m3s2-dev1-{seed}-{seat}-{run_id}"
        try:
            a = by_name[name]
            key = a["digest"].removeprefix("sha256:")
            path = local[key]
            m.require(key not in consumed and a["workflow_run"]["id"] == run_id and a["workflow_run"]["head_sha"] == commit
                      and a["size_in_bytes"] == path.stat().st_size, "Wrong archive identity")
            consumed.add(key)
            arm_games, issues, seq, exposure = audit.block(path, seed, seat, run_id, commit, digest)
            m.require(not issues, json.dumps(issues))
            with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(path) as z:
                folder = Path(td)
                audit.unpack(path, folder)
                archived = z.read(m.ARM + "/observations.jsonl.gz")
                m.require(gzip.decompress(views.recover(folder / m.ARM)) == gzip.decompress(archived), "Public requests failed independent replay")
                g = arm_games[m.ARM]
                g.update(archive_id=a["id"], archive_path=path.name, public_observations_sha256=hashlib.sha256(archived).hexdigest())
                games.append(g)
            sources.append(dict(id=a["id"], seed=seed, seat=seat, name=name, bytes=a["size_in_bytes"], sha256=key,
                                host_sequence=seq, exposure=exposure))
        except audit.ERRORS as e:
            errors.append(dict(seed=seed, seat=seat, reason=str(e)))
            games.append(audit.missing(seed,seat,m.ARM,str(e)))
    m.require(set(local) == consumed, "Unregistered archive")
    return dict(schema="master3s2-derivative-dev1-results/1", run_id=run_id, run_url=run["html_url"],
                commit=commit, protocol_sha256=digest, workflow=run["path"], workflow_conclusion=run["conclusion"],
                games=games, archives=sources, verification_errors=errors, all=audit.outcomes(games), map_clusters=clustered(games),
                by_seat={str(s):audit.outcomes([r for r in games if r["seat"]==s]) for s in (0,1)},
                records_reconciled=not errors, formal_games=0, training_games=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--matrix-sha", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("archives", type=Path, nargs="*")
    args = p.parse_args()
    result = merge(args.run, args.commit, args.matrix_sha, args.archives)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.play.live.save(args.out, result)
    print(json.dumps(result["all"]))
    raise SystemExit(int(not result["records_reconciled"]))
