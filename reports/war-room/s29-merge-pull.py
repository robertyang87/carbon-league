"""C1 merge of war-room direct pull into sync/h29 (2026-11-13).

Merges ~/workspace/carbon-live/h29-pull/bigeval_live/<tag>.jsonl (machine
originals, 200 lines each, md5-verified by war room) with the existing
branch files synced/h29/bigeval_live/<tag>.jsonl on branch sync/h29,
keyed by seed, per-tag domain filter (_m2: 996000-996199,
_echo2: 999000-999199). On a content conflict for the same seed the
pulled machine original wins (it is the direct file copy). Merged files
are written exactly like the snaploops wrote them: lines sorted by seed,
"\n"-joined, trailing "\n".

Usage:
  python3 s29-merge-pull.py analyze   # read-only report, no writes
  python3 s29-merge-pull.py push      # per tag: fresh get -> put -> read-back verify
"""
import base64, hashlib, json, os, sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BRANCH = "robertyang87", "carbon-league", "sync/h29"
PULL = "/home/hatch/workspace/carbon-live/h29-pull/bigeval_live"
BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SNAP = BASE + "/exam_snapshots/h29"
TAGS = ["fabler1_m2", "fabler1_echo2", "fableg1_m2", "fableg1_echo2",
        "m7bsv2r1_m2", "m7bsv2r1_echo2", "m7bsv2g1_m2", "m7bsv2g1_echo2"]
STORE_OF = {"fabler1": "lines-fable.json", "fableg1": "lines-fableg1.json",
            "m7bsv2r1": "lines-m7v2r1.json", "m7bsv2g1": "lines-m7v2g1.json"}


def seed0(tag):
    return 996000 if tag.endswith("_m2") else 999000


def parse_lines(text):
    """Return {seed: line} plus list of unparseable lines."""
    out, bad = {}, []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
            out[int(rec["seed"])] = ln
        except Exception:
            bad.append(ln[:80])
    return out, bad


def load_pulled(tag):
    with open(os.path.join(PULL, tag + ".jsonl"), encoding="utf-8") as f:
        return parse_lines(f.read())


def fetch_branch(tag):
    r = gh_api.get_file(OWNER, REPO, f"synced/h29/bigeval_live/{tag}.jsonl", ref=BRANCH)
    raw = base64.b64decode(r["content"])
    d, bad = parse_lines(raw.decode("utf-8"))
    return d, bad, r["sha"], raw


def merge(tag):
    s0 = seed0(tag)
    domain = set(range(s0, s0 + 200))
    pulled, pbad = load_pulled(tag)
    branch, bbad, sha, _raw = fetch_branch(tag)
    conflicts = sorted(s for s in set(pulled) & set(branch) if pulled[s] != branch[s])
    merged = dict(branch)
    merged.update(pulled)  # machine original wins
    ood = sorted(s for s in merged if s not in domain)
    for s in ood:
        del merged[s]
    missing = sorted(domain - set(merged))
    report = {
        "tag": tag, "seed0": s0, "branch_sha": sha,
        "branch_n": len(branch), "pulled_n": len(pulled),
        "branch_only": sorted(set(branch) - set(pulled)),
        "pulled_only_n": len(set(pulled) - set(branch)),
        "conflicts": conflicts, "out_of_domain_dropped": ood,
        "merged_n": len(merged), "missing": missing,
        "bad_lines": {"pulled": pbad, "branch": bbad},
        "complete": not missing and len(merged) == 200,
    }
    data = ("\n".join(merged[s] for s in sorted(merged)) + "\n").encode()
    return report, data, merged


def analyze():
    total = 0
    for tag in TAGS:
        rep, _data, _m = merge(tag)
        total += rep["merged_n"]
        print(json.dumps(rep, ensure_ascii=False))
    print(f"TOTAL merged={total} target=1600")


def push():
    done = {}
    for tag in TAGS:
        rep, data, merged = merge(tag)  # fresh get inside (sha + counts)
        assert rep["complete"], f"{tag} not complete: {rep}"
        assert not rep["bad_lines"]["pulled"] and not rep["bad_lines"]["branch"], rep["bad_lines"]
        gh_api.put_file(OWNER, REPO, f"synced/h29/bigeval_live/{tag}.jsonl",
                        data, f"h29 backfill close {tag} 200 lines (direct pull merge)",
                        sha=rep["branch_sha"], branch=BRANCH)
        # read-back verification
        d2, bad2, _sha2, raw2 = fetch_branch(tag)
        ok = (raw2 == data and len(d2) == 200 and not bad2
              and set(d2) == set(range(seed0(tag), seed0(tag) + 200)))
        print(f"{tag}: pushed {rep['branch_n']}->{rep['merged_n']} "
              f"conflicts={rep['conflicts']} ood_dropped={rep['out_of_domain_dropped']} "
              f"readback_ok={ok} md5={hashlib.md5(raw2).hexdigest()}")
        assert ok, f"{tag} read-back mismatch"
        with open(os.path.join(SNAP, tag + ".jsonl"), "wb") as f:
            f.write(data)
        done[tag] = merged
    # refresh the four ring stores so local state is coherent (rings retired)
    for prefix, fname in STORE_OF.items():
        sub = {t: done[t] for t in TAGS if t.startswith(prefix)}
        with open(os.path.join(SNAP, fname), "w") as f:
            json.dump({t: {str(s): ln for s, ln in d.items()} for t, d in sub.items()}, f)
        print(f"store {fname} refreshed: " + ", ".join(f"{t}={len(d)}" for t, d in sub.items()))
    b = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/sync%2Fh29")
    print("branch head:", b["commit"]["sha"][:8], b["commit"]["commit"]["author"]["date"])
    print("PUSH_ALL_OK total=1600")


if __name__ == "__main__":
    {"analyze": analyze, "push": push}[sys.argv[1]]()
