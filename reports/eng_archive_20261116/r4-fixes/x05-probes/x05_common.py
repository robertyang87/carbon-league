"""Shared harness for the X05 reproduction probes (probe_adopt / probe_rmtree).

X05 (Codex r4, baseline ddc88081): cnet2_build's publish path decides whether
an existing generation can be adopted with `_gen_complete`, which only checks
that manifest.json parses and that every declared field file EXISTS. It never
compares the files against the manifest's own per-field sha256. A same-id
generation whose bytes were changed in place after publication (or written by
drifted code) is therefore adopted wholesale, re-pointed to, and mirrored into
the flat view; an INCOMPLETE same-id generation is deleted with shutil.rmtree
and replaced, destroying any forensic trace.

Boundary (per the r4 ruling): gen_key binds the replay fingerprints, so an
INPUT change mints a NEW generation id and is out of scope. These probes only
construct the post-publish, same-id / different-bytes case.

Both probes:
  * fetch code/cnet2_build.py + code/cnet2_feat.py from GitHub main at run
    time (via ~/workspace/gh_api.py) into a local vendor dir -- the repo is
    never written to, and the build under test is the current main bytes;
  * build a one-game corpus from a real ladder replay already on this VM
    (~/workspace/carbon-league/replays), with the index display name of one
    seat set to a teacher name so the scan keeps that side (the replay bytes
    themselves are untouched);
  * print PASS/FAIL-style assertion lines and exit 0 only if the X05 defect
    reproduced exactly as described. After a correct fix (verify every
    manifest hash before adoption; quarantine-rename instead of rmtree),
    these probes are EXPECTED to fail their defect assertions.
"""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/hatch/workspace")
from gh_api import get_file  # noqa: E402

OWNER, REPO = "robertyang87", "carbon-league"
REVIEWED_BUILD_MD5 = "8598f26de678450a2dd24e308107416e"  # code/cnet2_build.py at r4 baseline
SOURCE_CORPUS = "/home/hatch/workspace/carbon-league/replays"
TEACHER_NAME = "claude-AlphaCarbon-Master"


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def md5_file(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Probe:
    def __init__(self, name):
        self.name = name
        self.failures = []
        self.log_lines = []

    def log(self, msg):
        self.log_lines.append(msg)
        print(msg, flush=True)

    def check(self, label, cond, detail=""):
        mark = "PASS" if cond else "FAIL"
        self.log(f"[{mark}] {label}" + (f" -- {detail}" if detail else ""))
        if not cond:
            self.failures.append(label)
        return cond

    def finish(self, workdir):
        if self.failures:
            result = (f"RESULT {self.name}: NOT-REPRODUCED ({len(self.failures)} assertion(s) failed) "
                      f"-- expected on a FIXED build, a failure on the reviewed bytes")
            code = 2
        else:
            result = f"RESULT {self.name}: X05 DEFECT REPRODUCED (all assertions held)"
            code = 0
        self.log(result)
        log_path = os.path.join(workdir, "probe.log")
        with open(log_path, "w") as fh:
            fh.write("\n".join(self.log_lines) + "\n")
        return code


def fetch_vendor(vendor_dir):
    """Fetch the build under test + its only local dependency from main."""
    os.makedirs(vendor_dir, exist_ok=True)
    out = {}
    for name in ("cnet2_build.py", "cnet2_feat.py"):
        r = get_file(OWNER, REPO, "code/" + name)
        raw = base64.b64decode(r["content"])
        with open(os.path.join(vendor_dir, name), "wb") as fh:
            fh.write(raw)
        out[name] = md5_bytes(raw)
    return out


def make_corpus(corpus_dir):
    """One real ladder game, one seat renamed (index only) to a teacher.

    Selection is deterministic: among index entries with >=200 steps, all
    scores numeric, an existing replay file, and no producer-provenance keys
    (plain ladder record), take the first by (steps, file).
    """
    idx = json.load(open(os.path.join(SOURCE_CORPUS, "index.json")))
    cands = []
    for e in idx:
        fp = os.path.join(SOURCE_CORPUS, e["file"])
        if not os.path.exists(fp):
            continue
        if e.get("id") == "selfplay" or any(k in e for k in ("identity", "mirror_signature", "open_seed")):
            continue
        if e.get("steps", 0) < 200:
            continue
        if not all(isinstance(v, (int, float)) for v in e.get("scores", {}).values()):
            continue
        cands.append((e["steps"], e["file"], e))
    if not cands:
        raise SystemExit("no suitable source replay found in " + SOURCE_CORPUS)
    cands.sort(key=lambda x: (x[0], x[1]))
    entry = json.loads(json.dumps(cands[0][2]))  # deep copy
    os.makedirs(os.path.join(corpus_dir, "replays"), exist_ok=True)
    shutil.copyfile(os.path.join(SOURCE_CORPUS, entry["file"]),
                    os.path.join(corpus_dir, "replays", os.path.basename(entry["file"])))
    teacher_seat = entry["order"][0]
    entry["strategies"][teacher_seat]["name"] = TEACHER_NAME
    with open(os.path.join(corpus_dir, "index.json"), "w") as fh:
        json.dump([entry], fh, indent=1)
    return entry


def run_build(vendor_dir, out_dir, corpus_dir):
    env = dict(os.environ, BUILD_WORKERS="1")
    p = subprocess.run([sys.executable, os.path.join(vendor_dir, "cnet2_build.py"),
                        out_dir, corpus_dir],
                       capture_output=True, text=True, env=env, timeout=900)
    rep = None
    for line in p.stdout.strip().splitlines():
        try:
            rep = json.loads(line)
        except Exception:
            pass
    return p.returncode, rep, p.stdout.strip(), p.stderr.strip()


def prepare(probe_name):
    """Fresh workdir + vendor + corpus. Returns (probe, workdir, vendor, corpus)."""
    workdir = os.path.join(HERE, "runs", probe_name)
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    pr = Probe(probe_name)
    pr.log(f"== {probe_name}: X05 reproduction probe ==")
    vendor = os.path.join(workdir, "vendor")
    md5s = fetch_vendor(vendor)
    pr.log(f"[INFO] fetched code/cnet2_build.py md5={md5s['cnet2_build.py']} "
           f"(reviewed r4 bytes: {REVIEWED_BUILD_MD5})")
    if md5s["cnet2_build.py"] != REVIEWED_BUILD_MD5:
        pr.log("[WARN] main has drifted from the reviewed bytes; assertions below "
               "describe the reviewed (vulnerable) behaviour")
    pr.log(f"[INFO] fetched code/cnet2_feat.py md5={md5s['cnet2_feat.py']}")
    corpus = os.path.join(workdir, "corpus")
    entry = make_corpus(corpus)
    pr.log(f"[INFO] corpus game: file={entry['file']} steps={entry['steps']} "
           f"teacher seat={entry['order'][0]} renamed to {TEACHER_NAME} (index only)")
    return pr, workdir, vendor, corpus


def first_build(pr, vendor, out_dir, corpus):
    rc, rep, so, se = run_build(vendor, out_dir, corpus)
    pr.check("initial build exits 0", rc == 0, f"rc={rc} stderr_tail={se[-200:]}")
    if rc != 0 or not rep:
        raise SystemExit(pr.finish(os.path.dirname(out_dir)))
    ptr = json.load(open(os.path.join(out_dir, "current.json")))
    pr.check("stdout generation == current.json generation",
             rep.get("generation") == ptr["generation"], f"gen={ptr['generation']}")
    gen_dir = os.path.join(out_dir, "generations", ptr["generation"])
    man = json.load(open(os.path.join(gen_dir, "manifest.json")))
    return ptr["generation"], gen_dir, man
