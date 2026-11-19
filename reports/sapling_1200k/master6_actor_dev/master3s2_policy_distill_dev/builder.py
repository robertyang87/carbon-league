"""Map the original tracking-conditioned, pre-search Master3s2 policy into Sapling.

Only replay() can read private replay state. It returns the restored environment;
both front ends receive its public per-seat observation at every turn. No teacher
parameter is exported. Run 'smoke' for the fixed small test, 'build' to create
full training shards once approved.
"""
import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import types

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
import cnet2_feat as F
from cnet2_build import FIELDS
from fastsim import FastCarbon

ORIGINAL_BLOB = "9c46f6fa4ff401b05939150054c2f88f549c2f7e"
ORIGINAL_SHA = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
PRIOR = HERE.parent / "planter_site_transfer_dev" / "sources.json"
CACHE = Path("/workspace/sapling-all-top5-20261119")
STEPS = [0, 12, 38, 63, 89, 115, 143, 170, 199, 228, 261, 298]
PUBLIC = ("players", "step", "player", "trees", "carbon")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def save(path, data):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def git_blob(blob):
    return subprocess.check_output(["git", "cat-file", "blob", blob], cwd=ROOT)


def original():
    blob = git_blob(ORIGINAL_BLOB)
    assert sha(blob) == ORIGINAL_SHA
    m = types.ModuleType("original_master3s2_for_teacher")
    exec(compile(blob, "sources/master3s2/master3s2.py", "exec"), m.__dict__)
    assert m._SID == m._POLICY.sid("TJB-o18-rush-v41") and m._TEMP == 1
    assert m.WACT == F.MOVES and m.BACT == F.BASE_ACTS
    return m


def cards():
    source = json.loads(PRIOR.read_bytes())["records"]
    source = [r for r in source if r["split"] in ("fit", "held")]
    native = [r for r in source if r["domain"] == "native"]
    result = []
    for i, terrain in enumerate(sorted({r["terrain"] for r in native})):
        group = [r for r in native if r["terrain"] == terrain]
        pick = sorted(group, key=lambda r: (r["seat"] != i % 2, r["source"]))[0]
        for domain, side in [("original", pick["seat"]), ("sapling_visited", 1-pick["seat"])]:
            result.append(dict(pick, domain=domain, seat=side, actor=domain))
    result.extend(r for r in source if r["domain"] == "elite")
    for i, r in enumerate(result):
        r["id"] = f"r{i:03}"
        r["gid"] = i
        r["source_sha256"] = r.pop("sha256")
        r["steps"] = STEPS
    assert len(result) == 44 and len({(r["source_sha256"], r["seat"]) for r in result}) == 44
    assert not ({r["terrain"] for r in result if r["split"] == "fit"} &
                {r["terrain"] for r in result if r["split"] == "held"})
    return result


def replay(card):
    path = (CACHE if card["domain"] == "elite" else ROOT) / card["source"]
    if path.is_file():
        encoded = path.read_bytes()
    else:
        archive = git_blob(card["archive_git_blob"])
        assert sha(archive) == card["archive_sha256"]
        with tarfile.open(fileobj=io.BytesIO(archive)) as reader:
            encoded = reader.extractfile(card["source"]).read()
    assert sha(encoded) == card["source_sha256"]
    tape = json.loads(gzip.decompress(encoded))
    if card["domain"] == "elite":
        frames = tape["steps"]
        initial = dict(frames[0][0]["observation"])
        initial["carbon"] = initial.get("full_carbon", initial["carbon"])
        sim = FastCarbon(tape["configuration"]).load_observation(initial)
        pairs = [[seat.get("action") or {} for seat in row] for row in frames[1:]]
    else:
        assert not tape["errors"] and tape["examiner_md5"] == "57f9a594cce4f2ec9e3573d8d9d26ca1"
        assert tape["initial_terrain_u8_sha256"] == card["terrain"]
        assert card["seat"] == (1-tape["candidate_seat"] if card["domain"] == "original"
                                else tape["candidate_seat"])
        sim = FastCarbon(tape["game_configuration"]).reset(seed=card["seed"])
        pairs = tape["actions"]
    assert sha(np.asarray(sim.carbon, dtype=np.uint8).tobytes()) == card["terrain"]
    return sim, pairs


def physical_to_student(m, feat, x, obs, heads):
    raw = np.zeros((F.OWN, 5), np.float32)
    owner = obs["players"][feat.seat]
    for s, uid in enumerate(feat.own_uid):
        if uid is None:
            continue
        if s < 10:
            pos = owner[2][uid][0]
            p = m._probs(heads[0][pos])
            raw[s] = p[F.CANON_MOVE] if feat.seat else p
        else:
            pos = owner[1][uid]
            raw[s, :3] = m._probs(heads[1][pos])
        assert feat.cpos(pos) == x["tok_pos"][s]
    return raw


def projected(feat, x, raw, actual, mode, gid, step):
    pref = F.Prefix(x, feat.aux)
    uf, mf = np.zeros((12, F.UF), np.float16), np.zeros((12, 5, F.MF), np.float16)
    legal = np.zeros((12, 5), bool)
    q, y = np.zeros_like(raw), np.full(12, -1, np.int8)
    repaired = np.zeros(12, bool)
    for slot in np.flatnonzero(actual >= 0):
        uf[slot], mf[slot], legal[slot] = pref.feats(slot)
        p = raw[slot] * legal[slot]
        assert np.isfinite(p).all() and p.sum() > 0
        q[slot] = p / p.sum()
        repaired[slot] = mode == 0 and not legal[slot, actual[slot]]
        y[slot] = actual[slot] if mode == 0 and not repaired[slot] else np.argmax(p)
        pref.advance(slot, y[slot])
    row = dict(x, uf=uf, mf=mf, legal=legal, y=y, carbon=np.zeros(F.N, np.uint8),
               value=0., teacher=2, gid=gid, step=step)
    row = {k: np.asarray(row[k], dtype=dt) for k, (dt, _) in FIELDS.items()}
    row.update(q=q, teacher_raw=raw, src_act=actual.astype(np.int8),
               prefix_mode=np.int8(mode), prefix_repaired=repaired)
    assert np.allclose(q.sum(1), actual >= 0, atol=2e-6) and not q[~legal].any()
    return row


def verify_reference(m, tr_before, prev, obs, encoded, heads):
    # Capture the original entry's own network invocation without running its search.
    calls = []
    bound = m._POLICY.forward
    def capture(x, sid):
        out = bound(x, sid)
        calls.append((x.copy(), sid, out))
        return out
    old_state = m._ST
    m._ST = dict(tr=tr_before, prev=prev, rng=np.random.RandomState(123))
    m._POLICY.forward = capture
    try:
        m._act(obs, configuration=None)
    finally:
        del m._POLICY.forward
        m._ST = old_state
    assert len(calls) == 1 and calls[0][1] == m._SID
    assert np.array_equal(calls[0][0], encoded)
    assert all(np.array_equal(a, b) for a, b in zip(heads, calls[0][2]))


def process(card, m, times, check=False):
    s = card["seat"]
    sim, tape = replay(card)
    feat, tr, prev = F.Feat(s), m.BeliefTracker(m._Bel, s), None
    rows, examples, reference_calls = [], [], 0
    for t, pair in enumerate(tape):
        obs = {key: val for key, val in sim.obs(s).items() if key in PUBLIC}
        before = copy.deepcopy(tr) if check and t in times else None
        est, known = tr.step(obs["players"], t, obs["carbon"])
        x = feat.observe(obs)
        if t in times:
            encoded = np.clip(np.rint(m.encode(obs["players"], t, s, est, known, prev) * 100), 0, 255) * .01
            heads = m._POLICY.forward(encoded, m._SID)
            if check:
                verify_reference(m, before, prev, obs, encoded, heads)
                reference_calls += 1
            raw = physical_to_student(m, feat, x, obs, heads)
            actual = feat.labels(pair[s])
            assert feat.to_actions(actual) == {uid:a for uid,a in pair[s].items() if a and a != "NONE"}
            if check and s:
                for slot, uid in enumerate(feat.own_uid[:10]):
                    if uid is not None:
                        pos = obs["players"][s][2][uid][0]
                        assert raw[slot, F.MOVES.index("LEFT")] == m._probs(heads[0][pos])[m.WIDX["RIGHT"]]
            both = [projected(feat, x, raw, actual, mode, card["gid"], t) for mode in (0,1)]
            rows.extend(both)
            if check:
                for slot in np.flatnonzero(actual >= 0):
                    a = int(actual[slot])
                    top = int(np.argmax(both[0]["q"][slot]))
                    cut = float(raw[slot][~both[0]["legal"][slot]].sum())
                    if top != a or cut > .01:
                        examples.append(dict(step=t, slot=int(slot), uid=feat.own_uid[slot],
                            raw=raw[slot].tolist(), legal=both[0]["legal"][slot].tolist(),
                            q=both[0]["q"][slot].tolist(), submitted=a, projected_argmax=top,
                            removed_mass=cut))
        tr.set_actions(obs["players"], pair[s])
        prev = obs["players"]
        if t == max(times):
            break
        sim.step(*pair)
    return rows, examples, reference_calls


def main(command):
    records, model = cards(), original()
    cert = dict(builder_sha256=sha(Path(__file__).read_bytes()), policy_blob=ORIGINAL_BLOB,
                source_path="sources/master3s2/master3s2.py", policy_sha256=ORIGINAL_SHA,
                prior_manifest_sha256=sha(PRIOR.read_bytes()), policy_style=model._SID,
                public_fields=PUBLIC, features_sha256=sha((ROOT/"code/cnet2_feat.py").read_bytes()))
    if command == "smoke":
        tests = []
        for dom, side in [("original",0),("original",1),("elite",1)]:
            r = next(v for v in records if v["domain"] == dom and v["seat"] == side and v["split"] == "fit")
            rs, ex, num = process(r, model, (0,12,38), True)
            stale = [z for z in ex if z["projected_argmax"] != z["submitted"]]
            masks = [z for z in ex if z["removed_mass"] > .01]
            tests.append(dict(id=r["id"], source=r["source"], source_sha256=r["source_sha256"],
                seat=side, domain=dom, steps=[0,12,38], public_observations_parity=num,
                projected_rows=len(rs), slots=int(sum((z["y"]>=0).sum() for z in rs)),
                actual_vs_argmax_examples=stale[:2], held_prefix_removed_mass_examples=masks[:2]))
        assert all(v["public_observations_parity"] == 3 for v in tests)
        assert any(v["actual_vs_argmax_examples"] for v in tests if v["domain"]=="original")
        assert any(v["held_prefix_removed_mass_examples"] for v in tests)
        save(HERE/"smoke.json",dict(**cert, tests=tests, actual_search_reruns=0,
             note="The original _act's pre-search forward was instrumented before arbitration. Exact input and both raw logit arrays compared to extraction. Two physical sides. Examples use canonical action numbers in Sapling order."))
        return
    records_out = []
    out = HERE/"shards"
    out.mkdir(exist_ok=True)
    for r in records:
        rows, _, _ = process(r, model, STEPS)
        assert rows
        file = out/(r["id"]+".npz")
        tmp = file.with_suffix(".partial")
        with tmp.open("wb") as fp:
            np.savez_compressed(fp, **{k:np.stack([z[k] for z in rows]) for k in rows[0]})
        with np.load(tmp) as check:
            assert np.isfinite(check["q"]).all() and len(check["q"]) == len(rows)
        tmp.replace(file)
        records_out.append(dict(r,file=str(file.relative_to(HERE)),sha256=sha(file.read_bytes()),
                                steps=sorted({int(z["step"]) for z in rows}),rows=len(rows)))
    save(HERE/"manifest.json",dict(**cert,records=records_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke","build"))
    main(parser.parse_args().command)
