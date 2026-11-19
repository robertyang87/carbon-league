"""Identity and actual-prefix verification for the frozen Sapling/it50 packages."""
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path("reports/sapling_1200k")
ACTORS = Path("data/weights/sapling_1200k")
PARENT = "b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca"
D1 = "429ec9a2bb84f77485b4f8830e71b252267705ffd8ce854a1cc35a03c446a367"
TERMINAL = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
COMPOUNDS = frozenset((PARENT, D1, TERMINAL))
STUDENT = "df0cf175db407712f31c188184c1395c0ac9b9c26ddc6b37b9dacda9fc43e5be"
STUDENT_PACK = "4cd4c1fcf40f05217ecba2a59095ba40286c5f96017d2b15285c942a3b0f5348"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def student_identity(identity):
    if identity == TERMINAL:
        from terminal_compound_formal import TRAINED
        return TRAINED
    if identity in (PARENT, D1):
        return STUDENT
    raise ValueError("unknown compound candidate")


def prove(weights, candidate, identity):
    if identity == TERMINAL:
        from terminal_compound_formal import prove as prove_terminal
        return prove_terminal(weights, candidate)
    from tree_win_gate import prove_single_pack
    if (identity not in COMPOUNDS or Path(weights).suffix.lower() != ".npz" or _hash(weights) != STUDENT):
        raise ValueError("this compound requires its exact trained cash V3 Sapling NPZ")
    with tempfile.TemporaryDirectory(prefix="sapling-compound-") as work:
        root = Path(work)
        def stage(relative):
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dest)
            return dest
        # prove_single_pack checks all tensor keys/shapes/finiteness, then actually exports the NPZ.
        # The student submitted to the second build is byte-identical to that independent export.
        student_pack = stage(ACTORS/"cash_v3_agent.py")
        if prove_single_pack(weights, student_pack) != (STUDENT, STUDENT_PACK):
            raise ValueError("independent student export differs")
        stage(Path("code/packs/ppou2_m_it50.py"))
        stage(REPORT/"it50_v3_runtime.py")
        hybrid_builder = stage(REPORT/"build_it50_v3.py")
        subprocess.run([sys.executable, str(hybrid_builder)], check=True, stdout=subprocess.DEVNULL)
        built = root / ACTORS / "hybrid_it50_v3.py"
        if _hash(built) != PARENT:
            raise ValueError("two-actor rebuild disagrees with frozen hybrid")
        if identity == D1:
            stage(Path("reports/search_dossier/iter2/echo4_search_int8_d1.py"))
            stage(REPORT/"build_it50_endhome.py")
            stage(REPORT/"d1_sapling_runtime.py")
            builder = stage(REPORT/"d1_sapling_build.py")
            built = root / ACTORS / "d1_sapling.py"
            subprocess.run([sys.executable, str(builder), "--out", str(built)], check=True, stdout=subprocess.DEVNULL)
        if _hash(built) != identity or _hash(candidate) != identity or built.stat().st_size > 10_000_000:
            raise ValueError("supplied compound differs from its reconstructed single-file package")
    return STUDENT, identity


@lru_cache(maxsize=1)
def reference():
    from tree_audited_game import prior
    parent = ROOT / ACTORS / "hybrid_it50_v3.py"
    if _hash(parent) != PARENT:
        raise ValueError("unrecognized reference dual policy")
    return prior.shared.module(parent, "formal_hybrid_reference")


def _live_d1(folder, row, raw):
    identity = row["candidate_search_audit"]
    name = f"d1_search_{row['seed']}_{row['candidate_seat']}.json.gz"
    if identity["file"] != name:
        raise ValueError("ambiguous D1 history filename")
    payload = (folder / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != identity["sha256"]:
        raise ValueError("D1 history checksum")
    trace = json.loads(gzip.decompress(payload))
    if ((trace["seed"], trace["seat"], trace["raw_sha256"], trace["pack_sha256"]) !=
            (row["seed"], row["candidate_seat"], row["raw_sha256"], D1) or
            len(trace["calls"]) != len(raw["actions"])):
        raise ValueError("unbound D1 search history")
    return trace["calls"]


def verify_actions(folder, row, raw, pack, record=False):
    """Independently rerun both original policies on every executed non-forced slot."""
    from tree_audited_game import exam
    from turning_point_ledger import sha, compact
    if pack == TERMINAL:
        from terminal_compound_formal import verify
        return verify(folder, row, raw, record)
    if pack not in COMPOUNDS:
        raise ValueError("unknown compound")
    model = reference()
    seat = row["candidate_seat"]
    sim = exam.FastCarbon(raw["game_configuration"]).reset(seed=row["seed"])
    live = _live_d1(Path(folder), row, raw) if pack == D1 else None
    unforced = contributions = changed = rounds = 0
    history = hashlib.sha256()
    for t, commands in enumerate(raw["actions"]):
        obs = sim.obs(seat)
        if t == 0:
            model.agent(obs, sim.config)
            features = model.sys.modules["cnet2_feat"]
            independent = features.Feat(seat)
        x = independent.observe(obs)
        executed = commands[seat]
        labels = independent.labels(executed)
        if independent.to_actions(labels) != executed:
            raise ValueError("compound executed an unknown command or unit")
        first = -1
        if live is not None:
            event = live[t]
            if event["root"] != t or event["executed_sha256"] != sha(compact(executed)):
                raise ValueError("D1 per-turn action history differs from complete game")
            h1 = event["completed_h1"]
            if type(h1) is not int or h1 not in (0,1) or type(event["search_changed"]) is not bool:
                raise ValueError("invalid D1 round accounting")
            rounds += h1
            if event["search_changed"]:
                base = independent.to_actions(model._mixed_actions(x, independent.aux))
                if base != event["base"] or base == executed or not h1:
                    raise ValueError("D1 changed an incorrect baseline or returned a partial round")
                first = next(int(s) for s in np.flatnonzero(x["tok_mask"][:len(labels)])
                             if labels[s] != independent.labels(base)[s])
                changed += 1
            elif event["base"] != executed:
                raise ValueError("unreported D1 intervention")
            if event["cumulative_rounds_changes"] != [rounds, changed]:
                raise ValueError("wrong cumulative D1 search count")
        single = [n.act(x, independent.aux, teacher=2, forced=labels)[1] for n in model._NETS]
        joint = np.argmax(0.75*single[0] + 0.25*single[1], axis=1)
        expert = np.argmax(single[0], axis=1)
        active = np.flatnonzero(x["tok_mask"][:len(labels)])
        prefix = features.Prefix(x, independent.aux)
        decisions = []
        for s in active:
            a = int(labels[s])
            if a < 0 or not prefix.feats(s)[2][a]:
                raise ValueError("illegal compound choice on actual prefix")
            prefix.advance(s, a)
            if s == first:
                continue
            if a != joint[s]:
                raise ValueError(f"unforced choice differs from both-policy recomputation at {t}/{s}")
            unforced += 1
            if s > first and a != expert[s]:
                decisions.append([independent.own_uid[s], a, int(expert[s])])
        if live is not None and live[t]["unforced_sapling_decisions"] != decisions:
            raise ValueError("D1 live student decisions differ from independent policy scores")
        contributions += len(decisions)
        history.update(compact([t, first, decisions]) + b"\n")
        sim.step(*commands)
    if sha(compact(sim.get_state())) != row["final_state_sha256"]:
        raise ValueError("compound attribution used a different final game state")
    if live is not None:
        actual = row["candidate_search_audit"]
        if (changed, rounds, contributions) != (actual["search_changes"], actual["completed_h1"], actual["unforced_sapling_decisions"]):
            raise ValueError("D1 game totals differ from its immutable per-turn records")
    return dict(pack_sha256=pack, transcript_sha256=row["raw_sha256"], verified_roots=len(raw["actions"]),
                verified_unforced_slots=unforced, student_changes=contributions, d1_changes=changed,
                d1_rounds=rounds, decisions_sha256=history.hexdigest())
