"""Recover the referee and public prefix of a signed cargo-defense game before the selected action."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parents[1]
ROOT = DERIVATIVE.parents[1]
DATA = DERIVATIVE / "ladder/cargo_defense/development"
HOSTED = DATA / "hosted"
MATRIX = "244d7cc3f531ab0806db3c859a651298cddff962aa63c81e63fc51cebe1fac6e"
RUN = "44683302304"
ARCHIVES = {
    (621758594, 0): ("14606205398", "8676db2481e1fe103c2db8e01d35543b12b3029ce067a42388771b4a946c62f6"),
    (621758594, 1): ("14605597113", "68b0ca77a89b6b7ebecfcf5fb326ed2c3ca0a7ebd61562a7e14db5eb9ebd9fcf"),
    (621758595, 0): ("14604754279", "daa631f1fb3882260e02232095863739ad1cd35e677c5e818d6374176f97f751"),
    (621758595, 1): ("14606050152", "1c74c77c88641db7a5b9b740d96d44599601100aa4fe82774f2378b2ee39a6af"),
}
sys.path.insert(0, str(ROOT / "code/env_src/carbon_challenge-master"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freight = load("response_recovery_freight", DERIVATIVE / "freight_response/run.py")
live = load("response_recovery_referee", freight.STUDY / "cargo_tree_action_learning_dev/ladder_live/live.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _work(call, previous):
    """The native opponent alone emits depth; single and no-search must not reuse a stale last."""
    require(call["status"] == "ok", "invalid root response")
    rsp = call["response"]
    counts = rsp.get("search_counts")
    result = dict(available=False, K=None, H=None, rolls=None, searched=None, single=None,
                  public_wall_seconds=rsp["public_wall_seconds"], public_cpu_seconds=rsp["public_cpu_seconds"])
    if counts is None:
        result["reason"] = "historical candidate worker did not report search counters"
        return result
    prior = previous["response"]["search_counts"]
    changes = {key: counts[key] - prior[key] for key in ("searched", "single", "aborted", "changed")}
    require(all(v in (0, 1) for v in changes.values()) and changes["searched"] + changes["single"] <= 1,
            "invalid cumulative search counter transition")
    result.update(changes)
    if changes["searched"] or changes["single"]:
        current = rsp["search_last"]
        if changes["single"]:
            require((current["candidates"], current["depth"], current["rolls"]) == (1, 0, 0),
                    "single reply has stale depth")
        result.update(available=True, K=current["candidates"], H=current["depth"], rolls=current["rolls"])
    else:
        result["reason"] = "no fresh completed search or single-candidate counter"
    return result


def source(ticket):
    """Return both physical public histories and the official unadvanced root. No historic root orders are returned."""
    seed, seat, arm, step = (ticket[k] for k in ("seed", "seat", "arm", "step"))
    require(arm in ("parent", "defense") and step in (279, 285), "unexpected history arm or root")
    artifact, signed_zip = ARCHIVES[seed, seat]
    require(ticket.get("archive_sha256", signed_zip) == signed_zip and str(ticket.get("run_id", RUN)) == RUN,
            "archive/run differs from ticket")
    raw = (HOSTED / (artifact + ".zip")).read_bytes()
    require(sha(raw) == signed_zip, "archive digest differs")
    matrix_bytes = (DATA / "protocol.json").read_bytes()
    require(sha(matrix_bytes) == MATRIX, "source matrix changed")
    matrix = json.loads(matrix_bytes)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        top = json.loads(z.read("pair_status.json"))
        require((top["seed"], top["seat"], top["protocol_sha256"], top["run_id"]) == (seed, seat, MATRIX, RUN),
                "source matrix ticket differs")
        reg_bytes = z.read(arm + "/registration.json")
        reg = json.loads(reg_bytes)
        receipt_bytes = z.read(arm + "/receipt.json")
        receipt = json.loads(receipt_bytes)
        executions = [json.loads(x) for x in z.read("execution.jsonl").splitlines()]
        bound = [e for e in executions if e["arm"] == arm]
        require(len(bound) == 1 and bound[0]["receipt_sha256"] == sha(receipt_bytes), "execution did not bind receipt")
        require((reg["arm"], reg["seed"], reg["candidate_seat"], reg["opponent_seat"], reg["matrix_sha256"],
                 reg["workflow_run"]) == (arm, seed, seat, 1-seat, MATRIX, RUN), "wrong registration")
        require(receipt["protocol_sha256"] == sha(reg_bytes) and receipt["seed"] == seed
                and receipt["candidate_seat"] == seat, "wrong signed receipt")
        files = reg["files"]
        require(files["candidate"] == matrix["files"][arm] and files["opponent"] == matrix["files"]["opponent"],
                "historical actors differ from registered arms")
        for k, role in ((seat, "candidate"), (1-seat, "opponent")):
            init = receipt["initialization"][k]
            require(init["status"] == "ok" and init["response"]["sha256"] == files[role]["sha256"],
                    "worker loaded different actor")
        original = receipt["initialization"][1-seat]["response"]["original"]
        require(original == dict(_TEMP=1.0, _SEED=None, _S_ABORT=.58, _SEARCH_OK=True), "different original settings")
        turns = z.read(arm + "/turns.jsonl")
        require(sha(turns) == receipt["turns_sha256"], "historical order/timing digest differs")
        obs_gz = z.read(arm + "/observations.jsonl.gz")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        reg_file = directory / "registration.json"
        reg_file.write_bytes(reg_bytes)
        live.configure(reg_file, directory / "unused", directory / "unused.lock")
        env, cfg = live.official()
    sim = freight.econ.FastCarbon(cfg).reset(seed=seed)
    terrain = sha(bytes(env.states[0].observation.full_carbon))
    require(terrain == matrix["physical_boards"][str(seed)] == reg["physical_map_sha256"] ==
            sha(bytes(map(int, sim.carbon))), "initial terrain mismatch")
    sim.cash = freight.econ.CashEvents(sim.cash)
    bank = [reg["platform"]["overage_seconds_per_player"] * 1.0] * 2
    prefix = []
    request_digests = []
    order_digests = []
    previous = None
    with gzip.GzipFile(fileobj=io.BytesIO(obs_gz)) as observations, io.BytesIO(turns) as history:
        for i in range(step + 1):
            frame = json.loads(observations.readline())
            turn = json.loads(history.readline())
            require(frame["step"] == turn["step"] == i and not env.done, "misaligned or expired root")
            public = [freight.public(env, s, bank) for s in (0, 1)]
            hashes = [freight.digest(o) for o in public]
            require(hashes == frame["public_observation_sha256"] == turn["public_observation_sha256"]
                    and frame["observations"] == public, f"wire replay mismatch at {i}")
            require(list(sim.cash) == [p[0] for p in public[0]["players"]], f"referee cash mismatch at {i}")
            calls = turn["calls"]
            require(len(calls) == 2 and all(c["status"] == "ok" and c["response"]["step"] == i for c in calls),
                    f"incomplete replies at {i}")
            request_digests.append(hashes)
            if i == step:
                by_seat = [_work(c, p) for c, p in zip(calls, previous)]
                break
            for s, call in enumerate(calls):
                live.check_action(call, public[s])
            require(all(c["status"] == "ok" for c in calls), "illegal historical command")
            actions = [c["response"]["action"] for c in calls]
            # Store the decoded immutable-by-convention tape frames, independent of changing referee lists.
            prefix.append((frame["observations"], actions))
            order_digests.append(freight.digest(actions))
            sim.cash.events.clear()
            sim.step(*actions)
            env.step(actions, [])
            bank = turn["overage_remaining_seconds"]
            previous = calls
    sim.cash.events.clear()
    expected = dict(archive_sha256=signed_zip, artifact_id=artifact, workflow_run=RUN,
                    receipt_sha256=sha(receipt_bytes), registration_sha256=sha(reg_bytes),
                    historic_candidates=[files["candidate"]["sha256"], files["opponent"]["sha256"]],
                    captured_observations_sha256=sha(obs_gz), turns_sha256=sha(turns), terrain=terrain,
                    prior_public_history_sha256=freight.digest(request_digests[:-1]),
                    orders_prefix_sha256=freight.digest(order_digests), public_sha256=request_digests[-1],
                    historic_remaining_overage=list(bank), root_step=step)
    return dict(cfg=cfg, env=env, sim=sim, public=public, expected=expected, prefix=prefix,
                source_work=dict(by_seat=by_seat, examiner=by_seat[1-seat], **by_seat[seat]))
