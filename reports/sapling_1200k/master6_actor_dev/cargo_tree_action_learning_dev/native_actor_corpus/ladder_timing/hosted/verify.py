"""Independently replay public observations, resource constraints and all commands from the hosted receipt."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import variant

engine = variant.engine
live = engine.live
REGISTRATION = "c997aabef84fd387e6c9dd81a877b9b11a34e5d6351c1e5fcde9a2b5751006c4"


def verify():
    if not __debug__:
        raise RuntimeError("run without optimization")
    meta = json.loads((HERE / "archive.json").read_bytes())
    run = meta["run_id"]
    assert run == 44607506846 and meta["head_sha"] == "70a5857e4245cef6ef034b553cda9838b2b1a151"
    assert meta["run_attempt"] == 1 and meta["conclusion"] == "success"
    assert meta["name"] == f"sapling-belief-6c548995-{run}"
    artifact = HERE / "artifact.zip"
    assert artifact.stat().st_size == meta["size_in_bytes"] and "sha256:" + live.sha(artifact) == meta["github_digest"]
    with zipfile.ZipFile(artifact) as z:
        assert sorted(z.namelist()) == ["attempt.json", "calls.jsonl", "receipt.json"] and z.testzip() is None
        attempt = json.loads(z.read("attempt.json"))
        result = json.loads(z.read("receipt.json"))
        tape = z.read("calls.jsonl")
    assert attempt == {"protocol_sha256": REGISTRATION, "run_id": str(run)}
    assert result["git_commit"] == meta["head_sha"] and result["run_id"] == str(run) and result["registration"] == REGISTRATION
    engine.PLAN = HERE.parent / "protocol.json"
    engine.PLAN_SHA = REGISTRATION
    raw, ref, history = engine.check_inputs()
    files = live.PLAN["files"]
    assert result["adapter_sha256"] == files["standard_runner"]["sha256"]
    assert result["pack"] == files["candidate"] and result["pack"]["sha256"] == variant.PACKAGE_SHA
    assert result["neutral_overage_public_history"] == history and result["replay"] == files["replay"]
    assert not result["blockers"] and not live.eligibility(result["probes"], result["containers"])
    assert len(result["containers"]) == len({c["id"] for c in result["containers"]}) == 2
    for c in result["containers"]:
        assert c["image"] == result["image_id"] and c["network"] == "none" and c["read_only_root"]
        assert c["nano_cpus"] == 1_500_000_000 and c["memory"] == c["memory_swap"] == 536_870_912
    assert result["cold_loads_issued"] == len(result["initialization"]) == 2
    for call in result["initialization"]:
        assert call["status"] == "ok" and 0 < call["elapsed"] <= call["limit"] == 20
        assert call["response"]["ready"] and call["response"]["sha256"] == variant.PACKAGE_SHA
    assert hashlib.sha256(tape).hexdigest() == result["tape_sha256"]
    rows = [json.loads(line) for line in tape.splitlines()]
    assert len(rows) == 299 and result["measured_calls_per_seat"] == [299, 299]
    env, _ = live.official()
    banks, slow = [5., 5.], [0, 0]
    for t, row in enumerate(rows):
        assert row["step"] == t and not env.done and len(row["calls"]) == 2
        for seat, call in enumerate(row["calls"]):
            obs = live.observe(env, seat, banks[seat])
            assert set(obs) == engine.KEYS and hashlib.sha256(live.encoded(obs)).hexdigest() == row["public_observation_sha256"][seat]
            live.check_action(call, obs)
            assert call["status"] == "ok" and call["response"]["step"] == t
            assert call["response"]["action"] == ref[seat]["timing"]["rows"][t]["action"]
            assert 0 < call["elapsed"] <= call["limit"] == live.allowance(slow[seat], banks[seat])
            if call["elapsed"] > 1:
                slow[seat] += 1
                banks[seat] = max(0., banks[seat] + 1 - call["elapsed"])
        assert slow == row["slow_steps"] and all(math.isclose(a,b) for a,b in zip(banks,row["overage_remaining_seconds"]))
        env.step(raw["actions"][t])
    assert env.done and [s.reward for s in env.states] == raw["official_score_by_seat"]
    assert result["status"] == "complete" and result["command_match_counts"] == [299,299]
    timing = live.timing_by_seat(result,rows)
    assert timing == result["timing_by_seat"] and result["whole_wall_seconds"] <= 1500
    for seat, stats in enumerate(timing):
        waits = sorted(row["calls"][seat]["elapsed"] for row in rows)
        assert (stats["observed_wait_seconds_p95"], stats["observed_wait_seconds_p99"], stats["observed_wait_seconds_max"]) == (waits[284], waits[296], waits[298])
        assert stats["observed_waits_over_one_second"] == sum(w > 1 for w in waits)
        assert math.isclose(stats["observed_excess_seconds"], sum(max(0, w - 1) for w in waits))
        samples = [result["initialization"][seat]["response"]] + [r["calls"][seat]["response"] for r in rows]
        assert stats["reported_peak_cgroup_bytes"] == max(s["cgroup_peak_bytes"] for s in samples)
        assert stats["reported_peak_rss_kib"] == max(s["rss_peak_kib"] for s in samples)
    assert result["fixed_replay_limits_pass"] and all(t["timed_out_requests"] == t["error_requests"] == 0 and
            t["observed_waits_over_one_second"] <= 3 and t["observed_excess_seconds"] <= 5 and
            t["observed_wait_seconds_max"] <= 3 and t["reported_peak_cgroup_bytes"] <= 536_870_912 for t in timing)
    evidence = dict(run_id=run, frozen_main=meta["head_sha"], source_sha256=variant.PACKAGE_SHA,
        source_bytes=result["pack"]["bytes"], artifact_sha256=live.sha(artifact), registration_sha256=REGISTRATION,
        scope=result["scope"], fixed_replay_limits_pass=True, seed=raw["seed"], terrain=raw["initial_terrain_u8_sha256"],
        history_neutral_overage=history, actual_probes=result["probes"], image_settings=result["containers"],
        replayed_public_requests=598, command_matches_against_local=[299,299], differing_commands=[],
        step0_external_ms=[1000*c["elapsed"] for c in rows[0]["calls"]], whole_wall_seconds=result["whole_wall_seconds"],
        external_timing=timing, verifier_sha256=live.sha(Path(__file__)))
    live.save(HERE/"verification.json", evidence)
    template=engine.HERE/"hosted/verify.py"
    spec=importlib.util.spec_from_file_location("render_timing_report",template)
    renderer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.HERE=HERE
    renderer.RUN=run
    renderer.report(evidence)
    path=HERE/"README.md"
    path.write_text(path.read_text().replace("# Sapling 碳观测候选的双席天梯限额计时", "# Sapling 全骨干候选的双席天梯限额计时") +
                    "\n首次托管预检在加载模型前终止。该次未执行计时，修复检出文件清单后重新发车。[首次运行记录](preflight_failure.json)\n")


if __name__ == "__main__":
    verify()
