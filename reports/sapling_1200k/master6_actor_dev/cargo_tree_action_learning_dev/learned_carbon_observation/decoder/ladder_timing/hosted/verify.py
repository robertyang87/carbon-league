"""Verify and summarize a previously downloaded, GitHub-digest-checked decoder timing artifact."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import variant as V
T=V.engine
L=T.live
PLAN_SHA="8e12ed097ed5227f0aca109a4ac01ce176f7c7fc6e344b3caa39ad7caab34d02"


def check():
    meta=json.loads((HERE/"archive.json").read_bytes());package=HERE/"artifact.zip"
    run=meta["run_id"]
    assert meta["run_attempt"]==1 and meta["name"]==f"sapling-belief-61dae172-{run}"
    assert "sha256:"+L.sha(package)==meta["github_digest"] and package.stat().st_size==meta["size_in_bytes"]
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None and len(set(z.namelist()))==len(z.namelist())
        claim=json.loads(z.read("attempt.json"));result=json.loads(z.read("receipt.json"));tape=z.read("calls.jsonl")
    assert claim==dict(protocol_sha256=PLAN_SHA,run_id=str(run)) and result["registration"]==PLAN_SHA
    assert result["git_commit"]==meta["head_sha"] and str(run)==result["run_id"]
    raw,ref,neutral_history=V.configure(PLAN_SHA)
    assert result["pack"]==L.PLAN["files"]["candidate"] and result["adapter_sha256"]==L.PLAN["files"]["standard_runner"]["sha256"]
    assert neutral_history==result["neutral_overage_public_history"] and result["replay"]==L.PLAN["files"]["replay"]
    assert not L.eligibility(result["probes"],result["containers"]) and not result["blockers"]
    assert len({z["id"] for z in result["containers"]})==2
    assert all(z["image"]==result["image_id"] and z["nano_cpus"]==1_500_000_000 and z["network"]=="none" and z["memory"]==z["memory_swap"]==536_870_912 and z["read_only_root"] for z in result["containers"])
    assert result["cold_loads_issued"]==2 and len(result["initialization"])==2
    assert all(x["response"]["ready"] and x["status"]=="ok" and 0<x["elapsed"]<=x["limit"]==20 and x["response"]["sha256"]==V.PACKAGE_SHA for x in result["initialization"])
    rows=[json.loads(x) for x in tape.splitlines()]
    assert hashlib.sha256(tape).hexdigest()==result["tape_sha256"] and len(rows)==len(raw["actions"])==299
    env,_=L.official()
    bank=[5.,5.];slow=[0,0];matches=[0,0];misses=[]
    for t,row in enumerate(rows):
        assert row["step"]==t and not env.done
        for s,c in enumerate(row["calls"]):
            obs=L.observe(env,s,bank[s])
            assert set(obs)==T.KEYS and hashlib.sha256(L.encoded(obs)).hexdigest()==row["public_observation_sha256"][s]
            L.check_action(c,obs)
            assert c["status"]=="ok" and 0<c["elapsed"]<=c["limit"]==L.allowance(slow[s],bank[s])
            assert c["response"]["step"]==t
            expected=ref[s]["timing"]["rows"][t]["action"]
            if c["response"]["action"]==expected:matches[s]+=1
            else:misses.append(dict(seat=s,t=t,expected=expected,returned=c["response"]["action"]))
            if c["elapsed"]>1:slow[s]+=1;bank[s]=max(0.,bank[s]+1.-c["elapsed"])
        assert slow==row["slow_steps"] and all(math.isclose(a,b) for a,b in zip(bank,row["overage_remaining_seconds"]))
        env.step(raw["actions"][t])
    assert env.done and [x.reward for x in env.states]==raw["official_score_by_seat"]
    timing=L.timing_by_seat(result,rows)
    assert result["timing_by_seat"]==timing and result["command_match_counts"]==matches
    assert result["status"]=="complete" and result["fixed_replay_limits_pass"] and result["whole_wall_seconds"]<=1500
    assert all(z["timed_out_requests"]==z["error_requests"]==0 and z["observed_waits_over_one_second"]<=3 and z["observed_excess_seconds"]<=5 and z["reported_peak_cgroup_bytes"]<=536870912 for z in timing)
    out=dict(run_id=run,frozen_main=meta["head_sha"],source_sha256=V.PACKAGE_SHA,source_bytes=result["pack"]["bytes"],artifact_sha256=L.sha(package),
        registration_sha256=PLAN_SHA,adapter_sha256=result["adapter_sha256"],decoder_adapter_sha256=L.sha(Path(V.__file__)),
        scope=result["scope"],fixed_replay_limits_pass=True,seed=raw["seed"],terrain=raw["initial_terrain_u8_sha256"],
        actual_probes=result["probes"],image_settings=result["containers"],replayed_public_requests=2*len(rows),command_matches_against_local=matches,
        differing_commands=misses,step0_external_ms=[1000*c["elapsed"] for c in rows[0]["calls"]],whole_wall_seconds=result["whole_wall_seconds"],
        external_timing=timing,verifier_sha256=L.sha(Path(__file__)))
    L.save(HERE/"verification.json",out)
    # The immutable encoder renderer uses the same timing/result schema.
    renderer=T.HERE/"hosted/verify.py"
    spec=importlib.util.spec_from_file_location("encoder_timing_renderer",renderer)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.HERE=HERE;old.RUN=run;old.report(out)
    p=HERE/"README.md"
    p.write_text(p.read_text().replace("# Sapling 碳观测候选的双席天梯限额计时","# Sapling 碳观测解码器的双席天梯限额计时"))


if __name__=="__main__":
    check()
