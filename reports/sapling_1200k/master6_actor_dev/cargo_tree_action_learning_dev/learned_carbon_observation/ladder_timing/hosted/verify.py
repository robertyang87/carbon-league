"""Recompute frozen public inputs, hard timing limits and all returned commands from the GitHub artifact."""
import gzip
import hashlib
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import time_replay as T

RUN=44599538074
COMMIT="4541f3d945dce5159a311595273db8a1546ef20c"


def verify():
    L=T.live
    meta=json.loads((HERE/"archive.json").read_bytes())
    package=HERE/"artifact.zip"
    assert meta["run_id"]==RUN and meta["head_sha"]==COMMIT and meta["run_attempt"]==1
    assert meta["name"]==f"sapling-belief-bc69b55e-{RUN}"
    assert package.stat().st_size==meta["size_in_bytes"] and "sha256:"+L.sha(package)==meta["github_digest"]
    with zipfile.ZipFile(package) as z:
        assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
        claim=json.loads(z.read("attempt.json"));r=json.loads(z.read("receipt.json"))
        lines=z.read("calls.jsonl")
    assert claim==dict(protocol_sha256=T.PLAN_SHA,run_id=str(RUN))
    assert r["run_id"]==str(RUN) and r["git_commit"]==COMMIT and r["registration"]==T.PLAN_SHA
    assert r["adapter_sha256"]==L.sha(T.HERE/"time_replay.py")
    raw,ref,neutral_history=T.check_inputs()
    assert r["pack"]==L.PLAN["files"]["candidate"] and r["replay"]==L.PLAN["files"]["replay"] and neutral_history==r["neutral_overage_public_history"]
    assert not L.eligibility(r["probes"],r["containers"]) and not r["blockers"]
    assert len(set(c["id"] for c in r["containers"]))==2
    assert all(c["network"]=="none" and c["read_only_root"] and c["nano_cpus"]==1_500_000_000 and c["memory"]==536870912 and c["memory_swap"]==536870912 and c["image"]==r["image_id"] for c in r["containers"])
    assert r["cold_loads_issued"]==2
    for c in r["initialization"]:
        assert c["limit"]==20 and 0<c["elapsed"]<=20 and c["status"]=="ok" and c["response"]["sha256"]==r["pack"]["sha256"] and c["response"]["ready"]
    env,cfg=L.official()
    assert r["tape_sha256"]==hashlib.sha256(lines).hexdigest()
    rows=[json.loads(x) for x in lines.splitlines()]
    assert len(rows)==299 and r["measured_calls_per_seat"]==[len(rows)]*2
    bank=[5.,5.];slow=[0,0];matches=[0,0];differences=[]
    for t,row in enumerate(rows):
        assert row["step"]==t and not env.done
        for s in (0,1):
            obs=L.observe(env,s,bank[s]);call=row["calls"][s]
            assert set(obs)==T.KEYS and row["public_observation_sha256"][s]==hashlib.sha256(L.encoded(obs)).hexdigest()
            assert call["limit"]==L.allowance(slow[s],bank[s]) and 0<call["elapsed"]<=call["limit"]
            L.check_action(call,obs)
            assert call["status"]=="ok" and call["response"]["step"]==t
            cmd=call["response"]["action"]
            expected=ref[s]["timing"]["rows"][t]["action"]
            if cmd==expected:matches[s]+=1
            else:differences.append(dict(seat=s,t=t,proposed=cmd,archived_same_file=expected))
            if call["elapsed"]>1:
                slow[s]+=1;bank[s]=max(0.,bank[s]+1-call["elapsed"])
        assert row["slow_steps"]==slow and all(math.isclose(a,b) for a,b in zip(bank,row["overage_remaining_seconds"]))
        env.step(raw["actions"][t])
    assert env.done and [x.reward for x in env.states]==raw["official_score_by_seat"]
    assert r["command_match_counts"]==matches and r["reference_commands_by_seat"]==[299,299]
    timing=L.timing_by_seat(r,rows)
    assert r["timing_by_seat"]==timing and r["whole_wall_seconds"]<=1500
    fixed_pass=r['status']=="complete" and all(z["timed_out_requests"]==0 and z["error_requests"]==0 and z["observed_wait_seconds_max"]<=3 and z["observed_waits_over_one_second"]<=3 and z["observed_excess_seconds"]<=5 and z["reported_peak_cgroup_bytes"]<=536870912 for z in timing)
    assert r["fixed_replay_limits_pass"] is fixed_pass
    evidence=dict(run_id=RUN,frozen_main=COMMIT,source_sha256=r["pack"]["sha256"],source_bytes=r["pack"]["bytes"],artifact_sha256=L.sha(package),
        registration_sha256=T.PLAN_SHA,adapter_sha256=r["adapter_sha256"],verifier_sha256=L.sha(Path(__file__)),scope=r["scope"],
        current_receipt_status=r["status"],fixed_replay_limits_pass=fixed_pass,seed=raw["seed"],terrain=raw["initial_terrain_u8_sha256"],history_neutral_overage=neutral_history,
        image_id=r["image_id"],actual_probes=r["probes"],image_settings=r["containers"],replayed_public_requests=2*len(rows),command_matches_against_local=matches,
        differing_commands=differences,cold_commands=[x['response']['sha256'] for x in r['initialization']],
        step0_external_ms=[1000*rows[0]['calls'][s]['elapsed'] for s in (0,1)],whole_wall_seconds=r["whole_wall_seconds"],
        external_timing=timing,max_public_agent_seconds=[max(x["calls"][s]["response"]["public_wall_seconds"] for x in rows) for s in (0,1)],
        max_public_cpu_seconds=[max(x["calls"][s]["response"]["public_cpu_seconds"] for x in rows) for s in (0,1)])
    L.save(HERE/"verification.json",evidence)
    return evidence


def report(z):
    rows=["# Sapling 碳观测候选的双席天梯限额计时","",
        f"冻结单文件 SHA-256 为 `{z['source_sha256']}`，{z['source_bytes']:,} 字节。固定历史图 621471120 的两席各完成 299 次公开请求，包括第 0 步。宿主从写出完整请求到收到并解析动作计时。两席使用两个独立的 cgroup，各 1.5 核、512 MiB，禁用交换与网络。","",
        "|席位|冷启（秒）|第 0 步（毫秒）|P95 / P99 / 最慢（毫秒）|慢步 / 超时 / 错误|累计超额（秒）|容器 / 进程峰值（MiB）|同包固定轨迹动作一致|",
        "|---|---:|---:|---|---|---:|---|---|"]
    for s,t in enumerate(z["external_timing"]):
        q=" / ".join(f"{1000*t['observed_wait_seconds_'+k]:.2f}" for k in ('p95','p99','max'))
        errs=" / ".join(str(t[k]) for k in ('observed_waits_over_one_second','timed_out_requests','error_requests'))
        rows.append(f"|{s}|{t['init_seconds']:.3f}|{z['step0_external_ms'][s]:.2f}|{q}|{errs}|{t['observed_excess_seconds']:.3f}|{t['reported_peak_cgroup_bytes']/2**20:.1f} / {t['reported_peak_rss_kib']/1024:.1f}|{z['command_matches_against_local'][s]}/299|")
    rows += ["",f"整次用时 {z['whole_wall_seconds']:.2f} 秒。以上时限对该固定轨迹的实测结果为 **{'通过' if z['fixed_replay_limits_pass'] else '未通过'}**。所有局面由历史双方指令推进。这里没有新增对局胜负，独立新图的运行表现需要实际比赛检验。","",
        f"验算逐拍重放了 {z['replayed_public_requests']} 份公开观察，将原始 ZIP 大小及 SHA-256 与 GitHub 记录比对，核对了两次冷启身份、平台约束与返回动作。[运行记录](https://github.com/robertyang87/carbon-league/actions/runs/{RUN}) · [原始 ZIP](artifact.zip) · [复核结果](verification.json)"]
    (HERE/"README.md").write_text('\n'.join(rows)+'\n')


if __name__=="__main__":
    report(verify())
