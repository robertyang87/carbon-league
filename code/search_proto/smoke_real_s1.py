#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 真实环境冒烟：Echo4 包基线自对弈收根，逐根跑 Comparator.decide。

口径：
  - 权重来源：仓库 robertyang87/carbon-league main 树 code/packs/
    echo4_int8.py（整包 md5 须为 819250bfc82632786063dabd46a616bd，
    不合即停，不用任何替代权重）。经 gh_api 的 git blob 通道取包
    （contents API 对 4.6MB 文件返回空内容），缓存 /tmp/echo4_int8.py。
  - 对局：FastCarbon（包内嵌 fastsim，钉版 aefe6c83）自对弈，双座
    均由包按其自身口径构造的基线 agent 驱动（SEARCH=False 原样）；
    基线动作照常下发推进对局，比较器只在 seat0 的根上做判读。
  - 根：seat0 某步 key_units 触发且去重后候选 ≥2 的根。每根跑一次
    Comparator.decide（H=1；budget = cfg BUDGET 0.7s 总预算减该步
    基线步耗，见 real_env.decide_at_root 的口径注释）。
  - scenario_seed 用诊断种子 986200 起（仅透传登记，rollout 未消费，
    J=1 确定性——见 real_env notes）。
  - 汇总：根数、完成比较总数、正例改子数（changed=True；被守卫否决
    的候选按合同不可能被选，故 changed 恒为守卫一致的改子）、每根
    耗时分布、叶调用数、中止原因分布、外层规则层开关清单。

运行：python3 smoke_real_s1.py [目标根数=12] [最多局数=3]
退出码 0 = 跑通（至少 1 根完成比较），1 = 阻塞/无根。
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace"))

from telemetry import StatsCollector          # noqa: E402
import real_env                               # noqa: E402

PACK_REPO_PATH = "code/packs/echo4_int8.py"
PACK_CACHE = "/tmp/echo4_int8.py"
ROOTS_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MAX_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SEED0 = 986200


def fetch_pack() -> str:
    """经 gh_api git blob 通道取 Echo4 包，验 md5 后落缓存。"""
    import base64
    import hashlib

    if os.path.exists(PACK_CACHE):
        with open(PACK_CACHE, "rb") as fh:
            if hashlib.md5(fh.read()).hexdigest() == real_env.EXPECTED_PACK_MD5:
                return PACK_CACHE
    from gh_api import req
    tree = req("GET", "/repos/robertyang87/carbon-league/git/trees/main:code/packs")
    sha = None
    for t in tree["tree"]:
        if t["path"] == "echo4_int8.py":
            sha = t["sha"]
    if sha is None:
        raise RuntimeError(f"仓库 main 树 code/packs 下找不到 echo4_int8.py —— 停")
    blob = req("GET", f"/repos/robertyang87/carbon-league/git/blobs/{sha}")
    raw = base64.b64decode(blob["content"])
    got = hashlib.md5(raw).hexdigest()
    if got != real_env.EXPECTED_PACK_MD5:
        raise RuntimeError(f"Echo4 包 md5 不合：got {got} —— 停")
    with open(PACK_CACHE, "wb") as fh:
        fh.write(raw)
    return PACK_CACHE


def pct(vals, q):
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def main():
    print("== vendor md5 核验 ==")
    for k, v in real_env.verify_vendor().items():
        print(f"  {k}: {v}")
    pack_path = fetch_pack()
    print(f"== Echo4 包：{PACK_REPO_PATH} -> {pack_path} (md5 {real_env.EXPECTED_PACK_MD5}) ==")

    pack, agent0 = real_env.load_pack(pack_path)
    _pack2, agent1 = real_env.load_pack(pack_path)
    cfg = agent0.cfg
    print(f"== agent.cfg 关键项：SEARCH={cfg['SEARCH']} MODE={cfg['MODE']} "
          f"GUARD={cfg['GUARD']} BUDGET={cfg['BUDGET']} TEACHER={cfg['TEACHER']} "
          f"ROLL={cfg['ROLL']} ==")

    print("== 外层规则层开关清单（行 1975–2048，agent.cfg 实测） ==")
    for row in real_env.outer_layer_audit(cfg):
        print(f"  {row['key']:<18} {row['layer']:<28} 值={row['value']!r:<8} "
              f"{'活' if row['live'] else '关'}  （{row['effect']}）")

    collector = StatsCollector()
    harness = real_env.RealEnvSearch(agent0, collector=collector)
    FastCarbon = harness.inx.FastCarbon

    elapsed_roots = []
    steps_scanned = 0
    games_played = 0
    t_start = time.time()
    for g in range(MAX_GAMES):
        if len(elapsed_roots) >= ROOTS_TARGET:
            break
        games_played += 1
        sim = FastCarbon()
        sim.reset(seed=SEED0 + g)
        while sim.step_count < 300 and len(elapsed_roots) < ROOTS_TARGET:
            obs0 = sim.obs(0)
            harness.begin_step(obs0)
            t0 = time.perf_counter()
            a0 = agent0(obs0)
            base_cost = time.perf_counter() - t0
            steps_scanned += 1
            dec = harness.decide_at_root(
                obs0, a0, baseline_step_cost_s=base_cost, horizon=1,
                scenario_seed=SEED0 + len(elapsed_roots))
            if dec is not None:
                elapsed_roots.append(dec.elapsed_s)
                print(f"  root step={obs0['step']} rounds={dec.completed_rounds} "
                      f"comparisons={dec.completed_comparisons} changed={dec.changed} "
                      f"chosen={dec.chosen_index} abort={dec.abort_reason} "
                      f"elapsed={dec.elapsed_s * 1000:.1f}ms base_cost={base_cost * 1000:.1f}ms")
            a1 = agent1(sim.obs(1))
            sim.step(a0, a1)

    snap = collector.snapshot()
    print("== 冒烟汇总 ==")
    print(f"  对局数={games_played} 扫描步数(seat0)={steps_scanned} "
          f"墙钟={time.time() - t_start:.1f}s")
    print(f"  根数={len(elapsed_roots)}")
    print(f"  完成比较总数={snap['completed_comparisons']} "
          f"完成轮次={snap['completed_rounds']} 叶调用={snap['leaf_calls']}")
    print(f"  正例改子数={snap['changed_count']}（changed 且经守卫一致；"
          f"守卫复核异常计 veto 数={harness._guard_exceptions}）")
    if elapsed_roots:
        arr = np.asarray(elapsed_roots)
        print(f"  每根耗时 ms：mean={arr.mean() * 1000:.1f} p50={pct(elapsed_roots, 50) * 1000:.1f} "
              f"p95={pct(elapsed_roots, 95) * 1000:.1f} max={arr.max() * 1000:.1f}")
    print(f"  中止原因分布={snap['abort_reasons']} 深度直方图={snap['depth_hist']}")
    print(f"  叶 adapter 调用={harness.leaf_adapter.calls} 超域计数="
          f"{harness.leaf_adapter.out_of_range_count}（原始分差尺度直通的预期现象，见 real_env 头注）")
    print("  备注：")
    for n in harness.notes:
        print(f"    - {n}")
    ok = snap["completed_comparisons"] > 0
    print("SMOKE:", "PASS" if ok else "FAIL（无完成比较的根）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
