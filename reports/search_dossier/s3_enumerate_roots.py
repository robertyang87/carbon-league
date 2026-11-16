#!/usr/bin/env python3
"""S3 根单枚举脚本（协议 S3_ROOT_LIST_FREEZE_PROTOCOL.md §1/§2/§6/§10）。

功能：诊断种子块 986200–986499 上基线包（Echo4，SEARCH=False）自对弈，
每步以生产 cfg 原样复算 key_units 筛类一根（全步带 0–298，不设
step>=240 线），输出根单＋局总体册＋枚举报告＋见证 sidecar。
分片（--games A-B）多机并行，每机独立输出；--merge 合并验合。

**枚举期间不做任何比较器决策、不算任何 ΔU**（协议 §1.1）。
0.7 s decide 预算在枚举期不绑定（协议 §10.2）。

引用指针（执行时核对，见 §10.4）：
  - key_units：包内嵌 cnet2_search（钉版代码 md5 94f7fb64），
    vendor/cnet2_search.py:386–483；生产调用点 _search 保留段 :1506。
  - 生产 cfg ＝ DEFAULTS（:38–44）⊕ Echo4 包 CONF；
    cfg 组装 make_search_agent :621–623；包 CONF 逐字见触发清单 §0。
  - DIST 矩阵：cnet2_feat.py:54。
  - 权威结算 F01：嵌套 leaf :666–676（灭绝席 step_count−episodeSteps−1、
    存活席入步前现金）；终局分数组装式同源 s1_verify3.py:298–315
    （make_terminal_score_fn）；到时 F04＝真实终态现金差
    （real_env._raw_leaf 的 terminal_kind 分流同源）。
  - 候选生成：real_env._gen_candidates_raw（复刻 _search :1503–1551）；
    守卫准入 real_env._guard_screen；先验截断 lg0 序。
  - 自对弈推进：s1_verify3.collect_game 同口径（双 fresh agent、
    FastCarbon sim.reset(seed)、step<299、灭绝即停）。

脚本级解释口径（协议未逐字写死处，见审查记录；凡需策略组／搜索组
负责人裁定的标为 OPEN）：
  D1. sub_direction 并列：1a 与 1b 的 stake 相等时取 "1a"（受威胁优先，
      与 key_units 先算 1a 的次序一致）。[OPEN：请负责人裁定]
  D2. “合法候选”＝合法性＋去重（屏 ①）后、经守卫一致屏（屏 ②）、
      先验截断（屏 ③）前的候选集；cand_count 为该集大小（含 A0）。
      守卫 veto 的候选不计入合法候选——否则结算 argmax 可能落在永
      不可准入的候选上，§6 保底在测试时恒不可满足。
      [OPEN：请负责人裁定]
  D3. settlement_argmax 的下标空间＝合法候选表（生成序，A0＝0）；
      并列时取先验序靠前者，A0（差 0）优先于一切非正差候选。
      [OPEN：请负责人裁定]
  D4. 结算一步推进的对手动作＝该根步对手实际执行的动作
      （自对弈实测动作，J=1 确定性延续），不用 agent 的预测 acts_o。
      [OPEN：请负责人裁定]
  D5. settlement_decidable ⟺ 全部合法候选的一步后继皆为权威终局
      （灭绝 F01 或步 299 F04），此时叶 F01／F04 分支给出精确结算；
      任一候选后继非终局 → 不可判（叶启发式分支不是精确结算）。
      此口径与 lvnet33 验收 1 的 settlement-decidable 子集
      （step-298 根、全部候选后继终局）同源。
  D6. in_sample／pi_r_given_g／pi_g／sampled 在枚举时为 null，
      冻结时填（默认值普查，B／C 由策略组定数）。
  D7. 根单／局总体册文件容器＝{"meta": {...}, "roots"/"register": [...]}；
      meta 含协议版本、脚本 md5、包 md5、key_units 指针、结算指针、
      conf 摘要、分片信息。
  D8. 试跑 3 局计入全量（同一脚本、同一批），不另跑、不剔除。

停下条件（§1.4，任一触发即整趟停、写 HALT 记录、不许跳过继续）：
  - 独立复算与 key_units 不符（stake 差 >1e-9、uid 集不等、子方向不符）；
  - 返回 key 的工人非 COLLECTOR，或 1a／1b 两谓词均不成立；
  - 变体门在 live cfg 下开启（THREAT_FIRST／SPLIT／KEY_LATE_HOME／
    KEY_LATE_CARGO／PLANTER_KEY／BASE_KEY／MAX_KEY_LATE）；
  - 复制的截断结果与冻结 _gen_candidates_raw 输出不等；
  - 叶 F01 交叉核对不等；根去重键重复；包／vendor md5 不合。

用法：
  python3 enumerate_roots.py --games 0-2 --out s3_freeze/trial_out
  python3 enumerate_roots.py --games 0-299 --out s3_freeze/enum_out --shard-id s0
  python3 enumerate_roots.py --merge s3_freeze/enum_out_merged <shard-dir>...
  python3 enumerate_roots.py --games 0-0 --out /tmp/smoke --smoke-steps 20
    （--smoke-steps 仅开发调试：输出打 SMOKE 戳、不得入冻结。）
产出（每分片）：roots_<shard>.json、register_<shard>.json、
  report_<shard>.json、witness_<shard>.jsonl（见证 sidecar，非冻结件），
  以及逐局 roots_<shard>.jsonl（崩溃可查中间件，非冻结件）。
退出码：0＝分片跑完落盘；非 0＝停下／异常（见 HALT 记录与 stderr）。
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time

# BLAS 钉 1（在 numpy 导入之前；协议 §10.2 部署口径）。
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.join(_HERE, "..", "proto")
sys.path.insert(0, _PROTO)
sys.path.insert(0, os.path.expanduser("~/workspace"))

import numpy as np  # noqa: E402

import real_env  # noqa: E402

# 注：不 import smoke_real_s1——其模块级解析 sys.argv（位置参数当
# ROOTS_TARGET），从本脚本导入会误读本脚本参数。取包顺序与
# smoke_real_s1.fetch_pack（R2-06）逐字同：--pack → /tmp 缓存 →
# checkout 候选 → gh_api git blob 通道落缓存；逐级验整包 md5。
PACK_CACHE = "/tmp/echo4_int8.py"
CHECKOUT_PACK_CANDIDATES = [
    os.path.join(_PROTO, "..", "..", "packs", "echo4_int8.py"),
]


def _pack_md5_ok(path):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        return _md5(fh.read()) == real_env.EXPECTED_PACK_MD5


def fetch_pack(pack_arg=None):
    """取 Echo4 基线包并验整包 md5，返回本地路径。"""
    import base64
    if pack_arg is not None:
        if not _pack_md5_ok(pack_arg):
            raise RuntimeError(
                f"--pack 指定包 md5 不合（须 {real_env.EXPECTED_PACK_MD5}）："
                f"{pack_arg} —— 停")
        return pack_arg
    if os.path.exists(PACK_CACHE) and _pack_md5_ok(PACK_CACHE):
        return PACK_CACHE
    for cand in CHECKOUT_PACK_CANDIDATES:
        cand = os.path.normpath(cand)
        if os.path.exists(cand):
            if not _pack_md5_ok(cand):
                raise RuntimeError(f"checkout 钉版包 md5 不合：{cand} —— 停")
            return cand
    try:
        sys.path.insert(0, os.path.expanduser("~/workspace"))
        from gh_api import req
    except ImportError as e:
        raise RuntimeError(
            "gh_api 不可用且缓存/checkout 均无钉版包；先把机上已验 md5 的包"
            "拷为 /tmp/echo4_int8.py 缓存，或用 --pack 指定 —— 停") from e
    tree = req("GET", "/repos/robertyang87/carbon-league/git/trees/main:code/packs")
    sha = None
    for t in tree["tree"]:
        if t["path"] == "echo4_int8.py":
            sha = t["sha"]
    if sha is None:
        raise RuntimeError("仓库 main 树 code/packs 下找不到 echo4_int8.py —— 停")
    blob = req("GET", f"/repos/robertyang87/carbon-league/git/blobs/{sha}")
    raw = base64.b64decode(blob["content"])
    if _md5(raw) != real_env.EXPECTED_PACK_MD5:
        raise RuntimeError("git blob 取回包 md5 不合 —— 停")
    with open(PACK_CACHE, "wb") as fh:
        fh.write(raw)
    return PACK_CACHE

SEED_BASE = 986200
N_GAMES = 300
TERMINAL_STEP = 299
KEY_UNITS_PTR = {"file": "code/cnet2_search.py", "lines": "386-483",
                 "code_md5": "94f7fb6438bb7577b141e388f068e122",
                 "call_site": "_search retained segment :1506"}
LEAF_PTR = {"file": "code/cnet2_search.py", "lines": "666-676 (F01 branch)",
            "score_fn_source": "s1_verify3.py:298-315 (make_terminal_score_fn)",
            "time_terminal": "F04 real terminal cash (real_env._raw_leaf)"}
# 变体门：生产 cfg 下必须全关（触发清单 §0／§1–§3）；任一开启即停。
VARIANT_GATES_OFF = ("THREAT_FIRST", "SPLIT", "KEY_LATE_HOME",
                     "KEY_LATE_CARGO", "PLANTER_KEY", "BASE_KEY",
                     "MAX_KEY_LATE")
STAKE_TOL = 1e-9


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def script_md5() -> str:
    with open(os.path.abspath(__file__), "rb") as fh:
        return _md5(fh.read())


def make_fresh_agent(pack):
    """fresh 基线 agent（s1_verify3.py:120–134 同口径：每次重新反量化
    权重、make_search_agent 全新状态；与 real_env.load_pack 构造逐字同）。"""
    import base64
    import zlib
    w1 = pack._dequantize(zlib.decompress(base64.b64decode(pack._B1)), pack._Q1)
    w1["meta"] = np.array(pack._M1)
    return pack._S.make_search_agent(pack.CONF, weights=w1,
                                     rweights=None, oweights=None)


def terminal_scores(sim, prev_cash, extinct_fn):
    """权威逐席终局分数（F01；s1_verify3.make_terminal_score_fn 同源，
    vendor leaf :666–676 的 rew 组装式）：灭绝席＝step_count−episodeSteps−1，
    存活席＝入终局步前现金。"""
    pen = float(sim.step_count) - float(sim.config.get("episodeSteps", 300)) - 1.0
    pc = prev_cash if prev_cash is not None else sim.cash
    return [pen if extinct_fn(sim, p) else float(pc[p]) for p in (0, 1)]


def independent_predicates(obs, key_uid, cfg, F):
    """§1.3 独立复算：不用 key_units，从 obs 现值重算 1a／1b 两谓词。

    返回 (stake_1a, stake_1b, stake, sub_direction)。stake 为两向最大值；
    并列时 sub_direction 取 "1a"（D1）。任一谓词的成立条件与触发清单 §1
    逐字一致。"""
    me = obs["player"]
    w = obs["players"][me][2][key_uid]
    pos, cargo, kind = w
    if kind != "COLLECTOR":
        raise ValueError(f"key {key_uid} 工人 kind={kind}，非 COLLECTOR —— 停")
    enemies = [e for e in obs["players"][1 - me][2].values() if e[2] == "COLLECTOR"]
    key_r = cfg["KEY_R"]
    minc = cfg["MINC"]
    prio = cfg.get("PRIO_THREAT", 1.0)
    stake_1a = 0.0
    stake_1b = 0.0
    for e in enemies:
        if F.DIST[pos, e[0]] > key_r:
            continue
        if e[1] <= cargo and cargo >= minc:
            stake_1a = max(stake_1a, cargo * prio)
        elif cfg["HUNT"] and e[1] > cargo and e[1] >= minc:
            stake_1b = max(stake_1b, e[1])
    stake = max(stake_1a, stake_1b)
    sub = "1a" if stake_1a >= stake_1b else "1b"
    return stake_1a, stake_1b, stake, sub


def independent_key_set(obs, cfg, F):
    """独立重算 key_units 基础段全集：全部本席 COLLECTOR 按两谓词取
    stake，降序截断 MAX_KEY。返回 [(stake, uid)]（与 key_units 同序：
    元组降序，即 stake 降序、uid 降序）。"""
    me = obs["player"]
    out = []
    for uid, w in obs["players"][me][2].items():
        if w[2] != "COLLECTOR":
            continue  # 与 key_units 基础段同：只认 COLLECTOR
        _a, _b, stake, _s = independent_predicates(obs, uid, cfg, F)
        if stake > 0:
            out.append((stake, uid))
    out.sort(reverse=True)
    return out[:cfg["MAX_KEY"]]


def assert_variant_gates_off(cfg):
    """触发清单 §0／§1–§3：变体门在生产 cfg 下必须全关；任一开启即停。"""
    live = [k for k in VARIANT_GATES_OFF if cfg.get(k)]
    if live:
        raise RuntimeError(f"变体门开启：{live}——触发清单判定失效，停")
    if cfg.get("KEY_LATE_STEP", 999) != 999:
        raise RuntimeError("KEY_LATE_STEP 非默认 999——类三步门判定失效，停")
    if cfg.get("MAX_KEY") != 3 or cfg.get("KEY_R") != 2 \
            or cfg.get("MINC") != 10.0 or cfg.get("HUNT") is not True:
        raise RuntimeError("类一常数偏离触发清单（KEY_R／MINC／HUNT／MAX_KEY）——停")


def conf_summary(cfg):
    """enum_provenance 用的 conf 摘要（生产 cfg 关键项＋变体门状态）。"""
    return {
        "KEY_R": cfg.get("KEY_R"), "MINC": cfg.get("MINC"),
        "HUNT": cfg.get("HUNT"), "MAX_KEY": cfg.get("MAX_KEY"),
        "PRIO_THREAT": cfg.get("PRIO_THREAT", 1.0),
        "variant_gates_off": {k: cfg.get(k) for k in VARIANT_GATES_OFF},
        "KEY_LATE_STEP": cfg.get("KEY_LATE_STEP", 999),
        "SEARCH": cfg.get("SEARCH"), "ROLL": cfg.get("ROLL"),
        "MODE": cfg.get("MODE"), "BUDGET": cfg.get("BUDGET"),
    }


class Enumerator:
    """单分片枚举器：基线自对弈推进＋逐根处理。"""

    def __init__(self, pack_path, out_dir, shard_id, smoke_steps=None):
        self.pack_path = pack_path
        self.out_dir = out_dir
        self.shard_id = shard_id
        self.smoke_steps = smoke_steps
        self.script_md5 = script_md5()
        self.pack = None
        self.agents = None
        self.harnesses = None
        self.inx = None
        self.cfg = None
        self.F = None
        self.FastCarbon = None
        self.terminal_fn = None
        self.leaf_fn = None
        self.extinct_fn = None
        self.key_units = None
        self.settle_sim = None
        self.roots = []          # §2.1 记录
        self.witness = []       # 见证 sidecar（逐根）
        self.register = []      # §3.2 记录（产根局）
        self.game_summaries = []
        self.halts = []
        self.timing = {"agent_call_s": [], "root_proc_s": []}

    # -- 启动 ---------------------------------------------------------------
    def setup(self):
        print("== vendor md5 核验 ==", flush=True)
        for k, v in real_env.verify_vendor().items():
            print(f"  {k}: {v}", flush=True)
        self.pack, _probe = real_env.load_pack(self.pack_path)
        print(f"== 包核验：{self.pack_path} md5 "
              f"{real_env.EXPECTED_PACK_MD5} —— 合 ==", flush=True)
        self.agents = [make_fresh_agent(self.pack), make_fresh_agent(self.pack)]
        self.harnesses = [real_env.RealEnvSearch(a) for a in self.agents]
        self.inx = self.harnesses[0].inx
        self.cfg = self.inx.cfg
        assert_variant_gates_off(self.cfg)
        print("== 变体门核验：THREAT_FIRST／SPLIT／KEY_LATE_HOME／"
              "KEY_LATE_CARGO／PLANTER_KEY／BASE_KEY／MAX_KEY_LATE 全关；"
              "KEY_LATE_STEP 默认 999；KEY_R=2／MINC=10.0／HUNT=True／"
              "MAX_KEY=3 —— 合 ==", flush=True)
        self.F = self.inx.F
        self.FastCarbon = self.inx.FastCarbon
        self.terminal_fn = self.inx.terminal_fn
        self.leaf_fn = self.inx.leaf_fn
        self.key_units = self.inx.mod["key_units"]
        cells = real_env._cells(self.leaf_fn)
        self.extinct_fn = cells.get("_extinct")
        if self.extinct_fn is None:
            raise RuntimeError("leaf 闭包缺 _extinct（vendor 布局已变）—— 停")
        self.settle_sim = self.FastCarbon()
        print("== conf 摘要：" + json.dumps(conf_summary(self.cfg),
                                            ensure_ascii=False, default=str),
              flush=True)

    # -- 候选生成（复制 real_env._gen_candidates_raw 的截断前管线） --------
    def _full_candidates(self, harness, root, a0_actions):
        """返回截断前全量 (cand_actions, cand_labels, cand_meta) 或 None。

        与冻结 _gen_candidates_raw 逐行同口径（key_units→强制标签→
        MAX_BR 封顶→decode_b→去重），唯一差别：不做 K≤4 先验截断。
        调用方须以 _truncate + 冻结函数输出交叉核对。"""
        inx, cfg = harness.inx, harness.cfg
        F = inx.F
        obs, x, y, lg0 = root.obs, root.x, root.y, root.lg0
        keys = inx.mod["key_units"](obs, cfg)
        if not keys:
            return None
        tok, ctx = inx.pol.enc_for(x, root.t_now)
        uid2slot = {u: s for s, u in enumerate(root.f.own_uid) if u is not None}
        base_fix = inx.st.get("base_fix", {})
        guard_base = inx.st.get("guard_base", ())
        forced, meta = [], []
        for _, uid in keys:
            s = uid2slot.get(uid)
            if s is None or (s >= 10 and (not cfg.get("BASE_KEY")
                                          or s in base_fix or s in guard_base)):
                continue
            for m in range(3 if s >= 10 else 5):
                if m == y[s]:
                    continue
                fz = np.full(F.OWN, -1, np.int64)
                fz[:s] = y[:s]
                fz[s] = m
                for s_b, v_b in base_fix.items():
                    if s_b != s:
                        fz[s_b] = v_b
                forced.append(fz)
                meta.append((uid, m))
        mbr = cfg["MAX_BR"] if not (cfg.get("MAX_BR_LATE")
                                    and obs["step"] >= cfg.get("KEY_LATE_STEP", 999)) \
            else cfg["MAX_BR_LATE"]
        forced = forced[:mbr - 1]
        meta = meta[:mbr - 1]
        if not forced:
            return None
        B0 = len(forced)
        X1 = {k: np.repeat(x[k][None], B0, 0)
              for k in ("tok_pos", "tok_kind", "tok_mask", "tok_cargo")}
        X1["cash"] = [x["cash"]] * B0
        X1["slots"] = [x["slots"]] * B0
        X1["left"] = [x["left"]] * B0
        ys = inx.mod["decode_b"](inx.pol, X1, [root.f.aux] * B0,
                                 np.repeat(tok[None], B0, 0),
                                 np.repeat(ctx[None], B0, 0), np.stack(forced))
        cand_actions = [a0_actions]
        cand_labels = [y]
        cand_meta = [None]
        seen = {tuple(y)}
        for yy, mm in zip(ys, meta):
            k = tuple(yy)
            if k in seen:
                continue
            seen.add(k)
            cand_actions.append(root.f.to_actions(yy))
            cand_labels.append(yy)
            cand_meta.append(mm)
        if len(cand_actions) < 2:
            return None
        return cand_actions, cand_labels, cand_meta

    @staticmethod
    def _truncate(cand_actions, cand_labels, cand_meta, lg0, uid2slot):
        """冻结先验截断（_gen_candidates_raw :③）：K≤4，A0 恒留，其余
        按 lg0 降序、同分按生成序。返回截断后 (actions, labels)。"""
        order = list(range(1, len(cand_actions)))
        if len(cand_actions) > 4:
            def prior(i):
                uid, m = cand_meta[i]
                return float(lg0[uid2slot[uid]][m])
            order.sort(key=lambda i: (-prior(i), i))
            keep = [0] + sorted(order[:3])
        else:
            keep = [0] + order
        return ([cand_actions[i] for i in keep],
                [cand_labels[i] for i in keep])

    # -- 结算：一步推进＋F01／F04 精确值 --------------------------------------
    def _settle_one(self, root, cand_action, opp_action, me):
        """一候选一步推进，返回 (terminal_kind, settlement_value)；
        后继非终局时返回 (None, None)。settlement_value 为本席结算分差
        （原始分差尺度，与叶①同尺度）。"""
        sim = self.settle_sim
        sim.load_observation(root.full)
        sim._tid_counter = [c + 500 for c in sim._tid_counter]
        prev_cash = list(sim.cash)
        if me == 0:
            sim.step(cand_action, opp_action)
        else:
            sim.step(opp_action, cand_action)
        if self.terminal_fn(sim):
            scores = terminal_scores(sim, prev_cash, self.extinct_fn)
            value = float(scores[me]) - float(scores[1 - me])
            # 交叉核对：叶 F01 分支须给出同一分差（s1_verify3 回归口径）。
            leaf_value = float(self.leaf_fn(sim, me, 0, prev_cash))
            if leaf_value != value:
                raise RuntimeError(
                    f"叶 F01 交叉核对不等：leaf={leaf_value} vs scores={value} —— 停")
            return "extinct", value
        if sim.step_count >= TERMINAL_STEP:
            return "time", float(sim.cash[me]) - float(sim.cash[1 - me])
        return None, None

    # -- 单根处理 -------------------------------------------------------------
    def _process_triggered_seat(self, game, seed, step, seat, obs, a0,
                                opp_a0, keys, harness):
        """处理一席一触发步：根构造→候选全量→守卫准入→结算→记根。

        返回本席本步根数。任一核对不等即抛错（整趟停，见 §1.4）。"""
        t0 = time.monotonic()
        root = harness._build_root(obs, a0)
        # 独立复算：key_units 全集须与独立重算逐字合（stake＋uid 序）。
        mine = independent_key_set(obs, self.cfg, self.F)
        got = [(float(s), u) for s, u in keys]
        if len(mine) != len(got) or any(
                abs(m[0] - g[0]) > STAKE_TOL or m[1] != g[1]
                for m, g in zip(mine, got)):
            raise RuntimeError(
                f"独立复算与 key_units 不符 game={game} step={step} seat={seat}："
                f"mine={mine} got={got} —— 停（§1.4 触发判据误报）")
        # 候选全量（复制管线）＋冻结函数交叉核对。
        full = self._full_candidates(harness, root, a0)
        if full is None:
            legal_actions, legal_labels = [a0], [root.y]
            legal_idx = [0]
            cand_count = 1
            meta_full = [None]
        else:
            cand_actions, cand_labels, cand_meta = full
            frozen = harness._gen_candidates_raw(root, a0)
            if frozen is None:
                raise RuntimeError(
                    f"复制管线有候选、冻结函数无候选 game={game} step={step} "
                    f"seat={seat} —— 停（实现漂移）")
            uid2slot = {u: s for s, u in enumerate(root.f.own_uid)
                        if u is not None}
            tr_actions, tr_labels = self._truncate(
                cand_actions, cand_labels, cand_meta, root.lg0, uid2slot)
            f_actions, f_labels = frozen
            if len(tr_actions) != len(f_actions) or any(
                    tr_actions[i] != f_actions[i] or not np.array_equal(
                        np.asarray(tr_labels[i]), np.asarray(f_labels[i]))
                    for i in range(len(tr_actions))):
                raise RuntimeError(
                    f"截断交叉核对不等 game={game} step={step} seat={seat} —— "
                    "停（复制管线与冻结函数漂移）")
            vetoed = harness._guard_screen(root, cand_labels)
            legal_idx = [i for i in range(len(cand_actions)) if i not in vetoed]
            legal_actions = [cand_actions[i] for i in legal_idx]
            legal_labels = [cand_labels[i] for i in legal_idx]
            cand_count = len(legal_idx)
            meta_full = cand_meta
        # 结算：逐合法候选一步推进（对手动作为该步实际执行动作）。
        values = []
        kinds = []
        decidable = True
        for la in legal_actions:
            kind, value = self._settle_one(root, la, opp_a0, seat)
            kinds.append(kind)
            values.append(value)
            if kind is None:
                decidable = False
        # argmax：结算差最大者；并列取先验序靠前者，A0（差 0）优先。
        if decidable:
            diffs = [0.0] + [values[i] - values[0]
                             for i in range(1, len(values))]
            uid2slot = {u: s for s, u in enumerate(root.f.own_uid)
                        if u is not None}
            def prior_of(li):
                if li == 0:
                    return float("inf")
                uid, m = meta_full[legal_idx[li]]
                return float(root.lg0[uid2slot[uid]][m])
            best_li, best_key = 0, (0.0, float("inf"), 0)
            for li in range(1, len(values)):
                key = (diffs[li], prior_of(li), -li)
                if key > best_key:
                    best_key, best_li = key, li
            argmax = int(best_li)
        else:
            argmax = None
        # 逐 key 记根（key_units 的返回序＝stake 降序）。
        n = 0
        for stake, uid in keys:
            _a, _b, _s, sub = independent_predicates(obs, uid, self.cfg, self.F)
            if abs(_s - float(stake)) > STAKE_TOL:
                raise RuntimeError(
                    f"逐 key stake 复算不等 game={game} step={step} seat={seat} "
                    f"uid={uid}：key_units={stake} 复算={_s} —— 停")
            if _s <= 0:
                raise RuntimeError(
                    f"返回 key 复算 stake<=0 game={game} step={step} seat={seat} "
                    f"uid={uid} —— 停（§1.4）")
            w = obs["players"][seat][2][uid]
            enemies = [(e[0], e[1]) for e in
                       obs["players"][1 - seat][2].values() if e[2] == "COLLECTOR"]
            root_id = f"{seed}-{game}-{step}-{seat}-{uid}"
            if any(r["root_id"] == root_id for r in self.roots):
                raise RuntimeError(f"去重键重复 {root_id} —— 停（§2.2）")
            rec = {
                "root_id": root_id, "game": game, "seed": seed,
                "step": step, "seat": seat, "trigger_class": "class1",
                "sub_direction": sub, "key_uid": uid,
                "stake": float(stake), "cand_count": cand_count,
                "settlement_decidable": bool(decidable),
                "settlement_argmax": argmax,
                "in_sample": None, "pi_r_given_g": None,
                "enum_provenance": {
                    "script_md5": self.script_md5,
                    "key_units": dict(KEY_UNITS_PTR),
                    "package_md5": real_env.EXPECTED_PACK_MD5,
                    "conf": conf_summary(self.cfg),
                },
            }
            self.roots.append(rec)
            self.witness.append({
                "root_id": root_id,
                "key_worker": {"pos": w[0], "cargo": w[1], "kind": w[2]},
                "enemy_collectors": [{"pos": p, "cargo": c} for p, c in enemies],
                "stake_1a": _a, "stake_1b": _b,
                "legal_candidates": [
                    {"legal_index": li,
                     "action": legal_actions[li],
                     "label": [int(v) for v in np.asarray(legal_labels[li])],
                     "settlement_kind": kinds[li],
                     "settlement_value": values[li]}
                    for li in range(len(legal_actions))],
            })
            n += 1
        self.timing["root_proc_s"].append(time.monotonic() - t0)
        return n

    # -- 单局 ------------------------------------------------------------------
    def play_game(self, game):
        """一局基线自对弈＋全步带枚举。返回局摘要。"""
        seed = SEED_BASE + game
        t_game = time.monotonic()
        sim = self.FastCarbon()
        sim.reset(seed=seed)
        n_roots = {0: 0, 1: 0}
        reached = 0
        stop_reason = "time_limit_299"
        while sim.step_count < TERMINAL_STEP:
            if self.smoke_steps is not None and sim.step_count >= self.smoke_steps:
                stop_reason = f"SMOKE cap {self.smoke_steps} steps"
                break
            step = sim.step_count
            obs = [copy.deepcopy(sim.obs(0)), copy.deepcopy(sim.obs(1))]
            keys = [self.key_units(obs[s], self.cfg) for s in (0, 1)]
            trig = [bool(keys[s]) for s in (0, 1)]
            actions = [None, None]
            for s in (0, 1):
                if trig[s]:
                    self.harnesses[s].begin_step(obs[s])
                t0 = time.monotonic()
                actions[s] = self.agents[s](obs[s])
                self.timing["agent_call_s"].append(time.monotonic() - t0)
            for s in (0, 1):
                if trig[s]:
                    n_roots[s] += self._process_triggered_seat(
                        game, seed, step, s, obs[s], actions[s],
                        actions[1 - s], keys[s], self.harnesses[s])
            sim.step(actions[0], actions[1])
            reached = sim.step_count
            if self.terminal_fn(sim):
                stop_reason = "extinct"
                break
        total = n_roots[0] + n_roots[1]
        summary = {"game": game, "seed": seed, "reached_step": reached,
                   "stop_reason": stop_reason, "roots": total,
                   "roots_by_seat": {"0": n_roots[0], "1": n_roots[1]},
                   "wall_s": round(time.monotonic() - t_game, 1)}
        self.game_summaries.append(summary)
        print(f"  game {game} seed {seed}: reached {reached} ({stop_reason}), "
              f"roots {total} (seat0 {n_roots[0]} / seat1 {n_roots[1]}), "
              f"wall {summary['wall_s']}s", flush=True)
        return summary

    # -- 分片执行与落盘 ---------------------------------------------------------
    def run_shard(self, g0, g1):
        os.makedirs(self.out_dir, exist_ok=True)
        # 旧趟 HALT 残留不得带入本趟：开跑即清（成功落盘后亦无 HALT）。
        stale_halt = os.path.join(self.out_dir, f"HALT_{self.shard_id}.json")
        if os.path.exists(stale_halt):
            os.remove(stale_halt)
        jsonl_path = os.path.join(self.out_dir, f"roots_{self.shard_id}.jsonl")
        for game in range(g0, g1 + 1):
            summary = self.play_game(game)
            # 逐局落盘 JSONL（崩溃可查中间件；冻结件为最终 JSON）。
            with open(jsonl_path, "a", encoding="utf-8") as fh:
                for r in self.roots:
                    if r["game"] == game:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._write_shard_files(g0, g1)
        return self._report(g0, g1)

    def _write_shard_files(self, g0, g1):
        meta = {
            "protocol": "S3_ROOT_LIST_FREEZE_PROTOCOL.md",
            "protocol_md5": "4212f13afe71b4008579df5f0c605bf0",
            "script_md5": self.script_md5,
            "package_md5": real_env.EXPECTED_PACK_MD5,
            "key_units": dict(KEY_UNITS_PTR),
            "settlement_fn": dict(LEAF_PTR),
            "conf": conf_summary(self.cfg),
            "shard_id": self.shard_id,
            "games": [g0, g1],
            "seeds": [SEED_BASE + g0, SEED_BASE + g1],
            "mode": "SMOKE-DO-NOT-FREEZE" if self.smoke_steps is not None
                    else "ENUMERATION",
            "smoke_steps": self.smoke_steps,
        }
        roots_doc = {"meta": meta, "roots": self.roots}
        with open(os.path.join(self.out_dir, f"roots_{self.shard_id}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(roots_doc, fh, ensure_ascii=False, indent=1, default=str)
        reg = []
        for s in self.game_summaries:
            if s["roots"] > 0:
                reg.append({"game": s["game"], "seed": s["seed"],
                            "root_count": s["roots"],
                            "root_count_by_seat": s["roots_by_seat"],
                            "pi_g": None, "sampled": None,
                            "zero_valid_roots": None})
        reg_doc = {"meta": meta, "register": reg}
        with open(os.path.join(self.out_dir, f"register_{self.shard_id}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(reg_doc, fh, ensure_ascii=False, indent=1, default=str)
        with open(os.path.join(self.out_dir, f"witness_{self.shard_id}.jsonl"),
                  "w", encoding="utf-8") as fh:
            for w in self.witness:
                fh.write(json.dumps(w, ensure_ascii=False, default=str) + "\n")
        self._register = reg

    def _report(self, g0, g1):
        n_games = g1 - g0 + 1
        n_prod = len(self._register)
        n_zero = n_games - n_prod
        total_roots = len(self.roots)
        decidable = sum(1 for r in self.roots if r["settlement_decidable"])
        argmax_nonzero = sum(1 for r in self.roots
                             if r["settlement_decidable"]
                             and (r["settlement_argmax"] or 0) != 0)
        calls = self.timing["agent_call_s"]
        report = {
            "shard_id": self.shard_id, "games": [g0, g1],
            "seeds": [SEED_BASE + g0, SEED_BASE + g1],
            "mode": "SMOKE-DO-NOT-FREEZE" if self.smoke_steps is not None
                    else "ENUMERATION",
            "games_played": n_games,
            "producing_games_N": n_prod, "zero_root_games_M": n_zero,
            "N_plus_M_eq_games": (n_prod + n_zero) == n_games,
            "total_roots": total_roots,
            "roots_by_seat": {
                "0": sum(1 for r in self.roots if r["seat"] == 0),
                "1": sum(1 for r in self.roots if r["seat"] == 1)},
            "settlement_decidable_roots": decidable,
            "settlement_argmax_nonzero": argmax_nonzero,
            "trigger_events_1_4": "无",
            "halts": self.halts,
            "agent_call_s": {
                "n": len(calls),
                "mean": round(sum(calls) / max(1, len(calls)), 3),
                "max": round(max(calls), 3) if calls else 0},
            "game_summaries": self.game_summaries,
            "script_md5": self.script_md5,
            "package_md5": real_env.EXPECTED_PACK_MD5,
        }
        with open(os.path.join(self.out_dir, f"report_{self.shard_id}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
        print(json.dumps({k: report[k] for k in (
            "games_played", "producing_games_N", "zero_root_games_M",
            "total_roots", "settlement_decidable_roots",
            "settlement_argmax_nonzero", "trigger_events_1_4")},
            ensure_ascii=False), flush=True)
        return report


def merge_shards(out_dir, shard_dirs):
    """合并多分片：验并集恰＝种子 986200–986499、不交叠不缺不重、
    去重键唯一；写合并根单／局总体册／报告。"""
    all_roots, all_reg, reports = [], [], []
    seen_games = {}
    for d in shard_dirs:
        for name in sorted(os.listdir(d)):
            if name.startswith("roots_") and name.endswith(".json"):
                doc = json.load(open(os.path.join(d, name), encoding="utf-8"))
                all_roots.extend(doc["roots"])
            elif name.startswith("register_") and name.endswith(".json"):
                doc = json.load(open(os.path.join(d, name), encoding="utf-8"))
                all_reg.extend(doc["register"])
            elif name.startswith("report_") and name.endswith(".json"):
                reports.append(json.load(open(os.path.join(d, name),
                                              encoding="utf-8")))
    ids = [r["root_id"] for r in all_roots]
    if len(ids) != len(set(ids)):
        raise RuntimeError("合并去重键重复 —— 停")
    games = sorted({r["game"] for r in all_roots} |
                   {g["game"] for g in all_reg})
    seeds = sorted({SEED_BASE + g for g in games})
    want_seeds = list(range(SEED_BASE, SEED_BASE + N_GAMES))
    if seeds != want_seeds:
        missing = sorted(set(want_seeds) - set(seeds))
        extra = sorted(set(seeds) - set(want_seeds))
        raise RuntimeError(f"合并种子并集不合：缺 {missing} 多 {extra} —— 停")
    for g in all_reg:
        if g["game"] in seen_games:
            raise RuntimeError(f"局总体册局号重复 {g['game']} —— 停")
        seen_games[g["game"]] = g
    os.makedirs(out_dir, exist_ok=True)
    meta = {"protocol": "S3_ROOT_LIST_FREEZE_PROTOCOL.md",
            "protocol_md5": "4212f13afe71b4008579df5f0c605bf0",
            "merged_from": sorted(shard_dirs),
            "games": [0, N_GAMES - 1],
            "seeds": [SEED_BASE, SEED_BASE + N_GAMES - 1]}
    with open(os.path.join(out_dir, "S3_ROOT_LIST_FROZEN.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"meta": meta,
                   "roots": sorted(all_roots, key=lambda r: r["root_id"])},
                  fh, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(out_dir, "S3_GAME_REGISTER.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"meta": meta,
                   "register": sorted(all_reg, key=lambda g: g["game"])},
                  fh, ensure_ascii=False, indent=1, default=str)
    n_prod = len(all_reg)
    merged_report = {
        "games_played": N_GAMES, "producing_games_N": n_prod,
        "zero_root_games_M": N_GAMES - n_prod,
        "total_roots": len(all_roots),
        "shard_reports": reports,
    }
    with open(os.path.join(out_dir, "S3_ENUMERATION_REPORT.json"),
              "w", encoding="utf-8") as fh:
        json.dump(merged_report, fh, ensure_ascii=False, indent=1, default=str)
    print(f"合并验合：{N_GAMES} 局种子全集合，去重键唯一 "
          f"（{len(all_roots)} 根），产根局 {n_prod}、零根局 {N_GAMES - n_prod}",
          flush=True)
    return merged_report


def parse_games(spec):
    a, b = spec.split("-")
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser(description="S3 根单枚举（协议 §1/§2/§6/§10）")
    ap.add_argument("--games", default="0-2",
                    help="局号区间，如 0-2（试跑）或 0-299（全量）")
    ap.add_argument("--out", default=None, help="输出目录")
    ap.add_argument("--pack", default=None,
                    help="基线包路径（默认 smoke fetch_pack 口径）")
    ap.add_argument("--shard-id", default="s0", help="分片标识")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="合并模式：给出各分片目录")
    ap.add_argument("--smoke-steps", type=int, default=None,
                    help="仅开发调试：每局最多步数；输出打 SMOKE 戳、不得入冻结")
    args = ap.parse_args()
    if args.merge:
        out_dir = args.out or os.path.join(_HERE, "enum_merged")
        merge_shards(out_dir, args.merge)
        return 0
    g0, g1 = parse_games(args.games)
    if not (0 <= g0 <= g1 <= N_GAMES - 1):
        raise SystemExit(f"--games 须在 0–{N_GAMES - 1} 内：{args.games}")
    out_dir = args.out or os.path.join(_HERE, "enum_out")
    if args.smoke_steps is not None:
        print("!! SMOKE 模式：输出打 SMOKE-DO-NOT-FREEZE 戳，不得入冻结 !!",
              flush=True)
    pack_path = args.pack or fetch_pack()
    enum = Enumerator(pack_path, out_dir, args.shard_id,
                      smoke_steps=args.smoke_steps)
    t0 = time.monotonic()
    try:
        enum.setup()
        enum.run_shard(g0, g1)
    except Exception as exc:
        halt = {"error": repr(exc), "wall_s": round(time.monotonic() - t0, 1)}
        enum.halts.append(halt)
        with open(os.path.join(out_dir, f"HALT_{args.shard_id}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(halt, fh, ensure_ascii=False, indent=1, default=str)
        print(f"HALT: {exc!r}", flush=True)
        return 2
    print(f"分片完成：{g0}–{g1}，总墙钟 {time.monotonic() - t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
