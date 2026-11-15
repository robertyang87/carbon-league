"""真实环境接线：Comparator → Echo4 包 agent（钉版 cnet2_search 94f7fb64）。

把 comparator.py 的注册比较器合同接进真实 _search 内部：

  - A0 ＝ post-guard 动作（_inner 行 1004 定型版：网络解码 → 规则层 →
    bankrupt_guard 之后的 acts），不是裸网输出。本模块不重跑 _inner 主链：
    基线 agent 本体先按原样执行一步产出 A0，本模块随后在同一根上接线。
  - 比较器替换 _search 行 1571–1665 的推进＋判读段（固定深度 rollout、
    MIN_DEPTH 门、0.45 首层门、MARGIN=20 提交门槛全部不带入新接线；
    新比较器整轮 argmax 提交、平局优先 A0，与注册比较器合同一致）。
  - 保留段（行 1503–1562：key_units 触发、Prefix forced 重解码、去重、
    对手根动作、full 根状态）在本模块复刻为我方代码，调用的全部函数
    与状态对象均取自已构造 agent 的闭包/模块全局（见 extract_internals），
    不 monkeypatch vendor 文件、不改 vendor 任何字节。

嵌套函数提取办法与漂移风险（如实登记）：
  leaf / guard / pol / rpol / st / stats / pool / _terminal 等都是
  make_search_agent 的闭包内对象，模块级不可达。本模块从已构造的
  agent 对象沿闭包单元按名反查：外层 agent.__closure__['_inner'] →
  _inner.__closure__['_search'/'guard'/'pol'/'st'/'cfg'/'stats'/'T'] →
  _search.__closure__['leaf'/'rpol'/'pool'/'_terminal']。这是零代码复制
  的提取（leaf 本体不复刻、直接引用原函数对象），但依赖 CPython 闭包
  布局与自由变量名：vendor/pack 源码一旦刷新（md5 变化），必须重验
  提取名清单——本模块在提取时逐名断言，缺名即抛错停，不静默降级。
  构造 agent 的模块来自 Echo4 包内嵌源，其 _BUILD_HASHES 与 vendor/
  钉版四源 md5 逐项核对（feat 2a45b08b / np d8c15ad4 / fastsim
  aefe6c83 / search 94f7fb64），不合即停。

post-guard 标签 y 的重建：_inner 不保存 post-guard 的 y（只存行 867 的
裸网 y0）。本模块用 Feat.labels(A0 动作字典) 反演——decode_b 对无单位
槽位留 -1、labels() 对无单位槽位同样留 -1，工人槽经 CANON_MOVE 对合
互逆、基地槽经 BASE_ACTS 互逆，故反演与真 y 逐槽一致（vendor 自身的
SEQ_KEYS 段亦用同一反演法）。

对手视角观测 xo 的重建：_inner 行 1019 用 fo.observe(obs_o) 生成 xo
但不保存。observe() 在 step <= self.step 时会整表重init，直接对真 fo
重算会毁掉信念状态。本模块在基线调用**之前**先 feat_copy(fo) 存快照，
调用后按 _inner 行 1013–1019 原式构造 obs_o、再由快照 observe 得 xo：
快照状态与真 fo 在该步 observe 前的状态一致、输入一致，结果确定性
相同；且 Echo4 CONF 下 INFER_MIRROR/INFER_TREE 均关，feat_copy 置
False 的 infer 标志与真 fo 一致，此重建在本 CONF 下精确。oenc 路径
（MIX/mcts/OPPID）下 st['oenc'] 在调用后可直读，按 vendor 行 1553–1558
原式走，不用 xo。

CONF 纪律：一切参数只读 agent.cfg（构造后合并字典），不读 DEFAULTS
推断。scenario_seed：sweep 路径无种子轴、J=1 确定性——本模块把
scenario_seed 透传给 Comparator.decide，但 rollout 不消费它（对手根
动作已在根外定型、双座推进为贪心确定性），此事实记入 self.notes，
不声称与任何带种子行为等价。

叶尺度（裁定 3）：嵌套 leaf 返回原始分差尺度。本阶段 HeuristicLeaf-
Adapter 以 affine 恒等 (a=1, b=0) 直通——比较器 argmax 对单调变换
不变，叶①的比较只在同叶内进行、尺度即原始分差；不套用任何未注册
压缩函数。代价：adapter 的 out_of_range 计数会大量增长（原始分差
本就在 [0,1] 外），那是本阶段的预期登记现象，不是量纲错误信号；
正式 U 尺度校准走后续注册流程。

Codex 复核修复（2026-11-16，REVIEW_SEARCH_IMPLEMENTATION SI01/
SI02/SI04/SI05）：
  - SI01 权威终局：最后有效动作观察 step298、终态 step299
    （eval_v53 口径）。decide_at_root 入口拒绝已终局根（step≥299
    或根状态已灭绝）；rollout 每次联合推进后同时判灭绝＋时间边界，
    终态冻结、其后零策略调用；leaf 的 left 按权威剩余步计；结算
    口径 F01（灭绝：leaf 内按带入灭绝步前现金）／F04（到时：真实
    终态现金差，见 _raw_leaf 的 terminal_kind 分流）。
  - SI02 唯一 deadline：总预算 cfg BUDGET 覆盖基线步耗＋快照＋根
    构造＋候选生成＋守卫准入＋比较器全链，删除原 0.05s 预算下限
    补齐逻辑——余额不足直接回 A0 且不启动比较；遥测记总耗时与
    分阶段耗时（baseline/snapshot/root_build/candgen/guard/
    rollout_leaf）；比较器带 C_hi 整轮准入。R2-04 残余修正：
    deadline 锚点前移到公开入口 begin_step（其快照耗时计入同一
    总账，旧版漏计）、余额检查前置于全部昂贵准备之前（旧版基线
    已耗尽仍做准备工作）、C_hi 改按 (K,H) 分桶高成本界（旧版跨
    根跨 H 均值＋无样本返回 0 禁用准入，均废；冷启动口径见
    _est_candidate_cost_s）、无候选根也记全调用耗时账。
  - SI04 续弈生产守卫：rollout 的 H≥2 续弈步为双席各建独立
    BankruptcyGuard 状态（copy 自根守卫、随步演化，reset_step→
    apply 与 _inner 同序），act_b 后、step 前过守卫。R2-01 修正：
    两席历史归属分开——己方席 copy 自身生产 guard，对手席只来自
    对手自身历史（登记快照源或全新 guard＋prevw=对手根处现存单位
    的显式近似），不再两席同抄己方 guard（对手会凭空继承己方
    单位为其死亡、lostw 虚增）。
  - SI05 seat 初始化：begin_step(obs) 的快照席位从当前 observation
    的 player 明确取，不再默认己方为 0。
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from comparator import Candidate, Comparator, Decision
from leaf_adapters import HeuristicLeafAdapter
from telemetry import StatsCollector

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "vendor")

# vendor 钉版四源的期望 md5（search 为去 3 行头注释后的代码 md5，
# 与包 _BUILD_HASHES 同口径；其余三源为整文件 md5）。
EXPECTED_VENDOR_MD5 = {
    "cnet2_feat": "2a45b08b6f97508541e62867e408748e",
    "cnet2_np": "d8c15ad4fdd8ee079363cf215d52daf3",
    "fastsim": "aefe6c83826efae4de83bef8c643c80b",
    "cnet2_search": "94f7fb6438bb7577b141e388f068e122",
}
EXPECTED_PACK_MD5 = "819250bfc82632786063dabd46a616bd"

# SI01 权威终局（eval_v53 口径）：最后有效动作观察为 step298，
# 终态为 step299。step>=TERMINAL_STEP 的观察不再有合法动作；
# rollout 的联合推进绝不越过 TERMINAL_STEP。
TERMINAL_STEP = 299


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def verify_vendor() -> dict:
    """核 vendor/ 四源 md5（只读），不合即抛错停。返回实测 md5 表。"""
    out = {}
    for name, want in EXPECTED_VENDOR_MD5.items():
        path = os.path.join(_VENDOR, name + ".py")
        with open(path, "rb") as fh:
            raw = fh.read()
        if name == "cnet2_search":
            # 与 tail -n +4 口径一致：去掉前 3 行头注释后算代码 md5
            raw = b"".join(raw.splitlines(keepends=True)[3:])
        got = _md5(raw)
        if got != want:
            raise RuntimeError(f"vendor {name} md5 不合：got {got}, want {want} —— 停")
        out[name] = got
    return out


def load_pack(pack_path: str):
    """按包自身口径加载 Echo4 包并构造一个 agent。

    验整包 md5 == 819250bf…；权重取包内 _dequantize(_B1/_Q1) 解出，
    CONF 用包内 CONF，构造走包内嵌 cnet2_search 的 make_search_agent
    （rweights/oweights 为 None：包 CONF 为 ROLL='self'、无 rollout 网，
    与包内 agent() 的构造口径一致）。返回 (pack_module, agent)。
    """
    import base64
    import importlib.util
    import zlib

    with open(pack_path, "rb") as fh:
        raw = fh.read()
    got = _md5(raw)
    if got != EXPECTED_PACK_MD5:
        raise RuntimeError(f"Echo4 包 md5 不合：got {got}, want {EXPECTED_PACK_MD5} —— 停")
    spec = importlib.util.spec_from_file_location("echo4_pack", pack_path)
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    for name, want in EXPECTED_VENDOR_MD5.items():
        bh = pack._BUILD_HASHES.get(name)
        if bh != want:
            raise RuntimeError(
                f"包内嵌源 {name} build hash 与 vendor 钉版不合：got {bh}, want {want} —— 停")
    w1 = pack._dequantize(zlib.decompress(base64.b64decode(pack._B1)), pack._Q1)
    w1["meta"] = np.array(pack._M1)
    agent = pack._S.make_search_agent(pack.CONF, weights=w1, rweights=None, oweights=None)
    return pack, agent


def _cells(fn) -> dict:
    """函数对象的闭包单元 {自由变量名: 值}；无闭包返回 {}。"""
    if fn.__closure__ is None:
        return {}
    return {name: cell.cell_contents
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}


def _need(cells_map: dict, name: str, where: str):
    if name not in cells_map:
        raise RuntimeError(
            f"闭包提取失败：{where} 缺自由变量 {name!r}（vendor 布局已变，须重验提取清单）—— 停")
    return cells_map[name]


@dataclass
class Internals:
    """从已构造 agent 沿闭包反查出的原对象（全部为引用，无复制）。"""
    search_fn: Any            # 嵌套 _search（仅作闭包载体，不调用）
    leaf_fn: Any              # 嵌套 leaf 原函数（行 666，不复刻）
    terminal_fn: Any          # 嵌套 _terminal
    guard: Any                # agent 的 BankruptcyGuard 实例（只许 .copy() 后用）
    pol: Any                  # CachePolicy（主网）
    rpol: Any                 # rollout 网（ROLL='self' 时与 pol 同一对象）
    st: dict                  # agent 状态字典（活引用）
    stats: dict               # agent 内部 stats（活引用；本模块遥测不走它）
    cfg: dict                 # = agent.cfg（合并后 CONF，只读此份）
    pool: list                # agent 的 FastCarbon 池（活引用）
    T: int                    # cfg TEACHER 的闭包别名（rollout teacher 兜底）
    mod: Any                  # 包内嵌 cnet2_search 模块（全局函数来源）
    F: Any                    # 包内嵌 cnet2_feat 模块
    FastCarbon: Any           # 包内嵌 fastsim.FastCarbon


def extract_internals(agent) -> Internals:
    """外层 agent → _inner → _search 三级闭包按名提取；缺名即停。"""
    inner = _need(_cells(agent), "_inner", "outer agent")
    ci = _cells(inner)
    search_fn = _need(ci, "_search", "_inner")
    cs = _cells(search_fn)
    mod = search_fn.__globals__  # dict 即包内嵌 cnet2_search 模块的 __dict__
    return Internals(
        search_fn=search_fn,
        leaf_fn=_need(cs, "leaf", "_search"),
        terminal_fn=_need(cs, "_terminal", "_search"),
        guard=_need(ci, "guard", "_inner"),
        pol=_need(ci, "pol", "_inner"),
        rpol=_need(cs, "rpol", "_search"),
        st=_need(ci, "st", "_inner"),
        stats=_need(ci, "stats", "_inner"),
        cfg=agent.cfg,
        pool=_need(cs, "pool", "_search"),
        T=_need(ci, "T", "_inner"),
        mod=mod,
        F=mod["F"],
        FastCarbon=mod["FastCarbon"],
    )


# 外层规则层（vendor 行 1975–2048）开关清单：(cfg 键, 层名, 改写性质)。
# 在目标 CONF 下逐项以 agent.cfg 实测值判定活/关，见 outer_layer_audit()。
OUTER_LAYER_SWITCHES = [
    ("REC_CUT_ALL", "recruit_cut（禁招募测量层）", "删基地招募动作"),
    ("REC_CUT_PLANTER", "recruit_cut（禁 planter 招募）", "删 RECPLANTER 动作"),
    ("OPEN_PL", "open_planter_sensor（开局 planter 传感层）", "可改写首招/招募为 RECPLANTER"),
    ("TG_MAX", "tree_gate", "按树数上限改写动作"),
    ("TG_LATE_STEP", "tree_gate（晚期触发）", "按树数上限改写动作"),
    ("RG_STEP", "recruit_gate", "按步数门改写招募"),
    ("PX_MODE", "planter_explore", "开局期改写 planter 动作用于探索"),
    ("PLANTER_AVOID", "planter_avoid", "改写 planter 动作避险"),
    ("SE_STEP", "safe_entry", "按步数门改写进入动作"),
    ("CB_STEP", "camp_bust", "按步数门改写扎营动作"),
    ("EH_STEP", "endgame_home", "终局期改写返航动作"),
]


def outer_layer_audit(cfg: dict) -> list:
    """逐项列外层规则层开关在给定（合并后）cfg 下的活/关状态。

    判定与 vendor 外层代码一致：各层均以 cfg.get(键) 真值/非 None 为
    触发条件，键缺失即关。外层异常兜底（任何异常 → 返回 {} 空动作）
    是结构性的、无开关，单列一行恒记「活」。
    """
    rows = []
    for key, name, effect in OUTER_LAYER_SWITCHES:
        val = cfg.get(key)
        rows.append({"key": key, "layer": name, "effect": effect,
                     "value": val, "live": bool(val) if key not in
                     ("TG_MAX", "TG_LATE_STEP", "RG_STEP", "SE_STEP", "CB_STEP", "EH_STEP")
                     else val is not None})
    rows.append({"key": "<structure>", "layer": "outer exception fallback",
                 "effect": "_inner 抛异常时返回 {} 空动作", "value": None, "live": True})
    return rows


@dataclass
class LeafState:
    """rollout 的不透明叶状态：叶调用前 comparator 不触碰其内部。

    terminal_kind（SI01）：None＝未终局；"extinct"＝灭绝终局（结算走
    leaf 内 F01 分支，prev_cash 为带入灭绝步前现金）；"time"＝到时
    终局（step 到达 TERMINAL_STEP，结算用真实终态现金差）。
    """
    sim: Any
    me: int
    left: int
    prev_cash: Any
    terminal_kind: Optional[str] = None


@dataclass
class RootContext:
    obs: dict
    me: int
    step: int
    left: int
    f: Any
    fo: Any
    x: dict
    y: np.ndarray            # post-guard 标签（labels() 反演）
    lg0: np.ndarray          # 根先验 logits [OWN, 5]
    acts_o: dict             # 对手根动作（J=1 确定性单情景）
    full: dict               # FastCarbon 根状态（行 1564–1565 原式）
    t_now: int
    ot: int


class RealEnvSearch:
    """在一个已构造的基线 agent 上接线的比较器 harness（单座）。

    用法（每个真实步）：
        snap = harness.begin_step(obs)       # 基线调用前存 fo 快照
        a0 = agent(obs)                      # 基线本体原样执行（A0 定型）
        dec = harness.decide_at_root(obs, a0, baseline_step_cost_s=dt,
                                     horizon=1, scenario_seed=...)
    基线动作照常下发推进对局；dec 只是同根上的比较器判读（冒烟口径）。
    """

    def __init__(self, agent, collector: Optional[StatsCollector] = None):
        self.agent = agent
        self.inx = extract_internals(agent)
        self.cfg = self.inx.cfg
        self.collector = collector if collector is not None else StatsCollector()
        self.notes: list = [
            "scenario_seed 未消费：sweep 路径无种子轴，J=1 确定性（对手根动作根外定型、双座推进贪心确定性）",
            "叶①尺度为原始分差：HeuristicLeafAdapter affine 恒等直通，比较只在同叶内",
        ]
        self._fo_snap = None
        self._guard_exceptions = 0
        # R2-01：对手席 rollout guard 的快照源（对手自身历史）。None＝
        # 未登记，rollout 时以全新 guard＋prevw=对手根处现存单位近似
        # （见 _make_rollout），绝不拿己方 guard 顶替。
        self._opp_guard = None
        # SI01：被入口拒绝的已终局根计数（step>=299 或已灭绝）
        self.terminal_roots_rejected = 0
        # SI02：C_hi 整轮准入用的已测单候选成本（rollout+leaf 秒/次）样本
        self._cand_cost_sum = 0.0
        self._cand_cost_n = 0
        # R2-04 残余：C_hi 成本样本按 (候选数 K, 深度 H) 分桶（高成本
        # 界＝桶内最大值，不用跨根跨 H 均值）；_cand_cost_sum/_n 只作
        # 兼容汇总，不再参与准入估计。
        self._cand_cost_buckets: dict = {}
        # R2-04 残余：begin_step 的快照耗时（公开入口锚点）——decide_
        # at_root 的唯一 deadline 自 begin_step 起算，快照成本经
        # _snapshot_cost_s 计入同一总账，decide 消费后清 pending。
        self._snapshot_cost_s = 0.0
        self._snapshot_pending = False
        # SI04：最近一次 rollout 的双席守卫副本（供验收/测试检查状态演化）
        self._last_rollout_guards = None
        self.rollout_guard_rewrites = 0
        self._probe = None  # 根灭绝探针 sim（懒构造）
        # 叶①：原嵌套 leaf 函数 + affine 恒等适配（裁定 3）
        self.leaf_adapter = HeuristicLeafAdapter(self._raw_leaf, a=1.0, b=0.0)

    # -- 叶 ---------------------------------------------------------------
    def _raw_leaf(self, leaf_state: LeafState) -> float:
        """叶值分流（SI01）：

        - terminal_kind == "time"（到时终局）：F04 口径，用真实终态
          现金差。不能走嵌套 leaf 的启发式分支——它在 left=0 时仍含
          「已在基地格上的载货」等项，不等于终局结算（电池 B 的旧
          双证只在构造格上偶合，不构成一般等式）。
        - 其余（未终局 / 灭绝终局）：直接引用原嵌套 leaf（行 666），
          不复刻公式；灭绝终局由 leaf 内 F01 分支按 prev_cash 结算。
        返回原始分差尺度。
        """
        if leaf_state.terminal_kind == "time":
            sim = leaf_state.sim
            return float(sim.cash[leaf_state.me]) - float(sim.cash[1 - leaf_state.me])
        return float(self.inx.leaf_fn(leaf_state.sim, leaf_state.me,
                                      leaf_state.left, leaf_state.prev_cash))

    # -- 步首快照 -----------------------------------------------------------
    def begin_step(self, obs: Optional[dict] = None):
        """基线调用前：feat_copy(fo) 存快照（fo 尚不存在时造对座新 Feat）。

        见模块 docstring：真 fo 在本步内会被 _inner observe 一次，
        事后重算 xo 会触发 step<=self.step 的整表重 init；快照是唯一
        无损重建路径。快照同时把 infer 标志置 False——本 CONF 下
        INFER_MIRROR/INFER_TREE 均关，与真 fo 一致。

        SI05：快照席位从当前 observation 明确取（obs["player"]）。
        旧实现 fo 不存在时默认己方为 0，fresh seat1 根会把对手快照
        造成 Feat(1)，随后对手预测分支全是 player-1 动作（错方）。
        fo 已存在但席位与本观察对不上（换席新局、_inner 尚未重建）
        时同样按观察席位新造，不沿用陈旧快照。

        R2-04 残余：本方法是公开调用链的入口锚点——自进入起计时，
        快照耗时记 _snapshot_cost_s 并置 pending，供随后 decide_
        at_root 计入同一 deadline 总账（旧实现快照在时钟外、报告
        总耗时漏计它）。
        """
        _t0 = time.monotonic()
        fo = self.inx.st.get("fo")
        if obs is not None:
            my_seat = int(obs["player"])
        else:
            f = self.inx.st.get("f")
            my_seat = f.seat if f is not None else 0
        if fo is None or int(getattr(fo, "seat", 1 - my_seat)) != 1 - my_seat:
            self._fo_snap = self.inx.F.Feat(1 - my_seat)
        else:
            self._fo_snap = self.inx.mod["feat_copy"](fo)
        self._snapshot_cost_s = time.monotonic() - _t0
        self._snapshot_pending = True
        return self._fo_snap

    # -- 对手席 guard 快照源（R2-01） ---------------------------------------
    def set_opponent_guard(self, guard):
        """登记对手席 BankruptcyGuard 快照源（对手自身历史）。

        rollout 的对手席守卫只从此源 .copy() 取用：绝不改写源对象，
        更不许拿己方 guard 顶替——己方 lostw/prevw 是己方历史，复制
        给对手席后，对手首次 apply 就会把己方现存单位误计为对手死亡
        （prevw 与 cur 不交 → lostw 虚增），并无条件继承己方既往死亡
        数（R2-01 实包反例：对手 lostw 凭空 1、叶值 −117 vs 正确
        +209）。离线/verify3 路径传快照中对手席的 guard 对象。
        """
        self._opp_guard = guard

    def set_opponent_agent(self, agent):
        """decide 路径：对手 agent 对象在手边时，取其自身 guard 作
        对手席快照源（对手自身历史的活引用；rollout 时仍只 .copy()）。"""
        self._opp_guard = extract_internals(agent).guard

    # -- C_hi 单候选成本估计（R2-04 残余：按 (K,H) 分桶的高成本界） --------
    def _est_candidate_cost_s(self, n_candidates: int, horizon: int):
        """本 (K,H) shape 的单候选（rollout+leaf）高成本界（秒/次）。

        口径（写明，不许静默）：
          - 本 shape 有样本：界＝桶内最大值（高成本界，非均值）。
          - 本 shape 无样本、其它 shape 有：冷启动界＝全部桶最大值
            的最大者（以已测最贵 shape 兜底，不借用浅层低均值）。
          - 全部无样本（全冷启动）：返回 None——准入豁免（比较器
            est=None 语义），首轮实测后入桶；这不是「成本为 0」，
            轮内 deadline 检查照常生效。旧实现无样本返回 0.0 把准入
            永久禁用、且跨根跨 H 均值把 H1 的低成本借给 H3，均废。
        """
        buckets = getattr(self, "_cand_cost_buckets", None)
        if buckets is None:
            buckets = self._cand_cost_buckets = {}
        own = buckets.get((int(n_candidates), int(horizon)))
        if own:
            return max(own)
        if buckets:
            return max(max(v) for v in buckets.values() if v)
        return None

    def _probe_sim(self):
        if self._probe is None:
            self._probe = self.inx.FastCarbon()
        return self._probe

    # -- 根构造（复刻 _search 保留段的输入侧） ------------------------------
    def _build_root(self, obs: dict, a0_actions: dict) -> RootContext:
        inx, st, cfg = self.inx, self.inx.st, self.cfg
        F = inx.F
        me = obs["player"]
        f, fo = st["f"], st["fo"]
        x = st["x"]
        lg0 = st["lg0"]
        # SI01：left = 权威剩余步（到终态 step299 还能推进的联合步数）
        left = TERMINAL_STEP - obs["step"]
        # post-guard 标签反演（Feat.labels 与 decode/to_actions 逐槽互逆）
        y = f.labels(a0_actions)
        # 对手根动作（行 1553–1562 原式）
        if st.get("oenc") is not None:
            Xo, toko, preo = st["oenc"]
            yo = inx.mod["decode_b"](
                inx.pol, Xo, [fo.aux], toko,
                inx.mod["ctx_b"](inx.pol, preo, np.array([st["ot"]])))[0]
            acts_o = fo.to_actions(yo)
        else:
            if self._fo_snap is None:
                raise RuntimeError("begin_step() 未调用：缺 fo 快照，无法重建 xo —— 停")
            obs_o = dict(obs)
            obs_o["player"] = 1 - me
            vis = np.zeros(F.N)
            for wv in obs["players"][1 - me][2].values():
                cs = [wv[0]] if wv[2] == "COLLECTOR" else F.DEST[wv[0], 1:]
                vis[cs] = f.est[cs]
            obs_o["carbon"] = vis.tolist()
            xo = self._fo_snap.observe(obs_o)
            yo, _, _ = inx.pol.act(xo, self._fo_snap.aux, teacher=st["ot"])
            acts_o = self._fo_snap.to_actions(yo)
        full = {"step": obs["step"], "players": obs["players"],
                "trees": obs.get("trees") or {},
                "full_carbon": [float(v) for v in f.est],
                "carbon": obs["carbon"], "remainingOverageTime": 60}
        return RootContext(obs=obs, me=me, step=obs["step"], left=left,
                           f=f, fo=fo, x=x, y=y, lg0=lg0, acts_o=acts_o,
                           full=full, t_now=st["T_now"], ot=st["ot"])

    # -- 候选生成（复刻行 1503–1551）＋三道准入 ------------------------------
    # 拆分（SI02 分阶段计时）：_gen_candidates_raw 管生成＋截断（candgen
    # 阶段），_guard_screen 管守卫一致准入（guard 阶段），
    # _generate_candidates 是旧签名的合并包装（s1_battery 插桩版用）。
    def _gen_candidates_raw(self, root: RootContext, a0_actions: dict):
        """生成＋去重＋K≤4 截断，返回 (cand_actions, cand_labels) 或 None。

        准入：① 根合法性＝生成端合法掩码＋seen 去重（原式）；③ 战术
        相关性截断＝K≤4（含 A0，其余按根先验 lg0[偏离槽][强制标签]
        降序、同分按生成序）。②守卫一致在 _guard_screen。
        bk/pk 两段纯 stats 计数分支不复刻（无行为影响）。
        """
        inx, cfg = self.inx, self.cfg
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
        # ③ 战术相关性截断到 K≤4：其余候选按根先验 lg0 顺序
        order = list(range(1, len(cand_actions)))
        if len(cand_actions) > 4:
            def prior(i):
                uid, m = cand_meta[i]
                return float(lg0[uid2slot[uid]][m])
            order.sort(key=lambda i: (-prior(i), i))
            keep = [0] + sorted(order[:3])
        else:
            keep = [0] + order
        cand_actions = [cand_actions[i] for i in keep]
        cand_labels = [cand_labels[i] for i in keep]
        return cand_actions, cand_labels

    def _guard_screen(self, root: RootContext, cand_labels: list) -> set:
        """② 守卫一致准入：逐候选用 guard.copy() 跑 apply，被改写（或
        复核抛异常、无法证明一致）者进 veto 集（A0 豁免，不测）。"""
        inx = self.inx
        lg0 = root.lg0
        guard_vetoed = set()
        for idx in range(1, len(cand_labels)):
            try:
                g = inx.guard.copy()
                out = g.apply(root.x, root.f.aux, cand_labels[idx].copy(), lg0,
                              inx.pol, root.f.own_uid, root.t_now)
            except Exception:
                self._guard_exceptions += 1
                guard_vetoed.add(idx)  # 无法证明守卫一致 → 按不一致计
                continue
            if not np.array_equal(np.asarray(out), cand_labels[idx]):
                guard_vetoed.add(idx)
        return guard_vetoed

    def _generate_candidates(self, root: RootContext, a0_actions: dict):
        """旧签名合并包装：返回 (candidates, cand_labels, guard_vetoed)，
        候选不足 2 个返回 None。"""
        raw = self._gen_candidates_raw(root, a0_actions)
        if raw is None:
            return None
        cand_actions, cand_labels = raw
        guard_vetoed = self._guard_screen(root, cand_labels)
        candidates = [Candidate(index=i, action=a, is_a0=(i == 0))
                      for i, a in enumerate(cand_actions)]
        return candidates, cand_labels, guard_vetoed

    # -- rollout（复刻行 1571–1640 的单分支串行版＋SI01/SI04） -----------------
    def _make_rollout(self, root: RootContext):
        inx = self.inx
        cfg = self.cfg
        pool = inx.pool
        if not pool:
            pool.append(inx.FastCarbon())
        sim = pool[0]  # 串行复用：comparator 在下一次 rollout 前已消费叶状态

        def is_terminal(s_) -> bool:
            """SI01 权威终局：灭绝（vendor _terminal）或时间边界
            （step_count 到达 TERMINAL_STEP）。旧实现只判灭绝，
            H≥2 会越过 step299 继续推进并多调一拍策略。"""
            return inx.terminal_fn(s_) or s_.step_count >= TERMINAL_STEP

        def rollout(state, action, horizon, scenario_seed):
            # scenario_seed 未消费（J=1 确定性），见 self.notes
            s_ = sim
            s_.load_observation(state)
            s_._tid_counter = [c + 500 for c in s_._tid_counter]
            fm = inx.mod["feat_copy"](root.f)
            fop = inx.mod["feat_copy"](root.fo)
            # SI04：续弈步的双席独立生产守卫。各从根守卫 copy，随步
            # 演化各自的 lostw/prevw/guard_base，互不串扰（vendor 守卫
            # 类 docstring 明示：绝不跨分支共享同一实例）。guards[0]
            # 对应 fm 席（root.me），guards[1] 对应 fop 席（1-root.me）。
            # R2-01：两席历史归属必须分开——guards[0] 从己方生产
            # guard copy；guards[1] 只能来自对手自身历史：已登记对手
            # 快照源（set_opponent_guard/set_opponent_agent/离线快照）
            # 时从其 copy；未登记时用全新 guard 并把 prevw 初始化为
            # 对手在根处的现存单位（对手根前历史未知的显式近似：
            # lostw=0、不继承任何一方的既往死亡）。旧实现两席同抄
            # 己方 guard，对手首次 apply 即凭空继承己方单位为其死亡
            # （lostw 虚增、GUARD_NARROW 误触发改写对手动作）。
            # 根步动作已在根处定型（我方候选经准入守卫复核、对方为根
            # 预测），不在此重过；守卫只管 H≥2 的续弈步。
            guards = None
            if cfg.get("GUARD"):
                if self._opp_guard is not None:
                    g_opp = self._opp_guard.copy()
                else:
                    g_opp = inx.mod["BankruptcyGuard"](cfg)
                    # prevw 只取工人槽 [:10]，与 apply 的 cur 口径逐字
                    # 对齐：own_uid 的 10/11 槽是招募中心，不是工人，
                    # 混入 prevw 会在首次 apply 被误计为对手死亡
                    # （实测凭空 +2、GUARD_NARROW 误触发）。
                    g_opp.prevw = {u for u in fop.own_uid[:10]
                                   if u is not None}
                guards = (inx.guard.copy(), g_opp)
            self._last_rollout_guards = guards
            term_cash = list(s_.cash)     # F01：带入根步的 cash
            if root.me == 0:
                s_.step(action, root.acts_o)
            else:
                s_.step(root.acts_o, action)
            depth = 1
            term = is_terminal(s_)
            hh = 1
            while hh < horizon and not term:
                pc = list(s_.cash)        # F01：带入本步的 cash
                xs = [fm.observe(s_.obs(root.me)), fop.observe(s_.obs(1 - root.me))]
                lab = inx.mod["act_b"](inx.rpol, xs, [fm.aux, fop.aux],
                                       [root.t_now, root.ot])
                if guards is not None:
                    # SI04：act_b 后、step 前逐席过守卫（与 _inner 同序：
                    # 先 reset_step 清 guard_base，再 apply）。守卫排序
                    # 需要本步 logits：rpol.act 与 act_b 同源解码，取
                    # 其 logits，labels 仍以 act_b 为准（多一次前向，
                    # 成本记入 rollout_leaf 阶段）。
                    lab = [np.asarray(lab[0]), np.asarray(lab[1])]
                    for side, feat in ((0, fm), (1, fop)):
                        teacher = root.t_now if side == 0 else root.ot
                        _y, lg_seat, _v = inx.rpol.act(
                            xs[side], feat.aux, teacher=teacher)
                        g = guards[side]
                        g.reset_step()
                        out = g.apply(xs[side], feat.aux, lab[side], lg_seat,
                                      inx.rpol, feat.own_uid, teacher)
                        if not np.array_equal(np.asarray(out), lab[side]):
                            self.rollout_guard_rewrites += 1
                        lab[side] = out
                am = fm.to_actions(lab[0])
                ao = fop.to_actions(lab[1])
                if root.me == 0:
                    s_.step(am, ao)
                else:
                    s_.step(ao, am)
                depth += 1
                hh += 1
                if is_terminal(s_):
                    term = True
                    term_cash = pc        # R02：终局步冻结，存活方按带入 cash 计
            kind = None
            if term:
                # 灭绝优先（与 leaf 内 F01 分支的判定优先级一致）；
                # 纯到时才标 "time"（走真实终态现金结算）。
                kind = "extinct" if inx.terminal_fn(s_) else "time"
            leaf_left = max(0, root.left - depth)
            return LeafState(sim=s_, me=root.me, left=leaf_left,
                             prev_cash=term_cash, terminal_kind=kind)

        return rollout

    # -- 主入口 ----------------------------------------------------------------
    def decide_at_root(self, obs: dict, a0_actions: dict,
                       baseline_step_cost_s: float, horizon: int = 1,
                       scenario_seed: int = 0,
                       horizons: Optional[list] = None) -> Optional[Decision]:
        """在基线已定型的根上跑一次比较器；无候选根/已终局根返回 None。

        SI02 唯一 deadline 口径：总预算 = agent.cfg['BUDGET']（Echo4
        包 0.7s）覆盖全链。旧实现的 0.05s 预算下限补齐已删除：全链
        余额不足时直接回 A0（Decision, abort_reason="deadline"）且
        不启动比较。遥测记总耗时与分阶段耗时，Decision 的
        total_elapsed_s / phase_costs 同步填充。

        R2-04 残余（本版）：唯一 deadline 锚在公开入口 begin_step——
        其快照耗时（_snapshot_cost_s）与基线步耗（baseline_step_
        cost_s，调用方实测按值计入）同为前置成本（pre_cost），加上
        本入口内耗时即全调用总账。余额检查前置于全部昂贵准备：
        pre_cost 已耗尽预算时在根构造/灭绝 probe/候选/guard 之前
        直接回 A0（旧残余：基线 0.8>0.7 仍做 0.9 的准备工作、快照
        0.6 漏计仍提交改子，两反例均由此翻转）。准备块（根构造→
        probe→候选→guard）内部互相依赖、作为整体进入，块后在比较
        器启动前再查余额并过 C_hi 整轮准入（按 (K,H) 分桶高界）。
        无候选根同样记一笔全调用耗时账（返回 None 不变，遥测
        abort_reason="no_candidates"）。

        SI01：已终局根（step>=TERMINAL_STEP 或根状态已灭绝）在入口
        拒绝（返回 None、记 terminal_root 失败行与计数），不进比较。

        horizons 非 None 时走 anytime 多轮（Comparator.decide_multi），
        忽略 horizon 参数。
        """
        t_entry = time.monotonic()
        total_budget = float(self.cfg["BUDGET"])
        base_cost = float(baseline_step_cost_s)
        # 前置成本：begin_step 快照（公开入口锚点）＋基线步耗。
        if getattr(self, "_snapshot_pending", False):
            snapshot_cost = float(self._snapshot_cost_s)
            self._snapshot_pending = False
        else:
            snapshot_cost = 0.0
        pre_cost = snapshot_cost + base_cost
        phases: dict = {"baseline": base_cost, "snapshot": snapshot_cost}

        def _total() -> float:
            return pre_cost + (time.monotonic() - t_entry)

        def _record(dec_fields: dict, total_elapsed: float) -> None:
            # 遥测：显式发布到独立 StatsCollector（活对象），不读快照
            self.collector.record_decision({
                "completed": dec_fields.get("completed", False),
                "rounds": dec_fields.get("rounds", 0),
                "changed": dec_fields.get("changed", False),
                "depth": dec_fields.get("depth"),
                "leaf_calls": dec_fields.get("leaf_calls", 0),
                "cost_s": dec_fields.get("cost_s", 0.0),
                "total_cost_s": total_elapsed,
                "phase_costs": dict(phases),
                "abort_reason": dec_fields.get("abort_reason"),
            })

        def _finish(dec: Decision, leaf_calls: int,
                    total_elapsed: float) -> Decision:
            dec = replace(dec, total_elapsed_s=total_elapsed,
                          phase_costs=dict(phases))
            _record({
                "completed": dec.completed_rounds > 0,
                "rounds": dec.completed_rounds,
                "changed": bool(dec.changed),
                "depth": dec.deepest_completed_h if dec.completed_rounds > 0 else None,
                "leaf_calls": leaf_calls,
                "cost_s": dec.elapsed_s,
                "abort_reason": dec.abort_reason,
            }, total_elapsed)
            return dec

        def _a0_decision(abort_reason: str) -> Decision:
            return Decision(action=a0_actions, chosen_index=0,
                            completed_rounds=0, deepest_completed_h=0,
                            completed_comparisons=0, changed=False,
                            elapsed_s=0.0, abort_reason=abort_reason)

        # ---- SI01 入口拒绝：时间终局根 --------------------------------
        if int(obs["step"]) >= TERMINAL_STEP:
            self.terminal_roots_rejected += 1
            self.notes.append(
                f"step {obs['step']}: 已终局根（step>={TERMINAL_STEP}）拒绝比较")
            _finish(_a0_decision("terminal_root"), 0, _total())
            return None

        # ---- 余额检查前置：耗尽即回 A0，不做任何昂贵准备 ----------------
        if _total() >= total_budget:
            return _finish(_a0_decision("deadline"), 0, _total())

        # ---- 根构造（对手推断含在内） ------------------------------------
        t = time.monotonic()
        root = self._build_root(obs, a0_actions)
        phases["root_build"] = time.monotonic() - t

        # ---- SI01 入口拒绝：根状态已灭绝 --------------------------------
        probe = self._probe_sim()
        probe.load_observation(root.full)
        if self.inx.terminal_fn(probe):
            self.terminal_roots_rejected += 1
            self.notes.append(
                f"step {root.step}: 根状态已灭绝，拒绝比较")
            _finish(_a0_decision("terminal_root"), 0, _total())
            return None

        # ---- 候选生成（candgen）＋守卫准入（guard），分阶段计时 ---------
        t = time.monotonic()
        raw = self._gen_candidates_raw(root, a0_actions)
        phases["candgen"] = time.monotonic() - t
        if raw is None:
            # 无候选根：仍记全调用耗时账（旧残余：此类根不入账）。
            _record({"abort_reason": "no_candidates"}, _total())
            return None
        cand_actions, cand_labels = raw
        t = time.monotonic()
        guard_vetoed = self._guard_screen(root, cand_labels)
        phases["guard"] = time.monotonic() - t
        candidates = [Candidate(index=i, action=a, is_a0=(i == 0))
                      for i, a in enumerate(cand_actions)]

        # ---- 比较器启动前再查余额＋C_hi 整轮准入 -------------------------
        elapsed = _total()
        remaining = total_budget - elapsed
        if remaining <= 0:
            return _finish(_a0_decision("deadline"), 0, elapsed)
        deadline_abs = t_entry + (total_budget - pre_cost)
        shape_h = horizon if horizons is None else max(horizons)
        est = self._est_candidate_cost_s(len(candidates), shape_h)
        comp = Comparator(budget_s=remaining, max_candidates=4)
        rollout = self._make_rollout(root)
        leaf_calls_before = self.leaf_adapter.calls
        t = time.monotonic()
        if horizons is not None:
            dec = comp.decide_multi(
                root.full, candidates, rollout, self.leaf_adapter,
                horizons=horizons, scenario_seed=scenario_seed,
                guard_vetoed=guard_vetoed,
                est_candidate_cost_s=est,
                deadline_abs=deadline_abs)
        else:
            dec = comp.decide(
                root.full, candidates, rollout, self.leaf_adapter,
                horizon=horizon, scenario_seed=scenario_seed,
                guard_vetoed=guard_vetoed,
                est_candidate_cost_s=est,
                deadline_abs=deadline_abs)
        phases["rollout_leaf"] = time.monotonic() - t
        total_elapsed = _total()
        leaf_calls = self.leaf_adapter.calls - leaf_calls_before
        if leaf_calls > 0 and phases["rollout_leaf"] > 0:
            # C_hi 成本样本：本根单次（rollout+leaf）成本入 (K,H) 桶；
            # decide_multi 的样本记最深 shape（最贵轮口径，保守）。
            per_call = phases["rollout_leaf"] / leaf_calls
            buckets = getattr(self, "_cand_cost_buckets", None)
            if buckets is None:
                buckets = self._cand_cost_buckets = {}
            bucket = buckets.setdefault((len(candidates), int(shape_h)), [])
            bucket.append(per_call)
            del bucket[:-64]  # 每桶保留最近 64 个样本
            self._cand_cost_sum += per_call
            self._cand_cost_n += 1
        return _finish(dec, leaf_calls, total_elapsed)
