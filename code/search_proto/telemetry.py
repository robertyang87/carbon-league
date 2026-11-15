"""独立遥测通道（S0b 原型）。

设计教训：旧代码（cnet2_search 94f7fb64 系）的 stats 字典只在搜索闭包内部
累积，从未发布到 agent.last_stats，上报链是死的（K4 探针 search_games=0
误判即源于此）。本模块把遥测做成一个显式、独立的对象：比较器每做一次
决策就调用 record_decision()，外部（harness / 测试 / 验收）通过 snapshot()
读取，与被测代码的内部状态完全解耦——计数为 0 时先查通道是否接通，
不再把「没上报」误判成「没运行」。

纯 Python、无第三方依赖。
"""

from __future__ import annotations

import copy


class StatsCollector:
    """搜索决策遥测计数器。

    计数器：
      calls                  —— 进入比较器（agent 决策点触发搜索评估）的次数
      completed_comparisons  —— 完成整轮比较的根数（全部候选在同深度评完）
      completed_rounds       —— 完成的比较轮次数（一根可有多轮）
      changed_count          —— 最终动作 != 原策略动作 A0 的次数
      depth_hist             —— {深度: 次数} 直方图（按完成轮的深度记）
      leaf_calls             —— 叶评估调用总次数
      cost_accum_s           —— 比较器内部累计耗时（秒，浮点）
      total_cost_accum_s     —— 公开入口全链累计耗时（SI02：含基线与各
                                阶段，不再只报比较器内核那一段）
      phase_cost_accum_s     —— {阶段: 累计秒}，阶段口径由生产入口定义：
                                snapshot / baseline / candgen / guard /
                                rollout_leaf
      abort_reasons          —— {原因: 次数}（deadline / single_call_
                                overrun / budget_gate / nonfinite_leaf /
                                exception / terminal_root）

    decision_dict 约定键（record_decision 入参，缺键按默认处理）：
      completed (bool)        本根是否完成至少一轮完整比较
      rounds (int)            本根完成的轮数
      changed (bool)          本根最终动作是否改变
      depth (int | None)      本根完成轮的最大深度（记入 depth_hist）
      leaf_calls (int)        本根消耗的叶调用数
      cost_s (float)          本根比较器内部耗时（秒）
      total_cost_s (float)    本根公开入口全链耗时（秒，含基线；SI02）
      phase_costs (dict)      本根分阶段耗时 {阶段: 秒}（SI02）
      abort_reason (str|None) 未完成/中止原因；非 None 时记入 abort_reasons
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.completed_comparisons = 0
        self.completed_rounds = 0
        self.changed_count = 0
        self.depth_hist: dict[int, int] = {}
        self.leaf_calls = 0
        self.cost_accum_s = 0.0
        self.total_cost_accum_s = 0.0
        self.phase_cost_accum_s: dict[str, float] = {}
        self.abort_reasons: dict[str, int] = {}

    def record_decision(self, decision: dict) -> None:
        self.calls += 1
        rounds = int(decision.get("rounds", 0))
        self.completed_rounds += rounds
        if decision.get("completed", rounds > 0):
            self.completed_comparisons += 1
            depth = decision.get("depth")
            if depth is not None:
                d = int(depth)
                self.depth_hist[d] = self.depth_hist.get(d, 0) + 1
        if decision.get("changed", False):
            self.changed_count += 1
        self.leaf_calls += int(decision.get("leaf_calls", 0))
        self.cost_accum_s += float(decision.get("cost_s", 0.0))
        if decision.get("total_cost_s") is not None:
            self.total_cost_accum_s += float(decision["total_cost_s"])
        phase_costs = decision.get("phase_costs") or {}
        for phase, secs in phase_costs.items():
            self.phase_cost_accum_s[phase] = (
                self.phase_cost_accum_s.get(phase, 0.0) + float(secs))
        reason = decision.get("abort_reason")
        if reason:
            self.abort_reasons[reason] = self.abort_reasons.get(reason, 0) + 1

    def snapshot(self) -> dict:
        """返回可 JSON 序列化的深拷贝快照（depth_hist 键转 str 以兼容 JSON）。"""
        return {
            "calls": self.calls,
            "completed_comparisons": self.completed_comparisons,
            "completed_rounds": self.completed_rounds,
            "changed_count": self.changed_count,
            "depth_hist": {str(k): v for k, v in sorted(self.depth_hist.items())},
            "leaf_calls": self.leaf_calls,
            "cost_accum_s": self.cost_accum_s,
            "total_cost_accum_s": self.total_cost_accum_s,
            "phase_cost_accum_s": copy.deepcopy(self.phase_cost_accum_s),
            "abort_reasons": copy.deepcopy(self.abort_reasons),
        }
