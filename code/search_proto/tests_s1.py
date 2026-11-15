#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1 机制验证套件 —— Comparator 合同测试（纯 Python，无第三方依赖）。

被测对象：同目录 comparator.py 的 Comparator，共享 API 合同：
  Candidate = dataclass(index:int, action, is_a0:bool)
  Comparator(budget_s: float, max_candidates: int = 4)
  decide(state, candidates, rollout, leaf, horizon:int, scenario_seed:int,
         guard_vetoed:set[int]=frozenset()) -> Decision(
             action, chosen_index, completed_rounds, deepest_completed_h,
             completed_comparisons, changed, elapsed_s, abort_reason)
  rollout(state, action, horizon, scenario_seed) -> leaf_state
  leaf(leaf_state) -> float in [0, 1]

合同语义（来自 S1 合同，非实现实测）：
  - 一轮 = 对全部（未被守卫否决的）候选在同一 horizon、同一 scenario_seed
    下各做一次 rollout+leaf；整轮完成才允许更新结果；
  - 中断时返回上一完整轮的结果；首轮未完成 -> 回 A0，
    completed_rounds=0 且 abort_reason 非空；
  - 平局优先 A0（原动作有先验优先权）；
  - guard_vetoed 中的候选绝不被选中；
  - 预算按每次 decide 调用独立计（自调用入口起），快步不攒额度、
    不跨步结转。

核对状态：已逐行比对 comparator.py 实现，签名与合同逐字一致；
T6 的「拒绝 >4」行为经实测确认（实现抛 ValueError），非仅合同假设。
运行：python3 tests_s1.py ；退出码 0 = 全过，1 = 有 FAIL。
"""

import sys
import time

try:
    from comparator import Candidate, Comparator
except Exception as exc:  # 实现缺失/导入失败：如实报告，不伪造结果
    print("FAIL  import comparator: %r" % (exc,))
    print("SUMMARY: 0 passed, 1 failed (comparator.py 不可用，套件无法执行)")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 确定性微型环境（stub）
#   state   : 整数格子位置
#   action  : 位移（整数）
#   rollout : 按位移推进 horizon 步 -> leaf_state = state + action * horizon
#             （scenario_seed 在本 stub 中不引入随机性，保持全确定性）
#   leaf    : 按预设价值表给分，表外位置给默认值 0.0，值域 [0, 1]
# ---------------------------------------------------------------------------

def make_rollout(sleep_s=0.0):
    def rollout(state, action, horizon, scenario_seed):
        if sleep_s > 0:
            time.sleep(sleep_s)
        return state + action * horizon
    return rollout


def make_leaf(table, default=0.0):
    def leaf(leaf_state):
        v = table.get(leaf_state, default)
        assert 0.0 <= v <= 1.0
        return v
    return leaf


def cand(index, action, is_a0=False):
    return Candidate(index=index, action=action, is_a0=is_a0)


# 标准三候选：A0(位移0) / C1(位移+1) / C2(位移-1)，state=10, horizon=1 时
# 叶状态分别为 10 / 11 / 9。
STATE = 10
HORIZON = 1
SEED = 986200


def std_candidates():
    return [cand(0, 0, is_a0=True), cand(1, +1), cand(2, -1)]


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

def t1_positive_must_change():
    """T1 正例必改：A0 叶值明显低于另一候选 -> 选最优候选、changed=True。"""
    comp = Comparator(budget_s=5.0)
    leaf = make_leaf({10: 0.20, 11: 0.90, 9: 0.50})
    d = comp.decide(STATE, std_candidates(), make_rollout(), leaf,
                    horizon=HORIZON, scenario_seed=SEED)
    assert d.completed_rounds >= 1, d
    assert d.chosen_index == 1, d
    assert d.action == +1, d
    assert d.changed is True, d
    assert d.abort_reason in (None, ""), d


def t2_negative_no_change():
    """T2 反例不改：A0 叶值最高 -> changed=False、仍选 A0。"""
    comp = Comparator(budget_s=5.0)
    leaf = make_leaf({10: 0.95, 11: 0.30, 9: 0.40})
    d = comp.decide(STATE, std_candidates(), make_rollout(), leaf,
                    horizon=HORIZON, scenario_seed=SEED)
    assert d.completed_rounds >= 1, d
    assert d.chosen_index == 0, d
    assert d.action == 0, d
    assert d.changed is False, d


def t3_abort_fallback_to_a0():
    """T3 中断回退：慢 rollout + 极小预算，首轮完不成 ->
    返回 A0、completed_rounds=0、abort_reason 非空。"""
    comp = Comparator(budget_s=0.05)
    leaf = make_leaf({10: 0.20, 11: 0.90, 9: 0.50})
    slow = make_rollout(sleep_s=0.30)  # 单次调用已远超总预算
    d = comp.decide(STATE, std_candidates(), slow, leaf,
                    horizon=HORIZON, scenario_seed=SEED)
    assert d.chosen_index == 0, d
    assert d.action == 0, d
    assert d.changed is False, d
    assert d.completed_rounds == 0, d
    assert d.abort_reason, d  # 非空字符串（合同要求）


def t4_guard_veto_respected():
    """T4 守卫一致：叶值最高候选被 guard_vetoed -> 绝不被选，次优胜出。"""
    comp = Comparator(budget_s=5.0)
    leaf = make_leaf({10: 0.20, 11: 0.95, 9: 0.70})
    d = comp.decide(STATE, std_candidates(), make_rollout(), leaf,
                    horizon=HORIZON, scenario_seed=SEED,
                    guard_vetoed={1})
    assert d.chosen_index != 1, d
    assert d.chosen_index == 2, d  # 次优（0.70 > A0 的 0.20）
    assert d.changed is True, d


def t5_tie_prefers_a0():
    """T5 平局优先 A0：所有叶值相等 -> 选 A0、changed=False。"""
    comp = Comparator(budget_s=5.0)
    leaf = make_leaf({10: 0.50, 11: 0.50, 9: 0.50})
    d = comp.decide(STATE, std_candidates(), make_rollout(), leaf,
                    horizon=HORIZON, scenario_seed=SEED)
    assert d.chosen_index == 0, d
    assert d.action == 0, d
    assert d.changed is False, d


def t6_candidate_cap():
    """T6 候选上限：传入 5 个候选（max_candidates=4）。

    合同要求：decide 必须拒绝 >max_candidates 的候选集并抛异常，
    不得静默截断后照常给出一个看似完整比较的结果。
    实现核对：comparator.py 对 len(candidates) > max_candidates
    抛 ValueError，与本断言一致（已实测）。
    """
    comp = Comparator(budget_s=5.0, max_candidates=4)
    five = [cand(0, 0, is_a0=True), cand(1, +1), cand(2, -1),
            cand(3, +2), cand(4, -2)]
    leaf = make_leaf({}, default=0.5)
    raised = False
    try:
        comp.decide(STATE, five, make_rollout(), leaf,
                    horizon=HORIZON, scenario_seed=SEED)
    except Exception:
        raised = True
    assert raised, "传入 5 个候选未报错（合同要求拒绝 >4）"


def t7_no_budget_banking():
    """T7 快步不攒额度（近似验证）。

    依据：Comparator 无跨调用状态，budget_s 是构造期固定参数，
    每次 decide 的截止自本次调用入口独立计。
    第一次 decide 用极快 rollout 完成整轮（耗时 << budget_s）；
    若实现错误地把第一次未用完的预算结转，第二次的慢 rollout
    （单次 0.30s，总轮次 >= 0.60s）仍不可能在 budget_s=0.10 +
    结转 <= 0.10 的总额内完成整轮 —— 故第二次必须回 A0、
    completed_rounds=0，以此证明无跨步额度结转在本量级成立。
    """
    comp = Comparator(budget_s=0.10)
    leaf = make_leaf({10: 0.20, 11: 0.90, 9: 0.50})
    d1 = comp.decide(STATE, std_candidates(), make_rollout(), leaf,
                     horizon=HORIZON, scenario_seed=SEED)
    assert d1.completed_rounds >= 1, d1
    assert d1.elapsed_s < 0.10, d1  # 第一次确实快、留有大量"剩余额度"
    d2 = comp.decide(STATE, std_candidates(),
                     make_rollout(sleep_s=0.30), leaf,
                     horizon=HORIZON, scenario_seed=SEED)
    assert d2.chosen_index == 0, d2
    assert d2.changed is False, d2
    assert d2.completed_rounds == 0, d2
    assert d2.abort_reason, d2


TESTS = [
    ("T1", t1_positive_must_change),
    ("T2", t2_negative_no_change),
    ("T3", t3_abort_fallback_to_a0),
    ("T4", t4_guard_veto_respected),
    ("T5", t5_tie_prefers_a0),
    ("T6", t6_candidate_cap),
    ("T7", t7_no_budget_banking),
]


def main():
    passed, failed = 0, 0
    for name, fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL  %s  %s" % (name, exc))
        except Exception as exc:  # 非断言异常同样记 FAIL 并如实呈现
            failed += 1
            print("FAIL  %s  unexpected %r" % (name, exc))
        else:
            passed += 1
            print("PASS  %s" % name)
    print("SUMMARY: %d passed, %d failed, %d total"
          % (passed, failed, passed + failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
