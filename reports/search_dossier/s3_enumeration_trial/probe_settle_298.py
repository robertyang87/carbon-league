"""试跑结算验证探针（§10.4 门 3 补充验证，非冻结件）。

目的：独立验证 _settle_one 在 step-298 可辨根上的结算值。
方法：重放 game 0（seed 986200）至 step 298（agent-only，不枚举），
对每席每根的全部合法候选，同时跑：
  (a) 冻结代码 Enumerator._settle_one；
  (b) 独立一步推进：全新 FastCarbon  sim，load_observation(root.full)，
      _tid_counter 同式 +500，step(候选，对手实际动作)，直读现金差。
比较两者，并打印绝对现金（确认非零、与 verify3 独立终局分一致）。
同时核对 step-298 key 集与试跑根单一致（重放确定性）。
"""
import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import enumerate_roots as ER  # noqa: E402


def main():
    enum = ER.Enumerator("/tmp/echo4_int8.py", "/tmp/probe_settle_out",
                         "probe")
    enum.setup()
    sim = enum.FastCarbon()
    sim.reset(seed=986200)
    obs298 = None
    actions298 = None
    while sim.step_count < ER.TERMINAL_STEP:
        step = sim.step_count
        obs = [copy.deepcopy(sim.obs(0)), copy.deepcopy(sim.obs(1))]
        keys = [enum.key_units(obs[s], enum.cfg) for s in (0, 1)]
        trig = [bool(keys[s]) for s in (0, 1)]
        actions = [None, None]
        for s in (0, 1):
            if trig[s]:
                enum.harnesses[s].begin_step(obs[s])
            actions[s] = enum.agents[s](obs[s])
        if step == 298:
            obs298 = obs
            actions298 = actions
            keys298 = keys
            break
        sim.step(actions[0], actions[1])

    assert obs298 is not None
    # 核对 key 集与试跑根单一致
    trial = [json.loads(l) for l in
             open(os.path.join(_HERE, "trial_out", "roots_trial.jsonl"))]
    trial298 = {(r["seat"], r["key_uid"]): r for r in trial
                if r["game"] == 0 and r["step"] == 298}
    print("trial step-298 roots:", len(trial298))
    ind = enum.FastCarbon()
    n_checked = 0
    n_agree = 0
    for seat in (0, 1):
        got = {(s, u) for s, u in keys298[seat]}
        want = {(r["seat"], r["key_uid"]) for r in trial
                if r["game"] == 0 and r["step"] == 298 and r["seat"] == seat}
        # key_units 返回 (stake, uid)；取 uid 集比较
        got_uids = {u for _, u in keys298[seat]}
        want_uids = {k[1] for k in want}
        print(f"seat {seat}: replay keys {sorted(got_uids)} "
              f"trial keys {sorted(want_uids)} "
              f"{'MATCH' if got_uids == want_uids else 'MISMATCH'}")
        harness = enum.harnesses[seat]
        root = harness._build_root(obs298[seat], actions298[seat])
        frozen = harness._gen_candidates_raw(root, actions298[seat])
        f_actions, f_labels = frozen
        vetoed = harness._guard_screen(root, f_labels)
        legal_idx = [i for i in range(len(f_actions)) if i not in vetoed]
        print(f"seat {seat}: candidates {len(f_actions)}, vetoed {sorted(vetoed)}, "
              f"legal {len(legal_idx)}")
        for i in legal_idx:
            cand = f_actions[i]
            kind_a, val_a = enum._settle_one(root, cand, actions298[1 - seat],
                                             seat)
            # 独立路径
            ind.load_observation(root.full)
            ind._tid_counter = [c + 500 for c in ind._tid_counter]
            cash_before = list(ind.cash)
            if seat == 0:
                ind.step(cand, actions298[1])
            else:
                ind.step(actions298[0], cand)
            cash_after = list(ind.cash)
            kind_b = ("time" if ind.step_count >= ER.TERMINAL_STEP else
                      "extinct" if enum.terminal_fn(ind) else None)
            val_b = (float(cash_after[seat]) - float(cash_after[1 - seat])
                     if kind_b else None)
            agree = (kind_a == kind_b and val_a == val_b)
            n_checked += 1
            n_agree += int(agree)
            print(f"  cand {i}: frozen=({kind_a},{val_a}) "
                  f"independent=({kind_b},{val_b}) "
                  f"cash_before={cash_before} cash_after={cash_after} "
                  f"{'AGREE' if agree else '*** DISAGREE ***'}")
    print(f"checked {n_checked}, agree {n_agree}")
    print("RESULT:", "PASS" if n_checked and n_checked == n_agree else "FAIL")


if __name__ == "__main__":
    main()
