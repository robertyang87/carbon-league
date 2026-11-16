#!/usr/bin/env python3
"""TAU=0 equivalence check (iter1 acceptance): variant-comparator pack
in TAU=0 form (CTRL: variant comparator + cmp3 wiring, CONF identical
to resync, no CMP_TAU key) vs the frozen-kernel resync pack
cba5923c -- same diagnostic seed 7000, same diagnostic budget 30,
pack seat 0 vs the baseline pack, probe_runner form (batch-3 A/B
precedent). Decision streams must be byte-identical: full action
streams + comparator telemetry, wall-clock accumulators excluded
(cost_accum_s / phase_cost_accum_s / total_cost_accum_s).

Resync side: resync/probe_new_big.json (batch-3 filed run of
diag/new_diag_big.py = resync pack with CONF BUDGET=30).
Variant side: iter1/out/probe_ctrl30.json (diag/diag_ctrl30.py).
"""
import json
import sys

WALL_KEYS = {"cost_accum_s", "phase_cost_accum_s", "total_cost_accum_s"}


def game(path):
    d = json.load(open(path))
    assert len(d["games"]) == 1
    return d, d["games"][0]


def main():
    ref_d, ref = game(sys.argv[1])   # resync (frozen kernel)
    new_d, new = game(sys.argv[2])   # variant, tau=0 form
    report = {"ref_pack": ref_d["pack"], "new_pack": new_d["pack"],
              "seed": ref["seed"], "checks": {}}
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        report["checks"][name] = {"pass": bool(cond), "detail": detail}
        ok = ok and bool(cond)

    chk("same_seed_seat_steps",
        (ref["seed"], ref["pack_seat"], ref["steps"])
        == (new["seed"], new["pack_seat"], new["steps"]),
        f"ref seed/seat/steps={ref['seed']}/{ref['pack_seat']}/{ref['steps']} "
        f"new={new['seed']}/{new['pack_seat']}/{new['steps']}")
    chk("actions_byte_identical", ref["actions"] == new["actions"],
        f"steps compared: {len(ref['actions'])} vs {len(new['actions'])}")
    if ref["actions"] != new["actions"]:
        for i, (a, b) in enumerate(zip(ref["actions"], new["actions"])):
            if a != b:
                report["checks"]["actions_byte_identical"]["first_diff_step"] = i
                break
    chk("final_cash_equal", ref["cash"] == new["cash"],
        f"ref={ref['cash']} new={new['cash']}")
    ref_cmp = {k: v for k, v in (ref["cmp"] or {}).items() if k not in WALL_KEYS}
    new_cmp = {k: v for k, v in (new["cmp"] or {}).items() if k not in WALL_KEYS}
    chk("cmp_telemetry_equal_excl_wall", ref_cmp == new_cmp,
        "" if ref_cmp == new_cmp else
        f"diff keys: {[k for k in set(ref_cmp) | set(new_cmp) if ref_cmp.get(k) != new_cmp.get(k)]}")
    chk("per_step_trace_equal", ref["trace"] == new["trace"],
        f"trace entries: {len(ref['trace'])} vs {len(new['trace'])}")
    if ref["trace"] != new["trace"]:
        for i, (a, b) in enumerate(zip(ref["trace"], new["trace"])):
            if a != b:
                report["checks"]["per_step_trace_equal"]["first_diff_trace_idx"] = i
                report["checks"]["per_step_trace_equal"]["ref"] = a
                report["checks"]["per_step_trace_equal"]["new"] = b
                break
    report["verdict"] = "PASS" if ok else "FAIL"
    report["ref_cmp"] = ref_cmp
    report["new_cmp"] = new_cmp
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
