"""bigeval -- independent-sample evaluation: one game per seed (seat alternates by seed parity), since the
two seats of a seed are exact mirrors in this harness.
  python3 bigeval.py A.py B.py SEED0 N TAG     -> appends summary to ../data/bigeval.jsonl"""
import json, os, statistics as st, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
def _play(t):
    import eval_v53
    try: return eval_v53.play_game(t)
    except Exception as e: return {"error": repr(e)[:200], "seed": t[3]}
def main(a, b, s0, n, tag):
    tasks = [(tag, a, b, s, s % 2) for s in range(s0, s0 + n)]
    live = os.path.join(HERE, "..", "data", "bigeval_live"); os.makedirs(live, exist_ok=True)
    lf = open(os.path.join(live, tag + ".jsonl"), "a")
    rows = []
    with ProcessPoolExecutor(int(os.environ.get("EVO_WORKERS", "8"))) as ex:
        futs = [ex.submit(_play, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            g = {"seed": r.get("seed"), "side": r.get("policy_side"), "margin": round(r["margin"]) if "margin" in r else None,
                 "err": r.get("error"), "t": time.strftime("%H:%M:%S")}
            if "ledger_pol" in r:
                g.update(dep=round(r["ledger_pol"]["deposits"]), odep=round(r["ledger_opp"]["deposits"]),
                         lost=round(r["ledger_pol"]["death_cargo"]), olost=round(r["ledger_opp"]["death_cargo"]),
                         tinc=round(r["ledger_pol"]["tree_income"]), otinc=round(r["ledger_opp"]["tree_income"]),
                         pspend=round(r["ledger_pol"].get("plant_spend", 0)), opspend=round(r["ledger_opp"].get("plant_spend", 0)),
                         strand=r["ledger_pol"].get("stranded"), ostrand=r["ledger_opp"].get("stranded"),
                         bk=r.get("bankrupt_steps", 0), lx=r.get("ledger_extra"), wms=r.get("a_worst_step_ms"), ms=r.get("a_ms_per_step"),
                         **({"srch": r["a_stats"].get("searched"), "chg": r["a_stats"].get("changed"),
                             "shal": r["a_stats"].get("shallow"), "ab": r["a_stats"].get("aborted", 0),
                             "rerr": r["a_stats"].get("root_errors", 0), "eh_e": r["a_stats"].get("eh_edge", 0), "eh_h": r["a_stats"].get("eh_heavy", 0),
                             "cc21": r["a_stats"].get("cc_2to1", 0), "cc01": r["a_stats"].get("cc_0to1", 0), "pa": r["a_stats"].get("pa_moves", 0), "paerr": r["a_stats"].get("pa_errors", 0), "pb": r["a_stats"].get("pb_skip", 0), "mnp": r["a_stats"].get("mnp", 0), "eh_alt": r["a_stats"].get("eh_alt", 0), "fixdrop": r["a_stats"].get("fix_dropped", 0), "pk_in": r["a_stats"].get("pk_in", 0), "pk_cut": r["a_stats"].get("pk_cut", 0), "sims": r["a_stats"].get("sims", 0), "pk_rep": r["a_stats"].get("pk_rep", 0), "pk_rep3": r["a_stats"].get("pk_rep_plant3", 0), "pk_repc": r["a_stats"].get("pk_rep_cash", 0), "bk_in": r["a_stats"].get("bk_in", 0), "bk_cut": r["a_stats"].get("bk_cut", 0)} if r.get("a_stats") else {}))
            lf.write(json.dumps(g) + "\n"); lf.flush()
    m = [r["margin"] for r in rows if "margin" in r]
    if len(m) < 2:
        print(json.dumps({"tag": tag, "INCOMPLETE": True, "n": len(m), "want": n, "err": len(rows) - len(m)}), file=sys.stderr, flush=True)
        sys.exit(2)
    L = lambda side, k: round(st.mean([r[f"ledger_{side}"][k] for r in rows if "margin" in r]))
    se = st.stdev(m) / len(m) ** .5
    rec = {"tag": tag, "a": a, "b": b, "seed0": s0, "n": len(m), "err": sum("error" in r for r in rows),
           "wins": sum(x > 0 for x in m), "mean": round(st.mean(m)), "se": round(se), "lo2se": round(st.mean(m) - 2 * se),
           "worst": round(min(m)), "worst_ms": max(r.get("a_worst_step_ms", 0) for r in rows if "margin" in r),
           "dep": [L("pol", "deposits"), L("opp", "deposits")], "tree": [L("pol", "tree_income"), L("opp", "tree_income")],
           "deathcargo": [L("pol", "death_cargo"), L("opp", "death_cargo")], "plants": [L("pol", "plant_count"), L("opp", "plant_count")],
           "t": time.strftime("%H:%M"),
           "search_games": sum(1 for r in rows if (r.get("a_stats") or {}).get("searched")),
           "changed0_games": sum(1 for r in rows if (r.get("a_stats") or {}).get("searched") and not r["a_stats"].get("changed")),
           "games": [[r.get("seed"), r.get("policy_side"), round(r["margin"])] for r in rows if "margin" in r]}
    open(os.path.join(HERE, "..", "data", "bigeval.jsonl"), "a").write(json.dumps(rec) + "\n"); print(json.dumps({k: v for k, v in rec.items() if k != "games"}), flush=True)
    # unified exam rule: an incomplete result (missing or errored games) must never be marked as done by callers
    if rec["changed0_games"]:
        print(json.dumps({"tag": tag, "WARN_SEARCH_NEVER_CHANGED": rec["changed0_games"], "of": rec["search_games"]}), file=sys.stderr, flush=True)
    if rec["err"] or rec["n"] < n:
        print(json.dumps({"tag": tag, "INCOMPLETE": True, "n": rec["n"], "want": n, "err": rec["err"]}), file=sys.stderr, flush=True)
        sys.exit(2)
if __name__ == "__main__":
    a = sys.argv[1:]; main(a[0], a[1], int(a[2]), int(a[3]), a[4])
