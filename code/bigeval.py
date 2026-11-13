"""bigeval -- independent-sample evaluation: one game per seed (seat alternates by seed parity), since the
two seats of a seed are exact mirrors in this harness.
  python3 bigeval.py A.py B.py SEED0 N TAG     -> appends summary to ../data/bigeval.jsonl
EXAM_ARCHIVE_20261114: per-game archive records (live rows and the summary's
games/games_full) carry the RAW margin, both raw scores and U computed from
the raw margin, plus md5 identity of both policy files and of this harness.
Rounding is display-only (summary statistics); it never enters a per-game
record, so an archived exam can be re-derived exactly (wins and U).
F4_20261114: opponent-side internal errors are visible and gated. Live rows
carry berr (the opponent's internal-error total, mirroring serr) and bst
(its error-class stats summary); the summary counts b_search_error_games /
b_errors_total / b_root_errors_total and FAILs the exam on the same F09
caliber as the candidate side -- a margin earned against a silently
degraded opponent is not a measurement."""
import hashlib, json, math, os, statistics as st, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
def _md5_of(path):
    try:
        with open(path, "rb") as fh: return hashlib.md5(fh.read()).hexdigest()
    except OSError: return None
def _resolve(p):    # mirror eval_v53's loading: policy paths are relative to HERE
    q = os.path.join(HERE, p)
    return q if os.path.exists(q) else p
def _u_of(margin):
    # U = 0.5*W + 0.5*sigmoid(margin/1000), W = 1 / 0.5 / 0 for win/draw/loss
    # from the RAW margin's sign (eval_v53 returns margin at the sim's native
    # cent precision, so this is exact, not an approximation).
    w = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-margin / 1000.0))
def _play(t):
    import eval_v53, traceback
    try: return eval_v53.play_game(t)
    except Exception as e: return {"error": repr(e)[:200], "tb": traceback.format_exc()[-600:], "seed": t[3]}
def main(a, b, s0, n, tag):
    seeds = [int(x) for x in os.environ["EVO_SEEDS"].split(",")] if os.environ.get("EVO_SEEDS") else range(s0, s0 + n)
    # REVIEW_FIXES_20261113 S4 (P2): the expected game count had two
    # definitions -- missing games were counted against len(tasks) while the
    # failure gate compared against the parameter n. With EVO_SEEDS set the
    # two diverge (n=3, two seeds): two clean games were saved FAIL=false
    # yet the process exited 2. n is the caller's declared exam size, so
    # refuse the inconsistency up front (nonzero exit, explicit error)
    # instead of silently redefining the exam to len(tasks) -- a shrunken
    # exam must never pass gates sized for n. After this check,
    # n == len(tasks) by construction and both counts agree everywhere.
    if os.environ.get("EVO_SEEDS") and len(seeds) != n:
        print(json.dumps({"tag": tag, "ERROR": f"EVO_SEEDS has {len(seeds)} seeds but N={n}; refusing inconsistent exam size"}), file=sys.stderr, flush=True)
        sys.exit(2)
    # REVIEW_FULL_20261113 F07 (P2): a duplicated explicit seed silently
    # shrank the exam's independent sample (EVO_SEEDS=17,17 with N=2 exited 0
    # with n=2, se=0 but only ONE independent seed). The count check above
    # cannot see it (len still equals N), so reject duplicates up front too.
    if os.environ.get("EVO_SEEDS") and len(set(seeds)) != len(seeds):
        print(json.dumps({"tag": tag, "ERROR": "EVO_SEEDS contains duplicate seeds; refusing non-independent exam sample"}), file=sys.stderr, flush=True)
        sys.exit(2)
    tasks = [(tag, a, b, s, s % 2) for s in seeds]
    # EXAM_ARCHIVE_20261114: stamp the exam's identity on a start line
    # (stderr), on every live row (ah/bh/hh) and on the summary record, so
    # any archived line can be attributed to exact policy + harness bytes.
    harness_md5 = _md5_of(os.path.abspath(__file__))
    a_md5 = _md5_of(_resolve(a)); b_md5 = _md5_of(_resolve(b))
    print(json.dumps({"tag": tag, "start": True, "harness_md5": harness_md5,
                      "a": a, "a_md5": a_md5, "b": b, "b_md5": b_md5, "n": n}),
          file=sys.stderr, flush=True)
    live = os.path.join(HERE, "..", "data", "bigeval_live"); os.makedirs(live, exist_ok=True)
    lf = open(os.path.join(live, tag + ".jsonl"), "a")
    rows = []
    with ProcessPoolExecutor(int(os.environ.get("EVO_WORKERS", "8"))) as ex:
        futs = [ex.submit(_play, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result(); rows.append(r)
            g = {"seed": r.get("seed"), "side": r.get("policy_side"), "margin": r["margin"] if "margin" in r else None,
                 "err": r.get("error"), "t": time.strftime("%H:%M:%S"), "serr": r.get("a_errors", 0),
                 "berr": r.get("b_errors", 0),
                 "ah": a_md5, "bh": b_md5, "hh": harness_md5}
            if "margin" in r:
                g.update(sa=r.get("my_score"), sb=r.get("opp_score"), win=r.get("win"), u=_u_of(r["margin"]))
            if "ledger_pol" in r:
                g.update(dep=round(r["ledger_pol"]["deposits"]), odep=round(r["ledger_opp"]["deposits"]),
                         lost=round(r["ledger_pol"]["death_cargo"]), olost=round(r["ledger_opp"]["death_cargo"]),
                         tinc=round(r["ledger_pol"]["tree_income"]), otinc=round(r["ledger_opp"]["tree_income"]),
                         pspend=round(r["ledger_pol"].get("plant_spend", 0)), opspend=round(r["ledger_opp"].get("plant_spend", 0)),
                         sspend=round(r["ledger_pol"].get("seize_spend", 0)), osspend=round(r["ledger_opp"].get("seize_spend", 0)),
                         strand=r["ledger_pol"].get("stranded"), ostrand=r["ledger_opp"].get("stranded"),
                         bk=r.get("bankrupt_steps", 0), lx=r.get("ledger_extra"), wms=r.get("a_worst_step_ms"), ms=r.get("a_ms_per_step"),
                         **({"srch": r["a_stats"].get("searched"), "chg": r["a_stats"].get("changed"),
                             "shal": r["a_stats"].get("shallow"), "ab": r["a_stats"].get("aborted", 0),
                             "rerr": r["a_stats"].get("root_errors", 0), "eh_e": r["a_stats"].get("eh_edge", 0), "eh_h": r["a_stats"].get("eh_heavy", 0),
                             "cc21": r["a_stats"].get("cc_2to1", 0), "cc01": r["a_stats"].get("cc_0to1", 0), "pa": r["a_stats"].get("pa_moves", 0), "paerr": r["a_stats"].get("pa_errors", 0), "pb": r["a_stats"].get("pb_skip", 0), "mnp": r["a_stats"].get("mnp", 0), "eh_alt": r["a_stats"].get("eh_alt", 0), "fixdrop": r["a_stats"].get("fix_dropped", 0), "pk_in": r["a_stats"].get("pk_in", 0), "pk_cut": r["a_stats"].get("pk_cut", 0), "sims": r["a_stats"].get("sims", 0), "pk_rep": r["a_stats"].get("pk_rep", 0), "pk_rep3": r["a_stats"].get("pk_rep_plant3", 0), "pk_repc": r["a_stats"].get("pk_rep_cash", 0), "bk_in": r["a_stats"].get("bk_in", 0), "bk_cut": r["a_stats"].get("bk_cut", 0), "simsl": r["a_stats"].get("sims_steps"), "cb": [r["a_stats"].get(k, 0) for k in ("cb_seen", "cb", "cb_ok", "cb_errors")], "se": [r["a_stats"].get(k, 0) for k in ("se_seen", "se", "se_ok", "se_none", "se_errors")], "se_lat": r["a_stats"].get("se_lat"), "msl": r["a_stats"].get("ms_steps"), "px": [r["a_stats"].get("px", 0), r["a_stats"].get("px_errors", 0)], "opl": [r["a_stats"].get("opl", 0), r["a_stats"].get("opl_errors", 0)], "tg": r["a_stats"].get("tg", 0), "rg": r["a_stats"].get("rg", 0), "seqc": r["a_stats"].get("seq_changed", 0)} if r.get("a_stats") else {}),
                         # F4: opponent-side error-class summary (the *_errors
                         # categories of b_stats), so a live row shows WHICH
                         # opponent internals fired, not just the berr total.
                         **({"bst": {k: v for k, v in r["b_stats"].items() if (k == "errors" or k.endswith("_errors")) and isinstance(v, (int, float))}} if r.get("b_stats") else {}))
            lf.write(json.dumps(g) + "\n"); lf.flush()
    m = [r["margin"] for r in rows if "margin" in r]
    # REVIEW_FIXES_20261113 S4 (P3): the run's health is computed ONCE, here,
    # before any exit decision, and every record below carries it. Previously
    # the <2-valid-games path exited before this block ran, so its FAIL
    # record had no first_error and no internal error counts -- the failure
    # gate worked, but the saved record could not say WHY the exam failed.
    # R05: internal health is computed and merged into the summary BEFORE
    # it is saved. (Previously search_error_games was attached to rec only
    # after the record had been written and printed, so the saved summary
    # showed err=0 and callers marked the exam done even when the search /
    # rule layers had thrown inside games.)
    search_error_games = sum(1 for r in rows if r.get("a_errors"))
    a_errors_total = sum(r.get("a_errors", 0) for r in rows)
    root_errors_total = sum((r.get("a_stats") or {}).get("root_errors", 0) for r in rows)
    # F4: opponent-side internal health, computed and gated exactly like the
    # candidate side above. Before this, fb's last_stats were never read at
    # all (eval_v53 returned no b_* fields), so an opponent that degraded to
    # an internal fallback left no trace in any record and its games counted
    # as clean measurements.
    b_search_error_games = sum(1 for r in rows if r.get("b_errors"))
    b_errors_total = sum(r.get("b_errors", 0) for r in rows)
    b_root_errors_total = sum((r.get("b_stats") or {}).get("root_errors", 0) for r in rows)
    # F09: rows arrive in completion order; report the FIRST failure's stack
    # deterministically (lowest seed, traceback preferred over the repr).
    by_seed = sorted(rows, key=lambda r: (r.get("seed") is None, r.get("seed") or 0))
    first_error = next((r.get("tb") or r.get("error") for r in by_seed if r.get("error")), None)
    first_search_error_seed = next((r.get("seed") for r in rows if r.get("a_errors")), None)
    first_b_search_error_seed = next((r.get("seed") for r in rows if r.get("b_errors")), None)
    # F09: FAIL covers every failure class -- internal search/rule errors,
    # outer per-game exceptions and missing games. Previously it counted only
    # the internal classes, so a 2-of-3 exam with one crashed game was saved
    # with FAIL=false (err=1, n=2) while exiting 2.
    outer_errors = sum(1 for r in rows if r.get("error"))
    missing = len(tasks) - len(m)
    # F4: the internal-failure gate is symmetric -- opponent-side internal
    # errors fail the exam on the same caliber as candidate-side ones (the
    # result is marked FAIL and the run exits 3 below). A degraded opponent
    # invalidates the measurement just as a degraded candidate does.
    internal_failed = bool(search_error_games or a_errors_total or root_errors_total
                           or b_search_error_games or b_errors_total or b_root_errors_total)
    failed = bool(internal_failed or outer_errors or missing)
    if len(m) < 2:
        # F09: this path used to save nothing at all; an incomplete exam must
        # still leave a machine-readable FAIL record for downstream readers.
        # It now carries the same health fields as the full summary.
        bad = {"tag": tag, "FAIL": True, "INCOMPLETE": True, "n": len(m), "want": n, "err": len(rows) - len(m),
               "search_error_games": search_error_games, "a_errors_total": a_errors_total,
               "root_errors_total": root_errors_total, "first_error": first_error,
               "first_search_error_seed": first_search_error_seed,
               "b_search_error_games": b_search_error_games, "b_errors_total": b_errors_total,
               "b_root_errors_total": b_root_errors_total,
               "first_b_search_error_seed": first_b_search_error_seed,
               "a_md5": a_md5, "b_md5": b_md5, "harness_md5": harness_md5}
        open(os.path.join(HERE, "..", "data", "bigeval.jsonl"), "a").write(json.dumps(bad) + "\n")
        print(json.dumps(bad), file=sys.stderr, flush=True)
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
           "games": [[r.get("seed"), r.get("policy_side"), r["margin"]] for r in rows if "margin" in r],
           "games_full": [[r.get("seed"), r.get("policy_side"), r["margin"], r.get("my_score"), r.get("opp_score"), _u_of(r["margin"])] for r in rows if "margin" in r],
           "a_md5": a_md5, "b_md5": b_md5, "harness_md5": harness_md5}
    rec.update({"search_error_games": search_error_games, "a_errors_total": a_errors_total,
                "root_errors_total": root_errors_total, "first_error": first_error,
                "first_search_error_seed": first_search_error_seed,
                "b_search_error_games": b_search_error_games, "b_errors_total": b_errors_total,
                "b_root_errors_total": b_root_errors_total,
                "first_b_search_error_seed": first_b_search_error_seed, "FAIL": failed})
    open(os.path.join(HERE, "..", "data", "bigeval.jsonl"), "a").write(json.dumps(rec) + "\n"); print(json.dumps({k: v for k, v in rec.items() if k not in ("games", "games_full")}), flush=True)
    # unified exam rule: an incomplete result (missing or errored games) must never be marked as done by callers
    if internal_failed:      # R05: any exception inside the search / rule layers means the exam is not testing the intended code: fail it (the saved record above already carries the FAIL marker). F4: opponent-side internal errors take the same exit, with their counts named on the same line.
        print(json.dumps({"tag": tag, "FAIL_SEARCH_ERRORS": search_error_games, "a_errors_total": a_errors_total,
                          "FAIL_B_SEARCH_ERRORS": b_search_error_games, "b_errors_total": b_errors_total, "of": len(rows)}), file=sys.stderr, flush=True)
        sys.exit(3)
    if rec["changed0_games"]:
        print(json.dumps({"tag": tag, "WARN_SEARCH_NEVER_CHANGED": rec["changed0_games"], "of": rec["search_games"]}), file=sys.stderr, flush=True)
    if rec["err"] or rec["n"] < n or missing:
        print(json.dumps({"tag": tag, "INCOMPLETE": True, "n": rec["n"], "want": n, "err": rec["err"]}), file=sys.stderr, flush=True)
        sys.exit(2)
if __name__ == "__main__":
    a = sys.argv[1:]; main(a[0], a[1], int(a[2]), int(a[3]), a[4])
