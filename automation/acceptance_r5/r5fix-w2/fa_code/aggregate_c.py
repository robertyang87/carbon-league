"""aggregate_c: acceptance aggregation for case C step 2.

Reads eval_c_pricing_results.jsonl plus the two control sources
(eval_probe_results.jsonl seed-1000 base-vs-shadow; eval_v53_results
.jsonl v53v05 base-vs-v0.5) and prints the 稳/准/狠 tables:

  稳: per paired game, fork death_cargo minus base death_cargo
      (shadow pairs vs cbase/cparity-era controls, v0.5 pairs vs
      eval_v53 controls, mirror pairs in-game). Breach = delta > +50.
  狠: paired deposits+tree_income increments: v0.5 tier and mirror
      tier are the main evidence (bar +300/game); shadow tier reports
      the increment and the conversion rate vs the step-1 headline
      tax (3924.1/game) as direction only (calibration 2026-11-03).
  准: pricing-bucket audit aggregated over all-on fork games: per
      bucket n, strikes, raw rate, Wilson 90% CI, the prior the
      pricing rule used, and how often each posterior level priced.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HEADLINE_TAX = 3924.1


def load(fn):
    out = []
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        return out
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def wilson(k, n, z=1.645):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    res = load("eval_c_pricing_results.jsonl")
    by = {}
    for r in res:
        by.setdefault(r["tag"], {})[(r["seed"], r["policy_side"])] = r

    # ---- controls
    base_shadow = {}
    for r in load("eval_probe_results.jsonl"):
        if r["tag"] == "pbshadow":
            base_shadow[(r["seed"], r["policy_side"])] = r
    for r in by.get("cbase", {}).values():
        base_shadow[(r["seed"], r["policy_side"])] = r
    base_v05 = {}
    for r in load("eval_v53_results.jsonl"):
        if r["tag"] == "v53v05":
            base_v05[(r["seed"], r["policy_side"])] = r

    def econ(r):
        L = r["ledger_pol"]
        return L["deposits"] + L["tree_income"], L["death_cargo"]

    # ---- parity
    print("== PARITY (fork_c_off vs shadow, expect 14255.07:2.43 / "
          "15796.46:1.30)")
    for k in sorted(by.get("cparity", {})):
        r = by["cparity"][k]
        print(f"  seed{k[0]} side{k[1]}: {r['my_cash']}:{r['opp_cash']}")

    # ---- shadow tier
    print("\n== SHADOW TIER (all-on vs base control, paired)")
    d_es = []
    breaches = []
    for k in sorted(by.get("call", {})):
        r = by["call"][k]
        b = base_shadow.get(k)
        if b is None:
            print(f"  seed{k[0]} side{k[1]}: NO CONTROL")
            continue
        e_f, dc_f = econ(r)
        e_b, dc_b = econ(b)
        d_e = e_f - e_b
        d_dc = dc_f - dc_b
        d_es.append(d_e)
        if d_dc > 50:
            breaches.append(("call-shadow", k, d_dc))
        print(f"  seed{k[0]} side{k[1]}: margin {r['margin']:+.0f} "
              f"d_econ {d_e:+.0f} dc {dc_f:.0f} vs base {dc_b:.0f} "
              f"(delta {d_dc:+.0f})")
    if d_es:
        mean_de = sum(d_es) / len(d_es)
        print(f"  mean d_econ {mean_de:+.1f}/game; conversion vs "
              f"headline tax {mean_de / HEADLINE_TAX:+.1%}")

    # ---- ablations vs shadow
    for tag in ("csoft", "clethal"):
        print(f"\n== ABLATION {tag} vs shadow (paired)")
        for k in sorted(by.get(tag, {})):
            r = by[tag][k]
            b = base_shadow.get(k)
            e_f, dc_f = econ(r)
            if b:
                e_b, dc_b = econ(b)
                if dc_f - dc_b > 50:
                    breaches.append((tag, k, dc_f - dc_b))
                print(f"  seed{k[0]} side{k[1]}: margin {r['margin']:+.0f} "
                      f"d_econ {e_f - e_b:+.0f} dc delta "
                      f"{dc_f - dc_b:+.0f}")
            else:
                print(f"  seed{k[0]} side{k[1]}: margin {r['margin']:+.0f} "
                      f"(no control)")

    # ---- v0.5 tier (main 狠 evidence)
    print("\n== V0.5 TIER (all-on vs eval_v53 base control, paired)")
    d_es = []
    for k in sorted(by.get("callv05", {})):
        r = by["callv05"][k]
        b = base_v05.get(k)
        e_f, dc_f = econ(r)
        if b is None:
            print(f"  seed{k[0]} side{k[1]}: NO CONTROL")
            continue
        e_b, dc_b = econ(b)
        d_es.append(e_f - e_b)
        if dc_f - dc_b > 50:
            breaches.append(("callv05", k, dc_f - dc_b))
        print(f"  seed{k[0]} side{k[1]}: margin {r['margin']:+.0f} "
              f"d_econ {e_f - e_b:+.0f} dc {dc_f:.0f} vs base {dc_b:.0f} "
              f"(delta {dc_f - dc_b:+.0f})")
    if d_es:
        print(f"  mean d_econ {sum(d_es) / len(d_es):+.1f}/game "
              f"(bar +300)")

    # ---- mirror tier (main 狠 evidence, in-game paired)
    print("\n== MIRROR TIER (fork vs base, same game)")
    d_es = []
    for k in sorted(by.get("cmirror", {})):
        r = by["cmirror"][k]
        e_f = r["ledger_pol"]["deposits"] + r["ledger_pol"]["tree_income"]
        e_b = r["ledger_opp"]["deposits"] + r["ledger_opp"]["tree_income"]
        dc_f = r["ledger_pol"]["death_cargo"]
        dc_b = r["ledger_opp"]["death_cargo"]
        d_es.append(e_f - e_b)
        if dc_f - dc_b > 50:
            breaches.append(("cmirror", k, dc_f - dc_b))
        print(f"  seed{k[0]} side{k[1]}: margin {r['margin']:+.0f} "
              f"d_econ {e_f - e_b:+.0f} dc {dc_f:.0f} vs base {dc_b:.0f} "
              f"(delta {dc_f - dc_b:+.0f})")
    if d_es:
        print(f"  mean d_econ {sum(d_es) / len(d_es):+.1f}/game "
              f"(bar +300)")

    print("\n== RED-LINE BREACHES (dc delta > +50):", breaches or "none")

    # ---- bucket audit over all-on games
    print("\n== BUCKET AUDIT (aggregated over all-on fork games)")
    agg = {"pb1": {}, "pb1m": {}, "pbt": {}, "pb2": {}, "pb2m": {}}
    lvl = [0, 0, 0]
    ngames = 0
    for tag in ("call", "callv05", "cmirror"):
        for r in by.get(tag, {}).values():
            if not r.get("pb"):
                continue
            ngames += 1
            for tab in agg:
                for k, v in r["pb"][tab].items():
                    a = agg[tab].setdefault(k, [0, 0])
                    a[0] += v[0]
                    a[1] += v[1]
            for i in range(3):
                lvl[i] += r["pb"]["pb_lvl"][i]
    print(f"  games with stats: {ngames}; pricing evaluations by level "
          f"(distinct key/step): prior={lvl[0]} marginal={lvl[1]} "
          f"bucket={lvl[2]}")
    priors_note = {
        "pb1": "priors: C1 0.5 / Cr 0.15 / lethal 0.5 (by call site)",
        "pbt": "prior: 1.0",
        "pb2": "priors by n2 tier: 0.06 / 0.25 / 1.0",
    }
    for tab in ("pb1", "pb1m", "pbt", "pb2", "pb2m"):
        rows = sorted(agg[tab].items(),
                      key=lambda kv: -kv[1][0])[:14]
        print(f"  -- {tab} ({priors_note.get(tab, 'marginal')})")
        for k, (o, s) in rows:
            lo, hi = wilson(s, o)
            print(f"     key {k}: n={o} strikes={s} rate={s / max(1, o):.3f} "
                  f"Wilson90 [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
