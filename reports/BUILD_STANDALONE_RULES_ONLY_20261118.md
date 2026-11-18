# Standalone morphology package: pure Echo4 actor + rule layer, no search

Date: 2026-11-18. Order: 作战室 2026-11-18 20:06 (registration
f0a512764c §5: the independent morphology — pure Echo4 + rule layer,
no search — screened separately per the 11-18 schedule; F3's
with-search increment does not substitute). Execution seat rerouted
20:30 from C1 spot-34 (confirmed gone) to Studio after budget match
A (~20:50); delivery DGX LAN direct transfer, Studio self-reads.

## Package

- File: `code/packs/echo4_int8_homefix.py`
- md5 **c44f57720c89000d4eed09474b24d7f3**, 4,608,334 B (≤ 10,000,000 ✓)
- Anchor: frozen Echo4 baseline pack `code/packs/echo4_int8.py`,
  md5 819250bfc82632786063dabd46a616bd (CONF SEARCH=False, unchanged)
- Rule layer: `cnet2_homefix`, md5 9bb454703c830feb061821ded3c4e289 —
  the same pinned module as the composite package (25f144f3),
  HOME_BUFFER=2 hard-pinned, no CONF keys
- Patched embedded search md5 a91f1fdd34529a8e363965086025160c
  (+29/−3 lines vs the baseline's embedded search: header, HF
  import, three return hooks)
- Method: anchored text surgery on the baseline pack (generator
  `homefix_standalone/make_standalone_homefix.py`); repr() round-trip
  check proves byte-faithful re-serialization of untouched modules.
  Import anchor adapted: the baseline pack has no comparator module,
  so HF is imported after `from fastsim import FastCarbon`. CONF,
  weights (_B1/_Q1/_M1) byte-identical to baseline. Full build record:
  `homefix_standalone/build_record_standalone.json`.

## Verification (all PASS)

- V1: 5 embedded sources md5-match _BUILD_HASHES.
- V2: cnet2_feat / cnet2_np / fastsim md5s equal the baseline pack's.
- V3: weights + CONF byte-identical; CONF SEARCH=False confirmed.
- V4: pack lines differing from baseline: exactly [8 (_SRC), 9 (_MODS),
  10 (_BUILD_HASHES)].
- V5: 15/15 synthetic homefix cases against the module extracted
  from THIS pack.
- Timing (this VM): mean 962 ms, p95 1,575 ms, max 2,728 ms per step —
  inside the 3 s platform step cap (same actor code as baseline; the
  rule is a microsecond pure function).

## Diagnostic equivalence (seed 7000, FastCarbon self-play)

- Game A (rules-only seat 0 vs baseline seat 1): 300 steps,
  cash [4310.55, 4280.77]; rule checked 1,478 / triggered 4 /
  changed 1; first trigger step 296 (worker-10, cargo 29.781, d=2,
  moves 4, UP → RIGHT).
- Control game B (baseline vs baseline): 300 steps, cash
  [4280.77, 4280.77]. Seat-0 streams identical for **296 steps**;
  first difference at step 296 is exactly the rule override
  (worker-10 UP → RIGHT). Cash cross-check: game A seat 0 exceeds
  baseline by 29.78 — exactly the rescued cargo; the rule's single
  changed action delivered that load and touched nothing else.
- **Harness bug, disclosed**: the first probe's game B was invalid
  and is discarded — both seats were played through ONE imported
  baseline module, but a pack module caches a single _AG whose belief
  trackers (st["f"]/st["fo"]) and guard are per-seat state, so seat 1
  polluted seat 0 from step 4 (symmetric cash 2932.37 = 2932.37 was
  the fingerprint). The rerun loads the baseline twice under
  separate module names. Lesson: a pack module serves one seat only;
  a two-seat local probe must import it twice.

## Sieve purpose

Standalone (actor+rules) vs baseline (actor) = the rule layer's
independent increment; standalone vs composite (actor+rules+search,
25f144f3) = search's increment. Pairs with the Master dissection
(4513e9c7): HOMEFIX-style rules are the residual-tree-2.1 candidate,
and this sieve answers how much the rules alone are worth.
