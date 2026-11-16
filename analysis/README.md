# Scout-line analysis products (filed 2026-11-16)

Tools live in `tools/`: `parse.py` (v1 parser), `parse2.py` (v2, superset of v1),
`role_pack.py` (worker role-mix pack builder), `analyze243.py` (Echo4 243-game redo).

## Provenance

- Echo4 corpus: DGX direct pull, 243 games. `data/echo4_matches.json` md5
  `a1905f0b1596aeaed9f86b65e67d96ab`; corpus manifest md5
  `79678f11cb0cef5dcec47e8f1a182628` (243 files, gzip BAD 0).
- Echo5 corpus: DGX direct pull, placement 29 games. matches md5
  `dfafd7b9e57361ff3bfc8e75e4679fa1`; corpus manifest md5 `b5bcecf8` (29 files).
- parse2 regression vs parse.py: 3 corpora (Echo4 243 + Echo4 51 + Echo5 29 =
  323 games), 11,305 shared fields compared, 0 mismatches, max float diff 0.0.
  The `.skips.json` files are empty (`[]`) — zero skipped games in all runs.

## Files

Echo4 (`analysis/echo4/`):
- `echo4_parsed243.json` — v1 parse of the full 243 games (loss-analysis basis).
- `echo4_parsed243_v2.json` + `.series.json.gz` — v2 parse + per-step series
  (both players x 300 steps x cash/trees/workers/cargo/planters/collectors).
- `echo4_parsed51.json`, `echo4_parsed51_v2.json` + series — the earlier
  51-game snapshot set (regression corpus).
- `echo4_rows243.json` — parsed243 joined with the matches list (validated
  0 mismatches on opponent/margin/outcome).
- `role_pack_echo4.json` — planter/collector step means, checkpoints, phase
  means, per-game role table (input to reports/ROLE_PACK_ANALYSIS_20261116.md).
- `mode5_window_localization.json` — per-game window-localization cuts for the
  7 mode-5 signature games (4 Echo4 + 3 Echo5; input to
  reports/MODE5_SUPPLEMENT_INTERIM_20261116.md).

Echo5 (`analysis/echo5/`):
- `echo5_parsed29.json`, `echo5_parsed29_annotated.json` — v1 parse of the
  29 placement games (+ first-move/match-id annotation).
- `echo5_parsed29_v2.json` + `.series.json.gz` — v2 parse + per-step series.
- `role_pack_echo5.json` — role-mix pack, same schema as Echo4's.

Metric definitions are identical between parse.py and parse2.py; the terminal
unsold-cargo definition (final-step worker carried amounts) is the one pinned
by the strategy group on 2026-11-15.
