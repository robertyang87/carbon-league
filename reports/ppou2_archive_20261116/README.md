# ppou2 (L20 line) archive drop — 2026-11-16

Snapshot taken 2026-11-16 ~12:16 CST from gpu-l20-6, while training and the
adj1 signature measurement continued uninterrupted. Only files already landed
on disk are included.

## Contents

- `train.jsonl` — ppou2_m training log, rounds it=1..52 complete (52 lines).
  Snapshot of the live file (state as of it=52 landing, 12:07 CST).
  md5 `69f2ca1229f61ae98224106b610fffd8`.
- `league.json` — league table snapshot at the same moment.
  md5 `69912ea95fcb6e7ed494026e5b7c2a80`.
- `launch_ppou2_v7.py` — production launcher v7 (dual-server config).
  md5 `22d0f6bd6ea3c1178377cea572f684b2`.
- `kl/` — cumulative KL vs the frozen start (cnet2_m3s2r1L), rootset N=6336,
  per-slot categorical KL, direction KL(pi_ckpt || pi_start):
  it=20 0.0241 / it=25 0.0188 / it=30 0.0270 / it=35 0.0363 / it=37 0.0382 /
  it=40 0.0499 / it=45 0.0614 / it=50 0.0681 (mean; p99/max in the JSONs).
  Plus the measure script (`remote_kl_cum_m_it50.py`, same template for every
  checkpoint), `pace_read.py` (md5 `688efd4ac6a83e08391b2ea2ff568dcc`) and
  `kl_watcher.py` (md5 `14d837d9f794f44bfe5cd7d6cbd7fc00`).
- `adj1_sig/` — ppou2_adj1 Tier-1 signature measurement, **partial snapshot**:
  29 of 50 chunks landed at snapshot time (28 chunks x 200 games = 5,600 games,
  plus the 8-game smoke file `smoke_cand.jsonl`); the measurement was still
  running and the remainder follows with the Tier-1 aggregation.
  Diagnostic fixed set: seeds 20291101-20292100, one game per seed,
  seat = seed % 2, opponent Echo4.
  Scripts: `signature_measure.py` (md5 `5c4ea68e5e41deef3b3ec79823cf88ff`),
  `run_sig_adj1.sh` (md5 `975d9fb3bf8390abbef89bb754302d1e`).

## Checkpoint manifest

Weights stay on gpu-l20-6 under `/root/carbon/data/rl3/ppou2_m/` (not in git).
Each `ckpt_*.npz` is 13,972,532 B; each `state_frozen_*.pt` is 82,600,176 B.

| it | ckpt npz md5 | frozen state md5 | pack | pack md5 |
|----|--------------|------------------|------|----------|
| 37 | `9d6257a82111ada4441cd1be36db790e` | `state_frozen_it37.pt` (see run dir) | — | — |
| 40 | `72e2338fb2cd50c2f29c76ca52e78526` | `73f649aa98993f783396fa91e88554f6` | `code/packs/ppou2_m_it40.py` | `5460a318` (already in repo) |
| 45 | `43e166d486e5a1d790bfee2da683fe92` | `7af8a490e8ab09f608163ddd2c4b4716` | `code/packs/ppou2_m_it45.py` | `c89e57275c7cb37749dc1b24332393bd` |
| 50 | `ed604ac800e0ad7ab60440c6e9f72872` | `23bf24a84c79d8f204cabd729825f222` | `code/packs/ppou2_m_it50.py` | `c35305af7f8a7bc58afb9cfb4918518c` |

ckpt_37 is the current KL anchor (resume of 2026-11-16 04:49 CST).
Anchor-qualified checkpoints on record: ckpt_40, ckpt_45, ckpt_50
(see `reports/PPOU2_IT50_EXAM_DISPOSITION_20261116.md`).

## State at snapshot

Main arm running on gpu-l20-6 (PGID 368291), it=52 landed 12:07 CST,
KL vs anchor ckpt_37 = 0.1071 (per-anchor cap 0.15). it=50 checkpoint exam:
candidate U 0.5739 vs Echo4 — visible stage gain, anchor-qualified.
