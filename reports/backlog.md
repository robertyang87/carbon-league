# Backlog (agreed with owner, 2026-11-07)

## 1C-Zero-family held-out stand-in (local exam only)
- Trigger: a search/distillation candidate sits at +100..+200 paired vs v13 (undecidable), or ExIt round 1 is done and CPU is free.
- Build: re-implement the 1C-Zero-v5 architecture in torch (from the public inference source; verify logit parity with the NumPy original), warm-start from v5's public weights, fine-tune on 1C-Zero v8-preview + v9 + v10 ladder replays (~300 games; v7 only if more data is needed).
- Validation: hold out ~15 v10 games; must predict v10's moves clearly better than v5 itself; strength must be above v12.
- Use: HELD-OUT exam only. Never a training/generator opponent, never a search opponent-model component, never shipped (it contains another author's weights).

## Recruit decisions in search (mfhr)
- Only if the value head orders sibling recruit branches >= 10 percentage points better than the policy head (paired ranking test).

## Reviving time-limited search
- Only via lower fixed per-step overhead (skip opponent-model trunk on non-searched steps, s64 root re-decodes); at the platform-equivalent budget all three parameter cuts lost 700-1100/game.

## Hardware reality (2026-11-08)
- Only this Mac (10 cores, MPS, 32 GB). No DGX/Spark. Search-teacher self-play ≈ 60–80 games/h on 8 cores; pure-net games ≈ 11–15 s each per core.
- Step 0 (before any pipeline scaling): heavy teacher (≈3× sims, H=6, robust lower-quantile leaf over sampled opponent models) must reach ≥ +300 paired vs v13; otherwise fix the opponent model / search first, or rethink the direction.
- If step 0 passes: vectorize cnet2_feat (Python bottleneck) and batch many games per MPS forward.
