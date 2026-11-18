# Master family dissection — why they iterate fast, where the strength lives

Date: 2026-11-18. Object: Master3s2 published source, md5 57f9a594
(sources/master3s2/master3s2.py, 1,487 lines, weights embedded).
Read-only mechanism analysis per the exam-only registration — not a
teacher, not training input. Builds on reports/
MASTER3S2_SOURCE_ANALYSIS_20261116.md (§1–§3; that draft stops
mid-§3 — the endgame and iteration sections below are new, read from
the source this round). All numbers below were re-verified against
the file this round unless marked as from the prior analysis.

## 1. Network: small, and only a proposal machine

- Policy net **1,111,616 params** (fp16, verified by decoding the
  embedded npz): 54 hand-built input planes → stem 3×3 conv (96 ch,
  torus wrap) + sid embedding (121×96) → 6 residual blocks (96 ch,
  SE-style global-pool branch) → two 1×1 heads (worker 5, base 3).
  No attention, no recurrence, **no value head**.
- Critic is a separate **12,192-param** net: 1×1 conv 96→32, pooled
  mean+amax (64) + 5 scalars → 69→128→1; output = current margin/3000
  + MLP correction (linear, not tanh).
- The net is conditioned on **120 strategy identities** (snames):
  the author's own lineages (AgyCarbon v1–v21, Apex, Argon-Godslayer
  I–LXII; claude-opus v1–v71; TJB-o18 personas) plus ladder opponents.
  Deployment pins one persona (TJB-o18-rush-v41) and models the
  opponent as the same persona.
- Input is not the raw board: a ~300-line hand-written **belief
  tracker** infers the hidden carbon map (game-dynamics forward
  model with reverse-calibrated REGEN/CAP/ABS/COLLECT constants,
  symmetry-class inference, collision attribution), and 54 feature
  planes (v1 board / v2 per-action danger / v3 endgame ramps / v4
  reach-distance-seize-margin) are computed from it.

## 2. Search: every step, small, variance-engineered

- Every turn from step 0, hard wall **_S_ABORT = 0.58 s** (retuned
  from 0.62 on 11-12), plan capped at _S_FILL = 0.85 of remaining
  time; an online cost model (_S_COST: measured s/network-row and
  s/board, smoothed + last, _S_SAFE = 1.3) sizes the plan.
- Candidates ≤32, all inside the policy's own distribution: its own
  sampled move (always kept) + whole-joint-move samples at
  temperatures 2.5/1.0/3.5/5.0 + up to 6 single-worker deviations.
- Evaluation: one rollout per candidate (distilled small net inside
  rollouts, see §4), staged pruning (after 1 step keep 50%, after 4
  keep 25%, survivors to 8 steps), critic-scored in batches ≤32.
  Opponent root moves are sampled once per rollout index and
  **shared across all candidates (CRN)** — comparisons are paired,
  not independent. Switching threshold **_S_EPS = 0.0**: any strictly
  better 8-step critic mean replaces the policy move.
- The simulator is an embedded reimplementation of the official
  Board, verified byte-identical on 22,796 rollout boards (11-11);
  Winograd convs likewise verified before use.

## 3. Endgame: no solver, no tablebase — a machine cluster

There is **no exact endgame solver and no depth extension** (search
depth is 8 at every step). The ~2.1 residual trees come from a
cluster of small mechanisms:
- `go_home` hard rule: loaded collectors whose distance home fits
  the remaining turns (with a cargo-scaled spare) are forced home,
  choosing non-deadly closing moves; **search candidates may not
  override go-home moves (_S_HOMEFIX)**.
- Endgame feature planes 37/38 (ramps over the last 16 / 6 turns)
  and plane 41 ("cannot reach an own base before the end" boolean);
  _BANK = 12 (final-12-turn base stacking allowed).
- Terminal-shaped critic target: _terminal(m) = clip(m/3000, ±2.5)
  + 0.5·sign(m) whenever a rollout crosses the end.
- The go_home spare was patched 11-10 from a quantified loss
  post-mortem (Master2s vs 1C-Zero v10: 5,385 cargo aboard at T−15,
  3,468 never delivered; "online games left 258 a game").

## 4. Iteration fuel: the source names its own training loop

- Docstring: policy from **behaviour cloning + self-play RL**.
- `_S_REC` (line 995): every search records (step, player,
  candidates, values, the move played) — annotated **"for
  rl/search_label.py"**. The deployed search generates its own
  training labels: expert iteration, search → labels → retrain.
- `_S_ROLL` (line 1103): rollouts use a small distilled net
  (**"the distilled 64x4"**, built by **rl/distill_roll.py**,
  temperature-corrected for its flatter entropy 0.17 vs 0.08).
  Big net judges roots; a cheap student drives rollouts.
- Dated micro-patches in the comments: 11-04 v4 feature layer;
  11-10 go_home cargo-spare (loss post-mortem above); 11-11
  simulator + Winograd verification; 11-12 time-budget retune.
  Four shipped changes in eight days, three of them **not
  retraining at all** — a rule, a verification, a constant.
- Population self-play: 120 sid identities (§1) — dozens of own
  generations (Godslayer to LXII, opus to v71) are the training
  population, so each new version trains against its own ancestry.

Why it looks fast and cheap: the nets are tiny (1.1M + 12K), the
simulator is embedded and fast, inference is CPU numpy/torch inside
0.58 s/step on commodity cores, and most iterations are
measure-one-defect → patch-one-layer → redeploy. No large compute
is required anywhere in the loop; the compounding asset is the
codebase (belief tracker, features, verified simulator) and the
label flywheel, not FLOPs.

## 5. Verdict: where the strength lives

Not the network (small, and only proposes). The strength is a
**system**: (i) belief tracking = an information advantage from
step 0; (ii) hand-built tactical/endgame features — the net is
handed facts our nets must learn; (iii) a variance-engineered
short search (CRN pairing, staged pruning, zero-threshold switching
on 8-step means) that converts a mediocre proposal distribution
into strong play; (iv) the endgame rule/feature cluster; (v) the
iteration machinery (search labels + rollout distillation +
quantified loss post-mortems) that makes the other four compound.

## 6. What our KataGo loop can and cannot borrow

Mechanism-level only (§4 registration bounds; all code self-built):

**Borrowable**
1. **Search-label expert iteration** — record (state, candidates,
   search values, played) in searched self-play and distill into
   the student. This is the flywheel our loop lacks; it slots into
   the registered path next to the Step-3 leaf swap.
2. **Distilled rollout net** — big net at the root, Tier-C student
   (already §4-passed at 2.92 ms) inside rollouts. Their 64×4
   distill + temperature correction is the template.
3. **CRN shared opponent root moves + staged pruning** for our
   comparator/search: paired comparisons buy resolution cheaper
   than any threshold tuning (their _S_EPS = 0 works because the
   estimator is paired and averaged).
4. **Online-calibrated time management** (measured per-row/per-board
   costs, fill 0.85, safety 1.3, hard abort) — directly applicable
   to our 1.5-core timing discipline.
5. **Candidate protection for proven terminal rules** (HOMEFIX
   pattern): inside our own search, a verified endgame rule's moves
   are fixed and the search may not override them — as a mechanism,
   with our own rule and our own code.
6. **Process**: quantified loss post-mortem → single-layer patch →
   redeploy. Costs nothing, compounds.

**Not borrowable**
1. **The belief tracker itself** — 300 lines of reverse-engineered
   game-specific inference (symmetry classes, calibrated constants,
   collision attribution). The *idea* (explicit belief state as
   input) is borrowable; the machine is a multi-week build of our
   own if we ever choose it, and its constants are their work.
2. **The 54-plane stack as a whole** — four versions of accumulated
   feature engineering. Individual derived planes (danger,
   reachability) can be re-derived from the public game rules by
   our own analysis; the stack cannot be transcribed.
3. **sid/persona conditioning on ladder identities** — needs their
   population and names; our self-play population does not map.
4. **Their weights, labels, or games as training input** — barred
   by the exam-only registration, unchanged by this analysis.
5. **Their iteration speed itself** — it rests on a mature verified
   codebase; we can adopt the pattern, not the maturity.
