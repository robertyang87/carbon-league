# m2_bot -- whole-game rule bot fitted to RyanX-AIPowercore-M2 (2026-11-10)

No neural net. Play: `code/m2_bot.py` (`agent(obs, config)`, fresh state at step 0), wrapper `code/exp/m2bot.py`
(cfg override: env `M2BOT_CFG` json or `M2BOT_CFG_B64`). Trees: `code/m2_bot_ctrees.json` (collectors),
`code/m2_bot_ptrees.json` (planter target + base), `code/m2_bot_otrees.json` (option trees).
Fitting code in `code/m2fit/`: load.py (corpus -> games.pkl), extract_all.py, fit_all.py (collector stay/home/target),
fit_pb.py (planter target + base), extract_opt.py + fit_opt.py (option trees), feats.py / stage.py / policy.py (shared
feature + decision code used by both extractor and bot), diag.py / deaths.py / trees.py / events.py / rectime.py /
agree.py (diagnostics), push.sh / grid.sh / res.sh (host-4 runs). Data: 86 M2 games (M3/M4 not used for fitting).

## Decision structure
* base (each base, if cash>=30 and workers<10): tree b_recruit (d6) gives P(recruit), drawn with a per-game seeded RNG;
  tree b_ptype (d6) gives P(planter). Deterministic argmax (SAMPLE=0) was worse (-7515 vs -6358).
* collector, stage 1: on a carbon cell, tree c_stay (d8) on row features (car_here, cargo, step, dist to base, best
  carbon within 1/2/3/5/7, enemy collector distances, ...). Economic veto: never keep collecting next to enemy trees when
  the 10%/step absorption beats the 30% haul.
* collector, stage 2: tree c_home (d6) -> go to nearest own base (threshold HOME_P=0.3; 0.5 = imitation optimum but
  weaker: -7039 vs -5930). Forced home when steps left <= dist + 8.
* collector, stage 3: tree c_tgt (d8) scores every carbon cell within 7 steps (candidate formulation), argmax; two of
  our collectors never share a target.
* collector move: option tree c_opt (d8) scores stay + 4 moves (approach to target, enemy collectors with
  smaller/equal/larger cargo near the cell, enemy planters, trees, own units ...). Safety layer on top: never two own
  units in a cell, never onto an enemy base, penalty 10*(1+cargo/100) for cells an enemy collector with cargo <= ours can
  enter, ETW=2 penalty for loaded cars next to enemy trees; going home = Dijkstra path (threatened cell now: +30,
  threatened later: +2).
* planter: tree p_tgt (d8, mfit/extract_tgt candidate features) picks the next stay cell; option tree p_opt (d8) picks
  the move (it reproduces M2's planter hunting: M2 loses ~45 planters/game, almost all in planter-planter clashes).

## Held-out-by-game accuracy (GroupKFold 5 over 86 M2 games)
| decision | held-out | baseline |
|---|---|---|
| collector stay (on carbon cell) d8 | 0.811 | majority 0.524 |
| collector home (when not staying) d6 | 0.891 | majority 0.797 |
| collector target top-1 (moving to a carbon cell) d8 | 0.372 | - |
| collector pipeline action (stay/home/target -> approach) | 0.575 | majority 0.340 |
| collector option tree action | 0.683 | target-following 0.570, majority 0.319 |
| planter next-stay target top-1 d8 | 0.476 | - |
| planter option tree action | 0.591 | target-following 0.471, majority 0.221 |
| base recruit? (cash>=30, <10 workers) d6 | 0.738 | majority 0.683 |
| base planter-vs-collector d6 | 0.742 | majority 0.701 |
Playing code path fed M2 replay observations (m2fit/agree.py, belief carbon): collector action 0.684, planter 0.598;
belief vs full carbon makes no difference (0.686 / 0.600). So the bot executes what was fitted (no execution bug:
m2fit/trace.py shows deposits, correct UP/DOWN/LEFT/RIGHT mapping, both seats, 0 position mismatches).

## Strength history (host 4, bigeval, A = m2bot, vs exp/cnet2_dg5e5s57.py, seeds 998000+)
| version | games | wins | mean margin | key change |
|---|---|---|---|---|
| v1 pure imitation | 100 | 0 | -7825 | |
| v2a SAFE=1 ROUTE HM8 HOME_P .4 | 60 | 0 | -6943 | safety, routed home |
| v3a + ETW .5 | 60 | 0 | -6358 | |
| v4b SAFE=10 THRC=30 | 60 | 0 | -6269 | loaded deaths vs base_v53 2290 -> 610 (Dijkstra bug fixed) |
| v5a ETW=2 | 60 | 0 | -5930 | |
| v6a HOME_P=.3 (= DEFAULT) | 60 | 0 | -5899 | |
Tried and not better: TGTW (greedy rate blend), SIDEW (enemy-half bias), COMPW (avoid cells an enemy reaches first),
HYST (target hysteresis), NOBACK, LOCKW/PLOCKW (force target following), PLOPT=0, SEIZE_R (seize-first), MINC (keep
>= N collectors), SAFE2, STAY_P .4/.6, SAMPLE=0. Calibration: our own engineered base_v53 rule bot loses -5472/game
to dg5 (8 games) and m2bot v4b loses -1424 to base_v53 (8 games), so m2bot is roughly at hand-rule level.

## Diagnosis (why it is far weaker than the real M2)
* Collection share: vs dg5 we collect ~7100 and dg5 ~10400; in the real ladder M2 and our v17 each collect ~8800.
  Per collector-step: ours 3.7, dg5 4.9, M2 4.3 (vs its real opponents). Opening is the worst: cash at step 100
  ~700 vs dg5 ~2700 (real M2 1600), dep100 1320 vs 2620.
* Tree war: our trees live 24 steps and earn 1.8/step (dg5's: 37 steps, 2.6/step); tree income 1300 vs 3400
  (real M2 2900 vs our nets 1900).
* Per-step imitation errors compound: the option tree sometimes leaves a 80-100 carbon cell the stay tree chose
  (flees an adjacent equal-cargo enemy), targets flip (reverse-move rate 10-12% vs M2 5.7%).

## Final (DEFAULT cfg)
* vs dg5 (exp/cnet2_dg5e5s57.py), 100 games, seeds 998000+: 0 wins, mean -6029 (se 123); dep 5733 vs 10140,
  tree income 1351 vs 3446, death cargo 129 vs 56.
* vs v17d2 (exp/submission_cnet2_v17d2.py), 50 games: 0 wins, mean -6456 (se 174); dep 5647 vs 10332, tree 1327 vs 3633.

## Likeness vs M2's replays (M2: 86 corpus games; m2bot: 100 games vs dg5, collector stats from 8 local games)
| stat | M2 | m2bot |
|---|---|---|
| deposits | 8702 | 5733 |
| tree income | 2567 | 1351 |
| own trees at 50 / 100 / 150 / 200 / 250 | 5.1 / 3.1 / 4.0 / 4.8 / 4.7 | 2.4 / 3.8 / 3.7 / 3.3 / 3.4 |
| planters recruited | 45.6 | 42.2 |
| collectors recruited | 19.5 | 16.5 |
| plants / seizes | 29.1 / 12.8 | 18.0 / 19.4 (bigeval count) |
| cargo left at end | 32 | 269 |
| collectors alive (mean) | 7.2 | 6.7 |
| deposits per game / mean size | 76 / 116 | 69 / 87 |
| carbon under a collecting car | 45 | 38 |
| cargo lost to enemy trees / in deaths | 235 / 316 | 490 / 116 |
| planter deaths per game (almost all planter clashes) | 40-45 | ~45 |

## What would most improve it next
1. Collector targeting is the bottleneck (target top-1 only 0.37; car under stays 38 vs 45; per-step errors and
   flip-flops). A planner that values a whole trip (stay length, regen, competition with enemy cars, return) fitted to
   M2's realised trips, instead of per-step trees, is the next step; cheap per-step knobs all plateau near -6000.
2. Opening economy: cash at step 100 is 700 vs M2's 1600; M2 plants ~5 trees by step 50, we ~2.4 (planters hunt
   before planting, and cash is short).
3. Tree war: our trees live 24 steps (dg5's 37); protect own trees / seize young enemy trees with a learned rule.
