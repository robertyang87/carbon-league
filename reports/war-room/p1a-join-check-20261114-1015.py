#!/usr/bin/env python3
"""P1 A段完工 join 检查（只读分析）。口径：作战室 09:26 既定口径。
- join rows(game,step) 与 first_terminal_shard0(game->ft)
- 零行：du == 0 的行（预期 0）
- step>=ft 行：应被聚合前剔除的 padding 行（预期 0）
- 另算：du 均值、sigma_game、step 三分带、wall/根耗时、sidecar 绑定。
"""
import json, math
from collections import Counter

BASE = '/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/'
res = json.load(open(BASE + 'endgame_p1a-20261114-1015.json'))
pilot = json.load(open(BASE + 'endgame_p1a-20261114-1015.pilot.json'))
ftmap = json.load(open(BASE + 'first_terminal_shard0-20261114-1015.json'))

rows = res['rows']
gate = res['du_gate']
prereg = res['prereg']

print('=== JOIN CHECK ===')
print('rows n =', len(rows), '| gate n =', gate.get('n'), '| n_games =', gate.get('n_games'))
games = [r['game'] for r in rows]
print('unique games =', len(set(games)))
print('game sample:', games[:3], '| game key type:', type(games[0]).__name__)
# resolve game -> ft key
def ft_of(g):
    if g in ftmap:
        return ftmap[g]['ft']
    for k in (f'selfplay-{g}', str(g)):
        if k in ftmap:
            return ftmap[k]['ft']
    return None

missing = [g for g in set(games) if ft_of(g) is None]
print('games missing ft =', len(missing), missing[:5])

bad = []   # step >= ft
for r in rows:
    ft = ft_of(r['game'])
    if ft is not None and r['step'] >= ft:
        bad.append((r['game'], r['step'], ft))
print('step>=ft rows =', len(bad), bad[:10])

zero = [r for r in rows if r['du'] == 0]
print('zero-du rows =', len(zero), 'share = %.4f' % (len(zero)/len(rows)))
zero_u = [r for r in rows if r['u_guided'] == 0 and r['u_greedy'] == 0]
print('zero-u rows (both 0) =', len(zero_u))
print('du==0 but u nonzero:', sum(1 for r in zero if r['u_guided'] != 0 or r['u_greedy'] != 0))
print('du min/max =', min(r['du'] for r in rows), max(r['du'] for r in rows))
print('step range =', min(r['step'] for r in rows), max(r['step'] for r in rows))
print('ft range =', min(v['ft'] for v in ftmap.values()), max(v['ft'] for v in ftmap.values()))
print('step<ft all =', all(r['step'] < ft_of(r['game']) for r in rows if ft_of(r['game']) is not None))

print()
print('=== DU STATS ===')
dus = [r['du'] for r in rows]
n = len(dus)
mean = sum(dus)/n
var = sum((x-mean)**2 for x in dus)/(n-1)
print('du_mean (recomputed) = %.6f | gate du_mean = %s | du_mean_roots = %s' % (mean, gate.get('du_mean'), gate.get('du_mean_roots')))
print('sigma_game (gate) = %s | sigma_root (gate) = %s | pilot sigma_game = %s' % (gate.get('sigma_game'), gate.get('sigma_root'), pilot.get('sigma_game')))
print('du_ci95_t_cluster =', gate.get('du_ci95_t_cluster'), '| t_exact =', gate.get('t_exact'))
print('du_ci95_boot_cluster =', gate.get('du_ci95_boot_cluster'))
print('verdict =', gate.get('verdict'))
print('stageB_N_games =', gate.get('stageB_N_games'), '| formula =', gate.get('stageB_N_formula'), '| raw =', gate.get('stageB_N_raw'))
# step thirds within [200,250)
thirds = [(200,217),(217,234),(234,251)]
for lo,hi in thirds:
    sub = [r['du'] for r in rows if lo <= r['step'] < hi]
    if sub:
        m = sum(sub)/len(sub)
        print('step [%d,%d): n=%d du_mean=%.6f' % (lo,hi,len(sub),m))
    else:
        print('step [%d,%d): n=0' % (lo,hi))

print()
print('=== TIMING ===')
wall = res['wall_s']
print('wall_s =', wall, '| wall/root = %.2f s (n=%d)' % (wall/len(rows), len(rows)))
print('teacher_ms_per_call =', gate.get('teacher_ms_per_call'), '| teacher_calls =', gate.get('teacher_calls'))
tc = sum(r['teacher_calls'] for r in rows); tm = sum(r['teacher_ms'] for r in rows)
print('rows teacher_calls total =', tc, '| teacher_ms total =', tm, '| ms/call = %.2f' % (tm/tc if tc else float('nan')))
print('rows teacher_calls per row: min/max =', min(r['teacher_calls'] for r in rows), max(r['teacher_calls'] for r in rows))

print()
print('=== SIDECAR BINDING ===')
print('gate pilot_artifact =', gate.get('pilot_artifact'))
print('pilot artifact_id   =', pilot.get('artifact_id'))
print('artifact match =', gate.get('pilot_artifact') == pilot.get('artifact_id'))
print('prereg run_id =', prereg.get('run_id'), '| pilot pilot_run_id =', pilot.get('pilot_run_id'), '| match =', prereg.get('run_id') == pilot.get('pilot_run_id'))
print('weights_md5 prereg =', prereg.get('weights_md5'))
print('weights_md5 pilot  =', pilot.get('weights_md5'))
print('weights match =', prereg.get('weights_md5') == pilot.get('weights_md5'))
print('code_md5 prereg =', prereg.get('code_md5'))
print('code_md5 pilot  =', pilot.get('code_md5'))
print('code match =', prereg.get('code_md5') == pilot.get('code_md5'))
print('pilot grade =', pilot.get('grade'), '| grade_reasons =', pilot.get('grade_reasons'))
print('pilot locked_N_games =', pilot.get('locked_N_games'), '| raw =', pilot.get('locked_N_games_raw'))
print('pilot_n_roots =', pilot.get('pilot_n_roots'), '| pilot_n_games =', pilot.get('pilot_n_games'))
print('prereg K/M/H =', prereg.get('K'), prereg.get('M'), prereg.get('H'), '| pilot K/M/H =', pilot.get('K'), pilot.get('M'), pilot.get('H'))
print('scene_window prereg =', prereg.get('scene_window'), '| pilot =', pilot.get('scene_window'))
print('ds_overlap_check =', pilot.get('ds_overlap_check'))

print()
print('=== GATE VERDICT ===')
ok_zero = len(zero) == 0
ok_step = len(bad) == 0
ok_join = len(missing) == 0
print('JOIN PASS' if (ok_zero and ok_step and ok_join) else 'JOIN FAIL',
      '| zero=%d step>=ft=%d missing_ft=%d' % (len(zero), len(bad), len(missing)))
