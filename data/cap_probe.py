import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import eval_v53
for sb, sh, sa in ((99, 99, 99), (0.167, 0.25, 0.3), (0.25, 0.35, 0.4)):
    os.environ['SB'], os.environ['SH'], os.environ['SA'] = str(sb), str(sh), str(sa)
    os.environ['CS_STATS'] = f'/tmp/claude-501/probe_{sb}.jsonl'
    try: os.remove(os.environ['CS_STATS'])
    except OSError: pass
    import importlib
    r = eval_v53.play_game(('p', 'exp/cnet2_v7s_mfxc.py', 'exp/submission_cnet2_v13.py', 970500, 0))
    s = json.loads(open(os.environ['CS_STATS']).readline())
    print(sb, 'ms/step', r['a_ms_per_step'], 'worst', r['a_worst_step_ms'], 'margin', round(r['margin']),
          '| searched', s['searched'], 'rounds', s['rounds'], 'sims', s['sims'], 'changed', s['changed'], 'aborted', s.get('aborted', 0), flush=True)
