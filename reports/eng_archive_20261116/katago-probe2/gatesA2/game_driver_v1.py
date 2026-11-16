#!/usr/bin/env python3
"""Part 1 pre-field GATE 2 v2 (deployment-form feeding) + GATE 5 same-trip timing.

Re-registered caliber (strategy-group ruling 2026-11-15, option (a)):
the OLD gate-2 caliber (fresh agent per root, inherited from probe-2's
construction gate) is structurally void for a deployment pack -- agent()
takes t0 before all phase-1 machinery, the depth loop breaks at
0.45*BUDGET=0.315s, cold phase-1 on step-200+ observations takes seconds,
so depth==1 always and changed==0 by construction. v2 therefore feeds the
packs exactly as deployed:

  * per game, FRESH module instances of pack A (SEARCH=True, seat 0) and
    the original pack (SEARCH=False, seat 1); the pack-level agent()
    builds a new search agent at step 0 and keeps it for the whole game
    (persistent in-game, phase-1 incremental);
  * at every seat-0 step the SAME observation is also fed to a SHADOW:
    an independent instance of the original pack's search agent
    (make_search_agent, SEARCH=False conf), created fresh per game and
    fed every seat-0 observation in order (a parallel no-search player
    whose actions never touch the game). changed := A action != shadow
    action (canonical JSON) at roots in steps [200, 298];
  * per-root search stats are read as deltas of the pack agent's
    cumulative stats dict (searched / depth_sum / shallow / changed /
    errors), plus per-call wall + process-CPU for every A call (gate 5).

Games: FastCarbon, seeds 20270001+i (diagnostic domain only), serial,
one process. Landing per game (checkpoint / progress caliber):
  out/roots.jsonl   -- one line per A root (steps 200..298), appended
  out/timing.jsonl  -- one line per A call (all 300 steps), appended
  out/game_<i>.json -- per-game summary (counts, timing aggregates)
Progress = #game summaries + cumulative searched roots / TARGET.
Resume: completed games (summary present) are skipped; nothing replays.

Self-check gate after game 0 (driver stops and waits for a review file
out/REVIEW_OK before playing further games):
  (1) some searched A root has depth >= 3;
  (2) shadow identity on the no-search side: every root whose internal
      changed-delta is 0 must have changed_ext == False (A's untouched
      default move is reproduced by the shadow exactly);
  (3) no A errors and no shadow errors in game 0.
If the self-check verdict is FAIL the driver never proceeds.

Usage: game_driver.py <workdir>
Env: MAX_GAMES (default 10), TARGET_SEARCHED (default 400).
"""
import copy, hashlib, importlib.util, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK_A = os.path.join(HERE, 'echo3_search_pack.py')
PACK_O = os.path.join(HERE, 'echo3_ref.py')
MD5_A = 'ea20a2714209e1a58b3e435b41d943bb'
MD5_O = '7b7620499411a088cf4dc17c0e929bd7'
OUT = os.path.join(HERE, 'out')
FIRST_SEED = 20270001
WIN_LO, WIN_HI = 200, 298
MAX_GAMES = int(os.environ.get('MAX_GAMES', '10'))
TARGET = int(os.environ.get('TARGET_SEARCHED', '400'))
KEYS = ('searched', 'shallow', 'depth_sum', 'changed', 'errors', 'steps',
        'branches')


def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_pack(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_weights(m):
    import base64, zlib
    w = m._dequantize(zlib.decompress(base64.b64decode(m._B1)), m._Q1)
    w['meta'] = np.array(m._M1)
    return w


def snap(m):
    st = getattr(m.agent, 'last_stats', None) or {}
    return {k: int(st.get(k, 0)) for k in KEYS}


def canon(a):
    return json.dumps(a, sort_keys=True) if a is not None else None


def loadavg():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def play_game(gidx):
    seed = FIRST_SEED + gidx
    t0 = time.perf_counter()
    mA = load_pack(f'v2_g{gidx}_A', PACK_A)
    mO = load_pack(f'v2_g{gidx}_O', PACK_O)
    load_wall = time.perf_counter() - t0
    assert mA.CONF['SEARCH'] is True and mO.CONF['SEARCH'] is False
    shadow = mO._S.make_search_agent(dict(mO.CONF), weights=load_weights(mO),
                                     rweights=None, oweights=None)
    FC = mA._P['fastsim'].FastCarbon
    sim = FC(); sim.reset(seed=seed)
    roots, timing = [], []
    sum_shadow_wall = 0.0
    first_call_wall = None
    game_t0 = time.perf_counter()
    for step in range(300):
        obs0 = sim.obs(0)
        s0 = snap(mA)
        w0 = time.perf_counter(); c0 = time.process_time()
        a0 = mA.agent(obs0, None)
        w1 = time.perf_counter(); c1 = time.process_time()
        s1 = snap(mA)
        d = {k: s1[k] - s0[k] for k in KEYS}
        if first_call_wall is None:
            first_call_wall = w1 - w0
        ws0 = time.perf_counter()
        try:
            a0s = shadow(copy.deepcopy(obs0)); shadow_err = None
        except Exception as e:
            a0s, shadow_err = None, repr(e)[:200]
        ws1 = time.perf_counter()
        sum_shadow_wall += ws1 - ws0
        timing.append({'game': gidx, 'step': step, 'wall': round(w1 - w0, 4),
                       'cpu': round(c1 - c0, 4), 'searched': d['searched']})
        if WIN_LO <= step <= WIN_HI:
            st = getattr(mA.agent, 'last_stats', {}) or {}
            roots.append({
                'game': gidx, 'seed': seed, 'step': step, 'seat': 0,
                'wall': round(w1 - w0, 4), 'cpu': round(c1 - c0, 4),
                'searched': d['searched'],
                'depth': d['depth_sum'] if d['searched'] else 0,
                'shallow': d['shallow'], 'changed_int': d['changed'],
                'errors': d['errors'],
                'changed_ext': (canon(a0) != canon(a0s))
                               if shadow_err is None else None,
                'shadow_err': shadow_err,
                'shadow_wall': round(ws1 - ws0, 4),
                'err_text': st.get('last_error') if d['errors'] else None})
        obs1 = sim.obs(1)
        a1 = mO.agent(obs1, None)
        sim.step(a0, a1)
    game_wall = time.perf_counter() - game_t0
    searched = sum(r['searched'] for r in roots)
    summary = {
        'game': gidx, 'seed': seed, 'load_wall': round(load_wall, 3),
        'first_call_wall': round(first_call_wall, 4),
        'game_wall_loop': round(game_wall, 2),
        'game_wall_deploy': round(game_wall - sum_shadow_wall, 2),
        'shadow_wall_total': round(sum_shadow_wall, 2),
        'roots': len(roots), 'searched': searched,
        'changed_ext': sum(1 for r in roots if r['changed_ext']),
        'changed_int': sum(r['changed_int'] for r in roots),
        'errors': sum(r['errors'] for r in roots),
        'shadow_errors': sum(1 for r in roots if r['shadow_err']),
        'loadavg_end': loadavg()}
    with open(os.path.join(OUT, 'roots.jsonl'), 'a') as f:
        for r in roots:
            f.write(json.dumps(r) + '\n')
    with open(os.path.join(OUT, 'timing.jsonl'), 'a') as f:
        for r in timing:
            f.write(json.dumps(r) + '\n')
    with open(os.path.join(OUT, f'game_{gidx}.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    return summary


def selfcheck():
    roots = [json.loads(l) for l in open(os.path.join(OUT, 'roots.jsonl'))]
    g0 = [r for r in roots if r['game'] == 0]
    searched = [r for r in g0 if r['searched']]
    ident_viol = [r for r in g0
                  if r['changed_int'] == 0 and r['changed_ext'] is True]
    dir2 = [r for r in g0
            if r['changed_int'] > 0 and r['changed_ext'] is False]
    sc = {'game0_roots': len(g0), 'game0_searched': len(searched),
          'max_depth': max((r['depth'] for r in searched), default=0),
          'depth_ge3': sum(1 for r in searched if r['depth'] >= 3),
          'identity_violations': len(ident_viol),
          'dir2_mismatch': len(dir2),
          'errors': sum(r['errors'] for r in g0),
          'shadow_errors': sum(1 for r in g0 if r['shadow_err'])}
    sc['verdict'] = ('PASS' if sc['depth_ge3'] > 0
                     and sc['identity_violations'] == 0
                     and sc['errors'] == 0 and sc['shadow_errors'] == 0
                     else 'FAIL')
    with open(os.path.join(OUT, 'selfcheck.json'), 'w') as f:
        json.dump(sc, f, indent=1)
    return sc


def main():
    assert md5_of(PACK_A) == MD5_A, 'pack A md5 mismatch'
    assert md5_of(PACK_O) == MD5_O, 'pack orig md5 mismatch'
    os.makedirs(OUT, exist_ok=True)
    print(f'packs md5 ok; loadavg start {loadavg()}', flush=True)
    done = {}
    for i in range(MAX_GAMES):
        p = os.path.join(OUT, f'game_{i}.json')
        if os.path.exists(p):
            done[i] = json.load(open(p))
    cum = sum(s['searched'] for s in done.values())
    print(f'resumed: {len(done)} games landed, searched cum {cum}/{TARGET}',
          flush=True)
    sc_path = os.path.join(OUT, 'selfcheck.json')
    if 0 in done and not os.path.exists(sc_path):
        print(f'SELFCHECK (on resume) {json.dumps(selfcheck())}', flush=True)
    if os.path.exists(sc_path):
        verdict = json.load(open(sc_path))['verdict']
        if verdict != 'PASS':
            print(f'selfcheck verdict {verdict} -- refusing to proceed',
                  flush=True)
            return
    played = len(done)
    for i in range(MAX_GAMES):
        if cum >= TARGET:
            break
        if i in done:
            continue
        if i > 0 or done:
            if not os.path.exists(os.path.join(OUT, 'REVIEW_OK')):
                print('game 0 landed; waiting for REVIEW_OK', flush=True)
                return
        s = play_game(i)
        played += 1
        cum += s['searched']
        print(f"GAME {i} DONE seed={s['seed']} searched={s['searched']} "
              f"cum={cum}/{TARGET} changed_ext={s['changed_ext']} "
              f"deploy_wall={s['game_wall_deploy']}s", flush=True)
        if i == 0:
            sc = selfcheck()
            print(f'SELFCHECK {json.dumps(sc)}', flush=True)
            if sc['verdict'] != 'PASS':
                print('SELFCHECK FAIL -- stopping, do not review-ok', flush=True)
                return
            if not os.path.exists(os.path.join(OUT, 'REVIEW_OK')):
                print('selfcheck PASS; waiting for REVIEW_OK', flush=True)
                return
    print(f'ALL DONE games_landed={played} searched_cum={cum}', flush=True)


if __name__ == '__main__':
    main()
