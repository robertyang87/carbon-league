#!/usr/bin/env python3
"""Self-test 2 (gate-4 caliber): leaf-level terminal cases + value-leaf wiring.
States are built from the Part 1 synthetic boundary roots (SYNTH-BND777, steps
296-300, both seats alive at 300) loaded into the VARIANT's own embedded
FastCarbon. leaf/_terminal/_extinct are taken from the built agents' closures
(the shipped bytes themselves, no twin). Cases:
  T1 step-300 root states, both alive -> leaf == exact cash difference (arms A/B)
  T2 steps 297/298/299 root states -> _terminal False; arm A leaf == A_AFF*z+B_AFF
     with z recomputed independently through the pack's own encode path (wiring);
     z also cross-checked against the fp-weights pipeline (Part 2 caliber)
  T3 step-297 state played forward with the ORIGINAL pack (SEARCH=False
     instrument) to step_count 300 -> leaf == exact final cash difference
  T4 extinction mutation (side 1 wiped) -> leaf == F01 scoring, independent of
     the extinct side's cash
  T5 extinction + step_count=300 -> F01 takes precedence over the cash difference
  T6 step_count set to 300 on an alive state -> exact cash difference;
     step_count 299 -> not terminal
Contrast values from the unmodified base-A pack leaf are reported, not asserted.
"""
import base64, hashlib, importlib.util, json, sys, zlib
import numpy as np

A_AFF = 3687.9177926610573
B_AFF = -638.5907028300794
CASES = [json.loads(l) for l in open('/home/hatch/workspace/katago-probe2/boundary_cases.jsonl')]
PACKS = {'orig': '/home/hatch/workspace/katago-probe/impl/echo3_ref.py',
         'baseA': '/home/hatch/workspace/katago-probe2/modules/echo3_search_pack.py',
         'A': '/home/hatch/workspace/katago-probe2/part3/modules/echo3_part3_a_pack.py',
         'B': '/home/hatch/workspace/katago-probe2/part3/modules/echo3_part3_b_pack.py'}
ok = []
def chk(cond, msg):
    ok.append((bool(cond), msg))
    print(('PASS' if cond else 'FAIL'), '-', msg)
    return cond

def load_pack(name, path):
    spec = importlib.util.spec_from_file_location(f'pack3l_{name}', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    w1 = m._dequantize(zlib.decompress(base64.b64decode(m._B1)), m._Q1)
    w1['meta'] = np.array(m._M1)
    return m, w1

def cells(fn):
    return {n: c.cell_contents for n, c in zip(fn.__code__.co_freevars, fn.__closure__ or ())}

def grab(agent_fn, *names):
    # BFS the closure graph for the function whose cells hold 'leaf'
    seen, stack, owner = set(), [agent_fn], None
    while stack:
        fn = stack.pop()
        if id(fn) in seen or not hasattr(fn, '__code__'):
            continue
        seen.add(id(fn))
        c = cells(fn)
        if 'leaf' in c:
            owner = c
            break
        stack.extend(v for v in c.values() if hasattr(v, '__code__'))
    assert owner is not None, 'leaf not found in closure graph'
    lc = cells(owner['leaf'])
    out = {}
    for n in names:
        out[n] = lc[n] if n in lc else owner[n]
    return out

def build_sim(pack_mod, case):
    obs = case['obs']
    full = {'step': obs['step'], 'players': obs['players'], 'trees': obs.get('trees') or {},
            'full_carbon': [float(v) for v in obs['carbon']], 'carbon': obs['carbon'],
            'remainingOverageTime': 60}
    sim = pack_mod._P['fastsim'].FastCarbon()
    sim.load_observation(full)
    return sim

def lft_of(sim):
    return max(0, 299 - sim.step_count)

results = {}
mods, ws, agents, L = {}, {}, {}, {}
for name in ('A', 'B'):
    mods[name], ws[name] = load_pack(name, PACKS[name])
    agents[name] = mods[name]._S.make_search_agent(mods[name].CONF, weights=ws[name])
    want = ('leaf', '_terminal', '_extinct', 'pol', 'st', 'cfg', 'T') \
        if name == 'A' else ('leaf', '_terminal', '_extinct')
    L[name] = grab(agents[name], *want)
    if name == 'A':   # module-level helpers are globals of the embedded module, not cells
        for fn_name in ('feat_copy', 'encode_b', 'stack_x'):
            L[name][fn_name] = getattr(mods[name]._S, fn_name)
mods['baseA'], ws['baseA'] = load_pack('baseA', PACKS['baseA'])
agents['baseA'] = mods['baseA']._S.make_search_agent(mods['baseA'].CONF, weights=ws['baseA'])
L['baseA'] = grab(agents['baseA'], 'leaf')
mo, wo = load_pack('orig', PACKS['orig'])

case = {(c['step'], c['seat']): c for c in CASES}
steps = sorted({c['step'] for c in CASES})
print('boundary steps available:', steps, '| sim.config keys sample:',
      sorted(L['A']['cfg'] and build_sim(mods['A'], case[(296, 0)]).config.keys())[:8], '...')
sim296 = build_sim(mods['A'], case[(296, 0)])
print('episodeSteps in sim.config:', sim296.config.get('episodeSteps', '<absent -> default 300 used>'))

# ---------- T1: step-300 roots, exact cash difference ----------
for seat in (0, 1):
    c = case[(300, seat)]
    for arm in ('A', 'B'):
        sim = build_sim(mods[arm], c)
        assert sim.step_count == 300
        exact = float(sim.cash[seat]) - float(sim.cash[1 - seat])
        v = L[arm]['leaf'](sim, seat, 0, None)
        chk(abs(v - exact) < 1e-9, f'T1 arm {arm} seat {seat}: leaf(step300)={v:.4f} == exact cash diff {exact:.4f}')
        chk(L[arm]['_terminal'](sim), f'T1 arm {arm} seat {seat}: _terminal(step300) is True')
    sim_s = build_sim(mods['baseA'], c)
    v_s = L['baseA']['leaf'](sim_s, seat, 0)
    print(f'   contrast base-A pack leaf(step300, seat {seat}) = {v_s:.4f} (exact {exact:.4f}, delta {v_s - exact:+.4f})')
    results[f'T1_seat{seat}'] = {'exact': exact, 'baseA_leaf': v_s}

# ---------- T2 + wiring: steps 297/298/299 not terminal; arm A leaf == a*z+b ----------
# fp pipeline (Part 2 caliber): standalone modules + fp npz weights, teacher pinned 2
sys.path.insert(0, '/home/hatch/workspace/katago-probe/impl/modules')
import cnet2_search as S1
def load_npz(p):
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}
fp_w = load_npz('/home/hatch/workspace/katago-probe/impl/smoke_assets/cnet2_m3s2r1L.npz')
s64_w = load_npz('/home/hatch/workspace/katago-probe/impl/smoke_assets/cnet2_s64.npz')
ag_fp = S1.make_search_agent({'SEARCH': False, 'TEACHER': 2}, weights=fp_w, rweights=s64_w)
pol_fp = grab(ag_fp, 'pol')['pol']
F_fp = sys.modules['cnet2_feat']

for step in (297, 298, 299):
    for me in (0, 1):
        c = case[(step, me)]
        sim = build_sim(mods['A'], c)
        assert sim.step_count == step
        chk(not L['A']['_terminal'](sim), f'T2 step {step} me {me}: _terminal False (both alive, below limit)')
        # branch Feat rooted at this root obs (mirrors _search: fs[i][0] = feat_copy(f), f observed the root)
        F = mods['A']._P['cnet2_feat']
        fm = F.Feat(me); fm.observe(c['obs'])
        # independent z through the pack's own encode path
        fc = L['A']['feat_copy'](fm)
        xm = fc.observe(sim.obs(me))
        ctx = L['A']['encode_b'](L['A']['pol'], L['A']['stack_x']([xm]),
                                 np.array([L['A']['st'].get('T_now', L['A']['T'])]))[1][0]
        z_ref = float(ctx @ L['A']['pol'].w['value.weightT'][:, 0] + L['A']['pol'].w['value.bias'][0])
        v = L['A']['leaf'](sim, me, lft_of(sim), fm)
        chk(abs(v - (A_AFF * z_ref + B_AFF)) < 1e-6,
            f'T2/wiring step {step} me {me}: arm A leaf {v:.6f} == a*z+b (z={z_ref:.6f})')
        # B arm: finite, non-terminal path
        sim_b = build_sim(mods['B'], c)
        fm_b = mods['B']._P['cnet2_feat'].Feat(me); fm_b.observe(c['obs'])
        v_b = L['B']['leaf'](sim_b, me, lft_of(sim_b), fm_b)
        chk(np.isfinite(v_b), f'T2 step {step} me {me}: arm B leaf finite ({v_b:.4f})')
        # fp cross-check of z (Part 2 pipeline weights, teacher 2)
        fm_f = F_fp.Feat(me); fm_f.observe(c['obs'])
        fc_f = S1.feat_copy(fm_f)
        xm_f = fc_f.observe(sim.obs(me))
        ctx_f = S1.encode_b(pol_fp, S1.stack_x([xm_f]), np.array([2]))[1][0]
        z_fp = float(ctx_f @ pol_fp.w['value.weightT'][:, 0] + pol_fp.w['value.bias'][0])
        dz = z_ref - z_fp
        print(f'   z step {step} me {me}: pack(int8) {z_ref:+.6f} | fp {z_fp:+.6f} | dz {dz:+.2e} | dM_eq {A_AFF * dz:+.2f}')
        results[f'z_step{step}_me{me}'] = {'z_pack': z_ref, 'z_fp': z_fp, 'dz': dz}

# ---------- T3: play a step-297 state forward to 300 with the orig pack ----------
c = case[(297, 0)]
sim = build_sim(mo, c)
orig_agents = [mo._S.make_search_agent(mo.CONF, weights=wo) for _ in range(2)]
np.random.seed(777)
while sim.step_count < 300:
    a0 = orig_agents[0](sim.obs(0)); a1 = orig_agents[1](sim.obs(1))
    sim.step(a0, a1)
assert sim.step_count == 300
for arm in ('A', 'B'):
    for me in (0, 1):
        exact = float(sim.cash[me]) - float(sim.cash[1 - me])
        v = L[arm]['leaf'](sim, me, 0, None)
        chk(abs(v - exact) < 1e-9, f'T3 arm {arm} me {me}: leaf(played-to-300)={v:.4f} == exact final cash diff {exact:.4f}')
results['T3'] = {'final_cash': [float(sim.cash[0]), float(sim.cash[1])]}

# ---------- T4/T5/T6: mutations on the step-296 state ----------
def wipe_side1(sim):
    sim.workers = {k: w for k, w in sim.workers.items() if w.player != 1}
    sim.trees = {k: t for k, t in sim.trees.items() if t.player != 1}
    sim.bases[1] = type(sim.bases[1])()
    return sim

for arm in ('A', 'B'):
    sim = wipe_side1(build_sim(mods[arm], case[(296, 0)]))
    chk(L[arm]['_extinct'](sim, 1) and not L[arm]['_extinct'](sim, 0) and L[arm]['_terminal'](sim),
        f'T4 arm {arm}: wiped side 1 is extinct, state terminal')
    pen = float(sim.step_count) - 300.0 - 1.0
    exp0 = float(sim.cash[0]) - pen
    v0 = L[arm]['leaf'](sim, 0, lft_of(sim), None)
    v1 = L[arm]['leaf'](sim, 1, lft_of(sim), None)
    chk(abs(v0 - exp0) < 1e-9 and abs(v1 + exp0) < 1e-9,
        f'T4 arm {arm}: F01 leaf me0 {v0:.4f} == cash0-pen {exp0:.4f}; me1 negated')
    cash1 = float(sim.cash[1]); sim.cash[1] = cash1 + 5000.0
    chk(abs(L[arm]['leaf'](sim, 0, lft_of(sim), None) - exp0) < 1e-9,
        f'T4 arm {arm}: F01 leaf independent of extinct side cash (+5000 no change)')
    sim.cash[1] = cash1
    sim.step_count = 300
    pen300 = 300.0 - 300.0 - 1.0
    exp300 = float(sim.cash[0]) - pen300
    v300 = L[arm]['leaf'](sim, 0, 0, None)
    chk(abs(v300 - exp300) < 1e-9 and abs(v300 - (float(sim.cash[0]) - float(sim.cash[1]))) > 1e-6,
        f'T5 arm {arm}: extinct at step 300 -> F01 {v300:.4f} (cash diff would be {float(sim.cash[0]) - float(sim.cash[1]):.4f})')
    sim2 = build_sim(mods[arm], case[(296, 0)])
    sim2.step_count = 300
    exact = float(sim2.cash[0]) - float(sim2.cash[1])
    chk(abs(L[arm]['leaf'](sim2, 0, 0, None) - exact) < 1e-9,
        f'T6 arm {arm}: alive state with step_count=300 -> exact cash diff {exact:.4f}')
    sim2.step_count = 299
    chk(not L[arm]['_terminal'](sim2), f'T6 arm {arm}: alive state with step_count=299 -> not terminal')

# ---------- T7: NON-DEGENERATE step-limit evidence ----------
# The synthetic boundary game is perfectly mirrored (identical deterministic
# agents, symmetric start): cash[0]==cash[1] at every recorded step, so the
# T1/T3/T6 exact differences are 0 by construction. Inject asymmetric cash so
# the step-limit branch is verified at nonzero values.
for arm in ('A', 'B'):
    sim7 = build_sim(mods[arm], case[(296, 0)])
    assert not L[arm]['_extinct'](sim7, 0) and not L[arm]['_extinct'](sim7, 1)
    sim7.cash = [5000.0, 3200.0]
    sim7.step_count = 300
    v0 = L[arm]['leaf'](sim7, 0, 0, None); v1 = L[arm]['leaf'](sim7, 1, 0, None)
    chk(abs(v0 - 1800.0) < 1e-9 and abs(v1 + 1800.0) < 1e-9,
        f'T7 arm {arm}: step-limit leaf with cash [5000,3200] -> +1800/-1800 exact (got {v0:.4f}/{v1:.4f})')
    sim7.step_count = 299
    chk(not L[arm]['_terminal'](sim7), f'T7 arm {arm}: same state at step_count=299 -> not terminal')
    sim8 = build_sim(mods[arm], case[(299, 0)])
    sim8.cash = [5000.0, 3200.0]
    np.random.seed(777)
    a0 = orig_agents[0](sim8.obs(0)); a1 = orig_agents[1](sim8.obs(1))
    sim8.step(a0, a1)
    assert sim8.step_count == 300
    expf = float(sim8.cash[0]) - float(sim8.cash[1])
    vf0 = L[arm]['leaf'](sim8, 0, 0, None); vf1 = L[arm]['leaf'](sim8, 1, 0, None)
    chk(abs(vf0 - expf) < 1e-9 and abs(vf1 + expf) < 1e-9,
        f'T7 arm {arm}: played-forward asymmetric game ends at exact final diff {expf:.4f} (got {vf0:.4f})')
    results[f'T7_{arm}'] = {'injected_diff': 1800.0, 'played_final_diff': expf}

print('\n%d/%d leaf checks passed' % (sum(1 for c_, _ in ok if c_), len(ok)))
json.dump(results, open('/home/hatch/workspace/katago-probe2/part3/data/leaf_tests.json', 'w'), indent=1)
sys.exit(0 if all(c_ for c_, _ in ok) else 1)
