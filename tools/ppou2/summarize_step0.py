import json, math

def load(p):
    rows = {}
    errs = 0
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            rows[r['seed']] = r
            if r.get('err') is not None or r.get('berr'):
                errs += 1
    return rows, errs

def U(m):
    w = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-m / 1000.0))

def win(r):
    m = r['margin']
    return 1 if m > 0 else (0 if m < 0 else 0.5)

def summ(rows, label):
    n = len(rows)
    u = sum(U(r['margin']) for r in rows.values()) / n
    w = sum(win(r) for r in rows.values())
    mm = sum(r['margin'] for r in rows.values()) / n
    print(f"{label}: n={n} wins={w} mean_margin={mm:+.1f} meanU={u:.4f}")
    return u

base = '/home/hatch/workspace/carbon-live/step0_results'
ma, ea = load(f'{base}/ppou2_base_m2_a.jsonl')
mb, eb = load(f'{base}/ppou2_base_m2_b.jsonl')
e2, ee = load(f'{base}/ppou2_base_e2_a.jsonl')
print(f"errors: m2a={ea} m2b={eb} e2a={ee}")
ua = summ(ma, 'M2 batch A (980000-981999)')
ub = summ(mb, 'M2 batch B (980000-981999)')
summ(e2, 'E2 batch A (982000-983999)')
common = sorted(set(ma) & set(mb))
ident = sum(1 for s in common if ma[s]['margin'] == mb[s]['margin'])
flips = sum(1 for s in common if win(ma[s]) != win(mb[s]))
dU = sum(U(mb[s]['margin']) - U(ma[s]['margin']) for s in common) / len(common)
maxd = max(abs(ma[s]['margin'] - mb[s]['margin']) for s in common)
print(f"M2 A-vs-B pair: n={len(common)} identical={ident}/{len(common)} flips={flips} dU(B-A)={dU:+.4f} max|dm|={maxd:.2f}")
print(f"M2 frozen-start mean U (two-batch mean) = {(ua + ub) / 2:.4f}")
