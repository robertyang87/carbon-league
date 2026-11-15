import json, math, sys

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

def pair(a, b, la, lb):
    common = sorted(set(a) & set(b))
    ident = sum(1 for s in common if a[s]['margin'] == b[s]['margin'])
    flips = sum(1 for s in common if win(a[s]) != win(b[s]))
    dU = sum(U(b[s]['margin']) - U(a[s]['margin']) for s in common) / len(common)
    maxd = max(abs(a[s]['margin'] - b[s]['margin']) for s in common)
    ua = sum(U(a[s]['margin']) for s in common) / len(common)
    ub = sum(U(b[s]['margin']) for s in common) / len(common)
    wa = sum(win(a[s]) for s in common); wb = sum(win(b[s]) for s in common)
    print(f"{lb} vs {la}: n={len(common)} identical={ident}/{len(common)} flips={flips} "
          f"dU({lb}-{la})={dU:+.4f} max|dm|={maxd:.2f} | U {la}={ua:.4f} {lb}={ub:.4f} | wins {la}={wa} {lb}={wb}")
    return ident, flips, dU

base = '/home/hatch/workspace/carbon-live/probe_master2_repro/results'
p1, e1 = load(f'{base}/probe_m2_p1.jsonl')
p2, e2 = load(f'{base}/probe_m2_p2.jsonl')
a85, _ = load('/home/hatch/workspace/it74_paired/h29_it85base_m2.jsonl')
a95, _ = load('/home/hatch/workspace/it74_paired/h29_it95base_m2.jsonl')
print(f"P1 rows={len(p1)} err={e1} | P2 rows={len(p2)} err={e2}")
print("--- primary ---")
pair(p1, p2, 'P1', 'P2')
print("--- secondary vs archived ---")
pair(a95, p1, 'A95', 'P1')
pair(a85, p1, 'A85', 'P1')
pair(a95, p2, 'A95', 'P2')
pair(a85, p2, 'A85', 'P2')
