import json, math, sys

def load(p):
    rows = {}; errs = 0
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

base = '/home/hatch/workspace/carbon-live/probe_master2_repro/results'
p1b, e1 = load(f'{base}/probe_m2_p1b.jsonl')
p2b, e2 = load(f'{base}/probe_m2_p2b.jsonl')
p1, _ = load(f'{base}/probe_m2_p1.jsonl')
a85, _ = load('/home/hatch/workspace/it74_paired/h29_it85base_m2.jsonl')
a95, _ = load('/home/hatch/workspace/it74_paired/h29_it95base_m2.jsonl')
print(f"P1B rows={len(p1b)} err={e1} | P2B rows={len(p2b)} err={e2}")
print("--- amendment primary: worker-count 62 self-reproduction ---")
pair(p1b, p2b, 'P1B', 'P2B')
print("--- worker count single variable: 62 vs 24 ---")
pair(p1, p1b, 'P1(w24)', 'P1B(w62)')
print("--- vs archived ---")
pair(a95, p1b, 'A95', 'P1B')
pair(a85, p1b, 'A85', 'P1B')
