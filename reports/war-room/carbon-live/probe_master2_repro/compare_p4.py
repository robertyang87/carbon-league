import json, math

def load(p):
    rows = {}; errs = 0; hhs = set()
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            rows[r['seed']] = r
            hhs.add(r.get('hh'))
            if r.get('err') is not None or r.get('berr'):
                errs += 1
    return rows, errs, hhs

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
p4, e4, hh4 = load(f'{base}/probe_m2_p4.jsonl')
p1, _, _ = load(f'{base}/probe_m2_p1.jsonl')
a85, _, _ = load('/home/hatch/workspace/it74_paired/h29_it85base_m2.jsonl')
a95, _, _ = load('/home/hatch/workspace/it74_paired/h29_it95base_m2.jsonl')
print(f"P4 rows={len(p4)} err={e4} hh={hh4}")
print("--- step 3 primary: harness swap (pack 11d80059 + old harness dceb1ebe) ---")
pair(p1, p4, 'P1', 'P4')
print("--- P4 vs archived ---")
pair(a85, p4, 'A85', 'P4')
pair(a95, p4, 'A95', 'P4')
