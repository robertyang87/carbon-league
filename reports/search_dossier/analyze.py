import json, math
from collections import Counter

BASE = '/home/hatch/workspace/search-group/winrate_screen/'
def load(fn):
    with open(BASE+fn) as f: return [json.loads(l) for l in f]
A = load('screen_search_a.jsonl')   # search = A side (pol)
B = load('screen_search_b.jsonl')   # baseline = A side (pol), search = B side (opp)

def quantiles(xs, ps=(0,10,25,50,75,90,100)):
    xs = sorted(xs); n = len(xs); out = {}
    for p in ps:
        if n == 1: out[str(p)] = xs[0]; continue
        pos = (p/100)*(n-1); lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        out[str(p)] = xs[lo] if lo==hi else xs[lo]+(xs[hi]-xs[lo])*(pos-lo)
    return out

def mean(xs): return sum(xs)/len(xs)
def median(xs):
    xs = sorted(xs); n=len(xs)
    return xs[n//2] if n%2 else (xs[n//2-1]+xs[n//2])/2
def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return num/den if den else None

R = {}

# ---------- unified per-game search-perspective records ----------
G = []  # (field, seed, margin_search, score_search, score_base, lost_search, lost_base, row)
for r in A: G.append(('A', r['seed'], r['margin'], r['sa'], r['sb'], r['lost'], r['olost'], r))
for r in B: G.append(('B', r['seed'], -r['margin'], r['sb'], r['sa'], r['olost'], r['lost'], r))
margins = [g[2] for g in G]

# ---------- 1. margin distribution ----------
bands = {'>0':0, '-500~0':0, '-1000~-500':0, '-2000~-1000':0, '<=-2000':0}
for m in margins:
    if m > 0: bands['>0'] += 1
    elif m > -500: bands['-500~0'] += 1
    elif m > -1000: bands['-1000~-500'] += 1
    elif m > -2000: bands['-2000~-1000'] += 1
    else: bands['<=-2000'] += 1
R['margin_dist'] = {
    'n': len(margins), 'mean': mean(margins), 'median': median(margins),
    'quantiles_pct': quantiles(margins), 'bands': bands,
    'wins': sum(1 for m in margins if m>0), 'losses': sum(1 for m in margins if m<0),
    'draws': sum(1 for m in margins if m==0),
    'mean_by_field': {'A': mean([g[2] for g in G if g[0]=='A']), 'B': mean([g[2] for g in G if g[0]=='B'])},
    'worst10_mean': mean(sorted(margins)[:10]), 'best10_mean': mean(sorted(margins)[-10:]),
    'share_of_total_loss_from_worst100games': sum(sorted(margins)[:100])/sum(m for m in margins if m<0),
}

# ---------- 2. dose-response (A field only) ----------
chg = [r['chg'] for r in A]; srch = [r['srch'] for r in A]; mA = [r['margin'] for r in A]
order = sorted(range(len(A)), key=lambda i: chg[i])
ter = {}
for name, idxs in (('low', order[:200]), ('mid', order[200:400]), ('high', order[400:])):
    ter[name] = {'chg_range': [chg[idxs[0]], chg[idxs[-1]]], 'mean_margin': mean([mA[i] for i in idxs]),
                 'mean_chg': mean([chg[i] for i in idxs])}
R['dose_response'] = {
    'pearson_chg_margin': pearson(chg, mA),
    'pearson_srch_margin': pearson(srch, mA),
    'pearson_chgrate_margin': pearson([c/s if s else 0.0 for c,s in zip(chg,srch)], mA),
    'chg_tertiles': ter,
    'chg0_all': {'n': sum(1 for c in chg if c==0), 'margins': [mA[i] for i in range(600) if chg[i]==0]},
    'chg0_searched_only': {'n': sum(1 for i in range(600) if srch[i]>0 and chg[i]==0),
                           'margins': [mA[i] for i in range(600) if srch[i]>0 and chg[i]==0]},
    'srch0': {'n': sum(1 for s in srch if s==0), 'margins': [mA[i] for i in range(600) if srch[i]==0]},
    'mean_chg_per_searched_game': sum(chg)/sum(1 for s in srch if s>0),
    'mean_srch_per_searched_game': sum(srch)/sum(1 for s in srch if s>0),
    'chg_rate_overall': sum(chg)/sum(srch),
    'low_chg_le20': {'n': sum(1 for c in chg if 0<c<=20), 'mean_margin': mean([mA[i] for i in range(600) if 0<chg[i]<=20]) if any(0<c<=20 for c in chg) else None},
}

# ---------- 3. disaster games ----------
ss = [g[3] for g in G]; bs = [g[4] for g in G]
ls_ = [g[5] for g in G]; lb_ = [g[6] for g in G]
R['disaster'] = {
    'search_score_le10': sum(1 for s in ss if s<=10), 'search_score_le50': sum(1 for s in ss if s<=50),
    'search_score_le100': sum(1 for s in ss if s<=100),
    'baseline_score_le100': sum(1 for s in bs if s<=100),
    'search_score_lt10pct_of_base': sum(1 for x,y in zip(ss,bs) if x < 0.1*y),
    'search_score_median': median(ss), 'baseline_score_median': median(bs),
    'search_score_mean': mean(ss), 'baseline_score_mean': mean(bs),
    'lost_search': {'mean': mean(ls_), 'median': median(ls_), 'max': max(ls_), 'dist': dict(sorted(Counter(ls_).items()))},
    'lost_baseline': {'mean': mean(lb_), 'median': median(lb_), 'max': max(lb_), 'dist': dict(sorted(Counter(lb_).items()))},
    'lost_diff_mean': mean([x-y for x,y in zip(ls_,lb_)]),
    'games_search_lost_more': sum(1 for x,y in zip(ls_,lb_) if x>y),
    'games_baseline_lost_more': sum(1 for x,y in zip(ls_,lb_) if x<y),
    'games_lost_equal': sum(1 for x,y in zip(ls_,lb_) if x==y),
}
# game length / extinction (A field msl only; B msl is None by construction)
short = [(r['seed'], len(r['msl']), r['margin'], r['lost'], r['olost'], r['sa'], r['sb']) for r in A if len(r['msl'])<299]
R['disaster']['game_length_Afield'] = {
    'full_299': sum(1 for r in A if len(r['msl'])==299), 'short_lt299': len(short),
    'short_games': [{'seed':s,'len':l,'margin_A':m,'lost_A_search':lo,'olost_B_base':oo,'sa':x,'sb':y} for s,l,m,lo,oo,x,y in short],
    'min_len': min(len(r['msl']) for r in A),
}

# ---------- 4. phase localization via lx ----------
PHASES = {
 'early_le100': ['cash100','dep100','haul100','cargo100','colsteps100','tree50','tree100','cc100_sum','camp100'],
 'mid_200_240': ['cash200','dep200','cargo200','dep240','haul240','cargo240','gain240','lost240'],
 'late_270_end': ['cash270','dep270','cargo270','late15','late30','rec259_275','rec276'],
 'final_totals': ['cash','gain_all','haul_all','tax_all','plants','planted','lost_self','lost_enemy','lost_unknown','life_sum','colsteps','treesteps','stall_max'],
}
phase_out = {}
for ph, feats in PHASES.items():
    phase_out[ph] = {}
    for f in feats:
        diffs, miss = [], 0
        for r in A:
            sv, bv = r['lx']['pol'].get(f), r['lx']['opp'].get(f)
            if isinstance(sv,(int,float)) and isinstance(bv,(int,float)): diffs.append(sv-bv)
            else: miss += 1
        for r in B:
            sv, bv = r['lx']['opp'].get(f), r['lx']['pol'].get(f)
            if isinstance(sv,(int,float)) and isinstance(bv,(int,float)): diffs.append(sv-bv)
            else: miss += 1
        if diffs:
            phase_out[ph][f] = {'mean_diff_search_minus_base': mean(diffs), 'n': len(diffs), 'missing_pairs': miss,
                                'mean_search': None, 'mean_base': None}
            svals, bvals = [], []
            for r in A:
                if isinstance(r['lx']['pol'].get(f),(int,float)): svals.append(r['lx']['pol'][f])
                if isinstance(r['lx']['opp'].get(f),(int,float)): bvals.append(r['lx']['opp'][f])
            for r in B:
                if isinstance(r['lx']['opp'].get(f),(int,float)): svals.append(r['lx']['opp'][f])
                if isinstance(r['lx']['pol'].get(f),(int,float)): bvals.append(r['lx']['pol'][f])
            phase_out[ph][f]['mean_search'] = mean(svals); phase_out[ph][f]['mean_base'] = mean(bvals)
        else:
            phase_out[ph][f] = {'all_missing': True}
R['phase_lx'] = phase_out
# cash/deposit trajectory gap summary
traj = {}
for f in ['cash100','cash200','cash270','cash','dep100','dep200','dep240','dep270','haul100','haul240','cargo100','cargo240','cargo270']:
    for ph in PHASES:
        if f in phase_out[ph] and 'mean_diff_search_minus_base' in phase_out[ph][f]:
            traj[f] = phase_out[ph][f]['mean_diff_search_minus_base']
R['trajectory_gap'] = traj

# ---------- 5. mirror structure ----------
Am = {r['seed']: r for r in A}; Bm = {r['seed']: r for r in B}
mirror, nonmirror = 0, []
for s in sorted(Am):
    ra, rb = Am[s], Bm[s]
    if rb['margin'] == -ra['margin'] and rb['sa'] == ra['sb'] and rb['sb'] == ra['sa']:
        mirror += 1
    else:
        nonmirror.append({'seed': s, 'margin_A': ra['margin'], 'margin_B': rb['margin'],
                          'sa_A': ra['sa'], 'sb_A': ra['sb'], 'sa_B': rb['sa'], 'sb_B': rb['sb'],
                          'chg_A': ra['chg'], 'srch_A': ra['srch']})
R['mirror'] = {'mirror_seeds': mirror, 'nonmirror_seeds': nonmirror}

# ---------- anchors: recompute paired dU and mean u ----------
def u_search(r, field): return r['u'] if field=='A' else 1-r['u']
ds = []
for s in sorted(Am):
    per = []
    for r, f in ((Am[s],'A'), (Bm[s],'B')):
        us = u_search(r, f); per.append(us-(1-us))
    ds.append(mean(per))
R['anchors'] = {'paired_delta_u_recomputed': mean(ds),
                'mean_u_search': mean([u_search(r,f) for r,f in [(r,'A') for r in A]+[(r,'B') for r in B]]),
                'seeds': len(ds)}

with open(BASE+'localization/numbers.json','w') as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
