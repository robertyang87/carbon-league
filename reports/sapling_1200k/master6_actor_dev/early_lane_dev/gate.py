"""Apply the registered map-balanced mechanism gate and select outcome-blind roots for future responses."""
from collections import Counter
from hashlib import sha256
from statistics import mean

from train import HERE, PROTOCOL, batch, check, extract


def load_games(number):
    games = []
    for spec in batch(number):
        path = HERE / f"validation_{spec['seed']}_{spec['seat']}.json.gz"
        g = extract.load(path)
        check(g['seed'] == spec['seed'] and g['seat'] == spec['seat'] and
              g['source_sha256'] == spec['raw_sha256'] and g['weights_sha256'] == extract.sha(HERE/'weights.json')
              and g['chooser_sha256'] == extract.sha(HERE/'chooser.py') and g['registered_source_and_counterfactuals_independently_checked'], 'detached validation artifact')
        g['artifact_sha256'] = extract.sha(path)
        games.append(g)
    return games


def summarize(games):
    counts = Counter()
    for game in games:
        counts.update(game['counts'])
    quality = {}
    for h in ('3','5'):
        quality[h] = {}
        for arm in ('model', 'baseline'):
            rows = [g['scores'][h][arm] for g in games]
            n = sum(r['cases'] for r in rows)
            check(n>0, 'no scored heldout enemy positions')
            metrics = {k:sum(r[k] for r in rows)/n for k in ('brier','correct','logloss','other','p_other')}
            quality[h][arm] = dict(cases=n, **metrics)
    return dict(counts=dict(counts), quality=quality,
                trace_gain_per_all_turns=counts['trace_gain_sum']/counts['source_turns'])


def one_batch(number, games):
    maps = {str(s):summarize([g for g in games if g['seed']==s]) for s in sorted({g['seed'] for g in games})}
    seats = {f"{g['seed']}:{g['seat']}":summarize([g]) for g in games}
    quality = {h:{arm:{k:mean(m['quality'][h][arm][k] for m in maps.values()) for k in ('brier','correct','logloss','other','p_other')}
                   for arm in ('model','baseline')} for h in ('3','5')}
    return dict(batch=number, total=summarize(games), maps=maps, by_map_seat=seats,
                map_equal_quality=quality, map_equal_trace_gain_per_turn=mean(m['trace_gain_per_all_turns'] for m in maps.values()))


def response_roots(game):
    ordered = game['decisions']; existing={r['t'] for r in ordered}
    base = {k:game[k] for k in ('seed','seat','source','source_sha256')}
    roots = {}
    for group,flag in [('issued',True),('eligible_unissued',False)]:
        choices = [r for r in ordered if r['issued'] == flag]
        row = min(choices,key=lambda r:r['signature']) if choices else None
        roots[group] = None if row is None else dict(t=row['t'],signature=row['signature'],history=row['history'],
            target_uid=row['candidate']['target_uid'],own_uid=row['candidate']['own_uid'],
            parent_order=row['candidate']['old'],alternative_order=row['candidate']['order'],
            corridor=row['candidate']['cells'],eligible_rows=len(choices))
    unused = set(range(game['steps'])) - existing
    def digest(t): return sha256(f"early-lane-v1|{game['seed']}|{game['seat']}|{t}|none".encode()).hexdigest()
    t = min(unused,key=digest) if unused else None
    roots['no_candidate'] = None if t is None else dict(t=t,signature=digest(t),alternatives=0,eligible_rows=len(unused))
    return dict(**base,bins=roots)


def main():
    spec = extract.load(HERE/'protocol.json')
    all_games = {b:load_games(b) for b in (4,5)}
    reports = {str(b):one_batch(b,games) for b,games in all_games.items()}
    gates = {'engine_and_identity':True}
    for b,r in reports.items():
        for h,q in r['map_equal_quality'].items():
            gates[f'B{b}_brier_t{h}'] = q['model']['brier'] < q['baseline']['brier']
        gates[f'B{b}_retrospective_gain'] = r['map_equal_trace_gain_per_turn'] > 0
    games = [g for values in all_games.values() for g in values]
    overall = summarize(games)['counts']
    gates['at_least_one_action_on_every_map'] = all(m['counts'].get('issued',0)>0 for r in reports.values() for m in r['maps'].values())
    gates['at_least_eight_issued_actions'] = overall.get('issued',0)>=8
    gates['no_newly_dead_focal_worker'] = overall.get('harm',0)==0
    gates['no_lost_factual_deposit'] = overall.get('lost_factual_delivery',0)==0
    status = 'pass_for_portable_parity' if all(gates.values()) else 'retired_before_native'
    summary = dict(protocol_sha256=PROTOCOL,weights_sha256=extract.sha(HERE/'weights.json'),
                   covered_artifacts={f"{g['seed']}:{g['seat']}":g['artifact_sha256'] for g in games},
                   overall=overall,batches=reports,gates=gates,decision=status)
    extract.write(HERE/'gate.json',summary)
    follow = dict(protocol_sha256=PROTOCOL,weights_sha256=summary['weights_sha256'],status='prospective design only',
            recruitment='Minimum frozen SHA within each terrain/physical seat/issuance bin. A no-candidate turn is a diagnostic with no admissible treatment.',
            execution=spec['sparse_followup'],
            estimands='Record initial-order difference, new examiner actions versus source and actual route at 3/5/10, both seats’ normal deposits, collision cargo by owner, final cash, official U and its difference. Keep separate streams and paired outcomes, including zero/negative. No claimed causal effect from the original routes.',
            roots=[response_roots(g) for g in games])
    extract.write(HERE/'response_plan.json',follow)
    print(status,gates,overall)


if __name__=='__main__':
    main()
