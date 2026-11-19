"""Independently reconcile the complete league action tapes, conditional choices and actor gradient."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parents[2]
REPO = next(p for p in HERE.parents if (p / '.git').exists())
PARENT = HERE.parents[1] / 'candidate.py'
sys.path.insert(0, str(ACTOR / 'blended_policy_return_dev'))
import policy


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()


def utility(scores, seat):
    margin = scores[seat] - scores[1 - seat]
    win = float(margin > 0) + .5 * float(margin == 0)
    # tanh form is independently stable for arbitrarily large signed margins.
    return .5 * win + .25 * (1. + math.tanh(margin / 2000.))


def probabilities(logits):
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def reconstructed_menu(module, feature, values):
    """Re-enumerate feasible physical orders from the uninstrumented parent on this history."""
    root = policy.Root(module, feature, values)
    a, q0 = root.decode(False)
    b, q1 = root.decode(True)
    greedy, reject = policy.guard(module, values, a, b)
    assert np.array_equal(module._mixed_actions(values, feature.aux), greedy)
    state_q = q0 if reject else q1
    greedy_order = dict(feature.to_actions(greedy))
    candidates = []
    for slot, law in state_q.items():
        if slot >= 10 or np.count_nonzero(law) < 2:
            continue
        ordered = np.argsort(-law, kind='stable')
        first, second = map(int, ordered[:2])
        gap = math.log(law[first] / law[second])
        if gap <= 1. and int(greedy[slot]) == first:
            candidates.append((gap, slot, first, second))
    for _, slot, first, second in sorted(candidates):
        alt0, _ = root.decode(False, slot, second)
        alt1, _ = root.decode(True, slot, second)
        alternative, new_reject = policy.guard(module, values, alt0, alt1)
        if reject != new_reject or int(alternative[slot]) != second:
            continue
        if not np.array_equal(greedy[:slot], alternative[:slot]):
            continue
        if module._return_conflicts(values, alternative) - module._return_conflicts(values, greedy):
            continue
        other_order = dict(feature.to_actions(alternative))
        if greedy_order == other_order:
            continue
        law = state_q[slot][[first, second]]
        return dict(slot=slot, ids=[first, second], options=[greedy_order, other_order],
                    prob=law / law.sum(), uid=feature.own_uid[slot], route=not reject,
                    prefix=greedy[:slot].tolist())
    return dict(options=[greedy_order])


def probability_audit(module, feature, values, event, advantage):
    menu = reconstructed_menu(module, feature, values)
    assert (len(menu['options']) == 2) == event['supported']
    if not event['supported']:
        assert menu['options'][0] == event['issued']
        return dict(supported=False)
    assert menu['options'] == event['options']
    assert menu['slot'] == event['slot'] and menu['ids'] == event['actions']
    assert menu['route'] == event['route'] and menu['prefix'] == event['prefix']
    assert menu['uid'] == event['uid']
    p = menu['prob']
    j = event['choice']
    prob_error = float(np.max(np.abs(p - event['probabilities'])))
    assert prob_error < 2.e-6, prob_error
    # Physically perturb the Sapling output layer of the parent, repeat guard, whole-team
    # alternatives and scores. Missing menu stability is reported and never repaired by
    # selecting a different result. Learnable parameter is Sapling bias, not mixed bias.
    direction = np.array([-.31, -.13, .02, .17, .25], dtype=np.float32)
    eps = .008
    bias = module._NETS[1].w['whead.bias']
    initial = bias.copy()
    sides = []
    try:
        for sign in (1., -1.):
            bias[:] = initial + sign * eps * direction
            sides.append(reconstructed_menu(module, feature, values))
    finally:
        bias[:] = initial
    stable = all(x['options'] == menu['options'] and x.get('ids') == menu['ids']
                 and x.get('route') == menu['route'] and x.get('slot') == menu['slot'] for x in sides)
    result = dict(supported=True, probability_error=prob_error, stable_for_derivative=stable,
                  advantage=advantage, conditional_entropy_nats=float(-(p * np.log(p)).sum()))
    if stable:
        measured = (math.log(sides[0]['prob'][j]) - math.log(sides[1]['prob'][j])) / (2*eps)
        d = direction[menu['ids']].astype(np.float64)
        expected = .25 * (d[j] - p @ d)
        assert abs(measured - expected) < .0004, (measured, expected)
        result.update(chosen_logp_derivative=float(measured), analytic=float(expected),
                      signed_negative_utility_loss_derivative=float(-advantage * measured))
    return result


def source_inputs():
    plan = read(HERE / 'protocol.json')
    maps = read(HERE / 'clearance.json')
    order = read(HERE / 'collection_registration.json')
    for n,k in [('protocol.json','protocol_sha256'),('clearance.json','clearance_sha256'),
                ('opponents.json','opponents_sha256'),('collect.py','collector_sha256')]:
        assert order[k] == file_sha(HERE/n)
    assert file_sha(PARENT) == order['parent_sha256'] == plan['initial_actor']
    sampler = PARENT.parent / 'onpolicy_native' / 'training_policy.py'
    assert order['sampler_sha256'] == file_sha(sampler)
    assert order['sampler_source_sha256'] == file_sha(sampler.with_name('sampler.py'))
    engine = REPO / 'code/env_src/carbon_challenge-master'
    for file, signature in order['engine_sha256'].items():
        assert file_sha(engine / file) == signature
    for o in order['opponents'].values():
        assert file_sha(REPO/o['path']) == o['sha256']
    all_seeds = sum(maps['groups'].values(), [])
    assert len(all_seeds) == len(set(all_seeds)) == len({maps['by_seed'][str(s)] for s in all_seeds}) == 26
    seeds = sorted(maps['groups']['train'])
    assert len(seeds) == 18 and maps['source_snapshot'] == plan['starting_commit']
    schedule = {(s,side,['it20','it40','it50'][i%3],arm) for i,s in enumerate(seeds)
                for side in (0,1) for arm in ('sampled','greedy')}
    assert len(order['tickets']) == len(schedule) and {
        (r['seed'],r['seat'],r['opponent'],r['arm']) for r in order['tickets']} == schedule
    assert all(r['terrain'] == maps['by_seed'][str(r['seed'])] for r in order['tickets'])
    assert plan['collection']['sample_steps'] == order['sample_steps'] == list(range(12,277,24))
    return maps, order, schedule, engine


def verify_draws(raw, card, episode_a):
    seat = card['seat']
    entropy, gradient = [], np.zeros(5)
    loss = 0.
    steps = [n for n in range(12,277,24) if n < raw['steps']] if card['arm'] == 'sampled' else []
    assert steps == [e['step'] for e in raw['events']]
    for e in raw['events']:
        assert e['seat'] == seat and raw['actions'][e['step']][seat] == e['issued']
        if not e['supported']:
            assert not {'choice','actions','options','probabilities'}.intersection(e)
            continue
        q = probabilities(e['logits'])
        actions = e['actions']
        chosen = int(e['uniform'] >= q[0])
        assert e['uniform'] == np.random.default_rng(
            np.random.SeedSequence([9173951,card['seed'],seat,e['step']])).random()
        assert 0.5 <= q[0] <= 1./(1.+math.exp(-1.000001))
        assert len(actions) == len(set(actions)) == len(e['options']) == 2
        assert all(e['legal'][a] for a in actions)
        assert e['options'][1] != e['options'][0]
        assert chosen == e['choice'] and e['options'][chosen] == e['issued']
        assert np.max(np.abs(q - e['probabilities'])) < 1e-12
        assert abs(e['logp'] - math.log(q[chosen])) < 1e-12
        entropy.append(float(-(q * np.log(q)).sum()))
        # For fixed behavior prefix the actual head bias only affects its own
        # logit. The partner and opponent have no parameters in this score.
        gradient[actions] += .25 * episode_a * (q - np.eye(2)[chosen])
        loss -= episode_a * math.log(q[chosen])
    assert len(entropy) == card['supported'] and sum(e.get('choice',0) for e in raw['events']) == card['alternate']
    return entropy, gradient, loss


def authoritative_replay(card, raw, registry, model, deep_checks, advantage):
    import random
    from zerosum_env import make
    seat, seed = card['seat'], card['seed']
    cfg = raw['game_configuration']
    assert cfg['randomSeed'] == seed
    np.random.seed(seed)
    random.seed(seed)
    env = make('carbon', configuration=cfg)
    initial = list(map(int, env.states[0].observation.full_carbon))
    assert initial == raw['initial_full_carbon']
    assert hashlib.sha256(bytes(initial)).hexdigest() == raw['initial_terrain_u8_sha256'] == registry['by_seed'][str(seed)]
    events = {e['step']: e for e in raw['events']}
    feature = sys.modules['cnet2_feat'].Feat(seat) if deep_checks else None
    checks = []
    if deep_checks:
        import opponents
        opponent = opponents.load(card["opponent"])
    for t, cmd in enumerate(raw['actions']):
        assert len(cmd) == 2 and not env.done
        if deep_checks and t < 16:
            visible = env.states[0].observation
            rival = env.states[1-seat].observation
            own_view = dict(player=1-seat, step=t, players=visible.players, trees=visible.trees,
                            carbon=rival.carbon, remainingOverageTime=rival.remainingOverageTime)
            assert opponent.agent(own_view, cfg) == cmd[1-seat], (seed, seat, t, card['opponent'])
        if t in events or feature is not None:
            shared = env.states[0].observation
            own = env.states[seat].observation
            public = dict(player=seat, step=shared.step, players=shared.players, trees=shared.trees,
                          carbon=own.carbon, remainingOverageTime=own.remainingOverageTime)
            assert public['step'] == t
            values = feature.observe(public) if feature is not None else None
            if t in events:
                event = events[t]
                assert event['public_sha256'] == digest(public)
                if t in deep_checks:
                    checks.append(dict(seed=seed,seat=seat,step=t,opponent=card['opponent'],
                                       **probability_audit(model, feature, values, event, advantage)))
        env.step(cmd)
    assert env.done and [s.status for s in env.states] == raw['status'] == ['DONE','DONE']
    scores = [float(s.reward) for s in env.states]
    assert scores == raw['official_score_by_seat'] == raw['official_scores_by_seat'] == card['official_score_by_seat']
    assert [p[0] for p in env.states[0].observation.players] == raw['cash_by_seat']
    margin = scores[seat] - scores[1-seat]
    assert margin == card['margin'] and card['result'] == ('win' if margin>0 else 'loss' if margin<0 else 'draw')
    assert abs(utility(scores, seat) + utility(scores, 1-seat) - 1.) < 1e-12
    assert abs(utility(scores, seat) - card['u']) < 1e-12
    return checks, min(16, len(raw["actions"])) if deep_checks else 0


def cohort():
    maps, order, schedule, engine = source_inputs()
    sys.path.insert(0,str(engine))
    catalog = read(HERE/'collection.json')
    binding = file_sha(HERE/'collection_registration.json')
    assert catalog['registration_sha256'] == binding and catalog['status'] == 'complete'
    assert catalog['protocol_sha256'] == order['protocol_sha256']
    allrows = {(c['seed'],c['seat'],c['opponent'],c['arm']): c for c in catalog['games']}
    assert set(allrows) == schedule and len(catalog['games']) == len(allrows) == 72
    assert len({c['sha256'] for c in allrows.values()}) == 72
    model = policy.load(PARENT)
    grad = np.zeros(5)
    objective = 0.
    probes, entropies, episodes = [], [], []
    opponent_openings = 0
    for seed,seat,opponent in sorted({token[:3] for token in allrows}):
        pair = {a:allrows[(seed,seat,opponent,a)] for a in ('sampled','greedy')}
        raw = {}
        for arm, c in pair.items():
            path = HERE/'train_games'/f'{seed}_{seat}_{arm}.json.gz'
            assert path == REPO/c['path'] and file_sha(path) == c['sha256']
            r = read(path)
            assert (r['seed'],r['candidate_seat'],r['arm'],r['registration_sha256']) == (seed,seat,arm,binding)
            assert r['policy_sha256'] == order['sampler_sha256' if arm=='sampled' else 'parent_sha256']
            assert r['opponent_sha256'] == order['opponents'][opponent]['sha256']
            assert r['steps'] == len(r['actions']) == c['actual_transitions']
            assert c['terrain'] == maps['by_seed'][str(seed)] and all(z['calls'] == r['steps']
                                                                    for z in r['local_per_player_diagnostic'])
            raw[arm] = r
        u_sample,u_base = (utility(raw[a]['official_scores_by_seat'],seat) for a in ('sampled','greedy'))
        a = u_sample - u_base
        nondefault = [e for e in raw['sampled']['events'] if e.get('choice') == 1]
        start = nondefault[0]['step'] if nondefault else raw['sampled']['steps']
        assert raw['sampled']['actions'][:start] == raw['greedy']['actions'][:start]
        if nondefault:
            assert raw['greedy']['actions'][start][1-seat] == raw['sampled']['actions'][start][1-seat]
            assert raw['greedy']['actions'][start][seat] == nondefault[0]['options'][0]
        else:
            assert raw['sampled']['actions'] == raw['greedy']['actions'] and a == 0.
        for arm,c in pair.items():
            moments = set()
            if arm == 'sampled' and seed in sorted(maps['groups']['train'])[:3]:
                # First supported pre-action turn per side and opponent, plus a late
                # turn per stratum. This is determined from support, independent of returns.
                eligible = [e for e in raw[arm]['events'] if e['supported']]
                unsupported = [e for e in raw[arm]['events'] if not e['supported']]
                moments = {e['step'] for e in eligible[:1] + eligible[-1:] + unsupported[:1]}
            verified, openings = authoritative_replay(c,raw[arm],maps,model,moments,a)
            probes.extend(verified)
            opponent_openings += openings
            h,g,l = verify_draws(raw[arm],c,a)
            entropies.extend(h)
            grad += g
            objective += l
        win_code = lambda m: 1. if m>0 else .5 if m==0 else 0.
        score_part = .25*(math.tanh(pair['sampled']['margin']/2000.)-math.tanh(pair['greedy']['margin']/2000.))
        wdl_part = .5*(win_code(pair['sampled']['margin'])-win_code(pair['greedy']['margin']))
        assert abs(a-wdl_part-score_part) < 1e-12
        episodes.append(dict(seed=seed,seat=seat,opponent=opponent,advantage=a,baseline_u=u_base,sampled_u=u_sample,
                             wdl_credit=wdl_part,margin_credit=score_part,
                             supported=pair['sampled']['supported'],alternate=len(nondefault),first_alternate=start if nondefault else None))
    n = len(episodes)*12
    assert n == 432 and len(probes) >= 6
    finite = [p for p in probes if p.get('stable_for_derivative')]
    assert any(p['advantage'] and abs(p['analytic']) > 1e-3 for p in finite)
    assert {p['seat'] for p in finite} == {0,1} and {p['opponent'] for p in finite} == {'it20','it40','it50'}
    signal = [abs(r['advantage'])*r['supported'] for r in episodes]
    assert sum(signal)>0
    return dict(pairs_with_wdl_change=sum(r['wdl_credit'] != 0 for r in episodes),
        positive_advantage_count=sum(r['advantage']>0 for r in episodes),
        negative_advantage_count=sum(r['advantage']<0 for r in episodes),
        largest_ticket_share_absolute_utility_credit=float(max(signal)/sum(signal)),
        source_sha256=file_sha(Path(__file__)), registration_sha256=binding,collection_sha256=file_sha(HERE/'collection.json'),
        official_tapes=72, paired_games=36, supported_draws=len(entropies), credited_per_episode=episodes,
        conditional_behavior_entropy_nats=dict(min=min(entropies),mean=float(np.mean(entropies)),max=max(entropies)),
        normalized_actor_loss=float(objective/n), gradient_whead_bias_unclipped=(grad/n).tolist(),
        normalization_denominator=n,signed_full_model_perturbations=probes,
        independently_emitted_opponent_opening_actions=opponent_openings,
        note='Full official replay of recorded commands; probes recompute feasible physical menus and perturb the actual serving head. League games use self-owned opponents. No original Master3s2 win or ladder timing qualification is assessed here.')


def compare_fit(qa):
    fit = read(HERE/'fit.json')
    audit = read(HERE/'decision_audit.json.gz')
    assert file_sha(HERE/'decision_audit.json.gz') == fit['audit_sha256']
    assert file_sha(HERE/'optimize.py') == fit['optimizer_sha256'] and file_sha(HERE/'candidate.py') == fit['package_sha256']
    assert fit['collection_sha256'] == qa['collection_sha256'] and fit['parent_sha256'] == file_sha(PARENT)
    tickets = {(r['seed'],r['seat'],r['opponent']):r for r in fit['tickets']}
    assert len(tickets) == len(fit['tickets']) == len(qa['credited_per_episode'])
    for r in qa['credited_per_episode']:
        t = tickets[(r['seed'],r['seat'],r['opponent'])]
        assert t['supported'] == r['supported'] and abs(t['A']-r['advantage']) < 1.e-12
    trainer = fit['training']
    denom = trainer['normalization_denominator']
    assert denom == qa['normalization_denominator'] and trainer['updates'] == 1
    assert len(audit) == fit['decisions'] == qa['supported_draws']
    idx = {(row['seed'],row['seat'],row['opponent'],row['step']):row for row in audit}
    assert len(idx) == len(audit)
    # Independent adjoint calculated from signed Torch projection recorded before the
    # update. Compare to the actual autograd that traversed both backbone and decoder.
    torch_gradient = np.zeros(5)
    losses, shifts = [], []
    for ep in qa['credited_per_episode']:
        seed,side,opp = ep['seed'],ep['seat'],ep['opponent']
        data = read(HERE/'train_games'/f'{seed}_{side}_sampled.json.gz')
        for ev in data['events']:
            if not ev['supported']:
                continue
            r = idx.pop((seed,side,opp,ev['step']))
            assert r['moves'] == ev['actions'] and r['pick'] == ev['choice'] and r['route'] == ev['route']
            assert r['parent_order'] == ev['options'][0] and r['old_logp'] == ev['logp']
            q = np.array([r['parent_torch_p0'],1.-r['parent_torch_p0']])
            assert abs(q[0]-ev['probabilities'][0]) <= .00075 and abs(r['advantage'] - ep['advantage']) < 1e-12
            a = ep['advantage']
            torch_gradient[ev['actions']] += .25 * a * (q - np.eye(2)[ev['choice']])
            losses.append(-a * math.log(q[ev['choice']]))
            shifts.append(abs(q[0] - ev['probabilities'][0]))
    assert not idx
    torch_gradient /= denom
    grad_diff = float(np.max(np.abs(np.array(qa['gradient_whead_bias_unclipped']) -
                                    np.array(trainer['unclipped_worker_bias_gradient']))))
    loss_diff = abs(qa['normalized_actor_loss'] - trainer['objective'])
    assert grad_diff < 5.e-8 and loss_diff < 5.e-12, (loss_diff,grad_diff)
    return dict(optimizer_sha256=fit['optimizer_sha256'],fit_sha256=file_sha(HERE/'fit.json'),
                trainer_objective=trainer['objective'], exact_behavior_loss_absolute_error=loss_diff,
                native_behavior_vs_autograd_bias_max_abs_error=grad_diff,
                uncorrected_torch_forward_loss_error=abs(math.fsum(losses)/denom-trainer['objective']),
                uncorrected_torch_forward_bias_max_abs_error=float(np.max(np.abs(
                    torch_gradient-np.array(trainer['unclipped_worker_bias_gradient'])))),
                max_parent_native_vs_torch_probability_error=max(shifts),
                conditional_distribution_has_sampling_entropy=True)


def verify_development():
    evidence = HERE/'evidence'
    result = read(evidence/'results.json')
    registered = read(evidence/'registration.json')
    assert result['reservation_sha256'] == file_sha(evidence/'registration.json')
    groups = read(HERE/'clearance.json')
    seeds = sorted(groups['groups']['development'])
    opponents = {s:['it20','it40','it50'][i%3] for i,s in enumerate(seeds)}
    assert [(r['seed'],r['seat'],r['opponent']) for r in registered['plan']] == [
        (s,seat,opponents[s]) for s in seeds for seat in (0,1)]
    signed, outcome, identity = {},{},set()
    for r in result['pairs']:
        s,side,opp = r['seed'],r['seat'],r['opponent']
        assert (s,side) not in identity and opp == opponents[s]
        identity.add((s,side))
        for role in ('parent','candidate'):
            arm = r['arms'][role]
            path = evidence/f'{s}_s{side}_{role}.json.gz'
            assert file_sha(path) == arm['sha256'] == arm['raw_sha256']
            raw = read(path)
            assert raw['initial_terrain_u8_sha256'] == groups['by_seed'][str(s)]
            assert raw['policy_sha256'] == registered['package_sha256'][role]
            assert raw['opponent_sha256'] == registered['opponents'][opp]['sha256']
            scores = raw['official_scores_by_seat']
            signed[(s,side,role)] = scores[side] - scores[1-side]
            assert signed[(s,side,role)] == arm['margin']
            outcome[(s,side,role)] = int(signed[(s,side,role)] > 0)
    assert identity == {(s,side) for s in seeds for side in (0,1)}
    gain = {s:sum(signed[(s,side,'candidate')]-signed[(s,side,'parent')] for side in (0,1))/2 for s in seeds}
    v = np.asarray(list(gain.values()))
    opp_gain = {o:float(np.mean([gain[s] for s in seeds if opponents[s]==o])) for o in set(opponents.values())}
    wins = {r:sum(x for (s,side,role),x in outcome.items() if role==r) for r in ('candidate','parent')}
    nmap = int((v>0).sum())
    nopp = sum(x>0 for x in opp_gain.values())
    passed = float(v.mean())>0 and wins['candidate']>=wins['parent'] and nmap>=4 and nopp>=2
    assert result['pass_held_league'] == passed and nmap == result['positive_map_count'] and nopp==result['positive_opponent_count']
    assert abs(result['overall']['mean_margin_delta'] - v.mean()) < 1.e-8
    assert all(wins[r] == result['overall'][r]['wins'] for r in wins)
    return dict(development_result_sha256=file_sha(evidence/'results.json'), self_owned_games=24, maps=6,
                candidate_wins=wins['candidate'],parent_wins=wins['parent'],
                mean_paired_official_margin_delta=float(v.mean()),positive_maps=nmap,
                positive_opponent_groups=nopp,advance=passed,original_master3s2_games=0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['collection','fit','development'],default='fit')
    phase = parser.parse_args().phase
    out = HERE/'audit_batch.json'
    if out.exists() and phase != 'collection':
        report = read(out)
        assert report['source_sha256'] == file_sha(Path(__file__)) and report['collection_sha256'] == file_sha(HERE/'collection.json')
    else:
        report = cohort()
    if phase in ('fit','development'):
        report['fit_check'] = compare_fit(report)
    if phase == 'development':
        report['development_check'] = verify_development()
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report,sort_keys=True,allow_nan=False,indent=2)+'\n')
    os.replace(tmp,out)
    print({k: report[k] for k in ('official_tapes','supported_draws','normalized_actor_loss','conditional_behavior_entropy_nats')})
    print(report.get('fit_check'),report.get('development_check'))


if __name__ == '__main__':
    main()
