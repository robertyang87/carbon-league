"""Evaluate frozen enemy-deposit forecasts on both seats of the two archived ladder panels."""
import argparse
from collections import defaultdict, deque
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
from pydantic_core import from_json

from predict import ACTOR, HERE, LONG, ROOT, SAP, SHORT, PublicDeposit, horizons, module, read, short

sys.path.insert(0, str(SAP))
from map_identity import local_board
from turning_point_ledger import CashEvents, FastCarbon

CACHE = Path('/workspace/sapling-all-top5-20261119')
MASTER6 = 'claude-AlphaCarbon-Master6'
SHORT_BATCH = 'appendix_4a4677c9'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def norm(name):
    return ' '.join(name.casefold().split())


def inventory():
    sources = read(SHORT/'sources.json')
    master = read(ACTOR/'data_split.json')
    long_fit = [g for g in master['games'] if g['split'] in ('train', 'val')]
    short_fit = [g for g in sources['native'] if g['split'] in ('train', 'calibration')]
    # The frozen short head also consumes probabilities produced by the old move prior.
    prior = read(short.RULES/'opponent_intent.json')['train']
    blocked_seeds = set(prior['seeds']) | set(read(HERE/'protocol.json')['excluded_seeds'])
    blocked_boards = {local_board(s) for s in blocked_seeds}
    blocked_seeds.update(g['seed'] for g in long_fit + short_fit)
    blocked_boards.update(g['terrain_sha256'] for g in long_fit)
    blocked_boards.update(g['terrain'] for g in short_fit)
    names = {norm(g['opponent']) for g in long_fit + short_fit}
    names.update(map(norm, (MASTER6, 'claude-AlphaCarbon-Master3s2', prior['policy'], 'Sapling')))
    batches = defaultdict(list)
    for g in master['games']:
        if g['split'] == 'holdout':
            batches[g['batch']].append((g['id'], g['sha256'], g['terrain_sha256'], g['seed'], 'long_holdout'))
    for g in sources['ladder']:
        if not g['seat']:
            batches[SHORT_BATCH].append((g['id'], g['sha256'], g['terrain'], g['seed'], 'short_ladder'))
    games, omitted = [], []
    seen, seed_seen = set(), set()
    for batch, candidates in sorted(batches.items()):
        manifest = read(SAP/'all_top5'/batch/'manifest.json')
        cards = {g['id']:g for g in manifest['games']}
        for gid, digest, board, seed, cohort in candidates:
            g = cards[gid]
            if (g['sha256'], g['seed'], g['clean_terminal'], g['frame_verified']) != (digest, seed, True, True):
                raise ValueError('manifest disagreement: ' + gid)
            if seed in blocked_seeds or board in blocked_boards or board in seen or seed in seed_seen:
                omitted.append(dict(id=gid, seed=seed, terrain=board, reason='overlap')); continue
            seen.add(board); seed_seen.add(seed)
            games.append(dict(id=gid, seed=seed, terrain=board, sha256=digest, panel=cohort,
                              names=g['seat_names'], ids=g['seat_strategy_ids'], archive=g['archive_location'],
                              turns=g['transitions'], scores=g['score_by_seat']))
    meta = dict(fit_or_calibration_terrains=len(blocked_boards), excluded_or_fit_seeds=sorted(blocked_seeds),
                short_fit_maps=len({g['terrain'] for g in short_fit}), long_train_val_maps=len(long_fit),
                move_prior_fit_seeds=prior['seeds'], known_participant_names=sorted(names),
                known_definition='Exact registration name in either training or calibration side; includes Master6 observer and val participants conservatively. Whitespace/case normalized. Other versions or registrant aliases are not proven independent.',
                excluded_games=omitted)
    return games, names, meta


def load_replay(card):
    location = card['archive']
    path = CACHE/location['member']
    if path.exists():
        packed = path.read_bytes()
    else:
        archive = ROOT/location['archive']
        binary = archive.read_bytes() if archive.exists() else subprocess.check_output(
            ['git', 'show', 'HEAD:' + location['archive']], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(binary)) as f:
            packed = f.extractfile(location['member']).read()
    if sha(packed) != card['sha256']:
        raise ValueError('compressed tape mismatch')
    return from_json(gzip.decompress(packed))


def collect(card, frozen):
    raw = load_replay(card)
    frames, cfg = raw['steps'], raw['configuration']
    init = frames[0][0]['observation']
    if sha(bytes(init['full_carbon'])) != card['terrain'] or cfg['randomSeed'] != card['seed']:
        raise ValueError('initial terrain or seed mismatch')
    if len(frames) != card['turns']+1 or [r['status'] for r in frames[-1]] != ['DONE','DONE']:
        raise ValueError('terminal or frame count mismatch')
    if not np.allclose([r['reward'] for r in frames[-1]], card['scores'], rtol=0, atol=.011):
        raise ValueError('terminal scores mismatch')
    sim = FastCarbon(cfg).load_observation(init)
    sim.cash = CashEvents(sim.cash)
    roots, history = [[], []], [deque(maxlen=11), deque(maxlen=11)]
    events = []
    cached_perspectives = [(card['id'], seat) in frozen for seat in (0,1)]
    for t, (pre, post) in enumerate(zip(frames, frames[1:])):
        if not all(cached_perspectives):
            shared = pre[0]['observation']
            for seat in (0, 1):
                # Only hand the feature code the actual seat's public carbon and the shared public roster.
                obs = dict(step=t, player=seat, carbon=pre[seat]['observation']['carbon'],
                           players=shared['players'])
                history[seat].append(obs)
                for uid, x, _, _, _ in short.public_rows(list(history[seat])):
                    roots[seat].append((t, uid, x))
        sim.cash.events.clear()
        sim.step(*(r['action'] or {} for r in post))
        players = post[0]['observation']['players']
        if not np.allclose(sim.cash, [p[0] for p in players], rtol=0, atol=.011):
            raise ValueError('official cash disagreement')
        for seat in (0, 1):
            if set(sim.player_workers[seat]) != set(players[seat][2]):
                raise ValueError('official worker roster disagreement')
            for uid, (position, carried, role) in players[seat][2].items():
                worker = sim.workers[uid]
                if worker.pos != position or worker.player != seat or worker.is_collector != (role == 'COLLECTOR') or abs(worker.carbon-carried) > .015:
                    raise ValueError('official worker state disagreement')
        deposits = {(seat,uid):value for kind,seat,value,*args in sim.cash.events
                    if kind == 'own_delivery' and value > 0 for uid in args[:1]}
        events.append(deposits)
    output = []
    for seat in (0, 1):
        cached = frozen.get((card['id'],seat))
        if cached is None:
            times = np.array([r[0] for r in roots[seat]], int)
            uids = np.array([r[1] for r in roots[seat]], dtype=str)
            x = np.asarray([r[2] for r in roots[seat]], np.float32).reshape(-1,len(short.FEATURES))
        else:
            times, uids, x, signed_targets, signed_cash = cached
        # Scores and realized monetary labels never enter either model.
        bank = np.zeros((len(times),3), np.float64)
        for row,(t,uid) in enumerate(zip(times, uids)):
            cash = np.cumsum([event.get((1-seat,uid),0.) for event in events[t:t+10]])
            bank[row] = [cash[min(h,len(cash))-1] for h in horizons]
        if cached is not None:
            if not np.array_equal(bank[:,:2]>0,signed_targets) or not np.allclose(bank[:,:2], signed_cash, rtol=0, atol=1e-7):
                raise ValueError('archived short labels differ from official full-tape ledger')
        output.append(dict(t=times, uid=uids, x=x, bank=bank))
    return output


def archive_short():
    sources = read(SHORT/'sources.json')
    all_cards = sources['native'] + sources['ladder']
    with np.load(SHORT/'ladder_features.npz', allow_pickle=False) as f:
        x, t, uid, events, bank, game = (f[k] for k in ('x','t','uid','events','bank','game'))
    result = {}
    for i in np.unique(game):
        g=all_cards[i]
        take = game==i
        key = g['id'],g['seat']
        if key in result or g['split']!='ladder':
            raise ValueError('ambiguous signed feature row')
        result[key] = (t[take], uid[take], x[take], events[take,:2], bank[take])
    return result


def raw_matrices(cache):
    cache.mkdir(exist_ok=True, parents=True)
    frozen = archive_short()
    cards, names, meta = inventory()
    recipe = sha(Path(__file__).read_bytes().split(b'def combine(')[0] + (SHORT/'model.py').read_bytes() +
                 (LONG/'predictor.py').read_bytes() + (SHORT/'ladder_features.npz').read_bytes() +
                 (SAP/'turning_point_ledger.py').read_bytes() + (ROOT/'code/fastsim.py').read_bytes())
    outputs = []
    for card in cards:
        path = cache/(card['id']+'.npz')
        if not path.exists():
            perspectives = collect(card,frozen)
            arrays = {k+str(s):v for s,p in enumerate(perspectives) for k,v in p.items()}
            with path.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f, **arrays, recipe=recipe, replay_sha256=card['sha256'])
            os.replace(path.with_suffix('.tmp'),path)
        with np.load(path,allow_pickle=False) as f:
            if f['recipe'].item()!=recipe or f['replay_sha256'].item()!=card['sha256']:
                raise ValueError('stale strict replay cache: '+str(path))
            for s in (0,1):
                z={k:f[k+str(s)] for k in ('t','uid','x','bank')}
                outputs.append((card,s,z))
        print('reconciled',card['id'],flush=True)
    return cards,names,meta,outputs
def combine(cache, destination):
    from metrics import assess, aggregate
    cards, known, coverage, raw = raw_matrices(cache)
    model = PublicDeposit()
    original_long = read(LONG/'results.json')['results']['holdout']['scores']['rich']['by_map']
    official_roots = {g['id']:g for g in from_json(gzip.decompress((ACTOR/'data_receipts.json.gz').read_bytes()))['games']}
    long_cards = {g['id']:g for g in read(ACTOR/'data_split.json')['games'] if g['split']=='holdout'}
    games, prior_brier_error = [], []
    clocks = {}
    for card, seat, data in raw:
        # Collect actual elapsed lead time separately from the forecasts. A move at root t
        # settling on the immediately following public observation has transition t+1.
        if card['id'] not in clocks:
            replay = load_replay(card)
            sim = FastCarbon(replay['configuration']).load_observation(replay['steps'][0][0]['observation'])
            sim.cash = CashEvents(sim.cash)
            timeline = defaultdict(list)
            for transition,frame in enumerate(replay['steps'][1:],1):
                sim.cash.events.clear()
                sim.step(*(r['action'] or {} for r in frame))
                for kind, owner, amount, *args in sim.cash.events:
                    if kind=='own_delivery' and amount>0:
                        timeline[owner,args[0]].append((transition,amount))
            clocks[card['id']] = timeline
        times, uids, x, bank = (data[k] for k in ('t','uid','x','bank'))
        ix = np.lexsort((uids,times))
        times,uids,x,bank = (z[ix] for z in (times,uids,x,bank))
        p = model.arrays(x)
        first = np.full((len(times),3),-1,int)
        first_steps = {uid:np.array([z[0] for z in seq],int) for (s,uid),seq in clocks[card['id']].items() if s==1-seat}
        for uid, ticks in first_steps.items():
            restricted = np.flatnonzero(uids==uid)
            tt = times[restricted]
            upcoming = np.searchsorted(ticks,tt,side='right')
            survives = upcoming<len(ticks)
            for j,h in enumerate(horizons):
                rr = restricted[survives]
                next_step = ticks[upcoming[survives]]
                within = next_step<=tt[survives]+h
                first[rr[within],j] = next_step[within]
        if not np.array_equal(first>=0,bank>0):
            raise ValueError('first deposit transition mismatch')
        g = dict(id=card['id'],seed=card['seed'],terrain=card['terrain'],observing_seat=seat,
                 opponent=card['names'][1-seat],opponent_id=card['ids'][1-seat],
                 focal=card['names'][seat],source_panel=card['panel'],terminal_turn=card['turns'],
                 official_scores=card['scores'],replay_sha256=card['sha256'],worker_turns=len(times),
                 turns_with_any_carrier=len(np.unique(times)),turns_without_loaded_opponent=card['turns']-len(np.unique(times)),
                 n_windows_ending_after_terminal={h:int(np.sum(times+h>card['turns'])) for h in horizons},
                 names_entirely_unseen=norm(card['names'][1-seat]) not in known)
        g['evaluation'] = assess(times,uids,x[:,0],x[:,1],p,bank,first=first)
        if card['panel']=='long_holdout' and seat==long_cards[card['id']]['seat']:
            base = official_roots[card['id']]
            retained = np.isin(times,base['steps'])
            orig = original_long[str(base['gid'])]
            value = np.mean((p[0][retained,2]-(bank[retained,2]>0))**2)
            err = abs(float(value)-orig['brier'])
            if not retained.any() or not np.isfinite(err) or err>1e-9:
                raise ValueError('signed original H10 selected-roots Brier disagreement '+str(err))
            prior_brier_error.append(err)
        games.append(g)

    panel = aggregate(games)
    panels = {'all':panel}
    for key in ('short_ladder','long_holdout'):
        subset=[g for g in games if g['source_panel']==key]
        panels[key]=aggregate(subset)
    for key, flag in [('seen_registration',False),('entirely_unseen_registration',True)]:
        subset=[g for g in games if g['names_entirely_unseen']==flag]
        panels[key]=aggregate(subset)
        panels[key]['opponent_names']=sorted({g['opponent'] for g in subset})
    for seat in (0,1):
        panels['observing_seat_'+str(seat)]=aggregate([g for g in games if g['observing_seat']==seat])
    by_name = {name:aggregate([g for g in games if g['opponent']==name]) for name in sorted({g['opponent'] for g in games})}
    # Exact prior selected-roots and short package comparisons are isolated from new full-trajectory metrics.
    oldshort=read(SHORT/'evaluation.json')['panels']['ladder']['events']
    short_brier_error={}
    for h in (3,5):
        for new_name,old_name in [('head','head'),('original_distance','distance')]:
            error=abs(panels['short_ladder']['horizons'][h]['map_weighted'][new_name]['brier']-
                      oldshort['bank'+str(h)][old_name]['brier'])
            short_brier_error[f'{h}_{new_name}']=error
            if error>1e-6:
                raise ValueError('signed original short-head Brier mismatch')
    examples = []
    for h in horizons:
        qualified=[]
        for g in games:
            event=g['evaluation']['alerts'][h]['head']['first_early_high']
            if event is None:continue
            qualified.append({**{k:g[k] for k in ('id','seed','observing_seat','opponent','source_panel','terrain')},**event})
        qualified.sort(key=lambda e:(e['id'],e['observing_seat'],e['t'],e['uid']))
        for hit in (False,True):
            selected=[r for r in qualified if (r['delivered']>0)==hit][:10]
            for r in selected:
                examples.append(dict(h=h,outcome='hit' if hit else 'miss',**r))
    if len(games)!=2*len(cards) or not panels['entirely_unseen_registration']['opponent_names']:
        raise ValueError('empty or incomplete evaluation')
    sources=[HERE/'protocol.json',HERE/'predict.py',HERE/'short_baseline.json',HERE/'metrics.py',Path(__file__),
             SHORT/'ladder_features.npz',SHORT/'sources.json',SHORT/'head.npz',SHORT/'model.py',
             LONG/'model.json',LONG/'predictor.py',LONG/'baseline.json',LONG/'dataset.json',
             ACTOR/'data_split.json',ACTOR/'data_receipts.json.gz',short.RULES/'opponent_intent.json',
             short.RULES/'opponent_intent.py',SAP/'turning_point_ledger.py',ROOT/'code/fastsim.py']
    report=dict(schema='public-ladder-deposit-horizons/1',code_and_data_sha256={str(f.relative_to(ROOT)):sha(f.read_bytes()) for f in sources},
                population='All 14 physical games of the signed short archive plus all 18 physical games of the signed long holdout. Both perspectives and every active worker-turn. Two observations of one physical match, not two played matches. Historical panels were already analyzed before these forecasts. No future or current actions enter model inputs.',
                label='Next 3/5/10 official transitions (cut at natural terminal). Only positive ordinary base cash by the same loaded opponent worker counts. Amount includes gains or losses since prediction. First transition is 1-based post-state on replay timeline. Zero is an observed no-deposit outcome.',
                money='A block alert is scored once from its pre-action start with known load >=100. Each method selects its own employee, hence alarm counts and identities differ. Exposure priority is load times probability; it is not a prediction of deposited cash. Distinct horizons or perspectives must not be summed. Brier and calibration are map-equal; alert counts are actual counts.',
                limitations=['An unseen registration name does not prove a different codebase or account. Anonymous strategy IDs can have only the public registration-name check.',
                             'Long public holdout originally sampled at most 32 times from the Master6 observer; here all decision times and both sides are examined. First-alarm failure cannot prove a successful interception.',
                             'Neither public cohort is a randomized sample of the full ladder and other historical replays were not assessed here.'],
                eligibility=coverage,games_inventory=cards,no_missing_signed_replays=True,
                parity=dict(signed_short_archive_worker_turns=panels['short_ladder']['horizons'][3]['worker_turns'],
                            short_historical_map_brier_error=short_brier_error,validated_long_maps=len(prior_brier_error),
                            long_selected_roots_historical_brier_max_absolute_error=max(prior_brier_error)),
                panels=panels,by_opponent_name=by_name,perspective_games=games,first_ten_success_failure_examples=examples)
    content=json.dumps(report,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    tmp=destination.with_suffix('.partial')
    tmp.write_bytes(gzip.compress(content,mtime=0));os.replace(tmp,destination)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('/tmp/sapling-deposit-ladder-v2'))
    parser.add_argument('--output',type=Path,default=HERE/'ladder_panel.json.gz')
    args=parser.parse_args()
    report=combine(args.cache,args.output)
    for key in ('all','seen_registration','entirely_unseen_registration'):
        panel=report['panels'][key]
        print(key,panel['maps'],panel['games'],{h:[round(x['brier'],6) for x in vals['map_weighted'].values()] for h,vals in panel['horizons'].items()})
