"""Single native court for the economic-head comparison after seven and the qualified head development."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import sys
import time

from priority_slot import TRAIN,HERE,ROOT,sha,load,retain,check
from continue_intakes import verify_sealed
from source_games import gate
import official_game_replay

SLOT=HERE/'economy_slot'
DEV=TRAIN.parent/'economy_aux_dev/native_dev'
HEAD=TRAIN/'response_head'
ARMS=('parent','control','candidate')
REG='553f899c47df00482d3fba1f1e41d09ef3860026034deff3db2005c425a084fc'


def predecessor():
    folder=HERE/'batches/intake_07'
    if not (folder/'aggregate.json').exists():return None
    a=load(folder/'aggregate.json')
    link=a['source_intakes'][-1]
    check(link=={'dir':str(folder.relative_to(TRAIN)),'labels_sha256':sha(folder/'training_labels.json')},'different seventh-intake aggregate')
    out={'seventh_intake_sha256':sha(folder/'aggregate.json'),'training_counts':[a['distinct_maps'],a['filled_pairs']]}
    if a['distinct_maps']<8 or a['filled_pairs']<32:
        return {**out,'tree_head':'insufficient_support'}
    acceptance=HEAD/'independent_acceptance.json'
    if not acceptance.exists():return None
    head=load(acceptance)
    check(head['completion_sha256']==sha(HERE/'completion.json')
          and head['counts']['paid_pairs']==a['filled_pairs'] and head['passed']==load(HEAD/'status.json')['eligible_for_development'],
          'tree eligibility does not belong to the achieved data cutoff')
    out['head_acceptance_sha256']=sha(acceptance)
    if not head['passed']:return {**out,'tree_head':'held_terrain_failed'}
    d=HEAD/'development_runs';comparison=d/'comparison.json'
    if not comparison.exists():return None
    result=load(comparison)
    check(result['games']==8 and len(result['pairs'])==4 and not result['formal']
          and result['release_sha256']==sha(d/'release.json'), 'tree head development has not settled')
    verify_tree_development(d,result)
    return {**out,'tree_head':'eight_games_complete','head_development_sha256':sha(comparison)}


def verify_tree_development(directory,result):
    """Link every paired result to its distinct raw trajectory, board, policy, and original-engine score."""
    release=load(directory/'release.json')
    source=load(TRAIN/'protocol.json')
    maps=release['maps']
    check(maps==source['development_maps'] and len(set(maps))==2 and release['arena_sources']['data/master3s2_source.py']==source['examiner_sha256']
          and release['development_protocol']==sha(HEAD/'protocol.json') and release['master3s2_sha256']==source['examiner_sha256']
          and release['source_protocol']==sha(TRAIN/'protocol.json'),'different tree head development registration or examiner')
    expected={(seed,seat) for seed in maps for seat in (0,1)}
    paired={(r['seed'],r['seat']):r for r in result['pairs']}
    check(set(paired)==expected and len(result['pairs'])==4,'tree head comparison must cover both seats of both maps')
    for arm in ('parent','candidate'):
        d=directory/arm
        native=load(d/'results.json');official=load(d/'official_replay.json')
        header=native['header'];replay=official['runs'][arm];package=release['packages'][arm]
        check(native['status']=='complete' and header['seeds']==maps and header['cpus']==release['cpus']
              and header['source_sha256']==release['arena_sources'] and header['examiner_md5']==source['examiner_md5']
              and header['pack_sha256']==package['pack_sha256']==replay['pack_sha256'] and header['weight_sha256']==package['weight_sha256']
              and official['engine_source_sha256']==release['official_engine'],'unfinished or different tree head development arm')
        games={(r['seed'],r['candidate_seat']):r for r in native['games']}
        confirmed={(r['seed'],r['seat']):r for r in replay['games']}
        check(set(games)==set(confirmed)==expected and len(native['games'])==len(replay['games'])==4,
              'tree head development repeated or missing physical map-seat')
        for key,r in games.items():
            o=confirmed[key];g=paired[key][arm];seed,seat=key
            raw=d/r['raw_file']
            check(raw.parent.resolve()==d.resolve() and sha(raw)==r['raw_sha256']==o['transcript_sha256']
                  and all(g[k]==v for k,v in o.items()) and o['map_sha256']==release['boards'][str(seed)]
                  and o['statuses']==['DONE','DONE'] and o['official_reward_by_seat']==r['official_score_by_seat']
                  and o['steps']==r['actual_transitions'] and o['result']==r['result'],'tree head trajectory, paired score or terminal differs')
            score=o['official_reward_by_seat'];margin=score[seat]-score[1-seat]
            check(margin==g['margin']==r['margin'],'tree head physical-seat margin differs')


def ready(require_pass=True):
    reg=load(SLOT/'reservation.json');p=load(DEV/'protocol.json')
    check(reg['protocol_sha256']==REG==sha(DEV/'protocol.json') and list(p['physical_seats'])==[0,1]
          and p['parent_pack_sha256']==load(TRAIN/'protocol.json')['parent_pack_sha256'] and len(set(p['maps']))==2, 'economic experiment registration changed')
    rel=load(DEV/'release.json');m=load(DEV/'mechanism.json');clear=load(DEV/'clearance.json')
    check(rel['protocol_sha256']==REG and rel['parent_pack_sha256']==p['parent_pack_sha256']
          and rel['parent_weight_manifest_sha256']==load(TRAIN/'protocol.json')['parent_weight_manifest_sha256']
          and len({rel[f'{a}_pack_sha256'] for a in ARMS})==3 and p['examiner_sha256']==sha(ROOT/'data/master3s2_source.py'),
          'original examiner, model or three distinct policy identities differ')
    check(clear['declaration_sha256']==REG and clear['verdict']=='eligible_in_checked_scope' and set(map(int,clear['boards']))==set(p['maps'])
          and len(set(clear['boards'].values()))==2 and not set(p['maps'])&set(load(TRAIN/'protocol.json')['development_maps']),
          'wrong development maps')
    bound=(m.get('release_sha256')==sha(DEV/'release.json') or (
           m.get('control_pack_sha256')==rel['control_pack_sha256'] and m.get('candidate_pack_sha256')==rel['candidate_pack_sha256']))
    check(m['protocol_sha256']==REG and bound and bool(m['passed']) is require_pass,'mechanism pass or packing identity wrong')
    return p,rel,clear


def grant():
    previous=predecessor()
    if previous is None or not all((DEV/f).exists() for f in ('release.json','mechanism.json','clearance.json')):return None
    if not load(DEV/'mechanism.json').get('passed'):return None
    with (SLOT/'.state.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX)
        f=SLOT/'granted.json';check(not (HERE/'batches/intake_08/source/results.json').exists(),'eighth source games have already started')
        if not f.exists():verify_sealed(HERE/'batches/intake_07')
        _,rel,_=ready()
        card={**previous,'reservation_sha256':sha(SLOT/'reservation.json'),'registration_sha256':REG,'release_sha256':sha(DEV/'release.json'),
              'mechanism_sha256':sha(DEV/'mechanism.json'),'map_clearance_sha256':sha(DEV/'clearance.json'),
              'packs':{a:rel[f'{a}_pack_sha256'] for a in ARMS},'native_cpus':[0,1]}
        retain(f,card);return f


@contextmanager
def native_session():
    while not grant():time.sleep(8)
    with (SLOT/'.native-owner.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(not any((SLOT/f).exists() for f in ('released.json','skipped.json')),'economic native court has already been released')
        yield


def verify_completion(official):
    p,rel,clear=ready();g=load(SLOT/'granted.json');result=load(DEV/'comparison.json')
    check(g['release_sha256']==sha(DEV/'release.json') and g['registration_sha256']==REG
          and result['schema']=='sapling-economy-aux-three-arm-native-v1' and set(result['competitors'])==set(ARMS)
          and len(result['paired_map_seats'])==4 and all(a in row for row in result['paired_map_seats'] for a in ARMS),
          'wrong three-arm study or incomplete comparison')
    expected={(s,v) for s in p['maps'] for v in (0,1)};arena=None;hashes={}
    for arm in ARMS:
        d=DEV/arm;h=load(d/'results.json');o=load(d/'official_replay.json');head=h['header']
        s=rel[f'{arm}_pack_sha256'];w=rel['parent_weight_manifest_sha256'] if arm=='parent' else sha(DEV/'release.json')
        files={k:sha(d/f) for k,f in [('results_sha256','results.json'),('official_sha256','official_replay.json')]}
        check(all(result['competitors'][arm][k]==v for k,v in files.items()) and h['status']=='complete' and
              head['pack_sha256']==s==o['runs'][arm]['pack_sha256'] and head['weight_sha256']==w and
              head['examiner_md5']==p['examiner_md5'] and head['cpus']==[0,1] and head['seeds']==p['maps'],
              'native report, policy, original opponent, map list or CPU allocation mismatch')
        common=head['source_sha256'];check(common['data/master3s2_source.py']==p['examiner_sha256'] and (arena is None or common==arena),
                                       'native arms used different source snapshots');arena=common
        idx={(x['seed'],x['candidate_seat']):x for x in h['games']};off={(x['seed'],x['seat']):x for x in o['runs'][arm]['games']}
        check(set(idx)==set(off)==expected and len(idx)==len(h['games'])==len(o['runs'][arm]['games'])==4,'native map-seat is missing or duplicated')
        for k,row in idx.items():
            raw=d/row['raw_file'];replay=off[k];gate.validate_game(row);gate.validate_examiner_configuration(load(raw)['configuration'])
            check(sha(raw)==row['raw_sha256']==replay['transcript_sha256'] and row['official_score_by_seat']==replay['official_reward_by_seat']
                  and replay['map_sha256']==clear['boards'][str(k[0])] and replay['statuses']==['DONE','DONE'],'native trajectory or outcome mismatch')
        if official:check(official_game_replay.audit(d)==o['runs'][arm],'original-engine economic score reconstruction differs')
        hashes[arm]=files
    return {'grant_sha256':sha(SLOT/'granted.json'),'comparison_sha256':sha(DEV/'comparison.json'),'signed_games':12,
            'arms':hashes,'verdict':result['verdict']}


def release():
    with (SLOT/'.state.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX);f=SLOT/'released.json'
        check(not (SLOT/'skipped.json').exists(),'economic study was skipped')
        retain(f,verify_completion(not f.exists()));return f


def skip():
    _,_,_=ready(require_pass=False)
    why=load(DEV/'mechanism_skip.json')
    check(why['protocol_sha256']==REG and why['mechanism_sha256']==sha(DEV/'mechanism.json') and not (SLOT/'released.json').exists(), 'no registered pre-play failure')
    for arm in ARMS:
        d=DEV/arm;check(not any(d.glob('raw_*')) and (not (d/'results.json').exists() or not load(d/'results.json')['games']),
                       'cannot skip a started economic comparison')
    f=SLOT/'skipped.json';retain(f,{'protocol_sha256':REG,'skip_sha256':sha(DEV/'mechanism_skip.json'),'signed_games':0});return f


def wait_before_source(wd,plan):
    if wd.resolve()!=HERE/'batches/intake_08':return
    while True:
        if (SLOT/'released.json').exists():
            check(load(SLOT/'released.json')==verify_completion(False),'economic development release differs');return
        if (SLOT/'skipped.json').exists():skip();return
        grant();time.sleep(8)
