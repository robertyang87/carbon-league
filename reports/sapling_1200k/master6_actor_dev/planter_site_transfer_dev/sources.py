"""Freeze intact authentic and elite replay identities by terrain; no outcomes choose the split."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import gzip
import json
import math
import sys

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
BASE=ACTOR.parent
ROOT=BASE.parents[1]
CACHE=Path('/workspace/sapling-all-top5-20261119')
sys.path.insert(0,str(BASE))
from tree_win_gate import validate_examiner_configuration

def sha(p):
    return sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())

def save(p,obj):
    b=json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if p.suffix=='.gz':b=gzip.compress(b,mtime=0)
    p.write_bytes(b)

def make():
    alt=read(ACTOR/'two_arm_plant_economy_dev/native_eval_rows.json.gz')
    ext={r['map'] for r in alt if r['historical_use']=='outside'}
    main={r['map'] for r in alt if r['map'] not in ext}
    assert len(main)==16 and len(ext)==2
    deny=set(main)|set(ext)
    for name in ('first_win_dev','first_win_blend_dev'):
        deny.update(read(ACTOR/name/'clearance.json')['physical_boards'].values())
    subset=read(ACTOR/'opponent_cargo_calibration_dev/sources.json')
    # The prior pay-policy register ties each of its booked seeds to a terrain.
    payseeds=set(read(ACTOR/'tree_tenure_dev/pay_policy/protocol.json')['new_dev']['seeds'])
    prev=read(ACTOR/'own_search_bridge_dev/opponent_response_dev/sources.json')['records']
    indexed={}
    def add(f,expected=None):
        raw=read(f); side=raw['candidate_seat']; g=raw['seed']; terrain=raw['initial_terrain_u8_sha256']
        external=terrain in ext and f.parent==ACTOR/'actor_partner_dev/native/candidate'
        if (terrain in deny and not external) or g in payseeds or (terrain,1-side) in indexed:return
        if expected:assert sha(f)==expected
        hf=f.parent; named=hf/f'receipt_{g}_{side}.json'
        if named.exists():cert=read(named)['official']
        elif (hf/'official_replay.json').exists():
            named=hf/'official_replay.json';cert=next(x for x in read(named)['runs'][hf.name]['games'] if x['seed']==g and x['seat']==side)
        else:
            named=hf/f'official_seat{side}.json';cert=read(named)['official']
        validate_examiner_configuration(raw['configuration'])
        assert raw['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1' and not raw['errors'] and not raw['audit_messages']
        assert cert['transcript_sha256']==sha(f) and cert['statuses']==['DONE','DONE'] and cert['map_sha256']==terrain
        assert list(cert['official_reward_by_seat'])==list(raw['official_score_by_seat'])
        indexed[terrain,1-side]=dict(domain='native',terrain=terrain,seed=g,seat=1-side,actor='original Master3s2',
            source=str(f.relative_to(ROOT)),sha256=sha(f),certificate=str(named.relative_to(ROOT)),certificate_sha256=sha(named),
            split='exposed_loss_panel' if external else '',teacher_scores=cert['official_reward_by_seat'])
    for d in ('actor_partner_dev/native/candidate','actor_partner_dev/replication/games/candidate','blended_actor_arena/games/parent'):
        for f in sorted((ACTOR/d).glob('raw_*.json.gz')):add(f)
    for g in prev:add(ROOT/g['raw'],g['raw_sha256'])
    for g in subset['native']:add(ROOT/g['source'],g['sha256'])
    elite={x['name'] for x in read(BASE/'all_top5/manifest.json')['top5']}
    for r in subset['ladder']:
        if r['focal'] not in elite:continue
        f=CACHE/r['source']; assert f.exists() and sha(f)==r['sha256'] and r['terrain'] not in deny
        assert (r['terrain'],r['seat']) not in indexed
        raw=read(f);assert raw['statuses']==['DONE','DONE']
        indexed[r['terrain'],r['seat']]=dict(domain='elite',terrain=r['terrain'],seed=r['seed'],seat=r['seat'],
            actor=r['focal'],other=r['opponent'],source=r['source'],sha256=r['sha256'],split='',teacher_scores=raw['rewards'],
            archive=r['archive'],archive_git_blob=subset['archive_git_blob'],archive_sha256=subset['archive_sha256'])
    rows=list(indexed.values())
    for domain in ('elite','native'):
        eligible=[r for r in rows if r['domain']==domain and not r['split']]
        maps=sorted({r['terrain'] for r in eligible},key=lambda g:sha256(('planter-site-v1|'+g).encode()).hexdigest())
        held=set(maps[:math.ceil(len(maps)/3)])
        for r in eligible:r['split']='held' if r['terrain'] in held else 'fit'
    for r in rows:
        a,b=r['teacher_scores'][r['seat']],r['teacher_scores'][1-r['seat']]
        r['teacher_result']='win' if a>b else 'loss' if a<b else 'draw'
    assert not ({r['terrain'] for r in rows if r['domain']=='native'} & {r['terrain'] for r in rows if r['domain']=='elite'})
    doc={'protocol_sha256':sha(HERE/'protocol.json'),'source_sha256':sha(Path(__file__)),'records':rows,
         'main_branch_terrains':len(main),'external_branch_terrains':len(ext),
         'counts':dict(Counter('/'.join([r['domain'],r['split'],r['teacher_result']]) for r in rows))}
    save(HERE/'sources.json',doc)
    print(doc['counts'],flush=True)
if __name__=='__main__':make()
