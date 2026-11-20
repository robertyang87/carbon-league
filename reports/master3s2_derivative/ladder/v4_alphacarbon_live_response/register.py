"""Freeze the exposed map, natural response streams and the three forced commands."""
from hashlib import sha256
from pathlib import Path
import json
import gzip


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
FAMILY = LADDER.parent
ROOT = FAMILY.parents[1]
RAW = LADDER / 'v4_observations/increment_0904/replays/6e994d80674dc2430a4925b71bb6a947-1.json.gz'
FREIGHT = FAMILY / 'freight_response/run.py'
V4 = FAMILY / 'advantage_policy_update/v2_bc.py'
ORIGINAL = ROOT / 'data/master3s2_source.py'
IDS = ('player-0-worker-53', 'player-1-worker-6')
PIN = ('91fc071bd9e395d02297c73f03835711832cec8b41b73ad562aa54268e3e3fbb',
       'cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a',
       '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5')
def sha(raw): return sha256(raw).hexdigest()
def encode(data): return (json.dumps(data,allow_nan=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def card(path): return {'path':str(path.relative_to(ROOT)), 'sha256':sha(path.read_bytes())}
def require(ok,msg):
    if not ok: raise ValueError(msg)
def save(path,doc):
    raw=encode(doc)
    if path.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    if path.exists():require(path.read_bytes()==raw,'Would replace existing evidence '+str(path))
    else:
        tmp=path.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(path)
    return sha(raw)
def register():
    game=json.loads(gzip.decompress(RAW.read_bytes()));steps=game['steps'];start=267
    require(tuple(sha(p.read_bytes()) for p in (RAW,V4,ORIGINAL))==PIN,'Inputs differ')
    views=[r['observation'] for r in steps[start]]
    units=[views[0]['players'][s][2][uid] for s,uid in enumerate(IDS)]
    require([a[0] for a in units]==[155,214] and len(steps)==300,'Different root or game')
    tickets=[]
    for ix,order in enumerate([['natural','three_moves'],['three_moves','natural']]):
        seeds=[int.from_bytes(sha256(f'6e994d80/t267/real-original/stream-{ix}/seat-{s}'.encode()).digest()[:4],'big') for s in (0,1)]
        tickets.append(dict(stream=ix,seed_by_seat=seeds,order=order))
    return {'schema':'single-exposed-ladder-position-natural-original-responses/1','seat':0,'step':start,
            'amendment':{'prior_sha256':sha((HERE/'preflight.json').read_bytes()), 'detail':'First local setup stopped at snapshot identity validation before any suffix transition. Bind the identical state-saver separately to the V4 and original SHA. Streams, arms, order, map and endpoints unchanged.'},
            'source_ladder_opponent':'claude-AlphaCarbon-Master3s2','archive_random_seed':game['configuration']['randomSeed'],
            'root_units':dict(zip(IDS,units)),'worker_by_seat':IDS,'v4_action_override':['DOWN','LEFT','NONE'],
            'root_cash':[p[0] for p in views[0]['players']],'map':sha(bytes(steps[0][0]['observation']['full_carbon'])),
            'prefix_orders':sha(encode([[row['action'] or {} for row in frame] for frame in steps[1:start+1]])),
            'executables':{'v4':card(V4),'official_original_copy':card(ORIGINAL),'archived_replay':card(RAW),
                           'suffix_runner':card(HERE/'run.py'),'public_search_and_economy':card(FREIGHT),
                           'historical_suffix_reference':card(FAMILY/'root_rotation_policy/response/run.py')},
            'streams':tickets,'physical_cpu':4,'maximum_transitions':128,'terminal_rule':'Official DONE or step cap. No resampling or dropped outcomes.',
            'intervention':'Only V4 worker 53 at 267/268/269; if no longer alive simply record absence. All its other decisions and those of the intact rival use native public agents, natural T1 and 0.58s each. Friendly collisions permitted by the game.',
            'matching':'Both arms per stream share full recorded public history, actual historical commands, fresh specified private random states, timing cost warmed at t266 and cached independent simultaneous root decisions. Thereafter each side uses its own public perspective. Wall budgets remain native.',
            'limits':'Exposed, retrospectively selected position. The original ladder contestant and its private RNG are unknown. Cash inputs and the terrain are checked by both engines; the private draws are new. A pinned development CPU is not a tournament sandbox.'}

if __name__=='__main__': print(save(HERE/'protocol.json',register()))
