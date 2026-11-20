"""Freeze two paired native response streams from the exposed 5906206b root."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[1]
FAMILY = LADDER.parent
ROOT = FAMILY.parents[1]
AA = LADDER/'v4_alphacarbon_live_response'
spec = importlib.util.spec_from_file_location('archived_registration', AA/'register.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
sha, encode, require, save, card = old.sha, old.encode, old.require, old.save, old.card
FREIGHT, V4, ORIGINAL = old.FREIGHT, old.V4, old.ORIGINAL
RAW = LADDER/'v4_observations/increment_0926/replays/5906206b61c94c29d8ea2f99d2b4d62d-1.json.gz'
IDS = ('player-0-worker-68','player-1-worker-56')
GUARD = 'player-1-worker-68'
COMMANDS = ('RIGHT','RIGHT','DOWN','NONE')


def register():
    game = json.loads(gzip.decompress(RAW.read_bytes()))
    start = 286
    players = game['steps'][start][0]['observation']['players']
    units = [players[s][2][u] for s,u in enumerate(IDS)]
    require(units == [[54,0,'COLLECTOR'],[72,132.9675,'COLLECTOR']] and
            players[1][2][GUARD]==[71,0,'COLLECTOR'] and game['rewards']==[6245.53,6493.12], 'Different root')
    require([sha(p.read_bytes()) for p in (V4,ORIGINAL)]==list(old.PIN[1:]),'Different policy')
    tickets=[]
    for i,order in enumerate((('natural','intercept_return'),('intercept_return','natural'))):
        seeds=[int.from_bytes(hashlib.sha256(f'5906206b/t286/intact/20261120/stream-{i}/seat-{s}'.encode()).digest()[:4],'big') for s in (0,1)]
        tickets.append(dict(stream=i,order=order,seed_by_seat=seeds))
    return dict(schema='exposed-krypton-root-two-native-streams/1',step=start,seat=0,
                recorded_opponent='Krypton (ladder registration); original source and private random state of entrant unknown',
                official_terminal_step=299,maximum_joint_transitions=52,cpu=4,streams=tickets,
                root_cash=[p[0] for p in players],root_units=dict(zip(IDS,units)),guard=GUARD,
                prefix_orders=sha(encode([[r['action'] or {} for r in frame] for frame in game['steps'][1:start+1]])),
                historical_margin=-247.59,historical_fixed_response_delta=454.32,
                policy={'forced_on_opening_if_feasible':COMMANDS,'afterwards':'From t290 until terminal the same living worker uses hurry from its own current public observation and native friendly commands; retain the native command if infeasible. No other order may change.'},
                comparison='Official terminal rewards, +/- v4 margin within private stream, worker 68 delivery and rivals 56/68 movements, deaths and receipts. No resampling. Historical scores are used for source verification only.',
                executables={'v4':card(V4),'official_original_copy':card(ORIGINAL),'replay':card(RAW),
                             'prefix_and_snapshots':card(AA/'run.py'),'native_search':card(FREIGHT),
                             'fixed_response_contract':card(HERE.parent/'actions/response.py'),
                             'hurry':card(LADDER/'v4_alphacarbon_decisions/branches.py'),
                             'feasible':card(LADDER/'v4_alphacarbon_decisions/scan.py'),'runner':card(HERE/'run.py')})

if __name__=='__main__':
    print(save(HERE/'protocol.json',register()))
