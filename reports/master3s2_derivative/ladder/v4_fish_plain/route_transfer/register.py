"""Freeze one decision point and one carrier rule before reading the eight held branch results."""
import json
from hashlib import sha256
from pathlib import Path
HERE=Path(__file__).resolve().parent
FAMILY=HERE.parents[2];LADDER=HERE.parents[1]
PARENT=HERE.parent/'late_cargo'
HELD=FAMILY/'v4_replication_dev/hosted'
def sha(raw):return sha256(raw).hexdigest()
def enforce(ok,why):
    if not ok:raise ValueError(why)
def protocol():
    source=HELD/'settlement.json';raw=source.read_bytes()
    enforce(sha(raw)=='f3c8212869dae39b1578728038e151bb1285c423c3dc5eaadf1e81bd4e723bf8','Different independent V4 games')
    d=json.loads(raw);games=[g for g in d['games'] if g['arm']=='challenger']
    expected={(s,k) for s in range(621758610,621758614) for k in (0,1)}
    enforce(d['records_reconciled'] and set((g['seed'],g['seat']) for g in games)==expected and len(games)==8,'Wrong map/seat panel')
    base=PARENT/'check.py';outcomes=PARENT/'outcomes.json';old=sha(base.read_bytes())
    enforce(old=='8df2c062ddf2c87172132a4d1065db0ab78a719145df9a25d50d5a809b97d69d','Changed source policy')
    return dict(scope='Retrospective fixed-response development check; same archived games measured earlier by other studies. No complete new match or formal map.',
        root_step=290,opponent='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5',
        v4='cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a',
        model='Choose one own collector with a geometric shortest path to a friendly base within remaining official transitions. Maximize visible cargo; tie by numeric worker suffix, ascending. Includes nonnegative cargo. If no eligible worker, retain original orders and count zero difference.',
        actions='From t290 through first positive own-base delivery by chosen worker, its disappearance or official termination, replace only that command with the least-tax procedure. Other live units take their original commands; no orders exist for newly created IDs, which stay. Removed IDs are skipped. Exact terminal payoff and actual cash reported separately.',
        outcomes='Primary equal-physical-map mean change in official margin (V4 minus opponent); report win flips and per-map/per-seat effects. If mean is nonpositive, stop this route without new games. Positive recorded-action results alone do not establish strength against responsive opponents.',
        additional_controls='At t290 reproduce recorded and least_tax official score, deposits and trace in earlier fish game for worker 27. Mirror physical player ordering on every checked root and compare commands. This fish root stays outside the eight-game mean.',
        qualifying_root_fraction='All eight included. Map/seat cannot be dropped for no opportunity, zero changes, collision or failure.',
        files={k:sha(v.read_bytes()) for k,v in {'base_policy':base,'fish_receipt':outcomes,'generic_policy':HERE/'policy.py','verification':HERE/'run.py','settlement':source}.items()},
        roots=[dict(seed=g['seed'],seat=g['seat'],terrain=g['terrain'],receipt=g['receipt_sha256']) for g in games],
        archive_cards=[dict(seed=a['seed'],seat=a['seat'],id=a['id'],sha256=a['sha256']) for a in d['archives']])
if __name__=='__main__':
    raw=(json.dumps(protocol(),indent=2,sort_keys=True,allow_nan=False)+'\n').encode();out=HERE/'protocol.json'
    enforce(not out.exists() or out.read_bytes()==raw,'Different freeze exists');out.write_bytes(raw);print(sha(raw))
