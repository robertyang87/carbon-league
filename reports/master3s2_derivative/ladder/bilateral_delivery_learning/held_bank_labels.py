"""Recover the three actual base credits separately from the frozen sixteen trajectories."""
import gzip
import json
import math
import zipfile
from held import HOSTED, SETTLEMENT_SHA, ECONOMY_SHA, RUN, STEPS, WINDOW, sha, read_bytes, require, cash_engine
from pathlib import Path

HERE=Path(__file__).resolve().parent
KINDS=('own_delivery','opponent_worker_delivery_to_this_base','base_collision_cash')
NARROW='fd4aae8f57f8af0be979baf5851e0c400f7f4f45673d4333a848e864897158c5'

def save_labels():
    old=json.loads(gzip.decompress(read_bytes(HERE/'held_predictions.json.gz',NARROW)))
    rows={(r['match'],r['step']):r for r in old['rows']}
    settle=json.loads(read_bytes(HOSTED/'settlement.json',SETTLEMENT_SHA))
    econ=json.loads(gzip.decompress(read_bytes(HOSTED/'economy.json.gz',ECONOMY_SHA)))
    accounts={(r['seed'],r['seat'],r['arm']):r for r in econ['games']}
    engine=cash_engine();cfg=engine.config();labels=[]
    ticket={(r['seed'],r['seat']):r for r in settle['archives']}
    for game in settle['games']:
        seed,side,arm=game['seed'],game['seat'],game['arm'];match=f'{RUN}:{seed}:{side}:{arm}'
        ar=ticket[seed,side];path=HOSTED/(str(ar['id'])+'.zip')
        require(sha(path.read_bytes())==ar['sha256'],'Source ZIP differs')
        with zipfile.ZipFile(path) as arc:
            book=engine.replay(game,arc,cfg);trace=book['transitions']
        for k in KINDS:
            for seat in (0,1):
                known=accounts[seed,side,arm]['by_seat'][seat]['cash_components'][k]
                engine.check_equal(book['signed_account_by_seat'][seat].get(k,0.),known,'Global booked '+k)
        for t in STEPS:
            if t+WINDOW>len(trace):continue
            r=rows[match,t];window=trace[t:t+WINDOW]
            components={k:[math.fsum(move['signed_cash_by_seat'][s].get(k,0.) for move in window)
                       for s in (side,1-side)] for k in KINDS}
            require(components['own_delivery']==r['target'],'Different ten-transition deposit window')
            labels.append({'match':match,'seed':seed,'terrain':r['terrain'],'seat':side,'source_arm':arm,'step':t,
                     'components':components,'all_bank_target':[sum(components[k][s] for k in KINDS) for s in (0,1)]})
    require(len(rows)==len(labels)==len({(r['match'],r['step']) for r in labels})==768,'Broken held denominator')
    doc={'source_settlement_sha256':SETTLEMENT_SHA,'v0_predictions_sha256':NARROW,
         'registration_sha256':sha((HERE/'bank_protocol.json').read_bytes()),'kinds':KINDS,'rows':labels}
    out=HERE/'held_bank_labels.json.gz'
    raw=gzip.compress((json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0)
    if out.exists() and out.read_bytes()!=raw:raise ValueError('Existing labels differ')
    if not out.exists():
        tmp=out.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(out)
    print(len(labels),sha(raw))
    return out
if __name__=='__main__':save_labels()
