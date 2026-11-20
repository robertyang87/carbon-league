"""Sign every frozen ticket using only the two public pre-action alternatives and the frozen model."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0,str(STUDY))
from model import rr_features, rr_predict, rr_scores


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction(delta):
    return 'A' if delta>0 else 'B' if delta<0 else 'tie'


def score(r,w):
    ticket=r['ticket'];out=dict(ticket=ticket,status=r['status'],public_sha=r.get('public_sha'),menu_sha=r.get('menu_sha'))
    choices=r.get('orders');native=r.get('native');dep=r.get('dep')
    if not choices or len(choices)!=2 or native is None or len(native)!=2:
        out.update(scored=False,reason='No complete pair of native orders and scores');return out
    if not r.get('players') or any(v is None for v in native):
        out.update(scored=False,reason='No public players or both comparable native values');return out
    assert isinstance(choices[0],dict) and isinstance(choices[1],dict) and choices[0]!=choices[1]
    s=ticket['seat'];step=ticket['step'];players=r['players'];assert s in (0,1) and len(players)==2
    expected=rr_scores(players,s,step,choices,w);worker=[]
    for a,command in enumerate(choices):
        features,carriers=rr_features(players,s,step,command);prob=rr_predict(features,w['enhanced']);arm=[dict(uid=u,current_cargo=load,order=command.get(u,'NONE'),estimated_receipt_fraction=float(p),estimated_money=float(p*load)) for p,(u,load) in zip(prob,carriers,strict=True)];assert abs(sum(row['estimated_money'] for row in arm)-expected[a])<1e-7;worker.append(arm)
    assert all(np.isfinite(expected)) and all(np.isfinite(native));difference=float(expected[0]-expected[1]);native_gap=float(native[0]-native[1]);terminal_gap=native_gap+w['pair_scale']*difference
    top_two=r.get('native_top_two');tags=r.get('tags',[])
    rolls=r.get('rolls'); cut=r.get('cut');inwindow=step in w['steps'] and dep==[w['depth'],w['depth']]
    if top_two is True and rolls==1 and cut is False and inwindow:
        support='inside original action/time/work domain; released model remains unqualified'
    elif not inwindow or top_two is False or (rolls is not None and rolls!=1) or cut is True:
        support='diagnostic extrapolation'
    else:
        support='domain metadata incomplete'
    out.update(scored=True,orders=choices,native=native,depth=dep,native_top_two=top_two,rolls=rolls,cut=cut,tags=tags,coverage=support,
        present_cargo=worker,predicted_money=list(map(float,expected)),money_A_minus_B=difference,predicted_receipt_order=direction(difference),
        native_A_minus_B=native_gap,pair_scale=w['pair_scale'],native_plus_correction_A_minus_B=terminal_gap,
        predicted_terminal_order=direction(terminal_gap),selected_if_scored='B' if terminal_gap<0 else 'A',baseline_native_pick='B' if native_gap<0 else 'A')
    return out


def main():
    prereg=json.loads((HERE/'model_protocol.json').read_bytes());path=STUDY/'weights.json';assert digest(path)==prereg['weight_sha256'] and digest(STUDY/'model.py')==prereg['inference_source_sha256'];w=json.loads(path.read_bytes());menus=HERE/'menus.json';reg=HERE/'protocol.json';data=json.loads(menus.read_bytes());rows=data['rows'];assert 0 < len(rows) <= 6 and len({r['ticket']['seed'] for r in rows})==len(rows) and data['protocol_sha256']==digest(reg)=='f8b83f5bd6f73c9116ffadd7921d3a14d70632c2363700f9ca49f60a91dc9793'
    training={x['terrain'] for k in ('ordinary','pairs') for x in json.loads((STUDY/'protocol.json').read_bytes())[k] if x['split']=='train'}
    known={x['seed']:x['terrain'] for x in json.loads(reg.read_bytes())['sources']}
    assert {known[r['ticket']['seed']] for r in rows}.isdisjoint(training)
    out=dict(protocol_sha256=digest(reg),model_protocol_sha256=digest(HERE/'model_protocol.json'),script_sha256=digest(Path(__file__)),weights_sha256=digest(path),model_sha256=digest(STUDY/'model.py'),menus_sha256=digest(menus),rows=[score(r,w) for r in rows]);raw=(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();dest=HERE/'model_blind.json';tmp=HERE/'.model_blind.tmp';
    with tmp.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    try:os.link(tmp,dest)
    except FileExistsError:
        if dest.read_bytes()!=raw:raise ValueError('A different blind prediction has already been frozen')
    finally:tmp.unlink()
    print(digest(dest),[(r['ticket']['seed'],r.get('money_A_minus_B'),r.get('predicted_terminal_order')) for r in out['rows']],flush=True)

if __name__=='__main__':main()
