"""Independently reapply the four command tapes with no model calls and settle their cash."""
import collections
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent; D=HERE.parents[1];sys.path.insert(0,str(D.parents[1]/'code/env_src/carbon_challenge-master'))


def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


f=module('rr_cash_replay',D/'freight_response/run.py');live=f.load('rr_cash_official',f.STUDY/'cargo_tree_action_learning_dev/ladder_live/live.py')
sha=lambda b:hashlib.sha256(b).hexdigest()


def reapply(game,c,zip_path):
    rows=game['rows'];s=c['seat'];t=c['step'];basename=c['arm']+'/'
    with zipfile.ZipFile(zip_path) as z:
        reg=z.read(basename+'registration.json'); turn_raw=z.read(basename+'turns.jsonl');receipt=json.loads(z.read(basename+'receipt.json'))
        assert sha(reg)==receipt['protocol_sha256']==c['registration_sha256'] and sha(turn_raw)==receipt['turns_sha256']
        turns=list(map(json.loads,turn_raw.splitlines()))
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);p=tmp/'r.json';p.write_bytes(reg);live.configure(p,tmp/'unused',tmp/'unused.lock');env,cfg=live.official()
    sim=f.econ.FastCarbon(cfg).reset(seed=c['seed']);sim.cash=f.econ.CashEvents(sim.cash);bank=[5.,5.]
    assert sha(bytes(env.states[0].observation.full_carbon))==c['terrain']
    for i in range(t):
        turn=turns[i];pair=[q['response']['action'] for q in turn['calls']];assert i==turn['step']
        assert [f.digest(f.public(env,k,bank)) for k in (0,1)]==turn['public_observation_sha256']
        env.step(pair,[]);sim.step(*pair);bank=turn['overage_remaining_seconds']
    assert list(sim.cash)==game['public_pair']['incoming_cash'] and rows[0]['orders'][1-s]==game['root_shared_original_order']
    totals=[collections.Counter(),collections.Counter()];before=tuple(sim.cash)
    for i,r in enumerate(rows,t):
        assert r['t']==i and not env.done and [f.digest(f.public(env,k,bank)) for k in (0,1)]==r['input_sha256']
        pair=r['orders'];sim.cash.events.clear();sim.step(*pair);env.step(pair,[])
        assert list(sim.cash)==r['end_cash']==[p[0] for p in env.states[0].observation.players]
        assert r['flow']==[dict(x) for x in f.econ.cash_flow(sim.cash.events)]
        for k in (0,1):totals[k].update(r['flow'][k])
        bank=[b-max(0,q['wall']-1) for b,q in zip(bank,r['clocks'])]
    assert env.done and all(st.status=='DONE' for st in env.states) and [float(st.reward) for st in env.states]==game['outcome']['scores']
    assert all(abs(sum(totals[k].values())-sim.cash[k]+before[k])<1e-5 for k in (0,1))
    return dict(official_scores=game['outcome']['scores'],verified_suffix_transitions=len(rows),verified_prefix_transitions=t,
                verified_by_seat_terminal_balances=list(sim.cash),two_seat_public_digests_checked=len(rows)*2)


def main():
    tickets=json.loads((HERE/'protocol.json').read_bytes())['tickets'];results=json.loads((HERE/'results.json').read_bytes())['roots'];assert len(results)==len(tickets)==2
    audit=[]
    for ticket,root in zip(tickets,results,strict=True):
        g=[];z=D/ticket['archive'];assert sha(z.read_bytes())==ticket['archive_sha256']
        for arm in ('parent','rotated'):
            path=HERE/f"{ticket['seed']}_{ticket['seat']}_{ticket['step']}_{arm}.json.gz";raw=path.read_bytes();e=json.loads(gzip.decompress(raw));g.append(e)
            assert e['rows'][0]['orders'][ticket['seat']]==ticket['orders'][arm] and sha(raw)==root['arms'][arm]['evidence_sha256']
            audit.append(dict(arm=arm,seed=ticket['seed'],seat=ticket['seat'],branch_sha256=sha(raw),**reapply(e,ticket,z)))
        for key in ('native_incoming_h','registered_v2_root_rng_sha','public_pair','root_shared_original_order'):
            assert g[0][key]==g[1][key]
    doc=dict(independent_action_only_replay=True,runner_sha256=sha(Path(__file__).read_bytes()),evidence=audit,
             verified_response_transitions=sum(r['verified_suffix_transitions'] for r in audit))
    p=HERE/'audit.json';r=(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();tmp=p.with_suffix('.partial');tmp.write_bytes(r);tmp.replace(p);print(sha(r),doc['verified_response_transitions'])


if __name__=='__main__':main()
