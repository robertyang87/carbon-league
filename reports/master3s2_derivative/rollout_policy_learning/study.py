"""Learn a tiny role-conditioned action residual from the last public physical move."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp

HERE = Path(__file__).resolve().parent
DERIV = HERE.parent
PREVIOUS = DERIV / 'opponent_rollout_calibration'
sys.path.insert(0, str(PREVIOUS))
from study import initialize, mapping, read, events
from runtime import adjust
PLAN = json.loads((HERE / 'protocol.json').read_text())
TEMP = json.loads((PREVIOUS / 'weights.json').read_text())['betas']['roles']
ROLE = ('COLLECTOR', 'PLANTER')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    raw = (json.dumps(obj, indent=2, allow_nan=False, ensure_ascii=False) + '\n').encode()
    if path.suffix == '.gz':
        raw = gzip.compress(raw, mtime=0)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f'Output changed: {path}')
    path.write_bytes(raw)
    return sha(raw)


def motion(m, units, uid, pos):
    phi = np.zeros((5, 4))
    if uid not in units:
        return phi
    prev = units[uid][0]
    matches = [i for i in range(5) if m.move(prev, i) == pos]
    if not matches:
        raise ValueError('motion exceeds unit speed')
    k, = matches
    if k == 0:
        phi[0, 3] = 1
    else:
        phi[k, 0] = 1
        back, = [i for i in range(1, 5) if m.move(pos, i) == prev]
        phi[back, 1] = 1
        phi[0, 2] = 1
    return phi


def ladder(card):
    raw = (DERIV / card['path']).read_bytes()
    assert sha(raw) == card['sha256']
    doc = json.loads(gzip.decompress(raw)); frames = doc['steps']; seat = card['seat']
    terrain = sha(bytes(frames[0][0]['observation']['full_carbon']))
    tape = []
    for t, (a, b) in enumerate(zip(frames, frames[1:])):
        pub = {k: a[0]['observation'][k] for k in ('players', 'trees', 'step')}
        pub.update({k: a[seat]['observation'][k] for k in ('carbon', 'player', 'remainingOverageTime')})
        assert pub['player'] == seat and pub['step'] == t
        tape.append((pub, [v['action'] or {} for v in b]))
    assert len(tape) == 299
    return tape, doc['configuration'], terrain


def gather(m, cards, hosted):
    schema = json.loads((DERIV.parents[1] / 'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_text())
    cfg = {k:v['default'] for k,v in schema['configuration'].items() if isinstance(v,dict) and 'default' in v}
    roots, skipped, terrains = [], [], []
    for ticket in cards:
        if hosted:
            tape, _ = read(ticket)
            group, terrain = ticket['map'], ticket['map']
        else:
            tape, cfg, terrain = ladder(ticket); group = ticket['opponent']
        terrains.append(terrain)
        seat = ticket['seat']; op = 1-seat
        tr = m.BeliefTracker(m._Bel,seat)
        for t,(pub,issued) in enumerate(tape):
            est,_ = tr.step(pub['players'],t,pub['carbon'])
            if 2 <= t <= 292 and (t-2)%5 == 0:
                nx, target = tape[t+1]
                proj = m._turn(dict(players=pub['players'],step=t,trees=pub['trees'],player=seat,
                                carbon=list(map(float,est)),remainingOverageTime=pub['remainingOverageTime']),cfg,issued)
                ps = proj['players']
                ids = mapping(ps, nx['players'])
                if ids is None:
                    skipped.append([group,seat,t+1]); tr.set_actions(pub['players'],issued[seat]); continue
                x = m._quant(m.encode(ps,t+1,op,proj['carbon'],[1.]*m.N,pub['players'])).astype(np.float32)
                w,b,_ = m._RNET(x[None],m._SID); w = w[0]; b = b[0]
                slots = []
                for uid,(pos,cargo,role) in ps[op][2].items():
                    assert uid.startswith('player-'+str(op)+'-')
                    slots.append(dict(id=uid,role=role,pos=pos,cargo=cargo,phi=motion(m,pub['players'][op][2],uid,pos),
                                      target=m.WIDX[target[op].get(ids[op][uid],'NONE')],z=w[pos].astype(np.float64)))
                for uid,pos in ps[op][1].items():
                    slots.append(dict(id=uid,role='BASE',pos=pos,cargo=0.,target=m.BIDX[target[op].get(ids[op][uid],'NONE')]))
                own_dest = {m.move(v[0],m.WIDX[target[seat].get(ids[seat][u],'NONE')]) for u,v in ps[seat][2].items()}
                roots.append(dict(group=group,terrain=terrain,seat=seat,step=t+1,op=op,players=ps,
                                  slots=slots,own_dest=own_dest,worker=w,base=b,public_unit_input=sha(x.tobytes())))
            tr.set_actions(pub['players'],issued[seat])
    return roots, skipped, sorted(set(terrains))


def fit(roots):
    out = {}
    for role in ROLE:
        rows = [(g['group'],s) for g in roots for s in g['slots'] if s['role']==role]
        label = np.array([s['target'] for _,s in rows]); phi = np.stack([s['phi'] for _,s in rows])
        z = TEMP[role]*np.stack([s['z'] for _,s in rows])
        groups, counts = np.unique([g for g,_ in rows],return_counts=True)
        wt = dict(zip(groups,1/(len(groups)*counts),strict=True)); weights = np.array([wt[g] for g,_ in rows])
        def objective(v):
            logits = z+phi@v
            ll = logsumexp(logits,axis=1) - logits[np.arange(len(label)),label]
            residual = softmax(logits,axis=1); residual[np.arange(len(label)),label]-=1
            return weights@ll+.01*(v@v),np.einsum('n,na,nai->i',weights,residual,phi)+.02*v
        opt=minimize(objective,np.zeros(4),jac=True,method='L-BFGS-B',bounds=[(-4.,4.)]*4)
        assert opt.success
        out[role]=dict(coef=opt.x.tolist(),n=len(rows),penalized_NLL_before=objective(np.zeros(4))[0],penalized_NLL_after=opt.fun)
    return out


def score(m, roots, weights):
    rows = []
    for g in roots:
        slots = g['slots']; true = np.array([s['target'] for s in slots])
        raw=(g['worker'],g['base'])
        calibrated=adjust(g['players'],g['op'],*raw,TEMP)
        rw=calibrated[0].copy()
        for s in slots:
            if s['role'] in ROLE:
                rw[s['pos']] += s['phi']@weights[s['role']]['coef']
        results = {}
        for method, logits in [('raw',raw),('temperature',calibrated),('motion',(rw,calibrated[1]))]:
            physical=[]
            for j in range(16):
                key=(1000003*g['step']+17*g['seat'])%(2**31)+7919+31*j+2
                cmd=m._sample(g['players'],g['op'],*logits,np.random.RandomState(key),g['step'])
                physical.append([(m.WIDX if s['role'] in ROLE else m.BIDX)[cmd.get(s['id'],'NONE')] for s in slots])
            simulated=np.array(physical); summary={}
            for name in ['workers',*ROLE,'loaded','BASE']:
                brier=[];top=[];nll=[]
                for ix,s in enumerate(slots):
                    role=s['role']
                    if not ((name=='workers' and role in ROLE) or name==role or (name=='loaded' and role=='COLLECTOR' and s['cargo']>=50)):continue
                    size=3 if role=='BASE' else 5
                    probs=np.bincount(simulated[:,ix],minlength=size)/16
                    brier.append(float(np.square(probs-np.eye(size)[true[ix]]).sum()))
                    top.append(int(probs.argmax()==true[ix]))
                    z=logits[role=='BASE'][s['pos']].astype(np.float64);nll.append(float(logsumexp(z)-z[true[ix]]))
                if brier:
                    summary[name]=[len(brier),float(np.mean(brier)),sum(top),float(np.mean(nll))]
            observed_event=events(m,g,slots,true)
            grouped=np.bincount([events(m,g,slots,a) for a in physical],minlength=8)/16
            results[method]=dict(metrics=summary,exact=sum(int(all(a==true)) for a in simulated),
                                 event_brier=float(np.square(grouped-np.eye(8)[observed_event]).sum()))
        rows.append({k:g[k] for k in ('group','terrain','step','seat','public_unit_input')}|dict(result=results))
    return rows


def summarize(rows):
    books={}
    for group in sorted({r['group'] for r in rows}):
        rr=[r for r in rows if r['group']==group]; books[group]={}
        for method in ('raw','temperature','motion'):
            rs=[r['result'][method] for r in rr];report={}
            for key in ['workers',*ROLE,'loaded','BASE']:
                a=[r['metrics'][key] for r in rs if key in r['metrics']]
                report[key]=dict(unit_rows=sum(v[0] for v in a),brier=float(np.mean([v[1] for v in a])),
                                 modal_hits=sum(v[2] for v in a),nll=float(np.mean([v[3] for v in a])))
            books[group][method]=dict(roots=len(rr),exact=sum(r['exact'] for r in rs),
                                      event_brier=np.mean([r['event_brier'] for r in rs]),slots=report)
    avg={}
    for method in ('raw','temperature','motion'):
        rs=[d[method] for d in books.values()]
        avg[method]=dict(exact=sum(r['exact'] for r in rs),event_brier=float(np.mean([r['event_brier'] for r in rs])),
                         slots={key:dict(brier=float(np.mean([r['slots'][key]['brier'] for r in rs])),
                                    nll=float(np.mean([r['slots'][key]['nll'] for r in rs])),
                                    top=sum(r['slots'][key]['modal_hits'] for r in rs),
                                    n=sum(r['slots'][key]['unit_rows'] for r in rs)) for key in rs[0]['slots']})
    wins=sum(v['motion']['slots']['workers']['brier'] < v['temperature']['slots']['workers']['brier'] for v in books.values())
    return dict(groups=books,average=avg,roots=len(rows),improved_groups=wins,
                qualifies=bool(avg['motion']['slots']['workers']['brier'] < min(avg[b]['slots']['workers']['brier'] for b in ('raw','temperature'))
                               and wins>=(3 if len(books)==4 else 4) and avg['motion']['exact']>=avg['temperature']['exact']))


def main():
    cli=argparse.ArgumentParser();cli.add_argument('split',choices=('fit','held','ladder'));cli.add_argument('--weights-sha');args=cli.parse_args()
    assert sha((PREVIOUS/'weights.json').read_bytes())==PLAN['temperature_sha']
    assert sha((PREVIOUS/'protocol.json').read_bytes())==PLAN['original_archived_matrix']
    m=initialize()
    assert m._RNET is m._NET and not getattr(m,'_ROLL',None)
    cards0=json.loads((PREVIOUS/'protocol.json').read_text())['tickets']
    cards=PLAN['external'] if args.split=='ladder' else [c for c in cards0 if c['split']==('train' if args.split=='fit' else 'held')]
    target=HERE/'weights.json'
    if args.split!='fit':
        assert sha(target.read_bytes())==args.weights_sha
    roots,skipped,terrains=gather(m,cards,args.split!='ladder')
    assert len(roots)+len(skipped)==59*len(cards)
    if args.split=='fit':
        print(save(target,dict(backend=m._S_STATS['rnet'],protocol=sha((HERE/'protocol.json').read_bytes()),
                    maps=terrains,roots=len(roots),miss=skipped,training=fit(roots))),flush=True)
    else:
        weights=json.loads(target.read_text());assert not set(terrains)&set(weights['maps'])
        scored=score(m,roots,weights['training'])
        receipt=save(HERE/(args.split+'_rows.json.gz'),scored)
        output=summarize(scored)|dict(terrains=terrains,skipped=skipped,weights_sha256=args.weights_sha,rows_sha256=receipt)
        save(HERE/(args.split+'.json'),output)
        print(args.split,output['roots'],output['improved_groups'],output['qualifies'],{k:(v['slots']['workers'],v['exact']) for k,v in output['average'].items()},flush=True)
if __name__=='__main__':
    main()
