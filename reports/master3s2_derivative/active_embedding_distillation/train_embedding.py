"""Distill original search decisions into one newly fitted root embedding."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys
import types
import zipfile

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
ROOT=FAMILY.parents[1]
sys.path.insert(0,str(FAMILY/'v2_original_cash_paths'))
from study import PARENT, EXAMINER, select, sha

FRAME=tuple(range(25,266,20))
HELD={'native_speed_dev','switch_margin_dev'}
EPOCHS=5


def save(name,data):
    raw=(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
    if name.endswith('.gz'):
        raw=gzip.compress(raw,mtime=0)
    p=HERE/name
    tmp=p.with_suffix('.partial')
    tmp.write_bytes(raw);tmp.replace(p)
    return sha(raw)


def registration(tickets):
    return dict(schema='original-active-embedding-1', source=sha(Path(__file__).read_bytes()),
                original=EXAMINER, apply_to_parent=PARENT, steps=FRAME, held=sorted(HELD),
                optimization=dict(seed=91021, epochs=EPOCHS,batch=8,lr=.004,max_delta_norm=.30,kl=.8),
                training='Only the original Master3s2 own active style row, initialized from the original current value. '
                         'The trainable increment uses actual original-issued moves, never another style row.',
                sources=[dict(terrain=x['game']['terrain'],archive=x['archive_sha256'],
                              receipt=x['game']['receipt_sha256'],side=1-x['game']['seat'],panel=x['panel']) for x in tickets])


def import_original():
    p=ROOT/'sources/master3s2/master3s2.py'
    raw=p.read_bytes()
    assert sha(raw)==EXAMINER
    bot=types.ModuleType('pinned_m3s2')
    exec(compile(raw,str(p),'exec'),bot.__dict__)
    assert bot._SEARCH_OK and bot._S_ABORT==.58 and bot._TEMP==1
    assert bot._POLICY.snames[bot._SID]=='TJB-o18-rush-v41'
    return bot


def collect(bot,tickets,split):
    import numpy as np
    rows=[]
    for t in tickets:
        if (split=='held')!=(t['panel'] in HELD):
            continue
        game=t['game']; side=1-game['seat']; prefix=game['arm']+'/'
        arc=ROOT/t['archive']
        assert sha(arc.read_bytes())==t['archive_sha256']
        with zipfile.ZipFile(arc) as z:
            tape=z.read(prefix+'turns.jsonl')
            record=z.read(prefix+'receipt.json')
            views=gzip.decompress(z.read(prefix+'observations.jsonl.gz'))
        assert sha(record)==game['receipt_sha256']
        receipt=json.loads(record)
        assert receipt['official_transitions']==299 and sha(tape)==receipt['turns_sha256']
        response=receipt['initialization'][side]['response']
        assert response['sha256']==EXAMINER and response['original']==dict(_S_ABORT=.58,_TEMP=1.,_SEED=None,_SEARCH_OK=True)
        tracker=bot.BeliefTracker(bot._Bel,side)
        previous=None
        for step,(p,c) in enumerate(zip(views.splitlines(),tape.splitlines(),strict=True)):
            record=json.loads(p); call=json.loads(c); ob=record['observations'][side]
            assert step==ob['step']==call['step'] and ob['player']==side and 'full_carbon' not in ob
            est,known=tracker.step(ob['players'],step,ob['carbon'])
            order=call['calls'][side]['response']['action']
            if step in FRAME:
                obs_sha=sha((json.dumps(ob,separators=(',',':'))+'\n').encode())
                assert obs_sha==record['public_observation_sha256'][side]==call['public_observation_sha256'][side]
                x=bot._quant(bot.encode(ob['players'],step,side,est,known,previous)).astype(np.float32)
                w,b=bot.labels(ob['players'],side,order)
                own=ob['players'][side]
                count=len(own[2]);cash=own[0]
                for uid,pos in own[1].items():
                    a=order.get(uid,'NONE')
                    if count>=10 or cash<30:
                        b[pos]=-1
                    if a!='NONE':
                        cash-=30;count+=1
                rows.append(dict(terrain=game['terrain'], seed=game['seed'], side=side, step=step, x=x, worker=w, base=b))
            tracker.set_actions(ob['players'],order)
            previous=ob['players']
            if step==FRAME[-1]:
                break
        print(split,game['seed'],side,flush=True)
    return rows


def forward(net,x,style):
    import torch
    n=len(x)
    v=x[:,:net.cin].permute(2,3,0,1).reshape(225,n,net.cin)
    h=torch.relu(net._conv(v[net.winb],*net.stem,n)+style)
    for u1,b1,g,gb,u2,b2 in net.blocks:
        v=torch.relu(net._conv(h[net.win],u1,b1,n))
        v=v+(v[net.board].mean(0)@g+gb)
        h=torch.relu(h+net._conv(v[net.win],u2,b2,n))
    h=h[net.board]
    return (h@net.hw[0]+net.hw[1]).permute(1,0,2), (h@net.hb[0]+net.hb[1]).permute(1,0,2)


def make(rows):
    import numpy as np
    import torch
    return [torch.from_numpy(np.stack([r[k] for r in rows])) for k in ('x','worker','base')]


def train(bot,rows,params):
    import torch
    from torch.nn import functional as f
    torch.manual_seed(params['seed'])
    net=bot._NET
    style=net.emb[bot._SID].detach()
    delta=torch.nn.Parameter(torch.zeros_like(style))
    opt=torch.optim.Adam([delta],lr=params['lr'])
    x,w,b=make(rows)
    counts=Counter(r['terrain'] for r in rows)
    mass=torch.tensor([1/counts[r['terrain']] for r in rows])
    trace=[]
    for epoch in range(params['epochs']):
        all_loss=[]
        for ix in torch.randperm(len(rows)).split(params['batch']):
            cur=forward(net,x[ix],style+delta)
            with torch.no_grad():
                old=forward(net,x[ix],style)
            loss=torch.zeros(())
            for proposal,anchor,target in zip(cur,old,(w[ix],b[ix]),strict=True):
                mask=target>=0
                if not mask.any():
                    continue
                log=f.log_softmax(proposal[mask],dim=-1)
                nll=f.nll_loss(log,target[mask].long(),reduction='none')
                kl=f.kl_div(log,f.softmax(anchor[mask],dim=-1),reduction='none').sum(1)
                unit=mass[ix,None].expand_as(mask)[mask]
                loss=loss+(unit*(nll+params['kl']*kl)).sum()/unit.sum()
            opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                delta.mul_(min(1.,params['max_delta_norm']/max(float(delta.norm()),1e-12)))
            all_loss.append(float(loss.detach()))
        trace.append(dict(epoch=epoch+1,loss=sum(all_loss)/len(all_loss),delta_norm=float(delta.detach().norm())))
        print(trace[-1],flush=True)
    return (style+delta.detach()).tolist(),trace


def measure(bot,rows,updated):
    from collections import defaultdict
    import numpy as np
    import torch
    from torch.nn import functional as f
    net=bot._NET
    frozen=net.emb[bot._SID]
    fitted=torch.tensor(updated,dtype=torch.float32)
    x,w,b=make(rows)
    by_map=defaultdict(Counter)
    checks=[]
    with torch.no_grad():
        for index in range(0,len(rows),8):
            sl=slice(index,index+8)
            a=forward(net,x[sl],frozen)
            c=forward(net,x[sl],fitted)
            legacy=net(x[sl].numpy(),bot._SID)
            checks.append(max(float(abs(a[k].numpy()-legacy[k]).max()) for k in range(2)))
            for i,row in enumerate(rows[sl]):
                values=by_map[row['terrain']]
                for k,target in enumerate((w[index+i],b[index+i])):
                    mask=target>=0
                    y=target[mask].long()
                    if not len(y):
                        continue
                    label=('worker','base')[k]
                    values[label+'_n']+=len(y)
                    for tag,series in (('original',a),('trained',c)):
                        z=series[k][i][mask]
                        values[f'{label}_{tag}_nll']+=float(f.cross_entropy(z,y,reduction='sum'))
                        values[f'{label}_{tag}_hit']+=int((z.argmax(-1)==y).sum())
    assert max(checks)<1e-5
    return dict(maps=by_map,example_count=len(rows),native_rebuild_error=max(checks))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('stage',choices=('register','fit','held'))
    cli.add_argument('--cpu',type=int,required=True)
    arg=cli.parse_args()
    os.sched_setaffinity(0,{arg.cpu})
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k]='1'
    tickets=select()
    lock=registration(tickets)
    p=HERE/'protocol.json'
    if arg.stage=='register':
        assert not p.exists() or json.loads(p.read_bytes())==json.loads(json.dumps(lock))
        print(save('protocol.json',lock))
    else:
        assert json.loads(p.read_bytes())==json.loads(json.dumps(lock))
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        bot=import_original();torch.set_num_threads(1)
        assert type(bot._NET).__name__=='_WinoNet'
        wr=HERE/'weights.json'
        if arg.stage=='held':
            fitted=json.loads(wr.read_bytes())
            assert fitted['protocol']==sha(p.read_bytes())
        data=collect(bot,tickets,arg.stage)
        maps=sorted({x['terrain'] for x in data})
        if arg.stage=='fit':
            updated,trace=train(bot,data,lock['optimization'])
            save('weights.json',dict(protocol=sha(p.read_bytes()),embedding=updated,maps=maps,trace=trace))
        else:
            assert not set(fitted['maps']).intersection(maps)
            updated=fitted['embedding']
        metrics=measure(bot,data,updated)
        save(arg.stage+'.json',dict(weights=sha(wr.read_bytes()),**metrics))
