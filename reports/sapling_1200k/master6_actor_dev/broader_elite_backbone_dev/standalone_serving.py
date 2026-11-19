"""Secondary normal-entry assessment of each 1,206,476-parameter network without a partner."""
import argparse,gzip,inspect,json,sys,tempfile
from pathlib import Path
from train import HERE,START,common,probe,imported,load,sha,save,assemble
from serve import by_split

def game(models,card,steps,feat):
    src=probe.CACHE/(card['id']+'.json.gz'); raw=src.read_bytes()
    if sha(raw)!=card['sha256']:raise ValueError('wrong source')
    doc=json.loads(gzip.decompress(raw));probe.intact(doc,card,card['terrain_sha256'])
    frames=doc['steps'];times={t for t in steps if t<=len(frames)-2}
    s=card['seat']; audit=feat.Feat(s); roots=[]
    for t in range(max(times)+1):
        obs=probe.seat_view(frames[t],s);x=audit.observe(obs)
        if t not in times:
            for m in models.values():
                f=inspect.getclosurevars(m._AG).nonlocals['st']['f']
                f.observe(obs)
            continue
        cmd=probe.cleaned(frames[t+1][s]['action']);target=audit.labels(cmd);arms={};slots={}
        for label,m in models.items():
            chosen=probe.cleaned(m.agent(obs,doc['configuration']));acts=audit.labels(chosen)
            if audit.to_actions(acts)!=chosen:raise ValueError('command roundtrip')
            arms[label]={'actual_command':chosen}
            prefix=feat.Prefix(x,audit.aux)
            for i,on in enumerate(x['tok_mask'][:feat.OWN]):
                if not on:continue
                uid=audit.own_uid[i]; action=int(acts[i]); legal=prefix.feats(i)[2]
                if not legal[action]:raise ValueError('illegal emitted command')
                row=slots.setdefault(uid,dict(role='base' if i>=10 else 'planter' if int(x['tok_kind'][i]) else 'collector',demo_canonical=int(target[i]),outputs={}))
                row['outputs'][label]={'actual':action,'legal':[a for a,v in enumerate(legal) if v]}
                prefix.advance(i,action)
        roots.append(dict(t=t,demonstration_command=cmd,arms=arms,slots=slots))
    return dict(**{k:card[k] for k in ('id','seat','split','teacher','opponent_new_to_ancestor')},roots=roots)

def main(evaluate):
    here_digest=sha(Path(__file__).read_bytes());protocol=load(HERE/'protocol.json')
    if not evaluate:
        return save(HERE/'standalone_protocol.json',dict(source_sha256=here_digest,panel_sha256=protocol['panel_sha256'],eval_steps=protocol['eval_steps'],
         purpose='Supplementary parent/candidate int8 standalone call using identical current package source and no partner/guards, same previously frozen eval steps.'))
    assert load(HERE/'standalone_protocol.json')['source_sha256']==here_digest
    pack=load(HERE/'package.json')
    assert sha((HERE/'standalone.py').read_bytes())==pack['standalone_sha256']
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp)/'parent_standalone.py';common.pack.main(str(START),str(parent),2)
        models={'parent':imported(parent,'bare_parent'), 'candidate':imported(HERE/'standalone.py','bare_candidate')}
        assert models['parent']._SRC_SHA256==models['candidate']._SRC_SHA256
        assert models['parent']._OPTS==models['candidate']._OPTS and models['candidate']._OPTS['teacher']==2
        feat=sys.modules['cnet2_feat'];cards=[g for g in assemble() if g['split'] in ('val','holdout')]
        out=[]
        for g in cards:
            out.append(game(models,g,protocol['eval_steps'],feat))
            if len(out)%10==0:print('standalone public serving',len(out),'/',len(cards),flush=True)
        results={k:by_split([g for g in out if g['split']==k]) for k in ('val','holdout')}
        payload=gzip.compress(json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),mtime=0)
        with (HERE/'standalone_roots.json.gz').open('xb') as f:f.write(payload)
        save(HERE/'standalone_serving.json',dict(scope='Expert command agreement on recorded histories at the normal public standalone agent entry; rollout outcomes remain unmeasured.',
             counterfactual_sample=False,trace_sha256=sha(payload),parent_package_sha256=sha(parent.read_bytes()),candidate_package_sha256=pack['standalone_sha256'],results=results))
        for k,v in results.items():print(k,v['overall'],v['macro_terrain_equal'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','evaluate']);main(p.parse_args().mode=='evaluate')
