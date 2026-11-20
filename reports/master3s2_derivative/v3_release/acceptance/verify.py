"""Independent byte-level release checks and same-private-root public-entry mechanism probes."""
import ast
import base64
import difflib
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
RELEASE=HERE.parent
LADDER=RELEASE.parent/'ladder'
PKG=LADDER/'prediction_feedback/package'
CASES={'bdbe59b04f71e43e53696874c8fd5958':('9d98e3a8afa6f793ea38485a3f2032bb1837c810eef93f6df3b1b38e87984772',[(152,28,'UP'),(269,72,'RIGHT'),(272,62,'UP'),(291,72,'LEFT')]),
       '4f97e17b6a190b493abac35ddb19f7dc':('6d1165c4483ffa8c5c4ed5b4c1f53ed53d30e75a08ef0cffa894bfeee3e9572e',[(191,25,'LEFT')])}

def sha(b):return hashlib.sha256(b).hexdigest()

def identity():
    manifest=json.loads((RELEASE/'manifest.json').read_bytes());deployed=json.loads((LADDER/'ds-pro-v2.json').read_bytes())
    before=(PKG/'candidate.py').read_bytes();after=(RELEASE/'submission.py').read_bytes()
    assert sha(before)==manifest['parent_sha256']==deployed['candidate_sha256'] and sha(after)==manifest['candidate_sha256']
    assert sha((RELEASE/'runtime.py').read_bytes())==manifest['runtime_sha256'] and sha((RELEASE/'build.py').read_bytes())==manifest['builder_sha256']
    assert len(after)==manifest['bytes']<=10_000_000 and manifest['new_weights']==0
    a,b=before.splitlines(keepends=True),after.splitlines(keepends=True);changes=[]
    for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
        if tag=='equal':continue
        assert tag=='insert' and i==j
        changes.append((i,k,l))
    assert len(changes)==2
    assert a[changes[0][0]].startswith(b'def _search(')
    assert a[changes[1][0]].strip() in (b'def begin(cs, ms):',b'')
    def fingerprint(data):
        tree=ast.parse(data); main_nodes={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        dependencies=sorted({x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names}|{str(n.module) for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)})
        return dict(weights={name:sha(base64.b64decode(main_nodes[name].value.value)) for name in ('_WEIGHTS','_VALUE')},
                    embedded_opponent_readout_sha=sha(ast.dump(main_nodes['_OPP_H']).encode()),feedback_coefficients_sha=sha(ast.dump(main_nodes['_FEEDBACK_WEIGHTS']).encode()),
                    imports=dependencies,public_agent_args=next([a.arg for a in n.args.args] for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='agent'))
    f=fingerprint(after);assert f==fingerprint(before)
    return dict(parent=sha(before),candidate=sha(after),file_bytes=len(after),inserted_range_zero_based=changes,**f)

def standalone(payload):
    program='''import importlib.abc,importlib.util,json,pathlib,sys,time
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','zerosum_env'): raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.addaudithook(lambda e,a: (_ for _ in ()).throw(PermissionError(e)) if e in ('socket.connect','socket.bind','socket.getaddrinfo') else None)
m=importlib.util.module_from_spec(s:=importlib.util.spec_from_file_location('isolated_agent','submission.py'));s.loader.exec_module(m)
p=json.loads(pathlib.Path('input.json').read_text());m._V3_AUDIT=[];out=m.agent(p['obs'],p['cfg']);print(json.dumps(dict(action=out,search=bool(m._SEARCH_OK),embedded_engine=bool(m._RX),trace=m._V3_AUDIT,has_external_board=m._NBoard is not None)))'''
    with tempfile.TemporaryDirectory() as dest:
        loc=Path(dest);(loc/'submission.py').write_bytes((RELEASE/'submission.py').read_bytes());(loc/'input.json').write_text(json.dumps(payload));env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        got=subprocess.run([sys.executable,'-I','-B','-c',program],cwd=loc,env=env,check=True,capture_output=True,text=True,timeout=30)
        return json.loads(got.stdout.splitlines()[-1])

def main():
    ident=identity()
    sys.path.insert(0,str(PKG));import probe
    os.sched_setaffinity(0,{int(os.getenv('V3_PROBE_CPU','2'))});probe.torch.set_num_threads(1);probe.torch.set_num_interop_threads(1)
    m,n=[probe.module(p.read_bytes()) for p in (PKG/'candidate.py',RELEASE/'submission.py')]
    costs=json.loads(gzip.decompress((PKG/'roots.json.gz').read_bytes()))['costs'];summary=[];raw_arms=[];smoke=None
    for match,(digest,tickets) in CASES.items():
        raw=(LADDER/'replays'/(match+'-1.json.gz')).read_bytes();assert sha(raw)==digest;tape=json.loads(gzip.decompress(raw));frames=tape['steps'];s=1;tr=m.BeliefTracker(m._Bel,s);prev=None
        cfg={k:v for k,v in tape['configuration'].items() if k!='randomSeed'}
        for t in range(max(c[0] for c in tickets)+1):
            frame=frames[t];ss=frame[s]['observation'];shared=frame[0]['observation'];obs=dict(step=t,player=s,carbon=ss['carbon'],players=shared['players'],trees=shared['trees'],remainingOverageTime=ss['remainingOverageTime']);shown=frames[t+1][s]['action'] or {}
            if t==0 and smoke is None:smoke=standalone(dict(obs=obs,cfg=cfg))
            for time,worker,wanted in tickets:
                if t!=time:continue
                wid=f'player-1-worker-{worker}';rng=int.from_bytes(hashlib.sha256(f'{match}/{t}'.encode()).digest()[:4],'big');order=['parent_a','child_a','child_b','parent_b'] if len(summary)%2==0 else ['child_a','parent_a','parent_b','child_b'];arms={};traces={}
                for name in order:
                    mod=n if name.startswith('child') else m
                    if mod is n:mod._V3_AUDIT=[]
                    arms[name]=probe.root_call(mod,tr,prev,obs,cfg,rng,costs)
                    if mod is n:traces[name]=mod._V3_AUDIT
                root={'match':match,'step':t,'target':wanted,'worker':wid,'historical':shown.get(wid,'NONE'),'public_sha':sha(json.dumps(obs,ensure_ascii=False,separators=(',',':')).encode()),'policy_seed':rng}
                a,b=arms['parent_a'],arms['child_a']; root.update(enemy_unchanged=all(q['opponent_samples']==a['opponent_samples'] for q in arms.values()),native_A0_unchanged=all(q['act0']==a['act0'] for q in arms.values()),within_version_repeat=all(arms[f'{side}_a'][key]==arms[f'{side}_b'][key] for side in ('parent','child') for key in ('issued','depth_by_arm','menu')),
                    original_menu=[x.get(wid,'NONE') for x in a['menu']],candidate_menu=[x.get(wid,'NONE') for x in b['menu']],parent_issued=a['issued'].get(wid,'NONE'),child_issued=b['issued'].get(wid,'NONE'),team_changed=a['issued']!=b['issued'],child_equals_history=b['issued']==shown,
                    child_trace=traces['child_a'],native_work={tag:[q['candidates'],q['depth'],q['rollouts']] for tag,q in arms.items()},caught_new_target=(not any(x.get(wid,'NONE')==wanted for x in a['menu']) and any(x.get(wid,'NONE')==wanted for x in b['menu'])),
                    returned_valid=all(isinstance(q['issued'],dict) and all(k in set(obs['players'][s][1])|set(obs['players'][s][2]) and v in {'UP','DOWN','LEFT','RIGHT','NONE','RECCOLLECTOR','RECPLANTER'} for k,v in q['issued'].items()) for q in arms.values()))
                raw_arms.append(dict(match=match,t=t,arms=arms,trace=traces));summary.append(root)
            tr.step(obs['players'],t,obs['carbon']);tr.set_actions(obs['players'],shown);prev=obs['players']
    z=gzip.compress(json.dumps(raw_arms,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':')).encode(),mtime=0);(HERE/'roots.json.gz').write_bytes(z)
    result=dict(identity=ident,numpy_only_public_smoke=smoke,source_scope='historical opposing actions are used only for labels; fixed action RNG and measured prior cost matched across four seeded, real native wall-clock public calls; historical private seeds and wall not observed',torch_available_public_entry=True,roots=summary,full_roots_sha=sha(z))
    (HERE/'receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print([(r['step'],r['original_menu'],r['candidate_menu'],r['parent_issued'],r['child_issued']) for r in summary],flush=True)

if __name__=='__main__':main()
