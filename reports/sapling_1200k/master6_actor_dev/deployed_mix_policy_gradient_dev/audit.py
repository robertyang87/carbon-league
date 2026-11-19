"""Verify deployment/student provenance and actual normal-entry decision change on held histories."""
import ast,base64,hashlib,zlib
from collections import Counter
import numpy as np
from study import HERE,BASE,PLAN,PARENT,OPP,arena,policy,World

def vars_of(path):
    out={}
    for n in ast.parse(path.read_bytes()).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('_B1','_Q1','_M1','_HYBRID','_W64','_QMETA','_WMETA'):
            out[n.targets[0].id]=ast.literal_eval(n.value)
    return out

def run():
    a=vars_of(PARENT);b=vars_of(HERE/'candidate.py')
    if not all(a[k]==b[k] for k in ('_B1','_Q1','_M1','_HYBRID')):raise ValueError('frozen partner changed')
    if a['_W64']==b['_W64']:raise ValueError("student deployment unchanged")
    with np.load(BASE,allow_pickle=False) as old,np.load(HERE/'candidate.npz',allow_pickle=False) as new:
        changes={k:int(np.count_nonzero(old[k]!=new[k])) for k in old.files if k!='meta'}
    h=lambda x:hashlib.sha256(zlib.decompress(base64.b64decode(x))).hexdigest()
    checks=[]
    for seed in PLAN['development_seeds']:
        for seat in (0,1):
            raw=arena.read(HERE/'games'/f'{seed}_{seat}_parent.json.gz')
            w=World(seed);base=policy.load(PARENT);challenger=policy.load(HERE/'candidate.py');delta=Counter();lex=Counter()
            moves=0
            for t,joint in enumerate(raw['actions']):
                obs=w.obs(seat);before=base.agent(obs,w.config);after=challenger.agent(obs,w.config)
                if before!=joint[seat]:raise ValueError(('parent serving differs',seed,seat,t))
                if before!=after:moves+=1
                for uid in before.keys()|after.keys():
                    if before.get(uid)!=after.get(uid):
                        role=obs['players'][seat][2][uid][2] if uid in obs['players'][seat][2] else 'BASE'
                        delta[role]+=1;lex[f'{role}/{before.get(uid)}->{after.get(uid)}']+=1
                w.advance(joint)
            checks.append(dict(seed=seed,seat=seat,turns=len(raw['actions']),changed_joint=moves,changed_by_role=delta,types=lex))
    arena.write(HERE/'deployment_audit.json',dict(learner_parent=arena.sha(PARENT),frozen_opponent=arena.sha(OPP),candidate=arena.sha(HERE/'candidate.py'),
        frozen_it50_quantized_weights_sha256=h(a['_B1']),parent_student_quantized_weights_sha256=h(a['_W64']),candidate_student_quantized_weights_sha256=h(b['_W64']),
        model=arena.sha(HERE/'candidate.npz'),change_by_tensor=changes,updated_fp32_parameters=sum(changes.values()),
        on_identical_parent_trajectories=checks,behavioral_changes=sum(r['changed_joint'] for r in checks)))
if __name__=='__main__':run()
