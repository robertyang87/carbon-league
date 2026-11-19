"""Independently reproduce all 96 student-visited public inputs under the official environment."""
from pathlib import Path
import json,ast
import collect as C
from collect import B,HERE,P,ROOT,panel
import math,numbers
def diff(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b);return max((diff(v,b[k]) for k,v in a.items()),default=0)
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b);return max((diff(x,y) for x,y in zip(a,b)),default=0)
    if isinstance(a,numbers.Real):return abs(a-b)
    assert a==b;return 0
def main():
    proof=[];total_new=0;times=0
    for c in P['cards']:
        game=B.read(ROOT/c['source']);labels=B.read(HERE/f"card_{c['id']:02d}.json.gz")['records'];env=B.begin(game);s=c['seat'];sim=panel.FastCarbon(game['game_configuration']).reset(seed=c['seed']);exact=0;err=0.
        for t,cmd in enumerate(game['actions'][:158]):
            if t>=150:
                obs=B.public(env,s);obs.pop('remainingOverageTime');tag=B.digest(obs)
                simobs={k:v for k,v in sim.obs(s).items() if k in obs};z=labels[t-150];assert B.digest(simobs)==z['public_hash'] and z['executed']==cmd[s]
                exact+=int(tag==z['public_hash']);err=max(err,diff(obs,simobs));assert err<1e-8,(c['id'],t,err)
                if z['clock'].get('reused'):
                    old=B.read(HERE.parent/f"{c['seed']}_{s}_teacher.json.gz")
                    assert z['teacher']==old['checks'][0]['submitted']
                else:
                    total_new+=1;times+=z['clock']['wall']
            env.step(cmd);sim.step(*cmd)
        proof.append(dict(id=c['id'],terrain=c['terrain'],source=B.sha(ROOT/c['source']),labels=B.sha(HERE/f"card_{c['id']:02d}.json.gz"),official_public_inputs=8,identical_bytes=exact,max_abs_numeric_error=err))
    def stable(path):
        tree=ast.parse(path.read_text());out={}
        for z in tree.body:
            if isinstance(z,ast.Assign) and len(z.targets)==1 and isinstance(z.targets[0],ast.Name):
                name=z.targets[0].id
                if name in ('_W64','_WMETA','_QMETA'):
                    out[name]=ast.literal_eval(z.value);z.value=ast.Constant('student payload')
        return ast.dump(tree),out
    old,ov=stable(B.PACK);new,nv=stable(HERE/'student.py');assert old==new and ov['_W64']!=nv['_W64']
    assert total_new==94 and len(proof)==12
    B.live.save(HERE/'audit.json',dict(inputs=proof,protocol=B.sha(HERE/'protocol.json'),program=B.sha(Path(__file__)),official_steps_verified=96,
         teacher_new=total_new,teacher_wall=times,entire_pack_ast_equal_except_student_payload=True,source_teacher_in_deployment=False))
if __name__=='__main__':main()
