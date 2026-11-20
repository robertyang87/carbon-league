"""Probe direct financial scalar pathways on actual archived H8 native inputs, without changing game state."""
import numpy as np

from study import D, P, REG, check, dump, freight, read, sha


def run():
    p=D/"cold_start/candidate.py"
    check(sha(p.read_bytes())==REG["parent"],"Wrong parent")
    mod=freight.load("tree_native_critic",p);net=mod._net()
    check(mod._SEARCH_OK,"Native search unavailable")
    raw=[]
    for ticket in P["tickets"]:
        f=D/"action_rank_data"/ticket["split"]/f"{ticket['seed']}_{ticket['seat']}_{ticket['step']}_leaf.npz"
        if not f.exists():continue
        identity=read(f.with_name(f.name.replace("_leaf.npz","_pair.json")))
        check(sha(f.read_bytes())==identity["leaf_sha256"],"Changed native leaf records")
        with np.load(f,allow_pickle=False) as z:
            for a in (0,1):
                check(not z["exact_terminal"][a,0] and z["depth"][a,0]==8,"Unexpected terminal/depth")
                x=z["X"][a,0].astype(np.float32)*np.float32(.01);s=z["scalars"][a,0]
                features=np.ascontiguousarray(np.broadcast_to(x,(3,*x.shape)))
                scalars=np.tile(s,(3,1))
                # Holding every spatial plane fixed isolates the direct scalar path, not a physical tree creation.
                scalars[1,3]+=np.float32(.1);scalars[2,0]+=np.float32(100/3000)
                v=net(features,mod._SID,scalars)[2]
                err=abs(float(v[0])-float(z["critic"][a,0]));check(err<3.e-5,"Cannot reproduce archived native value")
                raw.append(dict(seed=ticket["seed"],seat=ticket["seat"],arm=a,base=float(v[0]),replay_abs_error=err,
                                extra_tree_scalar=float(v[1]-v[0]),extra_100cash_scalar=float(v[2]-v[0])))
    result={"rows":raw,"parent_sha256":REG["parent"],"native_weights_unchanged":True,
            "method":"Real H8 spatial trunk fixed; perturb only tree count difference +1 or cash difference +100, separately. Nonphysical tests isolate direct scalar pathways, excluding the response of the spatial maps, future opposition or actual tree rent."}
    for k in ("extra_tree_scalar","extra_100cash_scalar"):
        z=[r[k] for r in raw]
        result[k]=dict(min=min(z),median=float(np.median(z)),max=max(z),nonzero=sum(abs(v)>1.e-6 for v in z))
    dump("critic_inputs.json",result)

if __name__=="__main__":run()
