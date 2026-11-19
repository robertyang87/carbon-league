"""Reconstruction and independent decoder for the frozen terminal-return Sapling policy."""
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

from compound_formal import ACTORS, REPORT, ROOT, PARENT, STUDENT, TERMINAL as PACK, _hash

TRAINED = "f82bfd47a904aa1fbfa4e95fbfcdd13e623c1f1eb216a8e736694a1d282cc713"
SINGLE = "fbccc5c8873849220eb696038feaa6c54666c77829453a2d5ea81bb2c06179f6"
BASE = "416cc31b8da7d38cc97f66903fbe2797d7eadda0ee175f105227057afe5d3874"
DEV = ROOT / REPORT / "master6_actor_dev"
FIT = DEV/"terminal_return_fit"
CHECKPOINT = "3efaf4b81f21620803ec5b29ba57e61408b547e3dbc4ab72457a8e09a3f5ac0a"
THETA = np.array([.6435502812730072,-.6435502812730072,.6435502812730072,.6435502812730072],dtype=np.float32)
RECEIPTS = {
    FIT/"learned_checkpoint.json":CHECKPOINT,
    DEV/"weight_receipt.json":"21fe203988d38cee5075469b7d4b40aca03dc6a5da088d7d88ac745235d09ad4",
    FIT/"train_plus/results.json":"c818594d43a4281dc3e146cfdda1258c9b6c4092a368ad3c2cacc067d83b7fb2",
    FIT/"train_minus/results.json":"df08fe33fc8912deef56a90961100fb366bd22a1cf86288d11e24d38c8bc5acc",
    FIT/"train_plus/official_replay.json":"6702a319ce7a930dfd53d4cd0e90cacb604991f4e2c7e26442029e8861585670",
    FIT/"train_minus/official_replay.json":"a7bcb9993c78f714208b94335c387a24ffd60f358dfbf9f2121793991e835f8f",
}


def prove(weights,candidate):
    from tree_win_gate import prove_single_pack
    from compound_formal import prove as prove_old
    if Path(weights).suffix.lower()!='.npz' or _hash(weights)!=TRAINED or _hash(candidate)!=PACK:
        raise ValueError("terminal actor requires the exact trained Sapling NPZ and frozen deployment package")
    if any(_hash(p)!=expected for p,expected in RECEIPTS.items()):
        raise ValueError("terminal training or student evidence changed")
    checkpoint=json.loads((FIT/"learned_checkpoint.json").read_bytes())
    if not np.array_equal(np.asarray(checkpoint["theta"],np.float32),THETA):
        raise ValueError("independent residual coefficients differ from the frozen terminal update")
    actors=ROOT/ACTORS
    if prove_single_pack(weights,actors/'master6_actor_v1.py')!=(TRAINED,SINGLE):
        raise ValueError("NPZ failed independent student export")
    if prove_old(actors/'cash_v3.npz',actors/'hybrid_it50_v3.py',PARENT)!=(STUDENT,PARENT):
        raise ValueError("original it50/Sapling reference failed reconstruction")
    with tempfile.TemporaryDirectory(prefix='sapling-return-rebuild-') as temp:
        rebuilt=Path(temp)/'base.py'
        subprocess.run([sys.executable,str(DEV/'hybrid_build.py'),str(actors/'master6_actor_v1.py'),
                        str(Path(weights).resolve()),str(rebuilt),str(Path(temp)/'base.json')],check=True)
        if _hash(rebuilt)!=BASE or _hash(actors/'master6_actor_hybrid.py')!=BASE:
            raise ValueError("trained 75/25 baseline failed reconstruction")
        # Isolate generic module names build/trial from other experiment scripts.
        code="""import json,sys
from pathlib import Path
import build,trial
build.PARENT=Path(sys.argv[1])
ckpt=json.loads((build.HERE/'learned_checkpoint.json').read_bytes())
if trial.compute_fit()!=ckpt:
    raise ValueError('four terminal outcomes did not reproduce the frozen checkpoint')
Path(sys.argv[2]).write_bytes(build.source(ckpt['theta']))
"""
        rebuilt_pack=Path(temp)/'agent.py'
        subprocess.run([sys.executable,'-c',code,str(rebuilt),str(rebuilt_pack)],cwd=FIT,check=True)
        if _hash(rebuilt_pack)!=PACK or rebuilt_pack.stat().st_size>10_000_000:
            raise ValueError("rebuilt residual package differs")
    return TRAINED,PACK


@lru_cache(maxsize=1)
def networks():
    """Read authenticated independent PPO/student payloads; do not invoke the candidate's agent."""
    from tree_audited_game import prior
    path=ROOT/ACTORS/'master6_actor_hybrid.py'
    if _hash(path)!=BASE:
        raise ValueError("frozen reference actors changed")
    mod=prior.shared.module(path,'formal_reference_trained_master6_dual')
    import base64,zlib
    policies=[]
    for blob,meta,desc in ((mod._B1,mod._Q1,mod._M1),(mod._W64,mod._QMETA,mod._WMETA)):
        tensors=mod._dequantize(zlib.decompress(base64.b64decode(blob)),meta)
        tensors['meta']=np.array(desc)
        policies.append(mod.sys.modules['cnet2_np'].Policy(tensors))
    return mod.sys.modules['cnet2_feat'],policies


def residual_scores(x,aux,slot,moves):
    # Independent expression on the actual public state. All three variants use identical residuals.
    cargo=float(x['tok_cargo'][slot]);loaded=cargo/(cargo+60.)
    d0=float(aux['dob'][int(x['tok_pos'][slot])]);d=np.rint(moves[:,10]*14)
    slack=np.clip((x['left']-1-d)/4,0,1)
    pressure=np.clip((d0+12-x['left'])/12,0,1)
    bank=loaded*(moves[:,5]+(.5+.5*pressure)*np.clip(d0-d,-1,1))
    danger=loaded*(moves[:,1]+.25*moves[:,4]+.25*moves[:,13])
    prey=np.expm1(moves[:,2]*6.5)
    raid=(prey/(prey+80))*(1-loaded)*(1-moves[:,1])*slack*(1-.5*moves[:,13])*(1-np.clip(moves[:,0],0,1))
    harvest=(moves[:,11]*moves[:,7]*(1-loaded)*(1-moves[:,1])*(1-moves[:,5])*(1-moves[:,6])*
             (1-moves[:,8])*(1-moves[:,9])*(1-.5*moves[:,13])*slack)
    return np.stack((bank,danger,raid,harvest),axis=1) @ THETA


class RootDecoder:
    def __init__(self,obs,feat,features,policies):
        self.feat,self.F,self.obs=feat,features,obs
        self.x=feat.observe(obs);self.aux=feat.aux
        self.active=list(map(int,np.flatnonzero(self.x['tok_mask'][:features.OWN])))
        self.encoded=[]
        for policy in policies:
            tokens,context=policy.encode(self.x,2)
            self.encoded.append((policy,tokens,context,policy.lin(context,'h0')))
        self.cache={}

    def orders(self,expert_weight,student_weight,residual):
        labels=np.full(self.F.OWN,-1,np.int64)
        history=()
        states=[e[-1] for e in self.encoded]
        prefix=self.F.Prefix(self.x,self.aux)
        prev=8
        for slot in self.active:
            uf,mf,legal=prefix.feats(slot)
            if history not in self.cache:
                scores=[];hidden=[]
                for (p,tokens,ctx,_),h0 in zip(self.encoded,states):
                    inp=p.lin(np.concatenate((tokens[slot],ctx,uf,mf.reshape(-1))),'din')+p.w['aemb.weight'][prev]
                    h=p.gru(inp,h0)
                    logits=p.lin(h,'whead') if slot<10 else np.concatenate((p.lin(h,'bhead'),np.zeros(2,np.float32)))
                    scores.append(logits+p.lin(mf,'V')@p.lin(h,'U'))
                    hidden.append(h)
                delta=(residual_scores(self.x,self.aux,slot,mf) if slot<10 and self.x['tok_kind'][slot]==0 else None)
                self.cache[history]=(hidden,scores,delta)
            states,scores,delta=self.cache[history]
            value=expert_weight*scores[0] + student_weight*scores[1]
            if residual and delta is not None:
                value=value+delta
            action=int(np.argmax(np.where(legal,value,-np.inf)))
            if not legal[action]:raise ValueError("decoder found no legal command")
            labels[slot]=action
            prefix.advance(slot,action)
            prev=action if slot<10 else 5+action
            history+=(action,)
        return labels

    def conflicts(self,labels):
        occ={}
        for s in self.active:
            if s>=10 and labels[s]==0:continue
            p=int(self.x['tok_pos'][s])
            cell=int(self.F.DEST[p,labels[s]]) if s<10 else p
            occ.setdefault(cell,[]).append(s)
        return {(a,b) for group in occ.values() for i,a in enumerate(group) for b in group[i+1:] if a<10}

    def choice(self,expert,student):
        base=self.orders(expert,student,False)
        altered=self.orders(expert,student,True)
        if any(base[s]!=altered[s] for s in self.active if s>=10 or self.x['tok_kind'][s]==1):
            return base,'protected_worker_or_hire',base
        if self.conflicts(altered)-self.conflicts(base):
            return base,'introduced_team_conflict',base
        return altered,'applied',base


def verify(folder,row,raw,record=False):
    from tree_audited_game import exam
    from turning_point_ledger import compact,sha
    from collections import Counter
    folder=Path(folder);seat=row['candidate_seat'];sim=exam.FastCarbon(raw['game_configuration']).reset(seed=row['seed'])
    F,policies=networks();feat=F.Feat(seat)
    turns=[];counts=Counter()
    for t,commands in enumerate(raw['actions']):
        root=RootDecoder(sim.obs(seat),feat,F,policies)
        final,guard,base=root.choice(.75,.25)
        # Primary ablation removes exactly the Sapling logit term; secondary measures the conventional normalized expert-only actor.
        no_sap,no_guard,_=root.choice(.75,0.)
        norm,norm_guard,_=root.choice(1.,0.)
        actual=feat.to_actions(final)
        if actual!=commands[seat]:
            raise ValueError(f"independent residual/guard joint differs from executed commands: {t}/{seat}")
        counts['slots']+=len(root.active)
        changed=[s for s in root.active if final[s]!=no_sap[s]]
        conventional=[s for s in root.active if final[s]!=norm[s]]
        different_base=[s for s in root.active if final[s]!=base[s]]
        counts['drop_roots']+=bool(changed);counts['drop_slots']+=len(changed)
        counts['normalized_roots']+=bool(conventional);counts['normalized_slots']+=len(conventional)
        counts['residual_roots']+=bool(different_base);counts['veto_roots']+=guard!='applied'
        # The first disagreement has the same preceding commands. If either both plans
        # applied the residual or both reverted, even its fallback status is controlled.
        shared=bool(changed) and (guard=='applied')==(no_guard=='applied')
        counts['direct_first']+=shared
        turns.append({'t':t,'full':actual,'drop_sapling':feat.to_actions(no_sap),'expert_normalized':feat.to_actions(norm),
                      'vetoes':[guard,no_guard,norm_guard],'changed_from_dual_no_residual':[feat.own_uid[s] for s in different_base],
                      'first_same_prefix_sapling':feat.own_uid[changed[0]] if shared else None})
        sim.step(*commands)
    if sha(compact(sim.get_state()))!=row['final_state_sha256']:
        raise ValueError('independent residual verification ends on another simulator state')
    name=f"terminal_policy_{row['seed']}_{seat}.json.gz"
    evidence=compact({'format':'sapling-terminal-ablation/v1','seed':row['seed'],'seat':seat,
                      'pack_sha256':PACK,'transcript_sha256':row['raw_sha256'],'turns':turns})
    path=folder/name
    if record and not path.exists():
        with tempfile.NamedTemporaryFile(dir=folder,delete=False) as out:
            out.write(gzip.compress(evidence,mtime=0));tmp=Path(out.name)
        try:os.link(tmp,path)
        except FileExistsError:pass
        finally:tmp.unlink(missing_ok=True)
    if gzip.decompress(path.read_bytes())!=evidence:
        raise ValueError('absent or changed immutable residual ablation evidence')
    return {'pack_sha256':PACK,'transcript_sha256':row['raw_sha256'],'verified_roots':len(turns),
            'verified_unforced_slots':counts['slots'],'student_changes':counts['direct_first'],
            'decisions_sha256':sha(evidence),'evidence_file':name,'evidence_sha256':_hash(path),
            'drop_sapling_changed_roots':counts['drop_roots'],'drop_sapling_changed_slots':counts['drop_slots'],
            'normalized_expert_changed_roots':counts['normalized_roots'],'normalized_expert_changed_slots':counts['normalized_slots'],
            'residual_changed_roots':counts['residual_roots'],'safety_veto_roots':counts['veto_roots']}
