"""Recover the unmodified 3s2 forward's own post-residual tensor from public replay history."""
import ast
import copy
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
RAW=HERE.parent
sys.path.insert(0,str(RAW))
import builder as B
F=B.F

def forward_capture():
    original=B.original()
    src=ast.parse(B.git_blob(B.ORIGINAL_BLOB))
    cls=next(n for n in src.body if isinstance(n,ast.ClassDef) and n.name=='_Policy')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    assert isinstance(fn.body[-1],ast.Return)
    fn.body[-1].value=ast.Tuple(elts=[fn.body[-1].value,ast.Name(id='h',ctx=ast.Load())],ctx=ast.Load())
    ast.fix_missing_locations(fn)
    ns={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'verbatim_forward_return_h','exec'),original.__dict__,ns)
    return original,ns['forward']

def extract():
    m,forward=forward_capture()
    proto=json.loads((HERE/'protocol.json').read_bytes())
    for key,digest in proto['identity'].items():
        assert B.sha((RAW/key).read_bytes())==digest
    model=m._POLICY;hw=model.p['hw.w'].reshape(5,-1);hb=model.p['hb.w'].reshape(3,-1)
    w_bias=model.p['hw.b'][:,None];b_bias=model.p['hb.b'][:,None];cards=json.loads((RAW/'manifest.json').read_bytes())['records']
    store=Path('/workspace/sapling-3s2-spatial-transfer-v1');store.mkdir(exist_ok=True)
    arr=[];witness=[];seats={0:0,1:0}
    for card in cards:
        sim,tape=B.replay(card);seat=card['seat'];feat=F.Feat(seat);tr=m.BeliefTracker(m._Bel,seat); prev=None
        with np.load(RAW/card['file'],allow_pickle=False) as z:
            mode=z['prefix_mode'];steps=z['step'][mode==0];prior=z['teacher_raw'][mode==0]; prior_steps=steps.tolist()
        assert prior_steps==card['steps']
        stored=[];eps=0.;approx=0.
        perm=F.CANON if seat else F.GRID
        assert np.array_equal(perm[perm],F.GRID)
        for t,pair in enumerate(tape):
            obs={k:v for k,v in sim.obs(seat).items() if k in B.PUBLIC}; x=feat.observe(obs)
            est,known=tr.step(obs['players'],t,obs['carbon'])
            if t in prior_steps:
                enc=np.clip(np.rint(m.encode(obs['players'],t,seat,est,known,prev)*100),0,255)*.01
                expected=model.forward(enc,m._SID);heads,h=forward(model,enc,m._SID)
                assert all(np.array_equal(a,b) for a,b in zip(expected,heads))
                cn=h.reshape(h.shape[0],-1)[:,perm].reshape(h.shape)
                reconstructed=[(w@cn.reshape(h.shape[0],-1)+b).T for w,b in [(hw,w_bias),(hb,b_bias)]]
                for raw,pred in zip(expected,reconstructed):
                    e=float(abs(raw[perm]-pred).max());eps=max(eps,e);assert e<1.e-4
                probs=B.physical_to_student(m,feat,x,obs,heads)
                j=prior_steps.index(t)
                np.testing.assert_allclose(probs,prior[j],rtol=0,atol=2.e-6)
                for slot,uid in enumerate(feat.own_uid):
                    if uid is None:continue
                    pos=(obs['players'][seat][2][uid][0] if slot<10 else obs['players'][seat][1][uid])
                    assert feat.cpos(pos)==x['tok_pos'][slot] and perm[feat.cpos(pos)]==pos
                    cq=m._probs(reconstructed[int(slot>=10)][feat.cpos(pos)])
                    if slot<10 and seat:cq=cq[F.CANON_MOVE]
                    q=prior[j,slot,:len(cq)];np.testing.assert_allclose(cq,q,atol=5.e-6,rtol=0)
                assert np.isfinite(cn).all()
                encoded=cn.astype(np.float16);assert np.isfinite(encoded).all()
                approx=max(approx,float(abs(cn-encoded.astype(np.float32)).max()));stored.append(encoded)
                seats[seat]+=1
            tr.set_actions(obs['players'],pair[seat]);prev=obs['players']
            if t==prior_steps[-1]:break
            sim.step(*pair)
        n=len(stored);assert n==len(prior_steps)
        file=store/(card['id']+'.npz')
        np.savez_compressed(file,board=np.stack(stored),step=steps)
        arr.append(dict(id=card['id'],sha256=B.sha(file.read_bytes()),source_sha256=card['source_sha256'],roots=n,seat=seat,split=card['split'],max_reconstructed_logit_error=eps,float16_max_error=approx))
        if len(arr)%8==0:print('3s2 maps/seats',len(arr),len(cards),flush=True)
    B.save(HERE/'labels.json',dict(original_sha256=B.ORIGINAL_SHA,original_blob=B.ORIGINAL_BLOB,protocol_sha256=B.sha((HERE/'protocol.json').read_bytes()),extract_sha256=B.sha(Path(__file__).read_bytes()),directory=str(store),channels=hw.shape[1],roots_by_seat=seats,records=arr))
if __name__=='__main__':extract()
