"""Build an own-root-only policy increment while leaving enemy and simulation/critic heads unchanged."""
import base64
import io
import json
import numpy as np
from data import D,HERE,P,sha,save

V2_PATH=D/'ladder/prediction_feedback/package/candidate.py'
V2_SHA='8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'

def assemble(method, online=False):
    parent=(V2_PATH if online else D/'cold_start/candidate.py').read_bytes()
    assert sha(parent)==(V2_SHA if online else P['parent'])
    if method=='zero':return parent
    assert method in ('bc','awr')
    weights=HERE/'weights.npz'
    assert sha(weights.read_bytes())==json.loads((HERE/'fit.json').read_bytes())['weights']
    fresh={}
    with np.load(weights) as f:
        for role in ('worker','base'):
            mu,sd,B=(f[method+'_'+role+'_'+k].astype(np.float64) for k in ('mu','sd','B'))
            w=B[:-1]/sd[:,None];b=B[-1]-mu@w
            fresh[role+'_w']=np.ascontiguousarray(w.T,dtype=np.float32);fresh[role+'_b']=np.ascontiguousarray(b[:,None],dtype=np.float32)
    out=io.BytesIO();np.savez_compressed(out,**fresh)
    install=(b'_OWN_AW_ARCHIVE = np.load(io.BytesIO(base64.b64decode('+repr(base64.b64encode(out.getvalue()).decode()).encode()+b')), allow_pickle=False)\r\n'
             b'_OWN_AW = {k: _OWN_AW_ARCHIVE[k] for k in _OWN_AW_ARCHIVE.files}\r\n_OWN_AW_ARCHIVE.close()\r\n\r\n')
    changes=[(b'class _Policy:',install+b'class _Policy:'),
             (b'    def forward(self, x, sid):',b'    def forward(self, x, sid, own_root=False):'),
             (b'        return lw.T, lb.T',b'        if own_root:\r\n'
              b"            lw += 1.25 * np.tanh((_OWN_AW['worker_w'] @ hf + _OWN_AW['worker_b']) / 1.25)\r\n"
              b"            lb += 1.25 * np.tanh((_OWN_AW['base_w'] @ hf + _OWN_AW['base_b']) / 1.25)\r\n"
              b'        return lw.T, lb.T'),
             (b'    lw, lb = _POLICY.forward(x, _SID)\r\n    workers,',b'    lw, lb = _POLICY.forward(x, _SID, own_root=True)\r\n    workers,')]
    source=parent
    for fr,to in changes:
        assert source.count(fr)==1
        source=source.replace(fr,to)
    restore=source
    for fr,to in reversed(changes):restore=restore.replace(to,fr)
    assert restore==parent and len(source)<10_000_000
    return source

if __name__=='__main__':
    candidate=assemble('awr')
    (HERE/'candidate.py').write_bytes(candidate)
    bc=assemble('bc');bc_v2=assemble('bc',True)
    (HERE/'bc_control.py').write_bytes(bc)
    (HERE/'v2_bc.py').write_bytes(bc_v2)
    save(HERE/'release.json',dict(parent=P['parent'],candidate_sha=sha(candidate),bc_control_sha=sha(bc),online_parent=V2_SHA,v2_bc_sha=sha(bc_v2),v2_bc_bytes=len(bc_v2),
                                  bytes=len(candidate),weights=sha((HERE/'weights.npz').read_bytes())))
