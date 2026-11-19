from setup import *
import ast,base64,zlib
def run():
    p=bridge.load(HERE/'protocol.json');assert bridge.sha(PARENT)==p['parent'];src=PARENT.read_text()
    node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_return_decode')
    seg=ast.get_source_segment(src,node).replace('_return_decode(x, aux, encoded, apply_residual)','_jb_decode(x, aux, encoded, apply_residual, overrides, record)')
    at=seg.index('        if not apply_residual:');end=seg.index('        labels[slot] = action',at)
    seg=seg[:at]+'''        record[slot]=(combined,legal)
        if slot in overrides:
            ix=np.flatnonzero(legal)
            order=ix[np.argsort(-combined[ix],kind="stable")]
            action=int(order[min(overrides[slot],len(order)-1)])
'''+seg[end:]
    def blob(f):return base64.b64encode(zlib.compress(f.read_bytes(),9)).decode()
    import inspect
    helper='\n'.join(inspect.getsource(getattr(bridge,k)) for k in ('public_enemy','estimate_view','leaf'))
    inj=f"""
import io, types, zlib, base64
_JB_FAST=types.ModuleType('_jb_fast')
exec(zlib.decompress(base64.b64decode({blob(ROOT/'code/fastsim.py')!r})),_JB_FAST.__dict__)
FastCarbon=_JB_FAST.FastCarbon
_JB_CFG=FastCarbon({{}}).config
F=sys.modules['cnet2_feat']
_SW=dict(np.load(io.BytesIO(zlib.decompress(base64.b64decode({blob(HERE/'shared.npz')!r}))),allow_pickle=False))
"""
    out=HERE/'candidate.py';out.write_text(src+'\n'+inj+'\n'+helper+'\n'+seg+'\n'+(HERE/'runtime.py').read_text())
    assert out.stat().st_size<=10_000_000 and out.read_bytes().startswith(PARENT.read_bytes())
    bridge.save(HERE/'binding.json',dict(parent=bridge.sha(PARENT),bundle=bridge.sha(out),runtime=bridge.sha(HERE/'runtime.py'),network=bridge.sha(HERE/'shared.npz'),bytes=out.stat().st_size,sim=bridge.sha(ROOT/'code/fastsim.py')))
if __name__=='__main__':run()
