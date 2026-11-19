"""Same-process save/restore of the signed native Master3s2. Call outside the timed agent request."""
import ast
import copy
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from types import ModuleType
import numpy as np

SOURCE="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
STATE=("_ST","_S_COST","_S_STATS","LAST_STATS","_SHIFT_T","_S_REC")
REFS=("_POLICY","_NET","_RNET")
OPTIONS=("_SEED","_TEMP","_SID","_RETIRE","_BANK","_HOME","_RX","_SEARCH_OK","_S_ENGINE","_S_START","_S_K",
         "_S_GEN","_S_DUPS","_S_M","_S_MMAX","_S_STAGES","_S_D2","_S_FILL","_S_ROWS","_S_EPS","_S_ABORT",
         "_S_SAFE","_S_CFIX","_S_HOMEFIX","_S_WINO","_S_ROLL","_S_RTEMP","DEFAULT_CONFIG")
HERE=Path(__file__).resolve().parent

def need(ok,msg):
    if not ok:raise ValueError(msg)

def digest(b):return hashlib.sha256(b).hexdigest()

@lru_cache(maxsize=32)
def source_sha(path):return digest(Path(path).read_bytes())

def canonical(x):
    if x is None or isinstance(x,(str,int,bool,float)):
        return x
    if isinstance(x,np.generic):return canonical(x.item())
    if isinstance(x,(bytes,bytearray)):return ['bytes',len(x),digest(x)]
    if isinstance(x,np.random.RandomState):return ['MT19937',canonical(x.get_state())]
    if isinstance(x,np.ndarray):
        need(not x.dtype.hasobject,'Opaque object array in native state')
        return ['ndarray',x.dtype.str,list(x.shape),digest(x.tobytes(order='C'))]
    if type(x).__module__.startswith('torch') and hasattr(x,'detach'):
        need(str(x.device)=='cpu','Only the registered CPU backend is supported')
        return ['tensor',str(x.dtype),list(x.shape),list(x.stride()),canonical(x.detach().contiguous().numpy())]
    if isinstance(x,dict):
        # Iteration/insertion order can determine per-worker RNG allocation.
        return ['dict',[[canonical(k),canonical(v)] for k,v in x.items()]]
    if isinstance(x,(tuple,list)):return ['tuple' if isinstance(x,tuple) else 'list',[canonical(y) for y in x]]
    if isinstance(x,(set,frozenset)):return ['set',sorted([canonical(y) for y in x],key=repr)]
    if isinstance(x,type):return ['class',x.__module__,x.__qualname__]
    if isinstance(x,ModuleType):return ['module',x.__name__]
    if hasattr(x,'__dict__'):return [type(x).__qualname__,canonical(vars(x))]
    raise TypeError(f'Unknown native snapshot field {type(x)}')

def h(x):
    return digest(json.dumps(canonical(x),allow_nan=False,ensure_ascii=False,separators=(',',':')).encode())

def prepare(m):
    need(source_sha(m.__file__)==SOURCE,'Wrong original module bytes')
    need(m._SEARCH_OK and m._RX and m._TEMP==1. and m._SEED is None and m._S_ABORT==.58 and m._NET is not None and m._RNET is not None,
         'Search, original sampler or warmed runtime unavailable')
    need(m._S_REC is None or isinstance(m._S_REC,list),'Unsupported native observation hook')
    need(isinstance(m._RX_CFG,list) and len(m._RX_CFG)==2,'Unknown engine configuration memo')
    return m

def static(m):
    # The signed forward functions only read these tensors; store no second network.
    # Verify their actual byte content on both capture and restore, independently of class names.
    return {'options':h({n:getattr(m,n) for n in OPTIONS}), 'policy':h(m._POLICY), 'net':h(m._NET),
            'rollout':'alias:_NET' if m._RNET is m._NET else h(m._RNET),
            'torch_threads':m._torch.get_num_threads(),'torch_interop':m._torch.get_num_interop_threads(),
            'torch_deterministic':m._torch.are_deterministic_algorithms_enabled(),'torch_dtype':str(m._torch.get_default_dtype())}

def rng(m):
    return (random.getstate(), np.random.get_state(),m._torch.get_rng_state().numpy().tobytes())

def dynamic(m):
    return {n:getattr(m,n) for n in STATE}

def fingerprint(m):
    prepare(m)
    return h((static(m),dynamic(m),m._RX_CFG,rng(m)))

def take(m):
    prepare(m)
    roots=dynamic(m)
    # Keeping the original cfg object matters: _turn tests "is configuration".
    # The caller's config is read-only throughout a registered pair; the derived config is private.
    ref=m._RX_CFG[0]
    return {'owner':m,'static_ids':tuple(id(getattr(m,k)) for k in REFS),'static':static(m),
            'state':copy.deepcopy(roots),'root_containers':dict(roots),'cfg0':ref,'cfg0_digest':h(ref),
            'cfg1':copy.deepcopy(m._RX_CFG[1]),'rx_ref':m._RX_CFG,'rng':copy.deepcopy(rng(m)),
            'expected':fingerprint(m)}

def restore(m,s):
    prepare(m)
    need(s['owner'] is m,'Snapshot belongs to another live native module')
    need(tuple(id(getattr(m,k)) for k in REFS)==s['static_ids'] and static(m)==s['static'],
         'Native network contents, aliases, or process execution settings changed')
    need(h(s['cfg0'])==s['cfg0_digest'],'Caller-mutated external configuration cannot be silently restored')
    restored=copy.deepcopy(s['state'])
    for n in STATE:
        original=s['root_containers'][n];v=restored[n]
        if isinstance(original,dict):original.clear();original.update(v)
        elif isinstance(original,list):original[:]=v
        elif original is not None:raise ValueError('Unrecognized native root container '+n)
        setattr(m,n,original)
    s['rx_ref'][:]=[s['cfg0'],copy.deepcopy(s['cfg1'])];m._RX_CFG=s['rx_ref']
    py,legacy,tensor=s['rng'];random.setstate(py);np.random.set_state(legacy)
    m._torch.set_rng_state(m._torch.from_numpy(np.frombuffer(tensor,dtype=np.uint8).copy()))
    need(fingerprint(m)==s['expected'],'Native same-process root restore did not reproduce every registered value')
    return s['expected']

def source_audit(path):
    raw=path.read_bytes();need(digest(raw)==SOURCE,'Unsigned source')
    tree=ast.parse(raw.decode())
    methods={}
    for name in ('_Policy','_TorchNet','_WinoNet'):
        obj=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
        rs={}
        for fn in obj.body:
            if not isinstance(fn,ast.FunctionDef) or fn.name=='__init__':continue
            stores=[(c.lineno,ast.unparse(c)) for c in ast.walk(fn) if isinstance(c,(ast.Attribute,ast.Subscript))
                    and isinstance(c.ctx,ast.Store) and ast.unparse(c).startswith('self.')]
            rs[fn.name]=dict(first=fn.lineno,last=fn.end_lineno,instance_stores=stores)
            need(not stores,'New writable network forward field')
        methods[name]=rs
    writes={}
    for name in ('_act','_turn','_shift_table','next_state','_net','_search'):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        writes[name]=[fn.lineno,fn.end_lineno]
    return dict(source_sha256=SOURCE,source_lines=writes,network_forward_source=methods,
        mutable_state={'_ST':'BeliefTracker, last_dest, every Belief cell/history/class/worker/tree field, physical prev players and full local NumPy RandomState.',
         '_S_COST':'Measured smoothed and last batch/physics costs. Affects candidate budget and adaptive cutoffs.',
         '_S_STATS':'Cumulative timing, depth, candidates, rollouts, counters and last search.',
         '_S_REC':'Optional full search records; None in original default. Preserve outer reference when recording.',
         '_RX_CFG':'Identity-keyed external configuration and derived merged config; preserve original external pointer and reject changed external contents.',
         '_SHIFT_T':'Mutable retained per-size geometry lookup used by embedded transitions; copied.',
         'LAST_STATS':'Embedded simulator’s last cell rent and worker withholding; restored even though the search does not select moves from it.',
         'ambient':'Python random, NumPy legacy global MT19937, CPU torch RNG. Root sampling has its own captured _ST RandomState; search-local per-depth RNGs are reconstructed from step and seat.'},
        immutable_runtime='Keep actual initialized policy, big/rollout Torch or Winograd instances with aliasing and signed weight bytes; pure forward uses local temporary tensors, no running BatchNorm state. Imported source has no assignment defining a _ROLL payload. Coordinate and symmetry tables built at module import are only read thereafter; all per-board rollout states are local.',
        boundary='Snapshots are at public-call boundaries in a single process; no active stack/forward/parallel request. PyTorch/BLAS allocator, oneDNN plans, OS scheduler, load and monotonic physical wall time cannot be rewound. Identical logical initial state does not guarantee identical future depths or chosen moves; retain each actual time/depth/score. Source temp choice at cold init is native; never reset global seeds during play.',
        ancestor='learned_carbon_observation/native_response/run.py capture/sig/restore for _ST, costs, stats and RNGs; extended to simulator memo, native diagnostic state, layer hashes and configuration identity without registering dynamic module names for pickle.')

def synthetic_control(path):
    """Isolated executable test of tracker, native diagnostics, caches, aliases, and three ambient RNG streams."""
    import sys
    spec=importlib.util.spec_from_file_location('unregistered_native_master_for_offline_control',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    need(m.__name__ not in sys.modules,'Synthetic control accidentally registered module')
    tracker=m.BeliefTracker(m._Bel,0);tracker.bel=m.Belief(0,[16,208]);tracker.last_dest={'w0':20}
    m._ST.update(tr=tracker,prev=[[2.45,{'base':16},{},{}],[3.03,{'base':208},{},{}]],rng=np.random.RandomState(51))
    config=dict(size=15);m._RX_CFG[:]=[config,dict(m.DEFAULT_CONFIG)];m._shift_table(15);m._S_REC=[]
    containers={n:getattr(m,n) for n in (*STATE,'_RX_CFG')};snap=take(m)
    for i in range(2):
        m._ST['tr'].bel.hist[17].append((2,1));m._ST['tr'].last_dest['w0']+=3;m._ST['rng'].rand()
        m._ST['prev'][0][1]['base']=22;m._S_COST[0]+=1;m._S_STATS['time'].append(9.1)
        m._S_REC.append('new');m._SHIFT_T[15].clear();m.LAST_STATS['worker_tax']=[4,5];m._RX_CFG[:]=[{},{}]
        random.random();np.random.rand();m._torch.rand(2)
        need(fingerprint(m)!=snap['expected'],'Did not detect intervention')
        need(restore(m,snap)==snap['expected'],'Repeated restore failed')
        need(all(getattr(m,n) is v for n,v in containers.items()) and m._RX_CFG[0] is config and m._ST['tr'].m is m._Bel,
             'Failed to restore original cross-object references')
    return dict(engine_requests=0,game_starts=0,native_module_registered_in_sys_modules=False,synthetic_restores=2,
                prior_mutation_detected=True,identical_state_and_container_aliases=True,torch_backend=type(m._NET).__name__,
                rollout_alias=m._RNET is m._NET,immutable_forward_hashes=snap['static'],
                changes_exercised=['tracker.bel.hist','tracker.last_dest','previous physical players','own NumPy RandomState',
                'recent/smoothed search costs','stats.time','optional search record','transition shift table','last worker tax',
                'configuration memo external identity','ambient Python/NumPy/Torch RNG streams'])


if __name__=='__main__':
    import argparse
    q=argparse.ArgumentParser();q.add_argument('examiner',type=Path);q.add_argument('--audit',type=Path,required=True)
    args=q.parse_args();out=source_audit(args.examiner);out['offline_mechanism_test']=synthetic_control(args.examiner)
    result=json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.audit.exists():need(args.audit.read_text()==result,'Existing snapshot source audit differs')
    else:args.audit.write_text(result)
