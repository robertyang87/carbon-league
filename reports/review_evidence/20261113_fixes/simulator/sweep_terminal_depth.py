"""Exercise production _search control flow with deterministic inference only.
Original game transitions, features, leaf and _search source are unchanged.
"""
import ast,pathlib,sys,json,time,types,tempfile
import os

def find_repository():
    explicit = os.environ.get("CARBON_REPO")
    if explicit:
        root = pathlib.Path(explicit).expanduser().resolve()
        if not (root / "code/fastsim.py").is_file():
            raise RuntimeError("CARBON_REPO must name a checkout containing code/fastsim.py")
        return root
    starts = (pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve())
    for start in starts:
        for root in (start, *start.parents):
            if (root / "code/fastsim.py").is_file():
                return root
    raise RuntimeError("Cannot locate code/fastsim.py; set CARBON_REPO to the checkout root")

ROOT = find_repository()
# Every execution writes into a NEW temporary directory, preserving all historical evidence.
review_base = os.environ.get("CARBON_REVIEW_OUT")
if review_base:
    pathlib.Path(review_base).expanduser().mkdir(parents=True, exist_ok=True)
    review_base = str(pathlib.Path(review_base).expanduser().resolve())
OUT = pathlib.Path(tempfile.mkdtemp(prefix="carbon-simulator-r3-", dir=review_base))
sys.path.insert(0,str(ROOT/'code'))
import numpy as np,cnet2_search as S,cnet2_feat as F
from fastsim import FastCarbon
src=ast.parse((ROOT/'code/cnet2_search.py').read_text())
fac=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='make_search_agent')
funcs=[n for n in fac.body if isinstance(n,ast.FunctionDef) and n.name in ('_extinct','_terminal','leaf','_search')]
sim=FastCarbon().reset(full_carbon=[0]*225);obs=sim.full_observation();obs['step']=100
for p in (0,1):obs['players'][p][0]=20
obs['players'][0][2]={'player-0-worker-0':[sim.bases[1][0][1]+15,0,'COLLECTOR']}
obs['players'][1][2]={'player-1-worker-0':[sim.bases[1][0][1]+14,100,'COLLECTOR']}
# Default A=UP enters enemy base 70 empty and dies. B=RIGHT enters 85.
# Candidate A=NONE stays 85, captures B (100), survives; B then extinct.
f=F.Feat(0);fo=F.Feat(1);x=f.observe(obs);oo=dict(obs,player=1);xo=fo.observe(oo)
y=np.zeros(F.OWN,np.int64);y[0]=F.MOVES.index('UP');yo=np.zeros(F.OWN,np.int64);yo[0]=F.MOVES.index('LEFT') # seat-1 canonical LEFT maps to world RIGHT
class Pol:
 def enc_for(self,*a):return np.zeros((1,1)),np.zeros(1)
 def act(self,*a,**kw):return yo.copy(),None,None

def decode(pol,X,aux,tok,ctx,forced=None,**kw):
 return np.stack([np.where(a>=0,a,y) for a in forced])
res=[]
for md in (3,1):
 cfg=dict(S.DEFAULTS,MAX_BR=2,MIN_DEPTH=md,H=5,LOG=True,BUDGET=100,HARD=100)
 stats={k:0 for k in ('searched','branches','branch_steps','depth_sum','shallow','changed','gain')};log=[]
 ns=dict(cfg=cfg,st={'f':f,'fo':fo,'T_now':1,'ot':1,'oenc':None},np=np,F=F,pol=Pol(),rpol=None,pool=[],stats=stats,log=log,
  T=1,key_units=S.key_units,decode_b=decode,FastCarbon=FastCarbon,feat_copy=S.feat_copy,clk=lambda:0,time=time)
 exec(compile(ast.Module(body=funcs,type_ignores=[]),'production_search_current','exec'),ns)
 try:
  a=ns['_search'](obs,x,y,{'player-0-worker-0':'UP'},xo,199,0,time.time())
 except Exception:
  print('DEBUG',f.own_uid,fo.own_uid,F.MOVES,sim.bases,[(z.cash,z._players_obs()) for z in ns['pool']],file=sys.stderr);raise
 terminal=[]
 for poolsim in ns['pool']:
  terminal.append({'step':poolsim.step_count,'extinct':[ns['_extinct'](poolsim,p) for p in (0,1)],'leaf':ns['leaf'](poolsim,0,198,[20,20])})
 res.append({'MIN_DEPTH':md,'returned_actions':a,'stats':stats,'branch_results':terminal,'log':log})
payload = json.dumps(res, indent=2)
(OUT / "sweep-terminal-depth.json").write_text(payload + "\n", encoding="utf-8")
print(payload)
print(f"Fresh evidence: {OUT / 'sweep-terminal-depth.json'}", file=sys.stderr)
