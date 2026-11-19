"""Join the 25-turn official cash account and replayed receipt categories by frozen game and turn."""
import gzip,hashlib,json
from pathlib import Path
import sys
import numpy as np
from pydantic_core import from_json
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent;BROAD=ACTOR/'broader_elite_backbone_dev'
sys.path.insert(0,str(ACTOR.parent))
from turning_point_ledger import FastCarbon,CashEvents
CACHE=Path('/workspace/sapling-all-top5-20261119/replays');SHARDS=Path('/workspace/sapling-broader-elite-v1-shards')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def one(g):
    p=CACHE/(g['id']+'.json.gz')
    if sha(p)!=g['sha256']:raise ValueError("source identity")
    game=from_json(gzip.decompress(p.read_bytes()));frames=game['steps'];s=g['seat']
    with np.load(SHARDS/(g['id']+'.npz'),allow_pickle=False) as shard:
        turns=shard['step'];ids=shard['gid']
    if any(ids!=g['gid']):raise ValueError("identity")
    opening=frames[0][0]['observation']
    if hashlib.sha256(bytes(opening['full_carbon'])).hexdigest()!=g['terrain_sha256']:raise ValueError('recorded opening identity')
    sim=FastCarbon(game['configuration']).load_observation(opening);ledger=np.zeros((len(frames),6))
    if not np.array_equal(sim.carbon,frames[0][0]['observation']['full_carbon']):raise ValueError('opening terrain')
    for t,frame in enumerate(frames):
        obs=frame[0]['observation']
        if any(abs(sim.cash[k]-obs['players'][k][0])>.011 for k in (0,1)):raise ValueError(('cash differs',g['id'],t,sim.cash,[v[0] for v in obs['players']]))
        own,other=sim.cash[s],sim.cash[1-s];ledger[t,0]=own;ledger[t,1]=other
        if t==len(frames)-1:break
        pair=[r.get('action') or {} for r in frames[t+1]]
        sim.cash=CashEvents(sim.cash);sim.step(*pair)
        ledger[t+1,2:]=ledger[t,2:]
        for kind,owner,val,*rest in sim.cash.events:
            side=int(owner!=s)
            if kind=='tree_absorption':ledger[t+1,2+side]+=val
            elif kind=='base_deposit':ledger[t+1,4+side]+=val
    future=np.minimum(turns.astype(int)+25,len(frames)-1)
    delta=ledger[future]-ledger[turns]
    labels=np.stack([delta[:,0]-delta[:,1],delta[:,2],delta[:,3],delta[:,4],delta[:,5]],1)
    return ids.astype('<i4'),turns.astype('<i4'),labels.astype('<f4'),future-turns,len(frames)-1
def main():
    panel=read(BROAD/'panel.json')['games'];result=[];frames=0
    for i,g in enumerate(panel):
        gid,t,targets,h,n=one(g);result.append((gid,t,targets,h));frames+=n
        if i%40==39:print(i+1,'maps',frames,'verified cash steps',flush=True)
    cols=list(zip(*result));arrays=[np.concatenate(v) for v in cols]
    np.savez_compressed(HERE/'labels_25.npz',gid=arrays[0],step=arrays[1],target=arrays[2],horizon=arrays[3])
    (HERE/'economic_labels.json').write_text(json.dumps(dict(cash_steps=frames,maps=len(panel),rows=len(arrays[0]),
        label_columns=['cash differential change','own tree receipts','opp tree receipts','own base deposits','opp base deposits'],
        source_panel_sha256=sha(BROAD/'panel.json'),labels_sha256=sha(HERE/'labels_25.npz'),label_code_sha256=sha(Path(__file__)),
        min_horizon=int(arrays[3].min()),max_horizon=int(arrays[3].max())),indent=2))
if __name__=='__main__':main()
