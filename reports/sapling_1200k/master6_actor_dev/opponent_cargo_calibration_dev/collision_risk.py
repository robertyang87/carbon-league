"""Optional supplementary horizon risks, derived from realized positive cargo transfers."""
import numpy as np
from model import HERE,public_rows,sigmoid

class CollisionRisk:
    def __init__(self):
        with np.load(HERE/'collision_head.npz',allow_pickle=False) as file:self.w=dict(file)

    def predict(self,history):
        rows=public_rows(history)
        if not rows:return {}
        w=self.w;x=np.stack([r[1] for r in rows]);z=np.clip((x-w['mu'])/w['scale'],-10,10)
        p=sigmoid((z@w['weights']+w['bias'])*w['platt_slope']+w['platt_bias'])
        p[:,1]=np.maximum(p[:,1],p[:,0]);p[:,3]=np.maximum(p[:,3],p[:,2])
        return {r[0]:dict(zip(('lose3','lose5','take3','take5'),map(float,p[i]))) for i,r in enumerate(rows)}
