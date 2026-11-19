"""Three public-state factors folded into the frozen Sapling worker output head."""
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Conditioning:
    hidden_mean: np.ndarray
    projection: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    ridge: float
    training_maps: dict
    fit_rmse_standardized: list

    @classmethod
    def fit(cls, hidden_samples, public_targets, terrains, ridge=0.01):
        h=np.asarray(hidden_samples,dtype=np.float64)
        p=np.asarray(public_targets,dtype=np.float64)
        ids=list(terrains)
        if h.ndim!=2 or not len(h) or not h.shape[1] or p.shape!=(len(h),3) or len(ids)!=len(h):
            raise ValueError('expected one hidden vector, three public targets and one terrain per training slot')
        if not np.isfinite(h).all() or not np.isfinite(p).all() or not math.isfinite(ridge) or ridge<=0:
            raise ValueError('finite input and positive ridge required')
        if any(not isinstance(k,str) or not k for k in ids) or np.any(p[:,:2]<0) or np.any(p[:,1]>14) or not np.isin(p[:,2],[0.,1.]).all():
            raise ValueError('invalid physical target or map id')
        counts=Counter(ids);w=np.array([1./(len(counts)*counts[t]) for t in ids])
        # Current cargo, toroidal home distance and a lighter enemy's possible
        # one-step entry into this cell are observable before the sampled action.
        y=np.column_stack((np.log1p(p[:,0]),p[:,1]/14.,p[:,2]))
        hm=w@h;ym=w@y
        x=h-hm;yc=y-ym
        scale=np.sqrt(w@(yc*yc))
        inactive=np.ptp(y,axis=0)==0
        scale[inactive]=1.
        target=yc/scale;target[:,inactive]=0.
        design=x.T@(w[:,None]*x)+float(ridge)*np.eye(h.shape[1])
        projection=np.linalg.solve(design,x.T@(w[:,None]*target))
        err=x@projection-target
        return cls(hm,projection,ym,scale,float(ridge),dict(sorted(counts.items())),np.sqrt(w@(err*err)).tolist())

    @staticmethod
    def _delta(delta):
        q=np.asarray(delta,dtype=np.float64)
        if q.shape not in ((15,),(5,3)) or not np.isfinite(q).all():
            raise ValueError('expected 15 finite worker-action-by-context coefficients')
        return q.reshape(5,3)

    def features(self,hidden):
        x=np.asarray(hidden,dtype=np.float64)
        if x.ndim<1 or x.shape[-1]!=len(self.hidden_mean) or not np.isfinite(x).all():
            raise ValueError('hidden feature order or dimension changed')
        return (x-self.hidden_mean)@self.projection

    def head_increment(self,hidden,delta):
        return self.features(hidden)@self._delta(delta).T

    def inject(self,student_initial_state_or_npz,delta):
        # The packer stores both of these small tensors in raw FP32; the
        # deployable result is the rounded increment of each original tensor.
        source=student_initial_state_or_npz
        if isinstance(source,(str,Path)):
            with np.load(source,allow_pickle=False) as z:
                s={k:z[k] for k in z.files}
        else:
            s={k:(v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)) for k,v in source.items()}
        w,b=s['whead.weight'],s['whead.bias']
        if w.shape!=(5,len(self.hidden_mean)) or b.shape!=(5,) or w.dtype!=np.float32 or b.dtype!=np.float32:
            raise ValueError('expected the original raw-FP32 output tensors')
        add=self._delta(delta)@self.projection.T
        bias_add=-(add@self.hidden_mean)
        s['whead.weight']=(w.astype(np.float64)+add).astype(np.float32)
        s['whead.bias']=(b.astype(np.float64)+bias_add).astype(np.float32)
        if not np.isfinite(s['whead.weight']).all() or not np.isfinite(s['whead.bias']).all():
            raise ValueError('nonfinite deployed output layer')
        return s

    def write(self,path):
        obj={'version':1,'targets':['log1p_public_cargo','own_base_distance_div_14','strict_lighter_enemy_can_enter_current_cell'],
             'ridge':self.ridge,'hidden_mean':self.hidden_mean.tolist(),'projection':self.projection.tolist(),
             'target_mean':self.target_mean.tolist(),'target_scale':self.target_scale.tolist(),
             'training_maps':self.training_maps,'fit_rmse_standardized':self.fit_rmse_standardized}
        dst=Path(path);data=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        if dst.exists() and dst.read_bytes()!=data:
            raise ValueError('frozen conditioning already differs')
        tmp=dst.with_suffix(dst.suffix+'.tmp');tmp.write_bytes(data);tmp.replace(dst)

    @classmethod
    def load(cls,path):
        a=json.loads(Path(path).read_bytes())
        if a['version']!=1:raise ValueError('unsupported conditioning revision')
        mean,mat,ym,ys=[np.asarray(a[k],dtype=np.float64) for k in ('hidden_mean','projection','target_mean','target_scale')]
        if mean.ndim!=1 or mat.shape!=(len(mean),3) or ym.shape!=(3,) or ys.shape!=(3,) or np.any(ys<=0) or not all(np.isfinite(x).all() for x in (mean,mat,ym,ys)):
            raise ValueError('invalid factor state')
        return cls(mean,mat,ym,ys,float(a['ridge']),a['training_maps'],a['fit_rmse_standardized'])
