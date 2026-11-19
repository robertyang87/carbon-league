"""Count actual gradient-batch exposure for opening decisions; bind the exact published training schedule."""
from collections import Counter
import numpy as np
from shared import CACHE,HERE,TRAIN,read,require,save,sha


def run():
    panel=read(TRAIN/'panel.json')['games'];plan=read(TRAIN/'protocol.json');report=read(TRAIN/'fit.json')
    blocks=[];split_counts={}
    for split in ('train','val','holdout'):
        gg=[x for x in panel if x['split']==split];frame=[]
        for g in gg:
            p=CACHE/(g['id']+'.npz')
            require(sha(p)==g['shard_sha256'],'cache identity')
            with np.load(p,allow_pickle=False) as z:
                frame+=list(zip(z['gid'].astype(int),z['step'].astype(int),z['y'][:,10].astype(int),z['y'][:,11].astype(int)))
        matrix=np.array(frame);sel=matrix[matrix[:,1]==0]
        split_counts[split]={'maps':len(gg),'roots':len(matrix),'turn0_roots':len(sel),'turn1_5_roots':int(np.count_nonzero((matrix[:,1]>=1)&(matrix[:,1]<=5))),
                  'teacher_base_labels_at_t0':dict(Counter(str((int(r[2]),int(r[3]))) for r in sel))}
        if split=='train':blocks=matrix
    idx=np.random.default_rng(plan['random_seed']).permutation(len(blocks))[:plan['batch_size']*plan['updates']]
    selected=blocks[idx];schedule_hash=sha(selected[:,:2].astype('<i8').tobytes())
    require(schedule_hash==report['sample_schedule_gid_step_sha256'],'actual fitted schedule')
    opener=selected[selected[:,1]==0]
    answer=dict(splits=split_counts,schedule_sha256=schedule_hash,actual_trained_roots=len(selected),actual_root0_count=len(opener),
                actual_root0_groups=len(set(opener[:,0])),actual_public_labels_at0=opener.tolist(),base_labels=['none','hire_collector','hire_planter'])
    save(HERE/'opening_coverage.json',answer);return answer

if __name__=='__main__':print(run())
