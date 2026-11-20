"""Describe independent simulated outcomes of one versus two draws at identical width."""
import numpy as np
from repeat import groups
from register import save
from pathlib import Path

if __name__ == '__main__':
    summary={}
    for split in ('train','held','extra'):
        rows=[]
        for seed,step,values in groups(split):
            first,second=values[:2,2,:]
            other_futures=values[2:,2,:].mean(axis=0)
            first_pick,second_pick,pooled_pick=(int(np.argmax(v)) for v in (first,second,(first+second)/2))
            rows.append(dict(seed=seed,step=step,first_pick=first_pick,second_pick=second_pick,
                             two_draws_pick=pooled_pick,
                             delta_vs_first=float(other_futures[pooled_pick]-other_futures[first_pick]),
                             delta_vs_second=float(other_futures[pooled_pick]-other_futures[second_pick])))
        terrains=sorted({r['seed'] for r in rows})
        map_means={s:{k:float(np.mean([r[k] for r in rows if r['seed']==s]))
                      for k in ['delta_vs_first','delta_vs_second']} for s in terrains}
        summary[split]=dict(maps=len(terrains),roots=len(rows),by_map=map_means,rows=rows,
                             equal_map_gain={k:float(np.mean([v[k] for v in map_means.values()]))
                                             for k in ['delta_vs_first','delta_vs_second']})
    save(Path(__file__).with_name('fixed_width.json'),summary)
