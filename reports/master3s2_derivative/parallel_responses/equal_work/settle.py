"""Intervene on 2 preselected pairs from exactly the same completed M2 search."""
import json
import os
from pathlib import Path
import sys
import importlib.util

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
spec=importlib.util.spec_from_file_location('shared_registered_suffix',HERE.parent/'settle.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
panel=base.panel
LOCK='7a739abf55e6c873f17eae2302ec3d0e5004ea9eab96b5b81c8bf447827e1ab2'


def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    os.environ['CARBON_DEBUG']='1'
    assert os.sched_getaffinity(0)=={4}
    raw=(HERE/'protocol.json').read_bytes();assert panel.sha(raw)==LOCK
    plan=json.loads(raw);assert plan['parent_protocol_sha']==panel.LOCK
    (HERE/'branches').mkdir(exist_ok=True);results=[]
    for i,selection in enumerate(plan['selected']):
        seed,s,t=(selection[k] for k in ('seed','seat','step'))
        cohort=next(c for c in plan['cohort'] if (c['seed'],c['seat'],c['step'])==(seed,s,t))
        card=next(c for c in panel.PLAN['tickets'] if (c['seed'],c['seat'])==(seed,s))
        rows,sha=base.signed_panel(card)
        assert sha==selection['panel_sha']==cohort['panel_sha']
        row=next(r for r in rows if r['step']==t);m2=row['M2'];assert row['public_sha']==cohort['public_sha']
        single=m2['menu'][m2['individual_winners'][0]];mean=m2['issued']
        assert single==cohort['single0'] and mean==cohort['mean2'] and single!=mean
        assert m2['clock']==cohort['work']
        assert selection['order']==(['single0','mean2'] if i==0 else ['mean2','single0'])
        row=dict(row,single0={'issued':single,'clock':m2['clock']},mean2={'issued':mean,'clock':m2['clock']})
        results.append(base.suffix(card,row,sha,i,variants=('single0','mean2'),scope=HERE,protocol=LOCK))
    assert sum(a['transitions'] for r in results for a in r['arms'].values())<=plan['max_joint_transitions']
    panel.put(HERE/'terminal.json',{'protocol':LOCK,'selected':results,'registered_cohort':len(plan['cohort']),
                     'selected_from_cohort':len(results),'scope':'Old-map counterfactuals with one common true root opponent response and identical M2 root work; remaining turns are newly played by the original opponent and the online v2.'})
    for r in results:print(r,flush=True)


if __name__=='__main__':main()
