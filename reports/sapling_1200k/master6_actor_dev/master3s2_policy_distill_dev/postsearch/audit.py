"""Bind the study's clock and final orders to the archived official receipts, and separate prefix from source-argmax mismatches."""
from collections import Counter
from dataset import HERE,ROOT,B,read,sha
import numpy as np
def run():
    manifest=read(HERE/'manifest.json');result={}
    for c in manifest['cards']:
        r=read(ROOT/c['receipt']);raw=read(ROOT/c['source']);wall=read(ROOT/c['wall'])
        assert r['binding']['original']==B.ORIGINAL_SHA
        assert r['game']['agent_errors_by_seat']==[0,0] and r['official']['statuses']==['DONE','DONE']
        assert r['official']['official_reward_by_seat']==raw['official_score_by_seat']
        assert r['official']['steps']==c['steps'] and r['official']['map_sha256']==c['terrain']
        with np.load(HERE/c['file'],allow_pickle=False) as a:
            assert np.array_equal(a['team_changed'],[v['changed'] for v in wall['calls']])
            for v,tag in [(True,'changed'),(False,'not_changed')]:
                rs=a['team_changed']==v;el=a['eligible']&rs[:,None]
                summary=result.setdefault(c['split']+'/'+tag,Counter())
                summary.update(roots=int(rs.sum()),actionable=int(el.sum()),recorded_argmax_diff=int(((a['q'].argmax(-1)!=a['src_act'])&el).sum()),
                               differing_joint_modal=int((((a['q'].argmax(-1)!=a['src_act'])&el).any(-1)&rs).sum()))
    B.save(HERE/'audit.json',dict(manifest_sha256=sha(HERE/'manifest.json'),totals=result))
if __name__=='__main__':run()
