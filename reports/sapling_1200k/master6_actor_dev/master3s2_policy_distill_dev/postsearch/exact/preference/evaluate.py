"""Compare normal deployed actions and free-prefix pair ranks on two frozen maps."""
from pathlib import Path
import sys,json,gzip,tempfile,subprocess
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import entry as E
B=E.B;build=E.build

def worker(arm):
    manifest=E.read(HERE.parent/'manifest.json');plan=E.read(HERE.parent/'protocol.json');bundles=build.bundles(plan)
    path=HERE/'preference.py' if arm=='student' else E.ROOT/'data/weights/sapling_1200k/master6_actor_v1.py'
    if arm=='student':assert E.sha(path)==E.read(HERE/'fit.json')['pack']
    records=[]
    for c in manifest['cards']:
        if c['split']!='held':continue
        src=bundles[c['gid']];raw=E.read(src['path']);s=c['seat'];sim=B.FastCarbon(raw['game_configuration']).reset(seed=c['seed'])
        pack=E.helper.import_pack(path);policy=sys.modules['cnet2_np'].Policy;original=policy.act;cache=[]
        def audit(self,*a,**k):
            result=original(self,*a,**k);cache.append(result[1].copy());return result
        policy.act=audit;feat=build.F.Feat(s);idx=c['first_step']
        with np.load(HERE.parent/c['file']) as z:
            d={k:z[k] for k in ('eligible','legal','unit_changed','src_act','pre_act')}
        try:
            for t,pair in enumerate(raw['actions'][:idx-1]+src['tape']):
                obs=sim.obs(s);feat.observe(obs);cache.clear();cmd=pack.agent(obs,sim.config);assert len(cache)==1
                if t>=idx:
                    i=t-idx;old=d['pre_act'][i];new=d['src_act'][i];changed=d['unit_changed'][i];mask=d['eligible'][i]
                    pred=feat.labels(cmd);live=new>=0;assert np.array_equal(new,feat.labels(pair[s]))
                    kind=mask&changed&d['legal'][i,np.arange(len(old)),old.clip(0)]
                    pairs={}
                    for j in np.where(kind)[0]:
                        a,b=int(old[j]),int(new[j]);lg=cache[0][j];ok=lg[a]>-1e8 and lg[b]>-1e8
                        pairs[feat.own_uid[j]]=[a,b,int(pred[j]),float(lg[b]-lg[a]) if ok else None]
                    kept=mask&~changed
                    records.append(dict(gid=c['gid'],seed=c['seed'],seat=s,t=t,cmd=cmd,joint=bool(np.all(pred[live]==new[live])),
                      kept=[int((pred[kept]==new[kept]).sum()),int(kept.sum())],pairs=pairs))
                sim.step(*pair)
        finally:policy.act=original
    return records
def main():
    if len(sys.argv)>1:
        Path(sys.argv[2]).write_text(json.dumps(worker(sys.argv[1])));return
    with tempfile.TemporaryDirectory() as td:
        arms=[]
        for arm in ('parent','student'):
            dest=Path(td)/arm;subprocess.run([sys.executable,__file__,arm,str(dest)],check=True);arms.append(json.loads(dest.read_text()))
    totals={};eligible_roots={}
    for a,b in zip(*arms,strict=True):
        assert all(a[k]==b[k] for k in ['gid','seed','seat','t']) and a['pairs'].keys()==b['pairs'].keys()
        k=(a['seed'],a['seat']);v=totals.setdefault(k,Counter())
        changed=a['cmd']!=b['cmd'];v['roots']+=1;v['changed_joints']+=changed
        for label,x in zip(['parent','student'],[a,b]):
            v[label+'_joint']+=x['joint'];v[label+'_kept_correct']+=x['kept'][0]
        v['kept_units']+=a['kept'][1]
        for uid,pair in a['pairs'].items():
            other=b['pairs'][uid];v['teacher_legal_pairs']+=1
            if pair[-1] is None or other[-1] is None:continue
            v['common_pairs']+=1
            for label,x in zip(['parent','student'],[pair,other]):
                v[label+'_ranked']+=x[-1]>0;v[label+'_chose']+=x[2]==x[1]
            if changed and pair[0]==pair[2] and other[1]==other[2] and k not in eligible_roots:
                eligible_roots[k]=dict(gid=a['gid'],t=a['t'],uid=uid,old=pair,new=other)
    maps={f'{s}_s{seat}':dict(v) for (s,seat),v in totals.items()}
    checks={key:dict(coverage=v['common_pairs']/v['teacher_legal_pairs']>=.95,
                    higher_rank=v['student_ranked']>v['parent_ranked'],actual_chosen=v['student_chose']>=v['parent_chose'],
                    retained=v['student_kept_correct']>=v['parent_kept_correct']-.02*v['kept_units']) for key,v in maps.items()}
    result=dict(checks=checks,passed=all(all(w.values()) for w in checks.values()),stats=maps,roots={f'{s}_s{seat}':v for (s,seat),v in eligible_roots.items()},
                scorer=E.sha(Path(__file__)),fit=E.sha(HERE/'fit.json'),parent=E.sha(E.ROOT/'data/weights/sapling_1200k/master6_actor_v1.py'))
    B.save(HERE/'results.json',result)
    with (HERE/'commands.json.gz').open('wb') as f:f.write(gzip.compress(json.dumps(arms,separators=(',',':')).encode(),mtime=0))
    print(result,flush=True)
if __name__=='__main__':main()
