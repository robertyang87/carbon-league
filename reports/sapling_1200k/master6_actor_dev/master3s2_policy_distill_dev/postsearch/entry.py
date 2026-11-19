"""Free-decoding check against final actual commands on held public histories, with native change groups."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,tempfile,subprocess,sys,os
from dataset import HERE,ROOT,B,read,sha
sys.path.insert(0,str(HERE.parent))
import importlib.util
spec=importlib.util.spec_from_file_location('frozen_parent_entry_tools',HERE.parent/'entry.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
def worker(arm):
    cards=read(HERE/'manifest.json')['cards'];out=[]
    pack_path=(ROOT/'data/weights/sapling_1200k/master6_actor_v1.py' if arm=='parent' else HERE/(arm+'.py'))
    for c in cards:
        if c['split']=='fit':continue
        sim,pairs=B.replay(c);mod=common.import_pack(pack_path);s=c['seat'];wall=read(ROOT/c['wall'])['calls'];rows=[]
        for t,pair in enumerate(pairs):
            obs=sim.obs(s);side=obs['players'][s];roles={uid:'base' for uid in side[1]}
            roles.update({uid:v[2].lower() for uid,v in side[2].items()})
            original=pair[s];output=mod.agent(obs,sim.config)
            if not set(original)|set(output)<=set(roles):raise ValueError('foreign command key')
            tally=Counter();correct=Counter()
            for uid,role in roles.items():
                tally[role]+=1;correct[role]+=int(output.get(uid)==original.get(uid))
            rows.append(dict(t=t,changed=wall[t]['changed'],exact=(output==original),counts=dict(tally),correct=dict(correct)))
            sim.step(*pair)
        out.append(dict(gid=c['gid'],map=c['terrain'],seat=s,seed=c['seed'],split=c['split'],steps=c['steps'],rows=rows))
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm');p.add_argument('--out',type=Path);a=p.parse_args()
    if a.arm:
        B.save(a.out,worker(a.arm));return
    fit=read(HERE/'fit.json');out={};stats={}
    with tempfile.TemporaryDirectory() as td:
        for arm in ('parent','raw_control','postsearch'):
            path=Path(td)/(arm+'.json')
            if arm!='parent':assert fit['artifacts'][arm]['.py']==sha(HERE/(arm+'.py'))
            subprocess.run([sys.executable,__file__,'--arm',arm,'--out',str(path)],check=True)
            out[arm]=read(path);per={}
            for g in out[arm]:
                m=per.setdefault(g['seed'],defaultdict(lambda:Counter()))
                for row in g['rows']:
                    z=m['changed' if row['changed'] else 'unchanged']
                    z['roots']+=1;z['joint_matches']+=row['exact'];z['live_slots']+=sum(row['counts'].values());z['matching_slots']+=sum(row['correct'].values())
            stats[arm]={str(k):{t:dict(v) for t,v in tab.items()} for k,tab in per.items()}
    import json,gzip
    path=HERE/'free_entry.json.gz';tmp=path.with_suffix('.partial')
    data=dict(fit_sha256=sha(HERE/'fit.json'),source_sha256=sha(Path(__file__)),stats=stats,records=out)
    tmp.write_bytes(gzip.compress(json.dumps(data,separators=(',',':'),allow_nan=False).encode(),mtime=0));tmp.replace(path)
    print(stats,flush=True)
if __name__=='__main__':main()
