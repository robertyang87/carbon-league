"""Evaluate independently loaded public agents on each held archived prefix+suffix trajectory."""
from collections import Counter
from pathlib import Path
import gzip,json,sys,importlib.util,tempfile,subprocess
import numpy as np
import build
from build import HERE,ROOT,B,read,sha
spec=importlib.util.spec_from_file_location('entry_helper',HERE.parents[1]/'entry.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def run():
    fit=read(HERE/'fit.json');manifest=read(HERE/'manifest.json');plan=read(HERE/'protocol.json')
    bundles=build.bundles(plan);out={}
    if len(sys.argv)>1:
        arm=sys.argv[1]
        path=HERE/(arm+'.py');assert sha(path)==fit['artifacts'][arm]['.py'];cards=[]
        for c in manifest['cards']:
            if c['split']!='held':continue
            src=bundles[c['gid']];raw=read(src['path']);s=c['seat'];sim=B.FastCarbon(raw['game_configuration']).reset(seed=c['seed'])
            pack=helper.import_pack(path);idx=c['first_step'];feat=build.F.Feat(s);rows=[]
            with np.load(HERE/c['file']) as sh:
                for t,pair in enumerate(raw['actions'][:idx-1]+src['tape']):
                    obs=sim.obs(s);feat.observe(obs);cmd=pack.agent(obs,sim.config)
                    if t>=idx:
                        k=t-idx;actual=feat.labels(pair[s]);pred=feat.labels(cmd);keep=sh['eligible'][k];new=sh['unit_changed'][k]
                        assert np.array_equal(sh['src_act'][k],actual) and np.all((sh['pre_act'][k]!=actual)==new)
                        row={'step':t,'joint':bool(build.clean(cmd)==build.clean(pair[s]))}
                        for name,mask in [('replaced',new&keep),('retained',~new&keep)]:
                            row[name]=[int((pred[mask]==actual[mask]).sum()),int(mask.sum())]
                        rows.append(row)
                    sim.step(*pair)
            assert len(rows)==c['steps'] and np.allclose(sim.cash,src['scores'])
            cards.append(dict(gid=c['gid'],seed=c['seed'],seat=s,rows=rows))
        Path(sys.argv[2]).write_text(json.dumps(cards));return
    with tempfile.TemporaryDirectory() as td:
        for arm in ('raw_control','team','unit'):
            dest=Path(td)/(arm+'.json');subprocess.run([sys.executable,__file__,arm,str(dest)],check=True);out[arm]=json.loads(dest.read_text())
    scores={}
    for arm,cards in out.items():
        per={}
        for c in cards:
            v=per.setdefault(c['seed'],Counter())
            for row in c['rows']:
                v['joint_matches']+=row['joint'];v['roots']+=1
                for name in ['replaced','retained']:
                    v[name+'_matches']+=row[name][0];v[name+'_units']+=row[name][1]
        scores[arm]={k:dict(a) for k,a in per.items()}
    result=dict(fit=sha(HERE/'fit.json'),entry=sha(Path(__file__)),scores=scores,rows=out)
    path=HERE/'free_entry.json.gz';tmp=path.with_suffix('.partial')
    tmp.write_bytes(gzip.compress(json.dumps(result,separators=(',',':'),allow_nan=False).encode(),mtime=0));tmp.replace(path)
    print(scores)
if __name__=='__main__':run()
