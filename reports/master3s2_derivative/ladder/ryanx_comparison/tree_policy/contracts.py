"""Check the packed hook plus its finite memory and target dimensions, using real histories only."""
import copy,gzip,json,time,types
from pathlib import Path
import numpy as np
from build import HERE,D,build,sha
from probe import assess,pub


def main():
    blob,identity=build();assert blob==(HERE/'submission.py').read_bytes()
    m=types.ModuleType('check_packed_tree');exec(compile(blob,'submission.py','exec'),m.__dict__)
    p=json.loads((HERE/'protocol.json').read_bytes());w=json.loads((D/'tree_control_response/weights.json').read_bytes());records=[]
    for g in p['games']:
        raw=(D/g['archive']).read_bytes();assert sha(raw)==g['sha256'];j=json.loads(gzip.decompress(raw));s=g['seat'];tr=types.SimpleNamespace();history={};negative=positive=0;times=[]
        for t,frame in enumerate(j['steps'][:-1]):
            ob=pub(frame,s);expected=assess(ob,j['configuration'],history,tr,m,w);negative+=sum(i['logit_change']<0 for i in expected.values());positive+=sum(i['predicted_net']>=0 for i in expected.values())
            baseline=np.full((225,5),.375,dtype=np.float32);target=baseline.copy();now=time.perf_counter();m._plant_reprice(ob,j['configuration'],tr,target);times.append(time.perf_counter()-now)
            for e in expected.values():baseline[e['cell'],0]+=e['logit_change']
            assert np.array_equal(target,baseline) and np.isfinite(target).all();past=tr._plant_public_history;assert list(past)==list(range(max(0,t-4),t+1))
            history[t]=ob['players'];history.pop(t-5,None)
        # Reuse the tracker across a fresh game's step zero, which must discard all prior history.
        base=np.zeros((225,5),np.float32);m._plant_reprice(pub(j['steps'][0],s),j['configuration'],tr,base);assert list(tr._plant_public_history)==[0]
        records.append(dict(game=g['name'],states=len(times),exact_estimate_and_offsupport_checks=len(times),negative_action_rows=negative,nonnegative_action_rows=positive,
                            overlay_median_ms=1000*float(np.median(times)),overlay_p99_ms=1000*float(np.quantile(times,.99)),overlay_max_ms=1000*max(times)))
    result={'identity':identity,'checks':records,'unit':'Only the pure overlay on real public boards; no search here. Includes every 299-step trajectory. Both startup and five-ply history windows checked; only own planter NONE logit may change and stays finite.'}
    (HERE/'contracts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(records)

if __name__=='__main__':main()
