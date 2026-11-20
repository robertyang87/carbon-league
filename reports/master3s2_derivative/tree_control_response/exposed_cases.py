"""Score previously named simultaneous positive and negative tree investments, without fitting to them."""
from collections import defaultdict
import gzip,json
from pathlib import Path
import sys
from build import HERE,sha
D=HERE.parent
sys.path[:0]=[str(D/"ladder/tree_payback"),str(D.parents[1]/"reports/sapling_1200k/all_top5")]
import labels,runtime as tree_features
from context import Game
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location("old_tree_features",D/"ladder/tree_payback/runtime.py");old=module_from_spec(spec);spec.loader.exec_module(old)

def run():
    w=json.loads((HERE/"weights.json").read_bytes())
    risk=json.loads((D/"ladder/tree_payback/weights.json").read_bytes())["full"]
    jobs=[("eec452f455f16f82bb432ce057daa25c","N34",[(137,159),(137,59),(142,30)]),
          ("3f0a1dfa0e1ac4a00a9f63f9cf644af9","Hajimi",[(246,214),(246,56),(255,89),(259,134),(261,120)])]
    rs=[]
    for gid,name,events in jobs:
        source=D/"ladder/replays"/(gid+"-1.json.gz");raw=gzip.decompress(source.read_bytes());data=json.loads(raw);game=Game(data);cfg=data["configuration"]
        receipts={(r["t"],r["cell"]):r for r in labels.tree_rows(data) if r["seat"]==1}
        for t,cell in events:
            assert (t,cell) in receipts
            r=receipts[t,cell];uid=r["planter"];obs=game._obs(t,1)
            x=old.features(obs,uid,cfg,{i:game._obs(t-i,1)["players"] for i in (3,5)})
            z=[(v-m)/s for v,m,s in zip(x,w["feature_mean"],w["feature_scale"])]
            net,hold=[mu+sum(a*b[j] for a,b in zip(z,w["coef"])) for j,mu in enumerate(w["means"])]
            enemies=obs["players"][0]
            near=min((old.distance(cell,v[0],cfg["size"]) for v in enemies[3].values() if v[1]<cfg["treeLifespan"]-1),default=None)
            rs.append(dict(game=name,source=sha(source.read_bytes()),t=t,cell=cell,planter=uid,quoted_cost=r["cost"],actual_net=r["net"],actual_hold=r["end_t"]-t,
                           actual_exit=r["exit"],predicted_net=net,predicted_hold=hold,nonpayback=old.predict(x,risk),distance_enemy_tree=near))
    (HERE/"exposed_cases.json").write_text(json.dumps(rs,indent=2,ensure_ascii=False)+"\n")
    print([(r['game'],r['t'],r['cell'],round(r['actual_net']),round(r['predicted_net']),r['distance_enemy_tree']) for r in rs])
if __name__=="__main__":run()
