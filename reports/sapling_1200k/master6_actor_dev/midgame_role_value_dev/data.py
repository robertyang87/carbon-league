"""Compute identifiable fifty-turn receipts net of wages and tree payments for each actual midgame hire."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'tree_response_training'))
import select_roots as sr
spec=importlib.util.spec_from_file_location('worker_attribution',HERE.parent/'workforce_policy/baseline.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def extract():
    jobs=[]
    for c in sr.load(HERE.parent/'harvest_contact_dev/coverage.json.gz'):
        path=sr.ROOT/c['source'];raw=sr.load(path)
        raw['candidate_seat']=c['seat'];sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed'])
        raw['initial_terrain_u8_sha256']=base.H(bytes(map(int,sim.carbon)))
        r=base.audit(raw,dict(official_rewards=raw['official_score_by_seat']))
        for b in r['base_rows']:
            if 'uid' not in b or b['seat']!=c['seat'] or not 50<=b['t']<=200:continue
            w=r['workers'][b['uid']];inc=sum((v if kind in ('own_delivery','tree_absorption') else -v) for at,kind,v in w['events'] if at<=b['t']+50)
            jobs.append(dict(seed=c['seed'],seat=c['seat'],base=b,role=w['role'],net50=inc-w['price'],lifetime=w['lifetime_net']))
    sr.retain(HERE/'hires.json.gz',jobs)
if __name__=='__main__':extract()
