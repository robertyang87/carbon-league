"""Separate the observed same-map difference before the first action change from its later change."""
from data import sr,HERE
def run():
    result=[]
    for folder in (HERE.parent/'harvest_contact_dev',HERE):
        for seat in (0,1):
            if folder==HERE:root=sr.load(HERE/f'audit_candidate_{seat}.json.gz')['details']['root']
            else:root=next(r for r in sr.load(folder/'full_ledger.json') if r['arm']=='candidate' and r['seat']==seat)['root']
            v={};t=root['t']
            for arm in ('candidate','parent'):
                path=next((folder/'games'/arm).glob(f'raw_*_{seat}.json.gz'));raw=sr.load(path)
                sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
                for pair in raw['actions'][:t]:sim.step(*pair)
                pre=list(sim.cash);sc=raw['official_score_by_seat'];v[arm]=dict(pre=pre[seat]-pre[1-seat],final=sc[seat]-sc[1-seat])
            pre=v['candidate']['pre']-v['parent']['pre'];final=v['candidate']['final']-v['parent']['final']
            result.append(dict(seed=raw['seed'],seat=seat,first_change=t,arms=v,pre_difference=pre,final_difference=final,later_change=final-pre))
    sr.retain(HERE/'prefix_gap.json',result)
    for r in result:print(r['seed'],r['seat'],round(r['pre_difference'],2),round(r['later_change'],2),round(r['final_difference'],2))
if __name__=='__main__':run()
