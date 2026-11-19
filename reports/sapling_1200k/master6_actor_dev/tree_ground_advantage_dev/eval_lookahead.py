from fit import HERE,ACTOR,sr
from decision import consider,update_memory
from lookahead import further
from collections import Counter
def run():
    base=sr.load(HERE/'paired_decisions.json.gz')
    original={r['case']:r for r in sr.load(ACTOR/'two_arm_plant_economy_dev/native_eval_rows.json.gz')};w=sr.load(HERE/'weights.json')['coef'];cache={};out=[]
    for r in base:
        case=sr.ROOT/r['case'];q=sr.load(case/'live_protocol.json');s=r['seat'];t=r['t'];uid=r['worker'];src=q['record']['source']
        if (src,s) not in cache:
            raw=sr.load(sr.ROOT/src);cache[src,s]=raw,sr.FastCarbon(raw['game_configuration']).reset(seed=r['seed']),{}
        raw,sim,mem=cache[src,s]
        while sim.step_count<t:
            update_memory(sim.obs(s),mem);sim.step(*raw['actions'][sim.step_count])
        obs=sim.obs(s);update_memory(obs,mem);arms=original[r['case']];cmd=arms['own_root_joint']['move']
        new=r['pick'];entry=r['decision'];support=further(obs,mem,cmd,uid,sim.config,w)
        if entry and entry['accept'] and support['sites']:
            best=support['sites'][0]
            if best['net']>=entry['ground_stay']-entry['quote']:new='move'
        money=lambda k:arms['arms'][k]['official_margin']
        out.append(dict(seed=r['seed'],seat=s,t=t,case=r['case'],parent=r['parent'],current=r['pick'],extended=new,
             panel=r['panel'],priced=entry is not None,sites=support['sites'],excluded=support['excluded'],
             current_delta=money(r['pick'])-money(r['parent']),extended_delta=money(new)-money(r['parent'])))
    sr.retain(HERE/'lookahead_rows.json.gz',out)
    for tag in ('all','train','hold','outside'):
        rs=[r for r in out if r['panel']==tag or tag=='all']
        a=dict(roots=len(rs),priced=sum(r['priced'] for r in rs),alternative_available=sum(bool(r['sites']) for r in rs),available_on_priced=sum(bool(r['sites']) and r['priced'] for r in rs),
               old_override=sum(r['current']!=r['parent'] for r in rs),new_override=sum(r['extended']!=r['parent'] for r in rs),changed_vs_old=sum(r['extended']!=r['current'] for r in rs),
               margin_delta=sum(r['extended_delta'] for r in rs)/len(rs))
        print(tag,a)
if __name__=='__main__':run()
