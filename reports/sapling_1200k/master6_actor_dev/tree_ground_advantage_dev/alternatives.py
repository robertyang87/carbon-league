"""Price cells at the root with public memory; identify the parent's later purchases only as hindsight."""
from fit import HERE,sr,predicted,update_memory,values
from decision import _TO
def run(s):
    out=HERE/f'seat{s}';q=sr.load(out/'live_protocol.json');raw=sr.load(sr.ROOT/q['record']['source'])
    sim=sr.FastCarbon(raw['game_configuration']).reset(seed=q['seed']);mem={}
    for t,pair in enumerate(raw['actions']):
        ob=sim.obs(s);update_memory(ob,mem)
        if t==q['root_t']:break
        sim.step(*pair)
    rec=sr.load(out/'live_parent.json');tape=sr.load(out/rec['action_tape']);w=sr.load(HERE/'weights.json')['coef'];r=q['root']
    future=sr.load(out/'focal.json')[0]['t_first_acquire'];data=[]
    for idx,joint in enumerate(tape):
        if r['worker'] not in sim.workers or sim.step_count>(future if future is not None else q['root_t']+10):break
        cell=sim.workers[r['worker']].pos
        x=values(ob,mem,r['worker'],cell,sim.config)
        data.append(dict(actual_t=sim.step_count,cell=cell,actual_parent_order=joint[s].get(r['worker']),
                    root_neighbors=[dict(cell=int(nb),carbon_last=mem[int(nb)][0],last_t=mem[int(nb)][1]) if int(nb) in mem else dict(cell=int(nb),carbon_last=None,last_t=None) for nb in _TO[cell,1:]],
                    value_from_pre_move_public_history=None if x is None else predicted(x,w)))
        sim.step(*joint)
    sr.retain(out/'parent_path_at_root_prices.json',data)
if __name__=='__main__':
    for s in (0,1):
        if (HERE/f'seat{s}'/'focal.json').is_file():run(s)
