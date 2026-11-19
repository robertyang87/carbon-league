"""Coverage audit against historical observations; future positions are never policy inputs."""
from pure_screen import HERE,sr
from route import _TO,_DIRECTION,_D
def check():
    results=[]
    for entry in sr.load(HERE/'pure_screen.json'):
        s=entry['seat'];q=sr.load(HERE/f'seat{s}/live_protocol.json')
        raw=sr.load(sr.ROOT/q['record']['source']);sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
        pending=[]
        proposals={r['t']:r for r in entry['proposals']}
        for t,pair in enumerate(raw['actions']):
            if t in proposals:
                r=proposals[t];worker=sim.workers[r['worker']]
                cmd=r['alternative'];idx=_DIRECTION.index(None if cmd=='NONE' else cmd)
                pending.append(dict(root=r,first_cell=int(_TO[worker.pos,idx]),actual_path=[]))
            sim.step(*pair)
            for p in pending:
                if 0<=t-p['root']['t']<2:
                    enemy=sim.workers.get(p['root']['target'])
                    p['actual_path'].append(None if enemy is None else [enemy.pos,enemy.carbon])
        for p in pending:
            end=p['actual_path'][1];p['original_target_reachable']=bool(end and _D[p['first_cell'],end[0]]<=1)
            p['original_target_matches_plan']=bool(end and end[0]==p['root']['planned_cell'])
        results.append(dict(seed=entry['seed'],seat=s,opportunities=pending))
    sr.retain(HERE/'support.json',results)
if __name__=='__main__':check()
