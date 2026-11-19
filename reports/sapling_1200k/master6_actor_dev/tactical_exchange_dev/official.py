"""Replay each altered step and its control with the official interpreter from its actual prefix."""
from measure import HERE,sr
def run():
    out=[]
    for archive in ('paired','advance'):
        for game in sr.load(HERE/(archive+'.json.gz')):
            raw=sr.load(sr.ROOT/game['path']);s=game['seat']
            for row in game['rows']:
                t=row['t'];choice=row['choice']
                for arm in ('parent','changed'):
                    env=sr.live.begin(raw)
                    for pair in raw['actions'][:t]:env.step(pair)
                    pair=[dict(cmd) for cmd in raw['actions'][t]]
                    if arm=='changed':pair[s][choice['uid']]=choice['after']
                    env.step(pair);players=sr.live.public(env,s)['players']
                    holder=next((i for i,side in enumerate(players) if choice['tree'] in side[3]),None)
                    got=dict(holder=holder,balance=[p[0] for p in players],enemy_alive=choice['enemy'] in players[1-s][2],collector_alive=choice['uid'] in players[s][2])
                    if got!=row[arm]:raise ValueError((game['path'],t,arm,got,row[arm]))
                    out.append([archive,game['path'],t,arm])
    sr.retain(HERE/'official_checks.json',dict(count=len(out),checks=out))
    print(len(out))
if __name__=='__main__':run()
