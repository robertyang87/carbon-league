"""At the largest observed swing record previous 10/5/3 actions and balance sheets for both seats."""
import gzip,json
from intake import HERE,ladder
def main():
    result=[]
    for game in json.loads((HERE/'master5s_delta.json').read_bytes())['games']:
        path=ladder.fetch.CACHE/'replays'/(game['id']+'.json.gz')
        env,_,_=ladder.fetch.extract(path.read_bytes())
        assert ladder.fetch.sha(ladder.fetch.canonical(env))==game['canonical_sha256']
        end=game['end']; trail=[]
        for t in range(end-10,end):
            frame=env['steps'][t];nxt=env['steps'][t+1]
            trail.append(dict(t=t,players=frame[0]['observation']['players'],
                commands=[f.get('action') or {} for f in nxt],cash_after=[p[0] for p in nxt[0]['observation']['players']]))
        start=lambda n:trail[-n]['players']
        endcash=trail[-1]['cash_after']
        swing={str(n):round(endcash[0]-endcash[1]-start(n)[0][0]+start(n)[1][0],2) for n in (3,5,10)}
        assert abs(swing['10']-game['maximum_ten_turn_swing_seat0'])<0.01
        result.append(dict(id=game['id'],end=end,seat0_cash_margin_change=swing,trail=trail))
    (HERE/'ladder_windows.json.gz').write_bytes(gzip.compress(ladder.fetch.canonical(result),mtime=0))
if __name__=='__main__':main()
