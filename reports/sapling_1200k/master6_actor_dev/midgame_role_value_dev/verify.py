"""Replay each original response with both engines and attribute downstream production to the actual hire."""
from data import sr,HERE,base
def check(seat,arm):
    out=HERE/f'seat{seat}';q=sr.load(out/'live_protocol.json');r=q['root'];raw=sr.load(sr.ROOT/q['record']['source'])
    rec=sr.load(out/f'live_{arm}.json');wall=sr.load(out/f'clock_{arm}.json.gz')
    path=out/rec['action_tape'];tail=sr.load(path)
    assert sr.sha(path)==rec['action_sha256']==wall['action_sha256'] and q['pack_sha256']==rec['pack_sha256']
    assert rec['source_sha256']==sr.live.MASTER_SHA and len(wall['calls'])==len(tail)-1
    pair=raw['actions'][r['t']];expect=dict(pair[seat]);expect[r['worker']]=r['old'] if arm=='parent' else r['alternative']
    assert tail[0][seat]==expect and tail[0][1-seat]==pair[1-seat]
    raw['actions']=raw['actions'][:r['t']]+tail
    env=sr.live.begin(raw)
    for cmd in raw['actions']:env.step(cmd)
    assert env.done and [v.reward for v in env.states]==rec['score']
    sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);raw['initial_terrain_u8_sha256']=base.H(bytes(map(int,sim.carbon)))
    raw['candidate_seat']=seat
    audit=base.audit(raw,dict(official_rewards=rec['score']))
    h,=[v for v in audit['base_rows'] if v['base']==r['worker'] and v['t']==r['t']]
    w=audit['workers'][h['uid']]
    assert w['birth_visible_after_step'] and w['price']==r['price'] and w['role']==('P' if arm=='parent' else 'C')
    direct50=sum(v if kind in ('own_delivery','tree_absorption') else -v for at,kind,v in w['events'] if at<=r['t']+50)-w['price']
    return dict(seat=seat,seed=q['seed'],arm=arm,margin=rec['margin'],score=rec['score'],focal=w,direct50=direct50,
                calls=rec['search_calls'],cutoffs=wall['summary']['native_aborted'],summary=audit['summary'])
def main():
    rows=[check(s,a) for s in (0,1) for a in ('parent','approach')];sr.retain(HERE/'ledger.json.gz',rows)
    for x in rows:print(x['seat'],x['arm'],x['margin'],x['direct50'],x['focal']['lifetime_net'],x['cutoffs'])
if __name__=='__main__':main()
