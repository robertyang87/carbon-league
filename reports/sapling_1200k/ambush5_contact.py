"""Single-step alternatives after the 18 actually executed approaches."""
import gzip
import json
from pathlib import Path

from ambush5_probe import HERE, DIST, DEST, MOVES, arena
from interception_cash import collision_events
from interception_fit import intent


def evaluate(who,target,before,sim,commands,me):
    sim.set_state(before);sim.step(*commands)
    event=collision_events(before,sim,commands).get(target)
    if not event or event[1]!=me or who not in sim.workers:
        return 0.
    where=event[3];victim=before['workers'][target]
    arrival=DEST[victim['pos']][intent.MOVE_ID[commands[1-me].get(target) or 'NONE']]
    if arrival!=sim.workers[who].pos or not (where==('worker',who) or where==('cash',me)):
        return 0.
    return event[2]


def main():
    live=json.loads((HERE/'ambush5_live.json').read_text())
    coeff=json.loads((HERE/'interception_fit.json').read_text())['weights']
    games=[]
    for game in live['games']:
        side=game['seat'];seed=game['seed']
        raw=json.loads(gzip.decompress((HERE/'ambush5_dev'/f'raw_{seed}_{side}.json.gz').read_bytes()))
        sim=arena.exam.FastCarbon(raw['game_configuration']).reset(seed=seed)
        scratch=arena.exam.FastCarbon(raw['game_configuration'])
        previous=None;rows=[]
        for tick,joint in enumerate(raw['actions']):
            before=sim.get_state();obs=sim.obs(side)
            for start in game['starts']:
                if not start['step']<tick<=start['step']+5:
                    continue
                wid,uid=start['attacker'],start['target'];w=sim.workers.get(wid);f=sim.workers.get(uid)
                if not w or not f or w.carbon>=f.carbon or f.carbon<50 or DIST[w.pos][f.pos]>2:
                    continue
                likelihood=intent.predict(obs,coeff,previous)[uid]
                own=obs['players'][side];foe=obs['players'][1-side]
                claimed={DEST[p][intent.MOVE_ID[joint[side].get(other) or 'NONE']] for other,(p,_,_) in own[2].items() if other!=wid}
                claimed|={p for bid,p in own[1].items() if joint[side].get(bid) in ('RECCOLLECTOR','RECPLANTER')}
                played=joint[side].get(wid) or 'NONE'
                offers=[]
                for i,m in enumerate(MOVES):
                    dest=DEST[w.pos][i]
                    if dest not in DEST[f.pos] or dest in foe[1].values() or dest in claimed:
                        continue
                    q=likelihood[DEST[f.pos].index(dest)]
                    alt=[cmd.copy() for cmd in joint];alt[side][wid]=m if m!='NONE' else None
                    mass=evaluate(wid,uid,before,scratch,alt,side)
                    offers.append({'move':m,'where':dest,'p':round(float(q),6),'captured':round(mass,4)})
                base=evaluate(wid,uid,before,scratch,joint,side)
                top=max(offers,key=lambda o:(o['p'],o['move']==played),default=None)
                forecast=top if top and top['p']>=.25 else None
                gain=(forecast['captured'] if forecast else base)-base
                oracle=max([base]+[o['captured'] for o in offers])-base
                target_landing=DEST[f.pos][intent.MOVE_ID[joint[1-side].get(uid) or 'NONE']]
                if target_landing not in DEST[w.pos]:
                    why='opponent_recorded_destination_out_of_reach'
                elif target_landing in foe[1].values():
                    why='would_credit_opponent_base'
                elif target_landing in claimed:
                    why='allied_worker_or_recruit_already_reserved'
                else:
                    why='reachable'
                rows.append({'target_destination':target_landing,'target_geometry':why,'close_after_start':start['step'],'tick':tick,'attacker':wid,'target':uid,'distance':DIST[w.pos][f.pos],
                             'original':played,'actual_target_command':joint[1-side].get(uid) or 'NONE',
                             'baseline':base,'all_safe_offers':offers,'learned':forecast,'learned_gain':round(gain,4),'hindsight_gain':round(oracle,4)})
            sim.step(*joint);previous=obs
        if arena.prior.hashed(sim.get_state()) != json.loads((HERE/'ambush5_dev/results.json').read_text())['games'][side]['final_state_sha256']:
            raise ValueError('terminal digest')
        games.append({'seed':seed,'seat':side,'records':rows})
    rr=[v for g in games for v in g['records']]
    summary={k:{'roots':sum(r[k]>0 for r in rr),'separate_raids':len({(g['seat'],r['close_after_start']) for g in games for r in g['records'] if r[k]>0}),
                'mass':round(sum(r[k] for r in rr),4)} for k in ('learned_gain','hindsight_gain')}
    summary['roots_tested_within_1_or_2']=len(rr)
    summary['geometry_counts']={k:sum(r['target_geometry']==k for r in rr) for k in sorted({r['target_geometry'] for r in rr})}
    summary['learned_offers']=sum(r['learned'] is not None and r['learned']['move']!=r['original'] for r in rr)
    out={'scope':'Fixed genuine current opponent command at each root. Alternative command changes only the originally assigned pursuer. One-step public move weights fitted previously on separate Top5 training set. Windows and roots can overlap.',
         'games':games,'summary':summary}
    (HERE/'ambush5_contact.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
    print(summary)


if __name__=='__main__':
    main()
