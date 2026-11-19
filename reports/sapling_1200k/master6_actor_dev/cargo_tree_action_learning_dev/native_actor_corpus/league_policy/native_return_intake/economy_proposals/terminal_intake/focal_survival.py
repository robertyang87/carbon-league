"""Replay a sealed root to distinguish paid economics, survival and contact from a claimed intent."""
import argparse
import gzip
import json
from pathlib import Path
from select_root import play, runner

HERE=Path(__file__).resolve().parent
VECTORS={"UP":(-1,0),"DOWN":(1,0),"LEFT":(0,-1),"RIGHT":(0,1),"NONE":(0,0)}


def destination(cell,direction,width):
    y,x=divmod(cell,width)
    dy,dx=VECTORS[direction]
    return (y+dy)%width*width+(x+dx)%width


def board_details(spec,folder,arm):
    prefix=play.read(folder/'prefix.json')
    menu=prefix['root_menu']
    uid,seat=menu['focal_uid'],spec['seat']
    env,cfg,live=play.official(spec['seed'],spec['board_sha256'],seat)
    bank=[5.,5.]
    with gzip.open(folder/'prefix.jsonl.gz') as stream:
        for line in stream:
            row=json.loads(line)
            env.step(row['orders'])
            bank=row['bank_after']
    with gzip.open(folder/(arm+'.jsonl.gz')) as stream:
        row=json.loads(stream.readline())
    assert prefix['root_board_sha256']==play.board_hash(env,spec['board_sha256'])
    before=[live.observe(env,s,bank[s]) for s in (0,1)]
    assert [play.sha(live.encoded(obs)) for obs in before]==row['public_before_by_seat']
    players=before[seat]['players']
    width=int(cfg['size'])
    pos=players[seat][2][uid][0]
    affected=destination(pos,row['orders'][seat].get(uid,'NONE'),width)
    entries=[]
    for s,player in enumerate(players):
        for who,(cell,carried,role) in player[2].items():
            could_reach=any(destination(cell,a,width)==affected for a in VECTORS)
            if not could_reach:
                continue
            action=row['orders'][s].get(who,'NONE')
            target=destination(cell,action,width)
            entries.append(dict(seat=s,uid=who,public_start=cell,role=role,
                                visible_start_cargo=carried,actual_action=action,
                                actual_destination=target,enters_focal_cell=target==affected))
    env.step(row['orders'])
    assert row['board_after_sha256']==play.board_hash(env,spec['board_sha256'])
    after=live.observe(env,seat,row['bank_after'][seat])['players']
    for worker in entries:
        worker['worker_after_step']=after[worker['seat']][2].get(worker['uid'])
    return dict(uid=uid,position=pos,focal_cell=affected,focal_action=row['orders'][seat].get(uid,'NONE'),
                official_focal_after=after[seat][2].get(uid),nearby_public_workers_and_realized_response=entries,
                trees_on_cell_after_by_seat=[[k for k,v in s[3].items() if v[0]==affected] for s in after],
                official_balance_after_by_seat=[s[0] for s in after],official_board_after_sha256=row['board_after_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args()
    [case]=[v for v in play.read(HERE/'protocol.json')['fixtures'] if v['seed']==args.seed]
    folder=HERE/'outputs'/f"{case['seed']}_{case['seat']}"
    pre=play.read(folder/'prefix.json')
    if pre['root_step'] is None:
        return
    files=['prefix.json','prefix.jsonl.gz','shared.json','A.jsonl.gz','B.jsonl.gz','verified.json']
    data=dict(seed=case['seed'],seat=case['seat'],step=pre['root_step'],candidate=pre['root_menu']['category'],
              program_sha256=runner.fixture.hfile(Path(__file__)),source_sha256={x:runner.fixture.hfile(folder/x) for x in files},
              scope='Observed adjacent geometry precedes action; enemy commands, worker survival and cash follow the chosen move.',
              arms={arm:board_details(case,folder,arm) for arm in ('A','B')})
    play.save(folder/'focal_survival.json',data)


if __name__=='__main__':
    main()
