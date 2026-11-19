"""Reconstruct a live prediction and verify observation boundaries on a training trajectory."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from data import HERE,read_game,splits,view
from value import CashMargin30,History,DEST,SIZE,HORIZON


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old',type=Path,required=True);a=p.parse_args()
    item=splits()['train'][0];game=read_game(a.old,item)
    predictor=CashMargin30()
    n=0;max_hidden=max_swap=0.
    for s in (0,1):
        ref,masked,relabeled=History(),History(),History()
        for t,frame in enumerate(game['steps'][:90]):
            v=view(frame,s)
            polluted=deepcopy(v)
            visible=set()
            for pos,_,kind in v['players'][s][2].values():
                visible.update((pos,) if kind=='COLLECTOR' else DEST[pos][1:])
            polluted['carbon']=[x if pos in visible else 1e8+pos+t for pos,x in enumerate(polluted['carbon'])]
            polluted.update(full_carbon=[-1e16]*225,map_seed=-1,opponent_name='ignored')
            swapped=dict(v,players=list(reversed(v['players'])),player=1-s)
            for hist,obs in ((ref,v),(masked,polluted),(relabeled,swapped)):hist.update(obs)
            expected=predictor.predict(ref,v,game['configuration'])
            max_hidden=max(max_hidden,abs(expected-predictor.predict(masked,polluted,game['configuration'])))
            max_swap=max(max_swap,abs(expected-predictor.predict(relabeled,swapped,game['configuration'])))
            n+=1
    if max_hidden>1e-9 or max_swap>1e-9:raise ValueError('privacy or role convention changed')
    result={'training_game':item['id'],'evaluated_states':n,'largest_invisible_carbon_or_extra_keys_difference':max_hidden,
            'largest_players_and_seat_id_joint_permutation_difference':max_swap,'window':HORIZON}
    (HERE/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__': main()
