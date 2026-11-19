"""Rebuild both arms of the illustrative roots in Board from step zero, including generated identities."""
from pathlib import Path
import gzip,json,sys

from scan import HERE,ROOT,FastCarbon,sha,sources,read
sys.path.insert(0,str(ROOT/'code/env_src/carbon_challenge-master'))
from zerosum_env.envs.carbon.helpers import Board
from zerosum_env.envs.carbon import idgen


def compare(tape,row,branch):
    game=FastCarbon(tape['game_configuration']).reset(seed=tape['seed'])
    state=game.full_observation()
    idgen.reset()
    plan={t:(old,new) for t,old,new,_ in row[branch]['moves']}
    s=row['seat'];uid=row['defender'][2]
    for t,original in enumerate(tape['actions'][:row[branch]['post']]):
        joint=[dict(v) for v in original]
        if t in plan:
            old,new=plan[t]
            if joint[s].get(uid,'NONE')!=old:raise ValueError('original focal move mismatch')
            joint[s].pop(uid,None)
            if new!='NONE':joint[s][uid]=new
        state=Board(state,tape['game_configuration'],joint).next().observation
        game.step(*joint)
        if state['players']!=game._players_obs() or state['carbon']!=game.carbon:
            raise ValueError(f"independent simulators differ at {t}")
    if list(game.cash)!=row[branch]['cash']:raise ValueError('published cash mismatch')
    return {'verified_transitions':len(tape['actions'][:row[branch]['post']]),'physical_cash':list(game.cash)}


def main():
    data=read(HERE/'responses.json.gz')
    all_rows=[r for g in data['games'] for r in g['roots']
              if r['horizon']==5 and 'defense' in r and r['defense']['at_original_seizure']['held']]
    resolver={'best_short_cash':max(all_rows,key=lambda r:r['margin_delta']),
              'worst_short_cash':min(all_rows,key=lambda r:r['margin_delta']),
              'least_profitable_enemy_takeover':min(all_rows,key=lambda r:r['actual_next_enemy_tenure_net'])}
    manifest={(s['seed'],s['seat']):s for tag,s in sources() if tag=='training_source'}
    result={}
    for tag,r in resolver.items():
        src=manifest[r['seed'],r['seat']];tape=read(ROOT/src['raw_path'])
        result[tag]=dict(seed=r['seed'],seat=r['seat'],t=r['root'],tree=r['tid'],responses={
            b:compare(tape,r,b) for b in ('parent','defense')})
    out={'route_data_sha256':sha(HERE/'responses.json.gz'),'script_sha256':sha(Path(__file__)),
         'engine_sha256':sha(ROOT/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/helpers.py'),'checks':result}
    (HERE/'official_examples.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
