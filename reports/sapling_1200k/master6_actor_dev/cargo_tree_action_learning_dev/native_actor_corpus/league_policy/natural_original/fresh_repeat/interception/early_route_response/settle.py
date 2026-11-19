"""Independently replay both natural suffix tapes with the official physical and scoring rules."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run
sys.path.insert(0,str(HERE.parents[2]))
import economy as e

def data(path):
    return json.loads(gzip.decompress(path.read_bytes()))

def main():
    folder=HERE/'observed'
    plan=json.loads((HERE/'protocol.json').read_bytes())
    common=data(folder/'shared.json.gz')
    shared_sha=run.sha((folder/'shared.json.gz').read_bytes())
    with zipfile.ZipFile(run.ROOT/plan['source_archive']['path']) as z:
        archive=[json.loads(line) for line in z.read('sampled_child/turns.jsonl').splitlines()]
    result={}
    for arm,order in plan['arm_actions'].items():
        doc=data(folder/(arm+'.json.gz'))
        e.require(doc['shared_sha']==shared_sha and doc['registered_commit']==run.RELEASE and
                  doc['source_original_sig']==common['original_sig'] and doc['source_child_sha']==common['child_state_sha'],
                  'different reconstructed root')
        cfg=e.config()
        e.np.random.seed(plan['seed'])
        env=e.make('carbon',configuration={**cfg,'randomSeed':plan['seed']})
        for row in archive[:plan['t']]:
            env.step([c['response']['action'] for c in row['calls']])
        root=env.states[0].observation
        e.require(run.sha(json.dumps(dict(root),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())==common['physical'],'incorrect public root')
        target=plan['target'];worker=plan['defender']
        deliveries=[]; positions=[]; first_death=None; defender_death=None
        for row in doc['turns']:
            o=env.states[0].observation;t=row['t']
            e.require(o.step==t and not env.done,'invalid tape')
            if t==plan['t']:
                joint=[dict(c) for c in plan['initial_joint_commands']];joint[0][worker]=order
                e.require(row['joint']==joint,'unexpected root change')
            env.step(row['joint']);post=env.states[0].observation
            e.require(row['post_cash']==[x[0] for x in post.players] and row['target_after']==post.players[1][2].get(target)
                      and row['defender_after']==post.players[0][2].get(worker),'physical replay mismatch')
            if first_death is None and target not in post.players[1][2]:first_death=t
            if defender_death is None and worker not in post.players[0][2]:defender_death=t
            if t<=216:positions.append(dict(step=t,target_after=row['target_after'],defender_after=row['defender_after'],
                root_enemy_order=row['joint'][1].get(target,'STAY'),defender_order=row['joint'][0].get(worker,'STAY')))
            deliveries.extend([dict(t=t,owner=x[1],amount=x[2],cell=x[4]) for x in row['cash_events']
                               if x[0] in ('own_delivery','opponent_worker_delivery_to_this_base') and x[3]==target])
        e.require(env.done and [x.reward for x in env.states]==doc['official_scores'],'different terminal')
        clock=doc['clocks']['calls']
        e.require([x['step'] for x in clock]==list(range(plan['t']+1,env.states[0].observation.step)),'native search receipt incomplete')
        m=doc['official_scores'][0]-doc['official_scores'][1]
        result[arm]=dict(arm=arm,root_order=order,transitions=len(doc['turns']),scores=doc['official_scores'],
            margin=round(m,2),winner='learner' if m>0 else 'examiner' if m<0 else 'draw',
            target_deliveries=deliveries,target_delivered_through_216=sum(x['amount'] for x in deliveries if x['t']<=216),
            target_first_death=first_death,defender_first_death=defender_death,tracked_steps=positions,
            suffix_original_calls=len(clock),clock=doc['clocks']['summary'],local_wall=doc['wall'],
            max_public_original=max(x['public_wall_seconds'] for x in clock),
            max_public_child=max(x['public_wall'][0] for x in doc['turns'] if x['public_wall']),
            cpu=doc['host']['affinity'],torch_threads=doc['host']['torch_threads'],
            long_public_calls=[sum(x['public_wall'][s]>1 for x in doc['turns'] if x['public_wall']) for s in (0,1)],
            tape_sha=run.sha((folder/(arm+'.json.gz')).read_bytes()))
    delta=round(result['intervention']['margin']-result['control']['margin'],2)
    document=dict(protocol_sha=run.PROTOCOL,registered_commit=run.RELEASE,runner_sha=common['runner_sha'],
        shared_sha=shared_sha,prefix_original_calls=len(common['clock']['calls']),
        prefix_original_disagreements=common['original_prefix_disagreements'],prefix_child_disagreements=common['child_prefix_disagreements'],
        replayed_historical_inputs=common['historical_public_checked'],common_reconstructed_original_state=common['original_sig'],
        common_reconstructed_child_state=common['child_state_sha'],original_signed_root=plan['initial_joint_commands'],
        official_margin_delta_up_minus_down=delta,arms=result)
    e.write(HERE/'summary.json',document)
    print([(v['arm'],v['scores'],v['margin'],v['target_delivered_through_216']) for v in result.values()],delta)

if __name__=='__main__':main()
