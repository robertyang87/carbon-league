"""Count original global-head disagreements across previously studied maps without rerunning either policy."""
from collections import Counter
from preregister import HERE, PREV, read, save, sha


def audit():
    protocol=read(HERE/'protocol.json')
    prior=read(PREV/'development/comparison.json')
    assert sha(PREV/'development/comparison.json') == protocol['references']['historical_comparison']
    rows=[]
    for pair in prior['pairs']:
        seed,seat=pair['seed'],pair['seat']
        for arm in ('parent','candidate'):
            archive=PREV/'development'/f'{arm}_{seed}_{seat}.json.gz'
            assert sha(archive)==pair['arms'][arm]['tape_sha256']
            game=read(archive)
            times=[]
            for step,obs in enumerate(game['shadow']):
                assert obs['step']==step and game['actions'][step][seat]==obs[arm]
                if obs['candidate']!=obs['parent']:
                    times.append(step)
            rows.append(dict(seed=seed,seat=seat,trajectory=arm,source_sha256=sha(archive),
                         all_joint_differences=times,
                         before_training_span=sum(t<205 for t in times),
                         within_training_span=sum(205<=t<=271 for t in times),
                         after_training_span=sum(t>271 for t in times),
                         near_5_inclusive=sum(min(abs(t-205),abs(t-271))<=5 for t in times),
                         exact_anchors=sum(t in (205,271) for t in times)))
    save(HERE/'previous_timing.json',dict(protocol_sha256=sha(HERE/'protocol.json'),rows=rows,
        scope='Descriptive timing of previously observed global-head command differences on parent/global tracks, computed after registration. No localized candidate was measured by this script.'))


if __name__=='__main__':
    audit()
