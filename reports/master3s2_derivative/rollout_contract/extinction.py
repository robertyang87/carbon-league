"""Build an independent native search repair for official absorbing economic extinction."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'cold_start/candidate.py'
def build():
    data=PARENT.read_bytes()
    assert hashlib.sha256(data).hexdigest()=='7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed'
    src=data.decode().replace('\r\n','\n')
    old="""    def over(b):
        return b[2]['step'] > LAST_TURN

    def result(b):
        c = [float(pl[0]) for pl in b[2]['players']]
        return _terminal(c[me] - c[op])"""
    new="""    recruit_floor = min(configuration.get('recCollectorCost', 30), configuration.get('recPlanterCost', 30))
    episode_steps = configuration.get('episodeSteps', 300)

    def extinct(pl):
        return not pl[2] and not pl[3] and (not pl[1] or pl[0] < recruit_floor)

    def over(b):
        state = b[2]
        teams = state['players']
        return state['step'] >= episode_steps - 1 or extinct(teams[0]) or extinct(teams[1])

    def result(b):
        state = b[2]
        teams = state['players']
        gone = (extinct(teams[0]), extinct(teams[1]))
        if any(gone):
            # The official interpreter first marks the lost player DONE, then the
            # surviving ACTIVE player DONE before the ordinary wallet reward update.
            penalty = state['step'] - episode_steps - 1
            c = [penalty if gone[s] else float(b[3][s][0]) for s in (0, 1)]
        else:
            c = [float(pl[0]) for pl in teams]
        return _terminal(c[me] - c[op])"""
    assert src.count(old)==1
    return src.replace(old,new).replace('\n','\r\n').encode()
if __name__=='__main__':
    import sys
    raw=build();Path(sys.argv[1]).write_bytes(raw);print(hashlib.sha256(raw).hexdigest())
