"""Apply bit-equivalent kernel changes, then compare live and matched budgets."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load('balanced_root_probe', FAMILY/'single_worker_priority/probe.py')
previous = load('verified_winograd_changes', FAMILY/'search_efficiency/build.py')
check = baseline.check

# RandomState.choice uses a float64 cumulative sum, normalizes by the
# final entry, draws one double, and uses searchsorted(side="right").
CHOICE = b"""def _search_choice(rng, probabilities):
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative /= cumulative[-1]
    return int(np.searchsorted(cumulative, rng.random_sample(), side='right'))


"""


def build():
    raw = baseline.SOURCE.read_bytes()
    assert check.sha(raw)==baseline.PARENT_SHA
    start = raw.index(b'class _WinoNet:')
    end = raw.index(b'_S_WINO = True',start)
    body = raw[start:end]
    for before, after in previous.REPLACEMENTS:
        assert body.count(before.encode())==1, before
        body = body.replace(before.encode(),after.encode())
    raw=raw[:start]+body+raw[end:]
    start=raw.index(b'def _sample(')
    end=raw.index(b'def _deviations(',start)
    body=raw[start:end]
    for k in (3,5):
        before=f'rng.choice({k}, p=p)'.encode()
        assert body.count(before)==1
        body=body.replace(before,b'_search_choice(rng, p)')
    raw=raw[:start]+CHOICE.replace(b'\n',b'\r\n')+body+raw[end:]
    dst=HERE/'candidate.py'
    if dst.exists():
        assert dst.read_bytes()==raw
    else:
        dst.write_bytes(raw)
    return check.sha(raw)


def registration():
    p=baseline.registration()
    p['schema']='native-search-exact-speed/1'
    p['design']='Unchanged V2 policy and weights. Winograd reuse of buffers plus float64 equivalent inverse-CDF rollout sampling. Same 16 public history roots. First arm natural clocks, second arm same parent work. Under forced work, compare both actions and every recorded raw leaf score.'
    for ticket in p['tickets']:
        ticket['rng']=int(check.sha(f"exact-speed/1/{ticket['seed']}/{ticket['seat']}/{ticket['step']}".encode())[:8],16)
    return p


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    args=cli.parse_args()
    check.HERE,check.PARENT,check.CANDIDATE=HERE,baseline.SOURCE,HERE/'candidate.py'
    check.DIGESTS=(baseline.PARENT_SHA,build())
    check.registration=registration
    if args.register:
        raw=check.encode(registration())
        dst=HERE/'protocol.json'
        assert not dst.exists() or dst.read_bytes()==raw
        dst.write_bytes(raw)
        print(check.DIGESTS[1],check.sha(raw))
    else:
        check.execute(args.cpu)
