"""Compare actual published commands under native and replayed budgets."""
import argparse
import importlib.util
import json

from build import FAMILY, HERE, PARENT, PARENT_SHA, build

path = FAMILY / 'native_critic_update/public_search/check.py'
spec = importlib.util.spec_from_file_location('readout_public_command_check', path)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def registration():
    source = json.loads((FAMILY / 'modal_root_opponent/protocol.json').read_bytes())
    return dict(schema='unused-rollout-head-work/1', packages=list(shared.DIGESTS),
                design='Keep all weights and omit only unconsumed readouts at native H8 and in double-seat batches.',
                tickets=source['tickets'])


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--register', action='store_true')
    cli.add_argument('--cpu', type=int)
    args = cli.parse_args()
    shared.HERE, shared.PARENT, shared.CANDIDATE = HERE, PARENT, HERE / 'candidate.py'
    shared.DIGESTS = (PARENT_SHA, build())
    shared.registration = registration
    if args.register:
        raw = shared.encode(registration())
        frozen = HERE / 'protocol.json'
        assert not frozen.exists() or frozen.read_bytes() == raw
        frozen.write_bytes(raw)
        print(shared.sha(raw))
    else:
        shared.execute(args.cpu)
