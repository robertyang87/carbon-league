"""Replicate the accepted root-action comparison with a second fixed random stream."""
import argparse

from respond import HERE, plan, runner, sha


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    a=cli.parse_args()
    first=plan()
    raw=runner.check.encode(first)
    assert (HERE/'response/protocol.json').read_bytes()==raw
    p=first | dict(
        schema='joint-policy-beam-second-reply/1',
        initial_protocol_sha256=sha(raw),
        tickets=[t | dict(rival_rng=t['rival_rng'] ^ 0x8E23866B,
                          branch_order=list(reversed(t['branch_order']))) for t in first['tickets']],
    )
    runner.HERE=HERE/'response_repeat'
    runner.HERE.mkdir(exist_ok=True)
    filename=runner.HERE/'protocol.json'
    data=runner.check.encode(p)
    if a.register:
        assert not filename.exists() or filename.read_bytes()==data
        filename.write_bytes(data)
        print(sha(data),len(p['tickets']))
    else:
        assert filename.read_bytes()==data and p['tickets']
        runner.execute(p,a.cpu)


if __name__=='__main__':
    main()
