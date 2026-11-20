"""Build the separate conditional investment experiment without changing the original competition."""
from build import HERE,build,sha
import argparse
def build_investment():
    base=build();old=b"def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):"
    code=(HERE/"investment.py").read_text().replace("\n","\r\n").encode()+b"\r\n"
    assert base.count(old)==1
    # Investment decisions can arise before the opposition has any trees at all.
    guard=b'    if most<3 or not obs["players"][1-seat][3]:return\r\n'
    new=b'    if most<3:return\r\n'
    assert base.count(guard)==1
    ret=base.replace(old,code+old).replace(guard,new)
    assert ret.replace(code,b"").replace(new,guard)==base
    return ret
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out");args=p.parse_args();ret=build_investment();print(len(ret),sha(ret))
    if args.out:
        from pathlib import Path
        Path(args.out).write_bytes(ret)
