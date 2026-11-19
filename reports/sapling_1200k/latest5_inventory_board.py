"""Reconstruct every newly verified action history in the repository's original Board."""
import argparse
import json
from pathlib import Path

from ladder_cash_rules import audit
from latest5_inventory_github import CACHE,REPORT,dump,sha


def main(args):
    games=json.loads(args.inventory.read_bytes())["games"]
    checks=[]
    for g in games:
        f=args.cache/g["path"]
        if sha(f.read_bytes())!=g["sha256"]:raise ValueError("replay digest "+g["id"])
        checked=audit(f,True)
        if (checked["game"]!=g["id"] or checked["official_board_transitions_checked"]!=g["transitions"] or
            checked["official_rewards"]!=g["score_by_seat"] or checked["statuses"]!=g["terminal"]):
            raise ValueError("official Board result "+g["id"])
        checks.append(checked)
    dump(args.out,dict(engine="zerosum_env.envs.carbon.helpers.Board",games=checks,
                       total_transitions=sum(g["transitions"] for g in checks)))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inventory",type=Path,default=REPORT/"latest5_inventory.json")
    cli.add_argument("--cache",type=Path,default=CACHE)
    cli.add_argument("--out",type=Path,default=REPORT/"latest5_inventory_board.json")
    main(cli.parse_args())
