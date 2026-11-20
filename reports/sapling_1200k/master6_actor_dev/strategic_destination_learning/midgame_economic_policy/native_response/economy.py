"""Apply the independent official/cash ledger to both complete registered suffixes."""
import argparse
import importlib.util
from pathlib import Path
import select_roots

base=select_roots.common
src=select_roots.STUDY/"policy_window_returns/economy/reconcile.py"
spec=importlib.util.spec_from_file_location("window_cash_reconstruction",src)
ledger=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipts",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    a=parser.parse_args()
    verified=ledger.load(a.out/"verified.json")
    summary=ledger.load(a.out/"summary.json")
    seat,root,seed=(verified[k] for k in ("seat","root","seed"))
    signed=a.receipts/"fit"/f"{seed}_{seat}"/"source.json.gz"
    base.need(base.sha(signed.read_bytes())==summary["source_sha256"] and
              base.sha((a.out/"summary.json").read_bytes())==verified["summary_sha256"],"Unsigned input")
    header=ledger.load(signed)
    games={}
    for arm,meta in summary["arms"].items():
        path=a.out/(arm+".json.gz")
        base.need(base.sha(path.read_bytes())==meta["sha256"],"Changed branch")
        response=ledger.load(path)
        games[arm]=ledger.replay(header,header["rows"][:root]+response["rows"],response["terminal"],arm)
    diff=ledger.ledger.comparison(games["weighted"],games["uniform"])
    ledger.ledger.check_equal(diff["official_margin_delta"],verified["relative_official_margin"],"Official delta")
    docs={arm:ledger.shorten(game,root) for arm,game in games.items()}
    for arm,g in games.items():
        docs[arm]["cash_per_step"]=[{k:r[k] for k in ("t","post_cash","cash_events","signed_cash_by_seat",
            "tree_changes","collisions","hostile_transfer")} for r in g["transitions"]]
    for v in diff["relative_cash_windows"].values():
        w=next(iter(v["main"]["window_by_arm"].values()))
        for i in range(max(0,w["first_turn"]-10),w["after_last_turn_state"]):
            for name in games:
                docs[name]["relevant_steps"][i]=games[name]["transitions"][i]
    out={"seed":seed,"seat":seat,"root":root,"signed_source":summary["source_sha256"],
         "summary_sha256":verified["summary_sha256"],"ledger_sha256":base.sha(src.read_bytes()),
         "comparison":diff,"arms":docs}
    base.save(a.out/"economy.json.gz",out)
    print(diff["official_margin_delta"],diff["margin_components"])

if __name__=="__main__":
    main()
