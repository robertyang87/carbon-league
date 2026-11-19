"""Run the unchanged scripted match evaluation and audit cash carried out of each seizure."""
import argparse
import hashlib
import inspect
from pathlib import Path
import sys

import tree_practice as test


class ReservationObserver(test.PurchaseObserver):
    latest = None

    def __init__(self):
        super().__init__()
        self.seizures = []
        lines, first = inspect.getsourcelines(test.audit.FastCarbon.step)
        matches = [first+i for i,line in enumerate(lines) if line.strip()=='self.cash[w.player] -= cfg["seizeCost"]']
        if len(matches) != 1:
            raise ValueError("unknown seizure debit")
        self.seize_line, = matches
        ReservationObserver.latest = self

    def trace(self, frame, event, arg):
        if frame.f_code is self.code and event == "line" and frame.f_lineno == self.seize_line:
            f=frame.f_locals
            p=f["w"].player
            fee=float(f["cfg"]["seizeCost"])
            self.seizures.append(dict(step=f["self"].step_count+1, seat=p, fee=fee,
                                      cash_after_debit=float(f["self"].cash[p]-fee)))
        return super().trace(frame,event,arg)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack",type=Path,required=True)
    p.add_argument("--pack-sha256",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seizures",type=Path,required=True)
    args=p.parse_args()
    if digest(args.pack)!=args.pack_sha256 or args.seizures.exists():
        p.error("unexpected identity or existing output")
    play=test.play
    extras, games = [], []
    def audited_game(agent,rival,seed,side):
        game=play(agent,rival,seed,side)
        purchases=[s for s in ReservationObserver.latest.seizures if s["seat"]==side]
        if len(purchases)!=game["seizures_won"]:
            raise ValueError("seizure debit and actual ownership changes disagree")
        extras.append(dict(seed=seed,seat=side,opponent=rival.__name__,seizure_debits=purchases))
        games.append(game)
        if len(games)==8:
            print("v0.8",test.summarize(games),flush=True)
        return game
    test.PurchaseObserver=ReservationObserver
    test.play=audited_game
    sys.argv=["tree_practice.py","--pack",str(args.pack),"--out",str(args.out)]
    test.main()
    test.audit.shared.atomic_json(args.seizures,dict(pack_sha256=digest(args.pack),practice_sha256=digest(args.out),
                                 evaluator_sha256=digest(Path(__file__)),games=extras))


if __name__=="__main__":
    main()
