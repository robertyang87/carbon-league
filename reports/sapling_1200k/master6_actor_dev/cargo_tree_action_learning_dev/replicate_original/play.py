"""Run the unchanged winning policy and the same control on two more isolated maps."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent / "direct_original"))
import match

def main():
    native = match.native
    previous = match.HERE
    ticket = native.read(previous / "binding.json")
    match.HERE = native.HERE = HERE
    native.PLAN = HERE / "protocol.json"
    native.REG = native.sha(native.PLAN)
    native.OUT = HERE / "games"
    match.plan = native.read(native.PLAN)
    assert native.sha(previous / "scores.json") == match.plan["prior_results_sha256"]
    assert all(ticket[key] == match.plan[key] for key in ("candidate","parent","original"))
    ticket.update(protocol=native.REG, replication_wrapper=native.sha(Path(__file__)), original_ticket=native.sha(previous / "binding.json"))
    native.save(HERE / "binding.json",ticket)
    match.main()

if __name__ == "__main__":
    main()
