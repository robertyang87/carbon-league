"""Run the one frozen neural search pack against the unmodified public examiner."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

HERE = Path(__file__).resolve().parent
FIT = HERE.parent
ACTOR = FIT.parents[1]
FULL = ACTOR / "deployed_mix_policy_gradient_dev/full_episode"
sys.path.insert(0, str(FULL / "direct_original"))
import play as fixture

native = fixture.native

def main():
    fixture.HERE = native.HERE = HERE
    native.PLAN = HERE / "protocol.json"
    native.REG = native.sha(native.PLAN)
    native.OUT = HERE / "games"
    fixture.plan = plan = native.read(native.PLAN)
    assert plan["candidate"] == native.read(FIT / "binding.json")["bundle"]
    assert plan["source_contract"] == native.sha(FIT / "protocol.json")
    witness = native.read(FIT / "summary.json")
    assert witness["multi_changed"] > 0 and witness["timing"]["over_one"] == 0
    def bound():
        d = fixture.base_driver()
        d.CANDIDATE = FIT / "candidate.py"
        d.PARENT = FULL / "selected.py"
        assert native.sha(d.CANDIDATE) == plan["candidate"]
        assert native.sha(d.PARENT) == plan["parent"]
        assert native.sha(d.arena.prior.SOURCE) == plan["original"]
        return d
    fixture.driver = native.driver = bound
    with TemporaryDirectory() as scratch:
        native.HERE = Path(scratch)
        try:
            live = native.clearance()
        finally:
            native.HERE = HERE
    assert live["physical_boards"] == native.read(HERE / "clearance.json")["physical_boards"]
    assert all(str(g['seed']) not in live["physical_boards"] and g['terrain'] not in live["physical_boards"].values() for g in native.read(FIT/'protocol.json')['sources'])
    native.save(HERE / "runtime_clearance.json", live)
    native.clearance = lambda: live
    native.save(HERE / "evidence.json",dict(runner=native.sha(Path(__file__)), core=native.sha(Path(fixture.__file__)), evaluator=native.sha(FIT/"shared.npz"), witness=native.sha(FIT/"summary.json"), protocol=native.REG))
    fixture.main()

if __name__ == "__main__":
    main()
