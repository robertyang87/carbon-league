"""Confirm the sole native-trained update from the pre-registered opening on both seats."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
FIT = FULL / "direct_native_fit"
sys.path.insert(0, str(FULL / "direct_original"))
import play as fixture

native = fixture.native

def main():
    result = native.read(FIT / "selection.json")
    if not result["replaces_center"]:
        raise SystemExit("Unchanged center; no new-map games registered.")
    assert result["score_sha256"] == native.sha(FIT / "scores.json")
    assert result["protocol"] == native.sha(FIT / "protocol.json")
    assert result["bindings"] == native.sha(FIT / "binding.json")
    assert len(native.read(FIT / "scores.json")) == 8
    candidate = native.ROOT / result["selected_pack_file"]
    center = FULL / "selected.py"
    fixture.HERE = native.HERE = HERE
    native.PLAN = HERE / "protocol.json"
    native.REG = native.sha(native.PLAN)
    native.OUT = HERE / "games"
    fixture.plan = native.read(native.PLAN)
    fixture.plan["candidate"] = result["selected_pack_sha256"]
    assert native.sha(candidate) == fixture.plan["candidate"]
    assert native.sha(center) == fixture.plan["parent"]
    assert native.sha(candidate) != native.sha(center)
    def bound():
        driver = fixture.base_driver()
        driver.CANDIDATE, driver.PARENT = candidate, center
        assert native.sha(driver.arena.prior.SOURCE) == fixture.plan["original"]
        return driver
    fixture.driver = native.driver = bound
    native.save(HERE / "weight_binding.json", dict(wrapper=native.sha(Path(__file__)), fixture=native.sha(Path(fixture.__file__)), selection=native.sha(FIT / "selection.json"), candidate=native.sha(candidate), parent=native.sha(center), protocol=native.REG))
    fixture.main()

if __name__ == "__main__":
    main()
