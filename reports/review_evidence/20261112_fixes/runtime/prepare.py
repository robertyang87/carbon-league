"""Build current-code valid/invalid packages into CARBON_REVIEW_OUT."""
import os
from common import setup, fixture_weights


def main():
    _, out = setup()
    fixture_weights()
    os.environ["CNET2_NO_ROLL"] = "1"
    import cnet2_search_pack
    for name, teacher in (("valid", 1), ("invalid", 999)):
        conf = {"SEARCH": False, "ROLL": "self", "TEACHER": teacher, "GUARD": True}
        cnet2_search_pack.main(str(out / (name + ".py")), repr(conf), str(out / "baseline.npz"))
    (out / "idle.py").write_text("def agent(observation, configuration):\n return {}\n")
    (out / "conditional_error.py").write_text(
        'def agent(observation, configuration):\n'
        ' if observation["player"] == 0: raise RuntimeError("controlled outer error")\n'
        ' return {}\n'
    )


if __name__ == "__main__":
    main()
