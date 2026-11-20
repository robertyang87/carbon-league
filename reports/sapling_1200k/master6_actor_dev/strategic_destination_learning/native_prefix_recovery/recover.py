"""Restore decision state on an already observed training prefix of the signed Master3s2."""
import hashlib
import math
from pathlib import Path

SOURCE = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
PUBLIC = {"step", "player", "players", "trees", "carbon", "remainingOverageTime"}


class NativePrefix:
    """An in-memory continuation, initialized after a genuine public step zero.

    Each advance consumes an authenticated historic public view, its actually executed
    command, and the recorded native cost vector after that step. The caller verifies
    archive identity. Actual future requests still call the original public agent.
    """
    def __init__(self, native, seat, next_step=1):
        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != SOURCE:
            raise ValueError("Unsupported examiner bytes")
        if not (native._SEARCH_OK and native._RX and native._TEMP == 1
                and native._SEED is None and native._S_ABORT == 0.58):
            raise ValueError("Expected the original sampler and search")
        if (next_step < 1 or seat not in (0, 1) or set(native._ST) != {"tr", "prev", "rng"}
                or native._ST["tr"].me != seat or native._ST["prev"] is None):
            raise ValueError("Require the initialized public examiner for this seat")
        self.native = native
        self.seat = seat
        self.next_step = next_step

    def advance(self, observation, issued, cost):
        if (set(observation) != PUBLIC or observation["step"] != self.next_step
                or observation["player"] != self.seat):
            raise ValueError("Wrong seat, step, or public observation fields")
        players = observation["players"]
        _, bases, workers, _ = players[self.seat]
        if len(cost) != 4 or not all(math.isfinite(x) and x >= 0 for x in cost):
            raise ValueError("Missing or invalid recorded four-component search cost")
        base_order = {"NONE", *self.native.BACT[1:]}
        worker_order = {"NONE", *self.native.WACT[1:]}
        for uid, action in issued.items():
            allowed = base_order if uid in bases else worker_order if uid in workers else ()
            if action not in allowed:
                raise ValueError("Unexpected archived physical order")

        state = self.native._ST
        tracker = state["tr"]
        tracker.step(players, self.next_step, observation["carbon"])
        # The signed _act uses RandomState.choice(..., p=...) exactly once per
        # own base and once per own worker, before go_home, funds or occupancy
        # resolve the final command. The weighted choice consumes one uniform.
        draws = len(bases) + len(workers)
        state["rng"].random_sample(draws)
        tracker.set_actions(players, issued)
        state["prev"] = players
        self.native._S_COST[:] = cost
        self.next_step += 1
        return draws
