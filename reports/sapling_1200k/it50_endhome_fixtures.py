"""Exercise the built pack on both seats, including precise final deposits."""
import argparse
from pathlib import Path

from cash_practice import audit, np
from build_it50_endhome import OUTPUT, PARENT, PARENT_SHA


class FixtureNet:
    """No-op or chosen canonical worker move; log the real autoregressive inputs."""
    def __init__(self, move=0, recruit=False):
        self.move, self.recruit, self.inputs = move, recruit, []
        self.w = {"aemb.weight": np.arange(9, dtype=np.float32).reshape(9, 1)}

    def encode(self, x, style):
        return np.zeros((24, 1), np.float32), np.zeros(1, np.float32)

    def lin(self, x, key):
        if key == "whead":
            logits = np.zeros(5, np.float32)
            logits[self.move] = 10.
            return logits
        if key == "bhead":
            return np.array([0,10,0] if self.recruit else [10,0,0], np.float32)
        return np.zeros((5,1) if key == "V" else 1, np.float32)

    def gru(self, x, hidden):
        self.inputs.append(int(x[0]))
        return np.zeros(1, np.float32)


def main(out):
    if out.exists() or audit.digest(PARENT) != PARENT_SHA:
        raise ValueError("fresh output and frozen parent required")
    package = audit.shared.module(OUTPUT, "endhome_fixture")
    parent = audit.shared.module(PARENT, "frozen_fixture")
    F = package.sys.modules["cnet2_feat"]
    rows = []
    for seat in (0, 1):
        for name in ("cash_at_deadline", "early", "already_reachable", "too_far", "low_cargo",
                     "planter", "lethal_at_home", "unsafe_future_path", "same_turn_recruit", "expired"):
            sim = audit.FastCarbon(audit.CFG).reset(seed=620188308)
            state = sim.get_state()
            base = state["bases"][seat][0][1]
            # Physical cell immediately right of a base, independently of canonical seat.
            dist = 3 if name == "unsafe_future_path" else (2 if name=="too_far" else 1)
            pos = base
            for _ in range(dist):
                pos = int(F.DEST[pos,2])
            state["step"] = {"early": 240, "already_reachable":297, "unsafe_future_path":296,
                             "expired":299}.get(name, 298)
            state["cash"] = [80,80]
            state["workers"] = {f"player-{seat}-worker-0": dict(player=seat,pos=pos,
                    carbon=0 if name=="planter" else (5 if name=="low_cargo" else 100),
                    type="PLANTER" if name=="planter" else "COLLECTOR")}
            if name in ("lethal_at_home", "unsafe_future_path"):
                state["workers"][f"player-{1-seat}-worker-0"] = dict(player=1-seat, pos=base, carbon=0, type="COLLECTOR")
            state["wid_counter"] = [2,2]
            sim.set_state(state)
            obs = sim.obs(seat)
            netpair = [FixtureNet(recruit=name=="same_turn_recruit") for _ in range(2)]
            package._NETS = netpair
            package._FEAT = None
            package._RETURN_COUNTS.clear()
            package._RETURN_ENABLED = True
            action = package.agent(obs, sim.config)
            assert netpair[0].inputs == netpair[1].inputs
            actual = action.get(f"player-{seat}-worker-0", "NONE")
            if name == "cash_at_deadline":
                assert actual == "LEFT" and len(package._RETURN_LAST)==1
                # In the proposed decode the base sees the chosen worker move.
                canonical = 2 if seat else 4
                assert netpair[0].inputs[1] == canonical
                joint = [{},{}]
                joint[seat] = action
                log = audit.Observer()
                log.step(sim, joint)
                deposited = log.d[seat]["own_positive_delivery_carbon"]
                assert deposited > 0 and sim.step_count == 299
            else:
                assert actual == "NONE" and not package._RETURN_LAST, (seat,name,action)
                deposited = None
            if name == "same_turn_recruit":
                assert package._RETURN_COUNTS["rejected_joint_roots"] == 1
            package._NETS = None
            package._FEAT = None
            package._RETURN_ENABLED = False
            parent._FEAT = None
            off = package.agent(obs, sim.config)
            original = parent.agent(obs, sim.config)
            assert off == original and package._COUNTS == parent._COUNTS
            rows.append(dict(seat=seat, case=name, fixed_network_worker_action=actual,
                             observed_own_delivery=deposited, real_weights_disabled_parity=True))
    audit.shared.atomic_json(out, dict(parent_sha256=PARENT_SHA, candidate_sha256=audit.digest(OUTPUT), cases=rows))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    main(p.parse_args().out)
