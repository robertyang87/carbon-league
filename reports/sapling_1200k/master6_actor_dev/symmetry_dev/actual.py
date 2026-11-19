"""Attribute the executed candidate trajectories using only the repeated public history of each game."""
import gzip
import json
import sys
from time import perf_counter

from mechanism import HERE, FastCarbon, compare, digest, read_tape, unpack


def main():
    comparison = json.loads((HERE/"comparison.json").read_bytes())
    games = []
    for entry in comparison["paired_map_seats"]:
        source = entry["candidate"]
        parent = entry["parent"]
        tape_file = HERE/"candidate"/source["raw_file"]
        tape = read_tape(tape_file)
        original = read_tape(HERE/"parent"/parent["raw_file"])
        s = source["seat"]
        sim = FastCarbon(tape["game_configuration"]).reset(seed=source["seed"])
        package = unpack()
        changes = []
        sapling = []
        first = None
        for t, (a,b) in enumerate(zip(tape["actions"], original["actions"])):
            if a != b:
                first = dict(step=t,candidate_seat_command_changed=a[s]!=b[s],
                             examiner_command_changed=a[1-s]!=b[1-s])
                break
        for t, joint in enumerate(tape["actions"]):
            obs = sim.obs(s)
            candidate = package.agent(obs, sim.config)
            if candidate != joint[s]:
                raise ValueError("recorded order differs on identical public history")
            # Replaying this single state's primitive decoder avoids a second observation/history update.
            # Save the public features in the wrapped original entry so all ablations see that same state.
            x, aux = package._actual_inputs
            baseline = package._FEAT.to_actions(package._ONE_VIEW_PARENT(x,aux))
            without_student = package._FEAT.to_actions(package._ensemble_actions(x,aux,student_weight=0.))
            def changed(other):
                return [[uid, other.get(uid,"NONE"), candidate.get(uid,"NONE")]
                        for uid in sorted(set(candidate)|set(other)) if candidate.get(uid)!=other.get(uid)]
            for arm,other in ((changes,baseline),(sapling,without_student)):
                edits = changed(other)
                if edits:
                    arm.append([t,edits])
            sim.step(*joint)
        if list(sim.cash) != tape["last_post_cash"]:
            raise ValueError("cash differs from official replay")
        games.append(dict(seed=source["seed"],seat=s,public_roots=len(tape["actions"]),first_native_divergence=first,
                          tape_sha256=digest(tape_file),original_pack_changes=changes,sapling_ablation_changes=sapling))
    content = json.dumps({"comparison_sha256":digest(HERE/"comparison.json"),"code_sha256":digest(__file_path__),
                          "games":games},separators=(",",":"),allow_nan=False).encode()
    (HERE/"actual.json.gz").write_bytes(gzip.compress(content,mtime=0))


__file_path__ = HERE / "actual.py"
_original_unpack = unpack


def unpack():
    mod = _original_unpack()
    run = mod._mixed_actions
    def record(x,aux):
        mod._actual_inputs = (x,aux)
        return run(x,aux)
    mod._mixed_actions = record
    return mod


if __name__ == "__main__":
    main()
