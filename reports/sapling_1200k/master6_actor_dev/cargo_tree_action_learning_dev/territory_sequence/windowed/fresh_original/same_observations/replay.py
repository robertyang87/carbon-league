"""Replay identical public inputs from the four window agents, keeping the original joint-action trajectory."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
sys.path.insert(0, str(LANE))
import matrix

live = matrix.live
LAST_STEP = 160  # Includes one post-window action as an exact control.


def digest(data):
    return hashlib.sha256(data).hexdigest()


def public_inputs(archive, seed, seat):
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        ticket = path / "registration.json"
        ticket.write_bytes(bundle.read("window/registration.json"))
        live.configure(ticket, path / "attempt.json", path / ".lock")
        tape = bundle.read("window/turns.jsonl")
        receipt = json.loads(bundle.read("window/receipt.json"))
        assert digest(tape) == receipt["turns_sha256"]
        assert live.PLAN["seed"] == seed and live.PLAN["candidate_seat"] == seat
        env, config = live.official()
        bank = [5., 5.]
        requests, official, hashes, roles = [], [], [], []
        for row in map(json.loads, tape.splitlines()):
            t = row["step"]
            if t > LAST_STEP:
                break
            obs = live.observe(env, seat, bank[seat])
            expect = row["public_observation_sha256"][seat]
            assert t == len(requests) == obs["step"] and digest(live.encoded(obs)) == expect
            if any(call["status"] != "ok" for call in row["calls"]):
                break
            requests.append(live.encoded({"observation":obs,"configuration":config}))
            actions = [call["response"]["action"] for call in row["calls"]]
            official.append(actions[seat])
            hashes.append(expect)
            player = obs["players"][seat]
            roles.append({**{uid:"base" for uid in player[1]},
                          **{uid:"collector" if rec[2] == "COLLECTOR" else "planter"
                             for uid,rec in player[2].items()}})
            env.step(actions)
            bank = row["overage_remaining_seconds"]
        return requests, official, hashes, roles, receipt["turns_sha256"]


def infer(requests, pack, cpu):
    worker = matrix.play.CARGO / "ladder_timing/worker.py"
    command = [sys.executable, "-u", str(worker), str(live.ROOT / pack["path"]), str(cpu), "512"]
    local_env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(command, input=b"".join(requests), capture_output=True, check=True, env=local_env)
    ready, *rows = map(json.loads, result.stdout.splitlines())
    assert (ready["sha256"], ready["numpy"]) == (pack["sha256"], "1.26.4") and len(rows) == len(requests)
    assert all(row["step"] == t for t,row in enumerate(rows))
    return [row["action"] for row in rows],ready


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cpu", type=int, required=True)
    args = cli.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    plan = matrix.play.load(LANE/"protocol.json")
    verified = matrix.play.load(LANE/"hosted/verification.json")
    assert verified["independent_replay_matches_hosted_verification"]
    grouped, all_turns = [], []
    for item in plan["schedule"]:
        seed, seat = item["seed"], item["seat"]
        archive = LANE/"hosted"/f"{seed}-{seat}.zip"
        identity = next(x["artifact"] for x in verified["jobs_and_archives"] if x["archive"] == archive.name)
        assert "sha256:"+live.sha(archive) == identity["digest"]
        requests, official, hashes, roles, tape_sha = public_inputs(archive, seed, seat)
        reference, runtime = infer(requests,plan["files"]["parent"],args.cpu)
        window, auxiliary = infer(requests,plan["files"]["window"],args.cpu)
        counters = Counter()
        for t, (obs_sha, actual, w, p) in enumerate(zip(hashes, official, window, reference, strict=True)):
            inside = 128 <= t <= 159
            counters["available"] += 1
            counters["inside_opportunities"] += int(inside)
            counters["local_window_vs_hosted_disagreements"] += int(w != actual)
            counters["inside_changed" if inside else "outside_changed"] += int(w != p)
            edits = [{"unit":k,"role":roles[t][k],"parent":p.get(k),"window":w.get(k)}
                     for k in sorted(p.keys()|w.keys()) if p.get(k) != w.get(k)]
            if inside:
                counters.update("commands_"+x["role"] for x in edits)
            all_turns.append({"seed":seed,"seat":seat,"t":t,"observation_sha256":obs_sha,
                              "hosted_window_action_sha256":digest(live.encoded(actual)),
                              "local_window_action_sha256":digest(live.encoded(w)),
                              "parent_action_sha256":digest(live.encoded(p)), "edits":edits})
        grouped.append({"seed":seed,"seat":seat,"archive":identity["digest"],"tape_sha256":tape_sha,
                        "counts":dict(counters), "python":runtime["python"],"numpy":runtime["numpy"]})
        print(seed,seat,dict(counters),flush=True)
    content = b"".join(live.encoded(r) for r in all_turns)
    output = HERE/"turns.jsonl.gz"
    output.write_bytes(gzip.compress(content,mtime=0))
    summary = {"run":verified["run"]["id"],"parent_sha256":plan["files"]["parent"]["sha256"],
               "window_sha256":plan["files"]["window"]["sha256"],"last_requested_step":LAST_STEP,
               "trajectory":"Natural hosted window trajectory; opponent commands fixed. Counterfactual commands never executed.",
               "local_window_agrees_all":all(r["counts"]["local_window_vs_hosted_disagreements"] == 0 for r in grouped),
               "window_switch_isolated":all(r["counts"]["outside_changed"] == 0 for r in grouped),
               "turns_file_sha256":live.sha(output),"groups":grouped}
    live.save(HERE/"summary.json",summary)


if __name__=="__main__":
    main()
