"""Read seven published development tapes. This module runs neither an agent nor the game engine."""

from pathlib import Path
import gzip
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "reports/master3s2_derivative/dev1"
SETTLEMENT = "0b08537eaa97b0fb46a09003c260b04b72bc7d875576ee1338db258d5d2fcf78"
PLAN = "4468d809a4c308f5ccaf056963f6ad7f240a323c8cc70e312ff1cd43eedc1dde"
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
CANDIDATE = "245b5814d1855c5449435ee6bd4a53560017a8cc7a46d08729fa9dfdc13e9654"
ARM = "opponent_policy/"
PUBLIC_KEYS = {"step", "player", "carbon", "players", "trees", "remainingOverageTime"}


def check(value, why):
    if not value:
        raise ValueError(why)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stream_digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def obs_digest(obs):
    return digest(
        (json.dumps(obs, separators=(",", ":"), allow_nan=False) + "\n").encode()
    )


def inventory():
    raw = (DEV / "hosted/settlement.json").read_bytes()
    plan = (DEV / "protocol.json").read_bytes()
    check(
        digest(raw) == SETTLEMENT and digest(plan) == PLAN,
        "published protocol or settlement changed",
    )
    book = json.loads(raw)
    roster = json.loads(plan)
    check(
        book["records_reconciled"]
        and not book["verification_errors"]
        and book["protocol_sha256"] == PLAN,
        "unsettled",
    )
    a = {(g["seed"], g["seat"]) for g in roster["schedule"]}
    check(
        a == {(s, i) for s in range(621758580, 621758584) for i in (0, 1)}, "schedule"
    )
    check(len(book["archives"]) == len(book["games"]) == len(a) == 8, "count")
    check(
        {(g["seed"], g["seat"]) for g in book["games"]} == a,
        "missing or duplicate game",
    )
    records = {r["id"]: r for r in book["archives"]}
    for g in book["games"]:
        r = records[g["archive_id"]]
        seed, seat = g["seed"], g["seat"]
        path = DEV / "hosted" / g["archive_path"]
        check(
            (seed, seat) == (r["seed"], r["seat"])
            and path.name == str(r["id"]) + ".zip",
            "archive identity",
        )
        with path.open("rb") as f:
            check(
                path.stat().st_size == r["bytes"] and stream_digest(f) == r["sha256"],
                path.name,
            )
        with zipfile.ZipFile(path) as z:
            raw_r = z.read(ARM + "receipt.json")
            check(digest(raw_r) == g["receipt_sha256"], "worker receipt digest")
            receipt = json.loads(raw_r)
            binding = json.loads(z.read(ARM + "registration.json"))
            check(
                binding["files"]["opponent"]["sha256"] == ORIGINAL
                and binding["files"]["candidate"]["sha256"] == CANDIDATE,
                "strategy identity",
            )
            check(
                binding["arm"] == g["arm"] == "opponent_policy"
                and binding["matrix_sha256"] == PLAN
                and (
                    binding["seed"],
                    binding["candidate_seat"],
                    binding["opponent_seat"],
                )
                == (seed, seat, 1 - seat),
                "seat identity",
            )
            check(
                g["terrain"]
                == binding["physical_map_sha256"]
                == roster["physical_boards"][str(seed)],
                "terrain",
            )
            check(
                (
                    receipt["seed"],
                    receipt["candidate_seat"],
                    receipt["status"],
                    receipt.get("official_scores_by_seat"),
                )
                == (seed, seat, g["status"], g["official_scores_by_seat"]),
                "official score mismatch",
            )
            with z.open(ARM + "observations.jsonl.gz") as f:
                check(
                    stream_digest(f) == g["public_observations_sha256"],
                    "public transcript digest",
                )
            if g["status"] == "complete":
                check(
                    receipt["official_transitions"] == g["official_transitions"] == 299
                    and g["replayed_public_observations"] == 598
                    and receipt["action_calls"] == [299, 299],
                    "incomplete game",
                )
                with z.open(ARM + "turns.jsonl") as f:
                    check(
                        stream_digest(f) == receipt["turns_sha256"],
                        "order transcript digest",
                    )
            else:
                check(
                    (
                        seed,
                        seat,
                        g["status"],
                        g["official_transitions"],
                        g["replayed_public_observations"],
                    )
                    == (621758582, 1, "bilateral_failure", 0, 0),
                    "unrecognized failure",
                )
                check(
                    [x["status"] for x in receipt["initialization"]]
                    == ["timeout", "timeout"]
                    and receipt["action_calls"] == [0, 0],
                    "not an empty bilateral timeout",
                )
                check(
                    ARM + "turns.jsonl" not in z.namelist()
                    and not gzip.decompress(z.read(ARM + "observations.jsonl.gz")),
                    "unexpected timeout tape",
                )
        yield g, path, receipt


def rows(path, seat):
    with (
        zipfile.ZipFile(path) as z,
        z.open(ARM + "turns.jsonl") as turns,
        z.open(ARM + "observations.jsonl.gz") as packed,
        gzip.GzipFile(fileobj=packed) as observations,
    ):
        n = 0
        for n, (a, b) in enumerate(zip(turns, observations, strict=True), 1):
            t = n - 1
            turn, frame = json.loads(a), json.loads(b)
            check(t == turn["step"] == frame["step"], "timeline")
            inputs, calls = frame["observations"], turn["calls"]
            check(len(inputs) == len(calls) == 2, "both seats required")
            for i, obs in enumerate(inputs):
                check(
                    set(obs) == PUBLIC_KEYS and (obs["player"], obs["step"]) == (i, t),
                    "unexpected observation",
                )
                check(
                    obs_digest(obs)
                    == frame["public_observation_sha256"][i]
                    == turn["public_observation_sha256"][i],
                    "public request identity",
                )
                check(
                    calls[i]["status"] == "ok"
                    and isinstance(calls[i]["response"]["action"], dict),
                    "missing real orders",
                )
            yield dict(
                t=t,
                obs=inputs[seat],
                own=calls[seat]["response"]["action"],
                other=calls[1 - seat]["response"]["action"],
            )
        check(n == 299, "incomplete actions")


def games():
    for g, path, _ in inventory():
        if g["status"] == "complete":
            yield dict(
                seed=g["seed"],
                terrain=g["terrain"],
                seat=g["seat"],
                status=g["status"],
                rows=rows(path, g["seat"]),
            )


def failure():
    return next(
        dict(
            seed=g["seed"],
            terrain=g["terrain"],
            seat=g["seat"],
            status=g["status"],
            rows=(),
            initialization=receipt["initialization"],
        )
        for g, _, receipt in inventory()
        if g["status"] != "complete"
    )


if __name__ == "__main__":
    for game in games():
        print(game["seed"], game["seat"], sum(1 for _ in game["rows"]))
    missing = failure()
    print(missing["seed"], missing["seat"], missing["status"], 0)
