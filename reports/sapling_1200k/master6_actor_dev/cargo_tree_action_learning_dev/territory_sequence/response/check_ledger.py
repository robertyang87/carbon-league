"""Exercise account closure and first-tenure attribution on historical official tapes."""
import run as R


def main():
    rows = []
    plan = R.preflight()
    for case in plan["cases"]:
        raw = R.archive(case, plan)
        env = R.R.live.begin(raw)
        comparisons = []
        for t, joint in enumerate(raw["actions"]):
            if R.BEGIN <= t < R.END:
                s = case["seat"]
                _, bases, workers, _ = R.R.live.public(env, s)["players"][s]
                changes = {u: [joint[s].get(u, "NONE"), "mock"] for u in list(bases)+list(workers)}
                comparisons.append(dict(t=t, changed=changes))
            env.step(joint)
        cash = [p[0] for p in env.states[0].observation.players]
        ledger = R.accounting(raw, raw["actions"][R.BEGIN:], cash, comparisons)
        assert [state.reward for state in env.states] == raw["official_score_by_seat"]
        trees = [r["purchased_tree"] for r in ledger["changed_orders"] if r["purchased_tree"]]
        assert trees and sum(z["ground_rent"] for z in trees)>0
        rows.append(dict(seed=case["seed"], seat=case["seat"], terrain=case["terrain"], physical_replay_balance_ok=True,
                         official_terminal_replay_matches_archive=True, transitions=len(ledger["steps"]),
                         archive=case["source_sha256"], all_window_orders_traced=len(ledger["changed_orders"]), acquired_trees_traced=len(trees)))
    R.R.write(R.HERE/"ledger_preflight.json", dict(protocol=R.REGISTERED, runner=R.R.sha(R.HERE/"run.py"), test=R.R.sha(R.HERE/"check_ledger.py"),
        source=R.R.sha(R.R.live.SOURCE), method="Every historical window order flagged, accounting and first-tenure attribute trace only.", cases=rows))
    print(rows)


if __name__ == "__main__":
    main()
