import json, sys
arm = sys.argv[1]
path = f"/root/carbon/data/rl3/ppou2_{arm}/train.jsonl"
want = {int(x) for x in sys.argv[2:]}
for line in open(path):
    d = json.loads(line)
    if d.get("it") in want and not d.get("warm"):
        srv = d.get("srv", {})
        print(arm, "it", d["it"], "kl", d["kl"], "kl_coef", d["kl_coef"],
              "gpm", d["games_per_min"], "roll_s", d["roll_s"], "upd_s", d["upd_s"],
              "pre_s", d.get("pre_s"), "games", d["games"], "winrate", d["winrate"],
              "clipfrac", d["clipfrac"], "t", d["t"],
              "srv_models", srv.get("models"), "srv_servers", srv.get("servers"))
