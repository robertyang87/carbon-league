#!/usr/bin/env python3
"""Per-round KL drift guard: logs each new train row's kl; flags >=0.12 / >=0.13."""
import json, os, time
TRAIN = "/root/carbon/data/rl3/snipe_n1/train.jsonl"
STATUS = "/root/carbon/data/rl3/snipe_n1/kl_guard_status.txt"
def log(msg):
    with open(STATUS, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
def main():
    seen = 0
    log("GUARD_START")
    while True:
        try:
            rows = [json.loads(l) for l in open(TRAIN) if l.strip()]
        except Exception:
            rows = []
        for r in rows:
            if r["it"] > seen:
                seen = r["it"]
                flag = ""
                if r["kl"] >= 0.13: flag = " CRITICAL_RED_LINE"
                elif r["kl"] >= 0.12: flag = " ALERT_TRIGGER_0.12"
                log(f"it{r['it']} kl={r['kl']} kl_coef={r['kl_coef']} wr={r['winrate']}{flag}")
        time.sleep(60)
if __name__ == "__main__":
    main()
