import json, os, hashlib, shutil, sys

def status():
    for a in ("m", "c"):
        d = f"/root/carbon/data/rl3/ppou2_{a}"
        tp = os.path.join(d, "train.jsonl")
        done = os.path.exists(os.path.join(d, "state_anchor_frozen.pt"))
        warm3 = False
        formal = False
        sha = None
        if os.path.exists(tp):
            for line in open(tp):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("warm") and r.get("it") == 3:
                    warm3 = True
                    sha = r.get("checkpoint", {}).get("files", {}).get("state_last.pt")
                if not r.get("warm"):
                    formal = True
        tag = "anchor_done" if done else ("WARM3" if warm3 else ("formal_started" if formal else "waiting"))
        print(a, tag, "sha", sha or "-")

def capture():
    for a in ("m", "c"):
        d = f"/root/carbon/data/rl3/ppou2_{a}"
        dst = os.path.join(d, "state_anchor_frozen.pt")
        src = os.path.join(d, "state_last.pt")
        if os.path.exists(dst) or not os.path.exists(src):
            continue
        tp = os.path.join(d, "train.jsonl")
        warm3 = False
        formal = False
        for line in open(tp):
            r = json.loads(line)
            if r.get("warm") and r.get("it") == 3:
                warm3 = True
            if not r.get("warm"):
                formal = True
        if warm3 and not formal:
            shutil.copyfile(src, dst)
            h = hashlib.md5(open(dst, "rb").read()).hexdigest()
            print("CAPTURED", a, h)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        capture()
    status()
