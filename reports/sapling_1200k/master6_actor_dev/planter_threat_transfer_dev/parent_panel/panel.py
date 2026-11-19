"""Pre-register the complement of the previous pay panel inside the already signed exact-v1 source index."""
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1]
ROOT=ACTOR.parents[2]
PAY=ACTOR/"tree_tenure_dev/pay_policy"
INDEX=ACTOR/"stratified_action_value_dev/source/inventory.json"
OLD=PAY/"opportunities.json.gz"
BLOCKED=frozenset(json.loads((PAY/"protocol.json").read_bytes())["new_dev"]["seeds"])


def read(p):
    raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix==".gz" else raw)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def check(cond,detail):
    if not cond:
        raise ValueError(detail)


def save(p,v):
    if p.exists():
        check(read(p)==v,("immutable receipt changed",p))
        return
    raw=(json.dumps(v,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode()
    temp=p.with_name(f".{p.name}.{os.getpid()}.part")
    temp.write_bytes(gzip.compress(raw,mtime=0) if p.suffix==".gz" else raw)
    temp.replace(p)


def main():
    old=read(OLD)
    prior={g["terrain"] for g in old["games"]}
    check(len(prior)==16 and len(old["games"])==32 and old["changed"]==56,"wrong exclusion archive")
    src=read(INDEX);rows=src["v1_inventory"]
    reg=read(PAY/"protocol.json");package=read(PAY/"package.json")
    check(sha(PAY/"candidate.py")==package["candidate_sha256"] and sha(PAY/"protocol.json")==package["protocol_sha256"]
          and src["accepted_raw_pack_sha256"]==reg["parent_sha256"],"different whole-pay policy or origin")
    accepted=[]
    for row in rows:
        if row["map"] in prior or row["seed"] in BLOCKED:
            continue
        check(row["pack_sha256"]==reg["parent_sha256"] and row["examiner_md5"]==src["original_md5"],"mixed parent or opponent")
        accepted.append({k:row[k] for k in ("seed","seat","map","raw","raw_sha256","official","official_sha256","config","pack_sha256","examiner_md5","steps")})
    check(len(accepted)==18 and len({r["map"] for r in accepted})==9 and
          set(Counter(r["map"] for r in accepted).values())=={2} and
          {(r["map"],s) for r in accepted for s in (0,1)}=={(r["map"],r["seat"]) for r in accepted},"incomplete paired map source")
    check(not prior.intersection(r["map"] for r in accepted),"reused old terrain")
    p={"protocol_version":"1","parent_sha256":reg["parent_sha256"],"examiner_sha256":reg["original_sha256"],
       "pay_candidate_sha256":package["candidate_sha256"],"pay_protocol_sha256":sha(PAY/"protocol.json"),
       "inventory_sha256":sha(INDEX),"historical_exclusion_index_sha256":sha(OLD),
       "source_universe":{"description":"Previously signed 25-terrain v1 vs original Master3s2 development archive; exact-v1 index compiled before the new pay pilot.",
                          "map_seats":len(rows),"maps":len({r["map"] for r in rows})},
       "old_coverage":{"maps":len(prior),"map_seats":32,"selected":old["selected"],"changed":old["changed"]},
       "excluded_terrains":sorted(prior),"blocked_new_pilot_seeds":sorted(BLOCKED),
       "pre_action_exposure":"Any existing opponent collector or planter can reach the current plant/seize tile in zero or one own movement according to the toroidal board. Any affordable enemy spawn on that exact tile is a separate field. No action, payment, survival, tree switch or future frame enters this exposure.",
       "scope":"Use the unmodified two-window whole-pay package on recorded parent trajectories; stop and leave unchanged are evaluated with the same originally recorded simultaneous opponent commands. Capture parent and pay worker survival, payer-specific fees and tree ownership with the official Board. The original 56 changed roots and new whole-game pilot are excluded.",
       "primary":"Report all reached windows; selected and changed-only separately. Compare sustainment and focus payment by public reach. Aggregate on maps carrying each exposure. Show actual enemy arrivals separately as post-action explanation; no strength claim.",
       "games":sorted(accepted,key=lambda r:(r["seed"],r["seat"]))}
    save(HERE/"panel.json",p)
    print("Frozen",len(accepted),"seats;",len({r["map"] for r in accepted}),"distinct terrains.")


if __name__=="__main__":
    main()
