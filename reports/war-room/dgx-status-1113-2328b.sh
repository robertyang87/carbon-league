#!/bin/bash
# DGX status probe 2328B (read-only follow-up). Marker: DGXSTATUS2328B
MARK=DGXSTATUS2328B
echo "$MARK BEGIN $(date '+%F %T %Z')"
echo '--- GEN1R INDEX STATS (positions / endgame proxy / U labels / mirror) ---'
python3 - <<'EOF' 2>&1
import json, collections
d=json.load(open("/home/arm/junyang/carbon/data/gen1r_shard0/index.json"))
items=d if isinstance(d,list) else d.get("entries",[])
print("games:",len(items))
steps=[it.get("steps",0) for it in items]
print("steps sum:",sum(steps),"mean:%.1f"%(sum(steps)/len(steps) if steps else 0),"min:",min(steps) if steps else None,"max:",max(steps) if steps else None)
print("endgame_positions_proxy sum(max(0,steps-200)):",sum(max(0,s-200) for s in steps))
print("games_with_steps>=200:",sum(1 for s in steps if s>=200))
ua=[it.get("u",{}).get("a") for it in items if isinstance(it.get("u"),dict) and it["u"].get("a") is not None]
print("u.a n:",len(ua),"distinct:",len(set(round(x,4) for x in ua)),"mean:%.4f"%(sum(ua)/len(ua) if ua else 0),"min:%.4f"%min(ua) if ua else "","max:%.4f"%max(ua) if ua else "")
print("mirror_signature True count:",sum(1 for it in items if it.get("mirror_signature")))
print("end_reason:",dict(collections.Counter(it.get("end_reason") for it in items)))
print("weights_md5 set:",{it.get("identity",{}).get("weights_md5") for it in items if isinstance(it.get("identity"),dict)})
print("allow_mirror set:",{it.get("identity",{}).get("diversity",{}).get("allow_mirror") for it in items if isinstance(it.get("identity"),dict)})
print("max_game_seed:",max(it.get("game",0) for it in items))
EOF
echo '--- GEN1R CHAIN LOGS ---'
echo "== gen1r_gen_shard0.log (tail 12)"; tail -12 /home/arm/junyang/carbon/data/dq_logs/gen1r_gen_shard0.log 2>&1
echo "== gen1r_chain.log (tail 12)"; tail -12 /home/arm/junyang/carbon/data/dq_logs/gen1r_chain.log 2>&1
echo "== gen1r_chain_nohup.log (tail 5)"; tail -5 /home/arm/junyang/carbon/data/dq_logs/gen1r_chain_nohup.log 2>&1
echo '--- IN-PROGRESS TMP ---'
ls /home/arm/junyang/carbon/data/gen1r_shard0/replays/.*.tmp* 2>/dev/null | wc -l
echo '--- REPLAYS COUNT NOW ---'
ls /home/arm/junyang/carbon/data/gen1r_shard0/replays/*.json.gz 2>/dev/null | wc -l
date '+%F %T %Z'
echo "$MARK END $(date '+%F %T %Z')"
