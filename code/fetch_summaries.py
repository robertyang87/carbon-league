import json, subprocess, time, sys, os

def get(path):
    out = subprocess.run(["curl","-sk","--max-time","40","https://132.232.176.153"+path], capture_output=True, text=True)
    if out.returncode!=0: raise RuntimeError(f"curl rc={out.returncode} {out.stderr[:200]}")
    try: return json.loads(out.stdout)
    except Exception: raise RuntimeError("bad json: "+out.stdout[:300])

def fetch(mid, game, tag):
    pages=[]; offset=None; n=0
    while True:
        path=f"/api/summary/{mid}/{game}" + (f"?events_offset={offset}" if offset is not None else "")
        d=get(path); n+=1
        if "error" in d and "metadata" not in d: raise RuntimeError(str(d))
        pages.append(d)
        offset=d.get("next_events_offset")
        print(tag, "page", n, "events", len(d.get("events",{}) if isinstance(d.get("events"),dict) else d.get("events",[])), "total_events", d.get("total_events"), "next", offset, flush=True)
        if offset is None or n>=6: break
        time.sleep(2.0)
    merged=pages[0]
    if len(pages)>1:
        ev=merged["events"]
        for p in pages[1:]:
            pe=p["events"]
            if isinstance(ev, dict):
                for k,v in pe.items(): ev.setdefault(k,[]).extend(v)
            else: ev.extend(pe)
    json.dump(merged, open(f"summaries/{tag}.json","w"))
    return merged

jobs = [
 ("e3608fdacaaf8c9e25d51e04c35e2216",1,"a2_loss_alpha1"),
 ("748d5ddc0d7d1c1e935d610f123c75a5",1,"a2_loss_rlpolicy_1"),
 ("78d0fecd2e2bbd2f42ca332e26d80376",1,"a2_loss_rlpolicy_2"),
 ("087c5cbf",1,"PLACEHOLDER"),
]
# resolve full ids from matches file
ms=json.load(open("matches_alpha2.json"))
by_short={m["id"][:8]:m for m in ms}
picks = {
 "e3608fda":"a2_loss_vs_alpha1",
 "748d5ddc":"a2_loss_vs_rlpolicy_a",
 "78d0fecd":"a2_loss_vs_rlpolicy_b",
 "087c5cbf":"a2_closewin_godslayerLXIX",
 "d22f8ee7":"a2_win_vs_rllight",
 "5e8cf575":"a2_lowscore_win_rllight",
 "41e45fcb":"a2_win_vs_alpha1",
 "95c85b01":"a2_medianwin_anshengF9",
 "49bc48bc":"a2_lowtempo_win_nnlight3",
 "e3a1e91e":"a2_win_codexv9",
 "eb76e41f":"a2_highscore_win_v53sw0",
 "1db2e0f1":"a2_maxmargin_win_sweeper",
}
for short,tag in picks.items():
    m=by_short[short]
    if os.path.exists(f"summaries/{tag}.json"): print("skip",tag); continue
    fetch(m["id"], m["game"], tag)
    time.sleep(2.0)
print("done")
