"""Read the public match timeline back before the v3 attempt and select the exact strategy name."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,ssl,urllib.request
from urllib.error import HTTPError

HERE=Path(__file__).resolve().parent
BASE="https://117.72.210.134"
SINCE=datetime.fromisoformat("2026-11-20T07:25:00+00:00").timestamp()
def utc():
    return datetime.now(timezone.utc).isoformat()
def get(path):
    context=ssl.create_default_context(cafile="/tmp/carbon-managed-egress-public.pem")
    context.check_hostname=False
    req=urllib.request.Request(BASE+path,headers={"Cache-Control":"no-cache","Accept":"application/json","User-Agent":"Carbon League public replay reader"})
    with urllib.request.urlopen(req,context=context,timeout=18) as resp:
        raw=resp.read()
        return json.loads(raw),resp.headers.get("Date"),hashlib.sha256(raw).hexdigest()

def main():
    started=utc();offset=0;seen={};pages=[];v3=[]
    while True:
        path=f"/api/matches?limit=100&offset={offset}"
        data,server_date,digest=get(path);ms=data["matches"]
        assert len({m["id"] for m in ms})==len(ms)
        times=[m["created"] for m in ms]
        assert times==sorted(times,reverse=True), "timeline not in descending creation order"
        pages.append(dict(path=path,date=server_date,read_utc=utc(),sha256=digest,ids=[m["id"] for m in ms],
            first_created=max(times, default=None),last_created=min(times,default=None),next_offset=data["next_offset"]))
        for m in ms:
            seen[m["id"]]=m
        if not times or min(times) < SINCE or data["next_offset"] is None:break
        assert data["next_offset"]>offset
        offset=data["next_offset"]
    matches=[m for m in seen.values() if SINCE <= m["created"] <= datetime.fromisoformat(started).timestamp()]
    v3=[m for m in matches if any(m.get("name_"+seat,"").casefold()=="ds-pro-v3" for seat in ("a","b"))]
    report=dict(through_utc=started,returned_utc=utc(),earliest_utc=datetime.fromtimestamp(SINCE,timezone.utc).isoformat(),
        coverage="public matches by time of creation, from the first live page back through the first older card; no owner visibility inferred",
        pages=pages,distinct_public_in_window=len(matches),v3_matches=v3)
    (HERE/"recent_window.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print('window',len(pages),'pages',len(matches),'unique',len(v3),'ds-pro-v3')
if __name__=="__main__":main()
