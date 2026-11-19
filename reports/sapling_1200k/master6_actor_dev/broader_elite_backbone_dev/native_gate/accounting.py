"""Read the identical paid-cash and original-clock fields from the 75:25 matches."""
import importlib.util
from pathlib import Path
import sys

H=Path(__file__).resolve().parent
pure=H.parent/"pure_native_gate"
sys.path.insert(0,str(pure))
sp=importlib.util.spec_from_file_location("pure_finance_by_score",pure/"accounting.py")
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
m.H=H
r=m.r

if __name__=="__main__":
    rows=[m.summarize(q) for q in r.load(H/"scores.json")]
    by={(v["arm"],v["seat"]):v for v in rows}
    compare={}
    for side in (0,1):
        a,b=by["parent",side],by["candidate",side]
        compare[str(side)]=dict(margin_delta=b["official_margin"]-a["official_margin"],
            delta_normal_deposit_margin=b["cash"]["delivery_margin"]-a["cash"]["delivery_margin"],
            delta_tree_income_margin=b["cash"]["tree_income_margin"]-a["cash"]["tree_income_margin"])
    r.write(H/"accounting.json",dict(protocol_sha256=r.sha(H/"protocol.json"),scores_sha256=r.sha(H/"scores.json"),rows=rows,same_seat=compare))
