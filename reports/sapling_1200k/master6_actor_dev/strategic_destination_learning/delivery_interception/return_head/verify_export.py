"""Read each complete history anew and verify both frozen inference outputs."""
import gzip
import json
from pathlib import Path
import sys
import numpy as np

from dataset import histories
from predict import PublicReturn, sha

HERE=Path(__file__).resolve().parent


def run():
    learner=PublicReturn()
    packed=json.loads(gzip.decompress((HERE/"predictions.json.gz").read_bytes()))
    table={(r["terrain"],r["seat"],r["step"],r["prefix_sha256"],r["public_sha256"]):r for r in packed["rows"]}
    checked=set(); worst=0.
    for game,r,history in histories():
        key=(game["terrain"],game["seat"],r["step"],r["prefix_sha256"],r["public_sha256"])
        row=table[key]
        pred=learner.predict(history,r["target"]["target_uid"])
        for arm in ("geometry","neural"):
            delta=max(abs(pred[arm]-row["probability"][arm]),abs(pred[arm+"_cash"]-row["cash_prediction"][arm]))
            worst=max(worst,delta)
        if key in checked:
            raise ValueError("repeated root")
        checked.add(key)
    if len(checked)!=885 or checked!=set(table) or worst>1e-10:
        raise ValueError("frozen narrow inference and batched exported scores differ")
    for history_arg in (history[::-1],history[:-1],[*history[:-1],dict(history[-1],full_carbon=[0]*225)]):
        try:
            learner.predict(history_arg,r["target"]["target_uid"])
        except ValueError:
            continue
        raise ValueError("wrong history was accepted by frozen inference")
    receipt=dict(roots=len(checked),maximum_absolute_probability_or_cash_difference=worst,
                 rejected_inputs=["reversed_history","missing_final_observation","privileged_full_carbon"],
                 release_sha256=sha(HERE/"release.json"),source_sha256=sha(Path(__file__)),
                 predictions_sha256=sha(HERE/"predictions.json.gz"))
    (HERE/"export_verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt)


if __name__=="__main__":
    run()
