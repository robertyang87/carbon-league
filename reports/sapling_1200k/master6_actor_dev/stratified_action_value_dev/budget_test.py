"""A lock miss never opens a lease. Interrupted running pairs and two phases share one limit."""
import tempfile
from pathlib import Path
import budget
from core import save,HERE,read

def main():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'ledger.json'
        clock=[100.]
        now=lambda:clock[0]
        assert budget.remaining(13.,path,now)==13.
        for _ in range(4):clock[0]+=500
        assert budget.remaining(13.,path,now)==13. and not path.exists()
        with budget.reserve(13.,path,now) as a:
            assert a==13.
            clock[0]+=4
        assert budget.remaining(13.,path,now)==9.
        clock[0]+=100
        assert budget.remaining(13.,path,now)==9.
        d=read(path);d['active']={"start":clock[0],"pid":123,"phase":"train"}
        save(path,d,False)
        clock[0]+=3
        with budget.reserve(13.,path,now) as b:
            assert b==6.
            clock[0]+=2
        assert budget.remaining(13.,path,now)==4.
        with budget.reserve(13.,path,now) as c:
            assert c==4.
            clock[0]+=4
        assert budget.remaining(13.,path,now)==0.
        out={"initial_seconds":13,"waiting_without_arena":2100,"train_seconds":4,
             "interrupted_training_seconds_recovered":3,"resumed_train_seconds":2,
             "available_for_held":4,"held_seconds":4,"remaining_at_end":0,
             "charged_intervals":[r["seconds"] for r in read(path)["history"]]}
        save(HERE/'budget_fixture.json',out)
        print(out)

if __name__=='__main__':main()
