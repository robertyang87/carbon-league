from data import HERE,sr
def gate():
    rows=sr.load(HERE/'ledger.json.gz')
    assert {(r['seat'],r['arm']) for r in rows}=={(s,a) for s in (0,1) for a in ('parent','approach')} and len(rows)==4
    delta={k:sum((1 if r['arm']=='approach' else -1)*r[k] for r in rows) for k in ('margin','direct50')}
    deployed=sr.load(HERE/'deployment.json');plan=sr.load(HERE/'protocol.json')
    assert {r['seat'] for r in deployed}=={0,1} and all(r['pack']==plan['candidate'] for r in deployed)
    passed=all(x>0 for x in delta.values());sr.retain(HERE/'gate.json',dict(delta=delta,passed=passed));return passed
if __name__=='__main__':raise SystemExit(0 if gate() else 3)
