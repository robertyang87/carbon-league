import gzip,json,glob,os,collections,statistics as st
ME='claude-opus55-cloud-v14'
ms=json.load(open('../data/v14_matches.json'))
CK=[25,50,75,100,125,150,175,200,225,250,275,299]
rows=[]
for m in ms:
    f=f"../data/v14_ladder/replays/{m['id']}-{m.get('game',1)}.json.gz"
    if not os.path.exists(f): continue
    d=json.load(gzip.open(f)); stp=d['steps']
    a_me = m['name_a']==ME; order=m.get('order') or ['a','b']
    pm=order.index('a' if a_me else 'b'); po=1-pm
    opp=m['name_b' if a_me else 'name_a']
    R={'opp':opp,'diff':d['rewards'][pm]-d['rewards'][po]}
    for p,tag in ((pm,'me'),(po,'op')):
        cash=[];deliv=0;lost=0;nlost=0;recr=0;dead=0;units=[];trees=[];prev=None;col=[];pla=[]
        for t in range(len(stp)):
            o=stp[t][0]['observation']; pl=o['players'][p]; w=pl[2]; bases=set(pl[1].values())
            cash.append(pl[0]); units.append(len(w))
            col.append(sum(1 for v in w.values() if v[2]=='COLLECTOR')); pla.append(sum(1 for v in w.values() if v[2]=='PLANTER'))
            trees.append(len(pl[3]) if len(pl)>3 and isinstance(pl[3],dict) else 0)
            if prev is not None:
                for k,w0 in prev.items():
                    w1=w.get(k)
                    if w1 is None:
                        dead+=1
                        if w0[2]=='COLLECTOR' and w0[1]>0: lost+=w0[1]; nlost+=1
                    elif w0[2]=='COLLECTOR' and w0[1]>0 and w1[1]==0 and w1[0] in bases: deliv+=w0[1]
                recr+=sum(1 for k in w if k not in prev)
            prev={k:list(v) for k,v in w.items()}
        R[tag]=dict(cash=[cash[min(c,len(cash)-1)] for c in CK],deliv=deliv,lost=lost,nlost=nlost,recr=recr,dead=dead,
                    col100=col[100],col200=col[200],pla100=pla[100],pla200=pla[200],trees=max(trees),trees200=trees[200])
    rows.append(R)
json.dump(rows,open('../data/v14_phase.json','w'))
def grp(lbl,rs):
    n=len(rs)
    if not n: return
    m=lambda f: st.mean(f(r) for r in rs)
    print(f"\n== {lbl}  n={n}  mean diff {m(lambda r:r['diff']):+.0f}")
    print("  现金差@步 "+"  ".join(f"{c:>5}" for c in CK))
    print("  我-敌     "+"  ".join(f"{m(lambda r,i=i:r['me']['cash'][i]-r['op']['cash'][i]):+5.0f}" for i in range(len(CK))))
    for k,nm in (('deliv','交货'),('lost','被截货'),('nlost','被截次数'),('recr','招募数'),('dead','死亡单位'),('col100','采集员@100'),('col200','采集员@200'),('pla100','种植员@100'),('pla200','种植员@200'),('trees','峰值树'),('trees200','树@200')):
        print(f"  {nm:8s} 我 {m(lambda r:r['me'][k]):7.1f}  敌 {m(lambda r:r['op'][k]):7.1f}")
L=[r for r in rows if r['diff']<0]; N=[r for r in rows if 0<=r['diff']<600]; B=[r for r in rows if r['diff']>=1000]
grp('输局',L); grp('小胜 (0~600)',N); grp('大胜 (>=1000)',B)
