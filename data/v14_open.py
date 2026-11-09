import gzip,json,os,statistics as st
ME='claude-opus55-cloud-v14'
ms=json.load(open('../data/v14_matches.json'))
out=[]
for m in ms:
    f=f"../data/v14_ladder/replays/{m['id']}-{m.get('game',1)}.json.gz"
    d=json.load(gzip.open(f)); stp=d['steps']
    a_me=m['name_a']==ME; order=m.get('order') or ['a','b']; pm=order.index('a' if a_me else 'b'); po=1-pm
    R={'diff':d['rewards'][pm]-d['rewards'][po]}
    for p,tag in ((pm,'me'),(po,'op')):
        prev=None; dv={50:0,100:0,200:0,300:0}; ls={50:0,100:0,200:0,300:0}; rc={50:0,100:0}; tr={}
        for t in range(len(stp)):
            o=stp[t][0]['observation']; pl=o['players'][p]; w=pl[2]; bases=set(pl[1].values())
            if t in (50,100): tr[t]=len(pl[3]) if isinstance(pl[3],dict) else 0
            if prev is not None:
                for k,w0 in prev.items():
                    w1=w.get(k)
                    if w1 is None and w0[2]=='COLLECTOR' and w0[1]>0:
                        for c in ls:
                            if t<=c: ls[c]+=w0[1]
                    elif w1 is not None and w0[2]=='COLLECTOR' and w0[1]>0 and w1[1]==0 and w1[0] in bases:
                        for c in dv:
                            if t<=c: dv[c]+=w0[1]
                for c in rc:
                    if t<=c: rc[c]+=sum(1 for k in w if k not in prev)
            prev={k:list(v) for k,v in w.items()}
        R[tag]=dict(dv=dv,ls=ls,rc=rc,tr=tr)
    out.append(R)
def g(lbl,rs):
    m=lambda f: st.mean(f(r) for r in rs)
    print(f"== {lbl} n={len(rs)}")
    for c in (50,100,200,300):
        print(f"  到第{c:3d}步  交货 我{m(lambda r:r['me']['dv'][c]):6.0f} 敌{m(lambda r:r['op']['dv'][c]):6.0f}   被截 我{m(lambda r:r['me']['ls'][c]):5.0f} 敌{m(lambda r:r['op']['ls'][c]):5.0f}")
    for c in (50,100):
        print(f"  到第{c:3d}步  招募 我{m(lambda r:r['me']['rc'][c]):5.1f} 敌{m(lambda r:r['op']['rc'][c]):5.1f}   树 我{m(lambda r:r['me']['tr'][c]):4.1f} 敌{m(lambda r:r['op']['tr'][c]):4.1f}")
g('输局',[r for r in out if r['diff']<0]); g('小胜',[r for r in out if 0<=r['diff']<600]); g('大胜',[r for r in out if r['diff']>=1000])
