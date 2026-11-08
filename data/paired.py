import json,math,sys
def load(t):
    d={}
    for tt in t.split('+'):
        for r in map(json.loads,open(f'../data/bigeval_live/{tt}.jsonl')): d[r['seed']]=r
    return d
c=load(sys.argv[1]); b=load(sys.argv[2])
ks=[s for s in c if s in b]
d=[c[s]['margin']-b[s]['margin'] for s in ks]; n=len(d); m=sum(d)/n; se=math.sqrt(sum((x-m)**2 for x in d)/(n-1)/n)
f=lambda k,D: sum(D[s][k] for s in ks)/n
print(f"{sys.argv[1]} vs {sys.argv[2]}: n {n} paired {m:+.0f} ± {se:.0f} (m-2se {m-2*se:+.0f}) | wins {sum(c[s]['margin']>0 for s in ks)} vs {sum(b[s]['margin']>0 for s in ks)} | mean {f('margin',c):+.0f} vs {f('margin',b):+.0f} | worst {min(c[s]['margin'] for s in ks)} vs {min(b[s]['margin'] for s in ks)} | diff<-600 {sum(x<-600 for x in d)} | lost {f('lost',c):.0f} vs {f('lost',b):.0f} | olost {f('olost',c):.0f} vs {f('olost',b):.0f}")
