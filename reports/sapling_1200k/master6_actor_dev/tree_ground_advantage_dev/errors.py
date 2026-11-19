from fit import HERE,sr,predicted
from collections import defaultdict
from statistics import mean

def run():
    games=sr.load(HERE/'purchases.json.gz');w=sr.load(HERE/'weights.json')['coef'];fold={}
    for seed in (621758327,621758328):
        decisions=[];bins=defaultdict(list)
        for g in games:
            if g['seed']!=seed:continue
            for r in g['rows']:
                net=r['ground']-r['fee'];forecast=predicted(r['x'],w)-r['fee']
                key=('predict_profitable' if forecast>0 else 'predict_unprofitable')
                bins[key].append(net)
                decisions.append(dict(source=g['source'],t=r['t'],seat=r['seat'],uid=r['uid'],cell=r['cell'],tree=r['tree'],
                    predicted_net=forecast,real_ground_net=net,harvest_tax=r['tax'],price=r['fee'],first_tenure_end=r['end']))
        fold[seed]=dict(bins={k:dict(n=len(v),profitable=sum(n>0 for n in v),mean_ground_net=mean(v)) for k,v in bins.items()},
            worst_false_positive=max((r for r in decisions if r['predicted_net']>0 and r['real_ground_net']<=0),key=lambda r:r['predicted_net']-r['real_ground_net']),
            worst_false_negative=max((r for r in decisions if r['predicted_net']<=0 and r['real_ground_net']>0),key=lambda r:r['real_ground_net']-r['predicted_net']))
    sr.retain(HERE/'errors.json',{str(k):v for k,v in fold.items()});print(fold)
if __name__=='__main__':run()
