"""Regression fixtures for the ten-step timing, principal accounting, and set of observable fields."""
import copy
from data import target
from predictor import FreightPredictor


def event(t,remaining=1,payment=None,alive=True):
    return dict(t=t,alive={'w'} if alive else set(),ret={'w':remaining},payments={} if payment is None else {'w':payment})

def main():
    assert target('w',100.,1,0,[event(1,.8),event(2,payment=(1,200.)),event(3,payment=(1,40.))])==[.8,1,240.,80.,2]
    assert target('w',100.,1,0,[event(1,payment=(0,100.)),event(2,payment=(1,80.))])==[0.,1,80.,0.,2]
    assert target('w',50.,1,0,[event(1,alive=False)]+[event(t,alive=False) for t in range(2,11)])==[0.,0,0.,0.,-1]
    assert target('w',50.,1,0,[event(t) for t in range(1,11)]+[event(11,payment=(1,50.))])==[0.,0,0.,0.,-1]
    assert target('w',50.,1,0,[event(t) for t in range(1,10)]+[event(10,payment=(1,50.))])==[1.,1,50.,50.,10]
    m=FreightPredictor();a={'step':12,'player':0,'players':[[80,{'h':0},{},{'tr':[2,5]}],[180,{'h':224},{'w':[223,178,'COLLECTOR']},{}]]};history=[dict(a,step=t) for t in range(7,12)]
    b=copy.deepcopy(a);b.update(full_carbon=[100]*225,carbon=[100]*225,terminated_reward=5,episode_outcome='W',future={'w':[224,0,'COLLECTOR']})
    assert m.predict(a,history)==m.predict(b,history) and set(m.predict(a,history))=={'w'}
    for wrong in (history+[a],history+[dict(a,step=11,player=1)]):
        try:m.predict(a,wrong)
        except ValueError:pass
        else:raise AssertionError('invalid history accepted')
    print('8 causal/history/principal fixtures passed')

if __name__=='__main__':main()
