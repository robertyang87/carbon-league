"""Selector contracts using synthetic numbers; these tests do not supply or certify trained weights."""
import time
import numpy as np
from pathlib import Path
import runpy,json
here=Path(__file__).resolve().parent
select=runpy.run_path(str(here/"selector.py"),init_globals=dict(np=np,_S_ABORT=.58))["_rr_select"]
base=dict(original=0,alive=[0,1,2],mean={0:.32,1:.30,2:.25},depths={0:8,1:8,2:8},
          run=[[i,0,{"step":287},None,True] for i in range(3)],cands=[{},{"u":"DOWN"},{"u":"UP"}],
          step=279,me=0,players=[[80,{"b":0},{"u":[2,50,"COLLECTOR"]},{}],[80,{"c":10},{},{}]],
          depth=8,rolls=1,cut=False,start=0.,clock=lambda:0.,score=lambda p,m,t,cs:[20.,24.],scale=.01,epsilon=.01)
def main():
    checks={}
    def case(name,expect,changes):
        choice,reason,detail=select(**(base|changes))
        assert choice==expect,(name,choice,reason)
        checks[name]=dict(chosen=choice,reason=reason)
    case("genuine_positive_pair",1,{})
    case("no_action_difference",0,dict(score=lambda *x:[20,20]))
    case("wrong_direction",0,dict(score=lambda *x:[20,19]))
    case("epsilon_preserves_native_A0",0,dict(score=lambda *x:[20,22.5]))
    case("nonfinite_model",0,dict(score=lambda *x:[np.nan,50]))
    case("incomplete_model",0,dict(score=lambda *x:[20]))
    case("uncovered_turn",0,dict(step=200))
    case("mixed_depth",0,dict(depths={0:8,1:7,2:8}))
    case("cutoff",0,dict(cut=True))
    case("missing_scenario",0,dict(run=[base["run"][0]]))
    case("historical_extinction",0,dict(run=[base['run'][0],[1,0,{"step":287},None,False]]))
    case("native_epsilon_already_held",0,dict(mean={0:.30,1:.32,2:.26}))
    case("insufficient_time_to_start",0,dict(clock=lambda:.578))
    case("expires_in_scorer",0,dict(clock=iter((.55,.59)).__next__))
    case("zero_or_unknown_scale",0,dict(scale=0.))
    case("duplicate_root_commands",0,dict(cands=[{},{},{"u":"UP"}]))
    case("no_owned_load",0,dict(players=[[20,{},{"u":[2,0,"COLLECTOR"]},{}],base['players'][1]]))
    out={"only_synthetic_function_contract":True,"tests":checks}
    (here/'fixture_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print('passed',len(checks))
if __name__=='__main__': main()
