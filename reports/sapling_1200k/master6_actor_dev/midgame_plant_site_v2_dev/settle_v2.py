"""Apply the recorded first-tenure, team receipts and official outcome gate."""
import runpy
from study import HERE,V1
def settle():
    verify=runpy.run_path(str(V1/'settle_sites.py'))['verify']
    verify.__globals__['HERE']=HERE
    verify()
if __name__=='__main__':settle()
