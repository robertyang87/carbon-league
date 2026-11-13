"""57557a8 dedicated trainer control-flow probe; no model training or remote operations.
Loads its source via git show against the isolated 62e0dee checkout (shared model unchanged).
"""
import contextlib, io, json, math, os, subprocess, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
repo=Path(os.environ['CARBON_REPO']).resolve();sys.path.insert(0,str(repo/'code'));os.environ['BC_DEVICE']='cpu';os.environ['BC_BS']='1';torch.set_num_threads(1)
REF='57557a83a73cd4d6b26b56ebebb4c0df8e13a155'
source=subprocess.check_output(['git','-C',str(repo),'show',REF+':code/stage0_train_heads.py'],text=True)
mod=types.ModuleType('review_full_head_train');mod.__file__=str(repo/'code/stage0_train_heads.py');exec(compile(source,REF+':stage0_train_heads.py','exec'),mod.__dict__)
net=mod.CNet2(C=16,NB=1,D=16,NH=4,L=1,E=8,G=8)
heads=('utility.','wdl.','margin.')
w={k:v.detach().numpy().copy() for k,v in net.state_dict().items() if not k.startswith(heads)};w['meta']=np.array(json.dumps({'cfg':net.cfg,'epoch':13,'data':'historical-fixture'}))
class NoUpdate:
    def __init__(self,*a,**kw):pass
    def zero_grad(self,*a,**kw):pass
    def step(self,*a,**kw):pass

def loss_stub(net,b):
    return torch.tensor(float('nan'),requires_grad=True),float('nan'),float('nan'),float('nan'),0.,float('nan'),(float('nan'),float('nan'))
def val_stub(*a,**kw):return {'val_l_u':float('nan'),'val_wdl_acc':0.,'val_margin_mae':float('nan'),'val_u_calib_mean':float('nan'),'val_u_calib_binned':float('nan'),'val_u_range':[float('nan'),float('nan')],'val_n':1}
res={'trainer_source':REF,'model_checkout':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'scope':'Actual main with injected NaN loss/validation and no-op optimizer; no actual weights update.'}
with tempfile.TemporaryDirectory(prefix='carbon-full-heads-') as td:
    td=Path(td);ip=td/'legacy.npz';op=td/'out.npz';np.savez(ip,**w)
    stream=io.StringIO()
    with patch.object(sys,'argv',['stage0_train_heads.py','stub_ds',str(ip),str(op),'1']),patch.object(mod.CT,'load',lambda p:({},np.array([False,True]),{})),patch.object(mod.CT,'loader',lambda *a,**kw:iter([{}])),patch.object(mod,'head_losses',loss_stub),patch.object(mod,'evaluate',val_stub),patch.object(torch.optim,'AdamW',NoUpdate),patch.object(torch.optim.lr_scheduler,'OneCycleLR',NoUpdate),contextlib.redirect_stdout(stream):
        returned=mod.main()
    res['nan_run']={'returned_normally':returned is None,'output_saved':op.exists(),'train_done_emitted':'TRAIN-DONE ' in stream.getvalue(),'stdout':stream.getvalue(),'saved_meta_matches_old':str(np.load(op)['meta'])==str(w['meta'])}
    # New trainer requires exactly six missing keys; its own fully populated output cannot be used as init.
    stream2=io.StringIO()
    with patch.object(sys,'argv',['stage0_train_heads.py','stub_ds',str(op),str(td/'out2.npz'),'1']),patch.object(mod.CT,'load',lambda p:({},np.array([False,True]),{})),contextlib.redirect_stdout(stream2):
        try:mod.main();res['own_output_as_init']={'raises':False}
        except AssertionError as e:res['own_output_as_init']={'raises':True,'exception':'AssertionError','message':str(e)}
    bad=dict(w);del bad['stem.weight'];bp=td/'bad.npz';np.savez(bp,**bad)
    with patch.object(sys,'argv',['stage0_train_heads.py','stub_ds',str(bp),str(td/'out3.npz'),'1']),patch.object(mod.CT,'load',lambda p:({},np.array([False,True]),{})):
        try:mod.main();res['bad_old_trunk_rejected']=False
        except AssertionError:res['bad_old_trunk_rejected']=True
text=json.dumps(res,ensure_ascii=False,indent=2)+'\n';print(text,end='')
if os.environ.get('CARBON_REVIEW_OUT'):(Path(os.environ['CARBON_REVIEW_OUT'])/'latest-full-heads-results.json').write_text(text)
