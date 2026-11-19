"""Reconcile the replication by calling the identical event accounting from the original experiment."""
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXP=HERE.parent.parent
sys.path.insert(0,str(EXP/'direct_original/accounts'))
import reconcile
import importlib.util
spec=importlib.util.spec_from_file_location('direct_cash_turns',EXP/'direct_original/accounts/turns.py')
turns=importlib.util.module_from_spec(spec)
spec.loader.exec_module(turns)
reconcile.EXAM=HERE.parent
reconcile.HERE=HERE
turns.HERE=HERE
if __name__=='__main__':
    reconcile.main()
    turns.main()
