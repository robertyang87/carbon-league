"""Seal the final Master5s update against eight existing frozen score/replay manifests."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from appendix_batch import prepare
REVISION='0137949d48ff0b954835abbba9a68afa058a281e'
PREVIOUS='99919cbf5889caa98303091e146aa210fa035d18'
BATCHES=('.', 'appendix_499886f1','appendix_4bfdabf5','appendix_555c0c56','appendix_bf049043','appendix_bd3c0021','appendix_5484590b','appendix_99919cbf')
if __name__=='__main__':
    prepare(REVISION,[HERE.parent/b/'manifest.json' for b in BATCHES],HERE,expected=(4,0))
