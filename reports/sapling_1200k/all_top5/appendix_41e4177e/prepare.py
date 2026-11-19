"""Download exactly the sixteen new score cards and seal their four independent identity keys."""
from intake import HERE, TOP, REV, BATCHES
from appendix_batch import prepare
if __name__=='__main__':
    prepare(REV,[TOP/b/'manifest.json' for b in BATCHES],HERE,expected=(16,0))
