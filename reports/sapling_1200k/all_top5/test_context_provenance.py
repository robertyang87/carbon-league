"""Check address and registration identity through both catalog entry points."""
import json
from pathlib import Path
import tempfile
import unittest

if __package__:
    from .context import Corpus
else:
    from context import Corpus


class IndexProvenance(unittest.TestCase):
    def test_final_appendix_index_and_renamed_copy_use_the_same_scorecards(self):
        folder=Path(__file__).resolve().parent
        manifest=folder/"appendix_4bfdabf5/manifest.json"
        index=folder/"context_appendix_4bfdabf5_index.tsv.gz"
        explicit=Corpus(manifest=manifest)
        cards=json.loads(manifest.read_bytes())["games"]
        with tempfile.TemporaryDirectory() as tmp:
            renamed=Path(tmp)/"my-index.tsv.gz"
            renamed.write_bytes(index.read_bytes())
            for path in (index,renamed):
                corpus=Corpus(index=path)
                self.assertEqual(corpus.size,8970)
                for r in cards:
                    gid=r["id"]
                    self.assertEqual(corpus.provenance(gid),explicit.provenance(gid))
                    self.assertEqual(corpus.provenance(gid)["seat_names"],r["seat_names"])
                    for seat in (0,1):
                        for turn in (0,r["transitions"]-1):
                            key=(gid,turn,seat)
                            i=corpus.ordinal(*key)
                            self.assertEqual(corpus.locate(i),key)
                            self.assertEqual(explicit.locate(i),key)


if __name__ == "__main__":
    unittest.main()
