# mkref.sh REV OUT.py CONF  -- build a pack from the code at git revision REV (reference for parity checks)
set -e; cd /Users/baggioustb/dev/carbon/carbon-league/code; R=$(mktemp -d ${TMPDIR:-/tmp}/ref.XXXX)
for f in cnet2_feat.py cnet2_np.py fastsim.py cnet2_search.py cnet2_search_pack.py cnet2_pack.py; do git show $1:code/$f > $R/$f; done
W=$PWD/../data/weights; export CNET2_QMODE=int8 CNET2_QMODE_ROLL=f16
(cd $R && python3 cnet2_search_pack.py out.py "$3" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz >/dev/null); cp $R/out.py $2; rm -rf $R
