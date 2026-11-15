echo "COMMIT-DGX $(date '+%F %T %Z') host=$(hostname)"
SRC=/home/arm/junyang/carbon/code/build_m9xj_w.py
ls -l "$SRC" && md5sum "$SRC" || { echo NO-SRC-FILE; exit 1; }
URL=$(git -C /home/arm/junyang/carbon-repo remote get-url origin); echo "ORIGIN $URL"
rm -rf /tmp/m9wx_commit
timeout 120 git clone -q /home/arm/junyang/carbon-repo /tmp/m9wx_commit || { echo CLONE-FAIL; exit 1; }
cd /tmp/m9wx_commit || exit 1
git remote set-url origin "$URL"
timeout 60 git fetch origin cnet2-search-v12 || { echo FETCH-FAIL; exit 1; }
git checkout -q -B cnet2-search-v12 FETCH_HEAD || { echo CHECKOUT-FAIL; exit 1; }
echo "BASE $(git rev-parse HEAD)"
if [ -f code/build_m9xj_w.py ]; then echo ALREADY-IN-REPO; md5sum code/build_m9xj_w.py; fi
cp "$SRC" code/build_m9xj_w.py
git add code/build_m9xj_w.py
git -c user.name='muse-chain' -c user.email='muse-chain@local' commit -q -m 'code: add build_m9xj_w.py (weighted dataset builder written by the it260 chain on DGX; was DGX-local only)' || { echo COMMIT-FAIL; git status --short; exit 1; }
echo "NEWCOMMIT $(git rev-parse HEAD)"
timeout 120 git push origin cnet2-search-v12 || { echo PUSH-FAIL; exit 1; }
echo "PUSH-OK $(git rev-parse HEAD) file-md5 $(md5sum code/build_m9xj_w.py | cut -d' ' -f1)"
echo COMMIT-DGX-DONE
