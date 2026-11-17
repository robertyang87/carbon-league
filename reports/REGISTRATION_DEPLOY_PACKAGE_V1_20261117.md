# REGISTRATION — Standard deployment package v1 (2026-11-17)

Owner rule 2026-11-17 14:47: new seats deploy from a standard package, never field-assembled environments. War-room dispatch (sole executor DGX), delivered same day.

## Package
- Name: carbon_deploy_pkg_v1_20261117.tar.gz (dir form: deploy_pkg_v1/)
- Location: DGX /home/arm/junyang/k0_dist/
- Version: 1.0.0 (VERSION file in package)
- Tarball: 81,536,616 B, md5 4e662646d6ac0217d0fa9eeeaf6c1f77
- MANIFEST.md5 (18 files, all verified at build): md5 ff32e2d401dff1e72a14bc3e0139dacf

## Contents
- venv.tar.gz (md5 da5d9ea313a5ee9928bd21aa47fcaa7b): prebuilt venv from spot-44's in-service venv — Python 3.12, numpy 2.5.4, scipy 1.18.1.
- code/: pinned annotation code set (k0_annotate.py 7f567a2e, stage0_teacher.py c6ade5ce, stage1_teacher_gate.py e922d9d8, cnet2_feat.py 2a45b08b, cnet2_search.py 94f7fb64, cnet2_np.py d8c15ad4, fastsim.py aefe6c83, eval_v53.py 45792dfb, env_src/ incl. carbon.json cd55531b).
- weights/cnet2_m3s2r1_heads_v2.npz (06a03d9c).
- iso_shard0.json (ee13caea), iso_test323.json (d5c31b1c).
- bench/: smoke replay 00f7e221872bf7bc75a3847676e17b56-1.json.gz (fdf1cb89) + games_bench.json.
- deploy.sh, README.md, VERSION, MANIFEST.md5.

## deploy.sh contract
1. Mask unattended-upgrades + apt-daily.service + apt-daily-upgrade.service + apt-daily.timer + apt-daily-upgrade.timer; verify each is-enabled == masked; verify dpkg lock free (fail-closed).
2. Unpack venv/code/weights/iso/bench into /root/carbon/ — no apt, no mirrors, no network beyond transferring the package.
3. Import hard-verify (numpy/scipy import + versions).
4. Environment smoke: annotate the bundled bench replay, window [100,101), single process; require rc=0 and exactly 2 roots in the produced shard.
5. Print DEPLOY_OK only after all steps pass; failures surface verbatim (no filtering).

## Verification status at registration
- Contents md5-verified against MANIFEST (18/18 OK); deploy.sh bash -n syntax OK; the same venv bits were functionally proven on spot-48 (transferred from spot-44, IMPORT_OK, then ran the 20-root bench to completion: 9.81 s/root, config 23f7e110 verified).
- The full deploy.sh end-to-end run on a virgin seat is still outstanding: the only idle x86_64 seat (spot-48) is under the owner's no-teardown standby order, so the first full script run happens on the next new seat (or on spot-48 if the owner orders a redeploy there).
