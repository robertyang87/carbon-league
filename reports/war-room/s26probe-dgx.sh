echo S26PDGX_BEGIN
date "+%F %T %Z"; hostname
echo "--- via L20 ssh to DGX (ZT) ---"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'hostname; df -h /home | tail -1; ls /home/arm/junyang/ ; ls -la ~/.ssh/ | head; du -sh /home/arm/junyang/carbon 2>/dev/null; ls /home/arm/junyang/carbon 2>/dev/null | head' 2>&1 | head -40
echo S26PDGX_END
