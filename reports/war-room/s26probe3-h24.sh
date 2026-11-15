echo S26P3H24_BEGIN
date "+%F %T %Z"
which rsync tar
timeout 120 du -sh /root/carbon/.venv 2>/dev/null
timeout 120 du -sh /root/carbon/code/* 2>/dev/null | sort -rh | head -8
timeout 60 du -sh "/root/carbon/code/root@47.116.17.244arbon" 2>/dev/null
echo S26P3H24_END
