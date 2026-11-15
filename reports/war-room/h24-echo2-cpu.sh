echo H24E2CPU1948_BEGIN
date "+%F %T %Z"
lscpu | grep -iE "model name|architecture|vendor id|^CPU\(s\)" | head -8
grep -m2 -iE "model name|hardware" /proc/cpuinfo || true
echo H24E2CPU1948_END
