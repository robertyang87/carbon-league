echo __MARK___BEGIN
date "+%F %T %Z"
hostname
md5sum /root/carbon/code/bigeval.py /root/carbon/code/eval_v53.py 2>&1
echo "cash270 count in eval_v53:"; grep -c "cash270" /root/carbon/code/eval_v53.py || true
echo "cash270 count in bigeval:"; grep -c "cash270" /root/carbon/code/bigeval.py || true
echo __MARK___END
