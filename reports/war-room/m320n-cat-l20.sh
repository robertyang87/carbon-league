#!/bin/bash
MARK=M320CAT1
echo "${MARK}_BEGIN $(date '+%F %T %Z') host=$(hostname)"
echo 'ZWNobyAiQ0FULURHWCAkKGRhdGUgJyslRiAlVCAlWicpIgpiYXNlNjQgLXcwIC9ob21lL2FybS9qdW55YW5nL2NhcmJvbi9jb2RlL2J1aWxkX205eGpfdy5weTsgZWNobwplY2hvICJDQVQtREdYLURPTkUiCg==' | base64 -d > /tmp/m320_inner_M320CAT1.sh
wc -c /tmp/m320_inner_M320CAT1.sh
timeout -k 30 200 ssh -i ~/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' < /tmp/m320_inner_M320CAT1.sh 2>&1
echo "ssh-rc=$?"
echo "${MARK}_END"
