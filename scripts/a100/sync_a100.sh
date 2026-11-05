#!/bin/bash
# every 30 min: copy A100 result folders (out*, small jsonl) to bjserver judgebench_local/a100_sync/ (tar stream via this
# machine; no keys are installed anywhere).  Raw datasets / models are NOT synced.
N=/tmp/claude-1000/-home-shawn-bjserver-papers-VLM/d20b9cde-2ffa-4407-8c30-50b441fc6461/a100
while true; do
  ts=$(date +%F_%T)
  timeout 1200 $N/a100.sh 'cd /media/vipuser/Data/eaglevlm && tar cf - out* 2>/dev/null' | \
    ssh bjserver 'mkdir -p /media/shawn/C/papers/VLM/judgebench_local/a100_sync && cd /media/shawn/C/papers/VLM/judgebench_local/a100_sync && tar xf - && du -sh . | cut -f1' \
    | sed "s/^/$ts synced /"
  sleep 1800
done
