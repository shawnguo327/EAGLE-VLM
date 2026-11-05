#!/bin/bash
# every 30 min: A100 result folders (out*, without model weights) -> bjserver judgebench_local/a100_sync/.
# Integrity-safe: archive on the A100 + md5, pulled with retries, verified, then extracted on bjserver; a failed transfer
# leaves the previous copy untouched (the old streaming version left truncated files on dropped connections).
D=/home/shawn/.claude/jobs/d20b9cde/tmp; L=$D/sync_tmp; mkdir -p $L
while true; do
  ts=$(date +%F_%T); ok=0
  sum=$(A100_TIMEOUT=600 $D/a100r.sh 'cd /media/vipuser/Data/eaglevlm && tar czf /media/vipuser/Data/tmp/out_sync.tgz --exclude="*.safetensors" --exclude="*.bin" --exclude="*.pt" out* 2>/dev/null; md5sum /media/vipuser/Data/tmp/out_sync.tgz | cut -c1-32')
  for i in 1 2 3 4 5 6; do
    rm -f $L/out_sync.tgz
    timeout 900 $D/a100pull.sh /media/vipuser/Data/tmp/out_sync.tgz $L/out_sync.tgz 2>/dev/null
    if [ -n "$sum" ] && [ "$(md5sum $L/out_sync.tgz 2>/dev/null | cut -c1-32)" = "$sum" ]; then ok=1; break; fi
    sleep $((i * 20))
  done
  if [ $ok = 1 ]; then
    scp -q $L/out_sync.tgz bjserver:/tmp/out_sync_d20b.tgz && \
    ssh bjserver 'cd /media/shawn/C/papers/VLM/judgebench_local && mkdir -p a100_sync && gzip -t /tmp/out_sync_d20b.tgz && tar xzf /tmp/out_sync_d20b.tgz -C a100_sync && rm /tmp/out_sync_d20b.tgz && du -sh a100_sync | cut -f1' | sed "s/^/$ts synced (verified) /"
  else
    echo "$ts sync FAILED (transfer could not be verified; previous copy kept)"
  fi
  sleep 1800
done
