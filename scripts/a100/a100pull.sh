#!/bin/bash
# scp FROM the A100 box (same settings as a100.sh): a100pull.sh <remote path> <local path>
T=/tmp/claude-1000/-home-shawn-bjserver-papers-VLM/d20b9cde-2ffa-4407-8c30-50b441fc6461/a100
H=$(grep -oE "root@[0-9.]+" $T/a100.sh); P=$(grep -oE "\-p [0-9]+" $T/a100.sh | cut -d" " -f2)
SSH_ASKPASS=$T/a100_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 exec scp -o ProxyCommand="python3 $T/mssproxy.py %h %p 1200" -o KexAlgorithms=ecdh-sha2-nistp256 -o ConnectTimeout=20 -o PubkeyAuthentication=no -P $P "$H:$1" "$2"
