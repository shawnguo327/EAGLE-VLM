#!/bin/bash
# a100.sh with retries (sshd on the rental drops some new connections under load)
T=/tmp/claude-1000/-home-shawn-bjserver-papers-VLM/d20b9cde-2ffa-4407-8c30-50b441fc6461/a100
for i in 1 2 3 4 5 6 7 8; do
  out=$(timeout ${A100_TIMEOUT:-300} $T/a100.sh "$@" 2>/tmp/a100r_err_$$); rc=$?
  if [ $rc -ne 255 ]; then printf '%s\n' "$out"; rm -f /tmp/a100r_err_$$; exit $rc; fi
  sleep $((i * 8))
done
cat /tmp/a100r_err_$$ >&2; rm -f /tmp/a100r_err_$$; exit 255
