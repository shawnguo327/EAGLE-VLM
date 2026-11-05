#!/bin/bash
D=$(dirname "$0"); for i in 1 2 3 4 5 6 7 8; do timeout 300 $D/a100scp.sh "$@" 2>/dev/null && exit 0; sleep $((i * 8)); done; exit 1
