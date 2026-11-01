#!/usr/bin/env bash
# Single entry point. Usage: ./run.sh {check|models|plan|start|status|stop|collect|logs} ...   (./run.sh -h for help)
cd "$(dirname "$0")"
PY=${CTL_PYTHON:-$(python -c 'import yaml' 2>/dev/null && echo python || echo python3)}
exec $PY code/pack/ctl.py --config config.yaml "$@"
