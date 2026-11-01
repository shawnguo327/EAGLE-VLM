#!/usr/bin/env bash
# Build the transfer folder on bjserver: code snapshot (git HEAD) + items + images (+ Place Pulse images used by pp).
# Usage: bash pack/build_pack.sh [PACK_DIR]      (default /media/shawn/C/papers/VLM/eaglev_pack)
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${EAGLEVLM_DATA:-/media/shawn/C/papers/VLM/judgebench}
PACK=${1:-/media/shawn/C/papers/VLM/eaglev_pack}
mkdir -p $PACK/data/items $PACK/data/images $PACK/code $PACK/models $PACK/runs $PACK/return
rm -rf $PACK/code/*; git -C $REPO archive HEAD | tar -x -C $PACK/code
echo "code: $(git -C $REPO log --oneline -1)" > $PACK/VERSION
cp $REPO/pack/run.sh $REPO/pack/setup_env.sh $REPO/pack/requirements.txt $REPO/pack/README.md $PACK/
[ -f $PACK/config.yaml ] && echo "keeping existing $PACK/config.yaml" || cp $REPO/pack/config.yaml $PACK/config.yaml
chmod +x $PACK/run.sh $PACK/setup_env.sh
# images: hard links when on the same filesystem (no extra space), copies otherwise
cp -al $DATA/images/. $PACK/data/images/ 2>/dev/null || rsync -a $DATA/images/ $PACK/data/images/
# items; pp uses absolute Place Pulse paths -> copy those images to images/pp/ and rewrite to relative paths
python3 - "$DATA" "$PACK" <<'PY'
import json, os, shutil, sys
data, pack = sys.argv[1:]
for f in sorted(os.listdir(f"{data}/items")):
    if not f.endswith(".jsonl"):
        continue
    out = open(f"{pack}/data/items/{f}", "w"); n = c = 0
    for l in open(f"{data}/items/{f}"):
        r = json.loads(l)
        def fix(p):
            global c
            if not os.path.isabs(p):
                return p
            d = f"{pack}/data/images/pp"; os.makedirs(d, exist_ok=True); dst = f"{d}/{os.path.basename(p)}"
            if not os.path.exists(dst):
                try: os.link(p, dst)
                except OSError: shutil.copy2(p, dst)
                c += 1
            return f"images/pp/{os.path.basename(p)}"
        r["context"] = [fix(p) for p in r["context"]]
        if r["kind"] == "img":
            r["cand"] = [fix(p) for p in r["cand"]]
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print(f"  items/{f}: {n} pairs" + (f", {c} images localised" if c else ""))
PY
du -sh --apparent-size $PACK/data $PACK/code | sed 's/^/  /'
echo "pack ready: $PACK"
echo "transfer:   rsync -avP --exclude models --exclude runs --exclude return $PACK/ <user>@<h100-host>:<dest>/eaglev_pack/"
