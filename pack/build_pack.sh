#!/usr/bin/env bash
# Build the transfer folder on bjserver: code snapshot (git HEAD) + items + images.
#   capped sets   from $EAGLEVLM_DATA        (items/<ds>.jsonl)
#   full sets     from $EAGLEVLM_FULL        (items/<ds>.jsonl -> <ds>_full.jsonl; pp_full and text sets keep their names)
# Absolute image paths (Place Pulse) are localised to images/pp/ and rewritten as relative paths.
# Usage: bash pack/build_pack.sh [PACK_DIR]      (default /media/shawn/C/papers/VLM/eaglev_pack)
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${EAGLEVLM_DATA:-/media/shawn/C/papers/VLM/judgebench}
FULL=${EAGLEVLM_FULL:-/media/shawn/C/papers/VLM/judgebench_full}
PACK=${1:-/media/shawn/C/papers/VLM/eaglev_pack}
mkdir -p $PACK/data/items $PACK/data/images $PACK/code $PACK/models $PACK/runs $PACK/return
rm -rf $PACK/code/*; git -C $REPO archive HEAD | tar -x -C $PACK/code
echo "code: $(git -C $REPO log --oneline -1)" > $PACK/VERSION
cp $REPO/pack/run.sh $REPO/pack/setup_env.sh $REPO/pack/requirements.txt $REPO/pack/README.md $REPO/pack/config.pro6000.yaml $PACK/
[ -f $PACK/config.yaml ] && echo "keeping existing $PACK/config.yaml (delete it to take the repo version)" || cp $REPO/pack/config.yaml $PACK/config.yaml
chmod +x $PACK/run.sh $PACK/setup_env.sh
# images: hard links when on the same filesystem (no extra space); identical names in both roots are the same image
for R in $DATA $FULL; do [ -d $R/images ] && (cp -aln $R/images/. $PACK/data/images/ 2>/dev/null || rsync -a --ignore-existing $R/images/ $PACK/data/images/); done
python3 - "$DATA" "$FULL" "$PACK" <<'PY'
import json, os, shutil, sys
data, full, pack = sys.argv[1:]
KEEP = {"pp_full", "mtbench_human", "arena55k"}
srcs = [(data, f, f) for f in sorted(os.listdir(f"{data}/items")) if f.endswith(".jsonl")]
if os.path.isdir(f"{full}/items"):
    srcs += [(full, f, f if f[:-6] in KEEP else f[:-6] + "_full.jsonl") for f in sorted(os.listdir(f"{full}/items")) if f.endswith(".jsonl")]
for root, f, name in srcs:
    out = open(f"{pack}/data/items/{name}", "w"); n = c = 0; kinds = {}
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
    for l in open(f"{root}/items/{f}"):
        r = json.loads(l)
        r["context"] = [fix(p) for p in r["context"]]
        if r["kind"] == "img":
            r["cand"] = [fix(p) for p in r["cand"]]
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    ties = 0
    print(f"  items/{name:28s} {n:6d} pairs {kinds}" + (f", {c} images localised" if c else ""))
PY
du -sh --apparent-size $PACK/data $PACK/code | sed 's/^/  /'
echo "pack ready: $PACK"
echo "transfer:   rsync -avP --exclude models --exclude runs --exclude return $PACK/ <user>@<h100-host>:<dest>/eaglev_pack/"
