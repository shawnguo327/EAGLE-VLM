"""Freeze the raw judge outputs and the item files into one self-contained, compressed snapshot.

    python -m eaglevlm.pipeline.freeze configs/freeze.yaml

Reads every (dataset, judge) through the same loader the analyses use (ordered sources, first source wins, malformed
lines skipped) and writes
    <snapshot>/outputs/dataset=<ds>/judge=<judge>.parquet   one row per item x condition x order, columns:
                                                            item_id, cond, order, then the normalised probabilities of
                                                            every answer token the prompt allows (A, B, T, 1..9)
    <snapshot>/items/<file>.jsonl.gz                        the item files (human labels) that were read
    <snapshot>/manifest.csv                                 source files (path, lines, md5) and snapshot files (rows, md5)
Analyses can then run from the snapshot alone: set paths.snapshot in configs/paths.*.yaml (Store reads it instead of the
scattered JSONL files)."""
import csv, gzip, hashlib, os, shutil, sys
import yaml

TOKENS = ["A", "B", "T"] + [str(d) for d in range(1, 10)]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import pyarrow as pa, pyarrow.parquet as pq
    from eaglevlm.pipeline.io import Store
    from eaglevlm.pipeline.run import load_cfg
    cfg = load_cfg(sys.argv[1]); out = cfg["snapshot"]
    os.makedirs(f"{out}/outputs", exist_ok=True); os.makedirs(f"{out}/items", exist_ok=True)
    st = Store(cfg["paths"]["items"], cfg["paths"]["outputs"]); files = []
    for ds in cfg["datasets"]:
        st.items(ds)
        for j in cfg["judges"]:
            R = st.load(ds, j)
            if not R:
                continue
            cols = {k: [] for k in ["item_id", "cond", "order"] + TOKENS}
            for i in sorted(R):
                for (c, o), p in sorted(R[i].items()):
                    cols["item_id"].append(i); cols["cond"].append(c); cols["order"].append(o)
                    for t in TOKENS:
                        cols[t].append(p.get(t))
            d = f"{out}/outputs/dataset={ds}"; os.makedirs(d, exist_ok=True); p = f"{d}/judge={j}.parquet"
            pq.write_table(pa.table({k: pa.array(v, type=(pa.string() if k in ("item_id", "cond", "order") else pa.float64())) for k, v in cols.items()}),
                           p, compression="zstd")
            files.append(dict(kind="snapshot", path=os.path.relpath(p, out), lines=len(cols["item_id"]), md5=md5(p)))
            print(ds, j, len(cols["item_id"]), flush=True)
    for v in st.manifest.values():
        if v["kind"] == "items":
            dst = f"{out}/items/{os.path.basename(v['path'])}.gz"
            with open(v["path"], "rb") as fi, gzip.open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            files.append(dict(kind="snapshot_items", path=os.path.relpath(dst, out), lines=v["lines"], md5=md5(dst)))
    with open(f"{out}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "path", "lines", "md5"]); w.writeheader()
        for v in sorted(st.manifest.values(), key=lambda v: (v["kind"], v["path"])):
            w.writerow(dict(v, kind="source_" + v["kind"]))
        for v in files:
            w.writerow(v)
    yaml.safe_dump(cfg, open(f"{out}/freeze_config.yaml", "w"), sort_keys=False)
    print("snapshot:", out)


if __name__ == "__main__":
    main()
