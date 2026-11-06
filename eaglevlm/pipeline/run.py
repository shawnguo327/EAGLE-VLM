"""Run one analysis experiment from a YAML config and save its results as CSV.

    python -m eaglevlm.pipeline.run configs/exp/certify_main.yaml [--workers 32] [--results-root DIR]

Config keys (see configs/exp/*.yaml): name, type (certify | router | conventional), include (shared paths file),
paths.items / paths.outputs (ordered source directories), paths.results, datasets, judges, and the experiment settings
(methods, alphas, delta, split, n_splits, seed, min_coverage).
Output: <results>/<name>/<timestamp>/
    config.yaml     the fully resolved config
    splits.csv.gz   one row per dataset x judge x method x alpha x split (experiments with random splits)
    summary.csv     aggregated over splits (the numbers the tables use)
    manifest.csv    every input file read (path, lines, md5)
    env.json        git commit + dirty flag, package versions, host, start / end time
    run.log
and <results>/<name>/latest -> <timestamp>.  No model is run here: inputs are the saved judge outputs."""
import argparse, concurrent.futures as cf, csv, datetime, gzip, importlib, json, os, platform, socket, subprocess, sys
import yaml

TYPES = {"certify": "eaglevlm.pipeline.certify", "router": "eaglevlm.pipeline.router", "conventional": "eaglevlm.pipeline.conventional"}


def load_cfg(path):
    cfg = yaml.safe_load(open(path))
    inc = cfg.pop("include", None)
    if inc:
        base = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(path)), inc)))
        for k, v in base.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**v, **cfg[k]}
            else:
                cfg.setdefault(k, v)
    return cfg


def _unit(args):
    mod, cfg = args
    from eaglevlm.pipeline.io import Store
    st = Store(cfg["paths"]["items"], cfg["paths"]["outputs"], cfg["paths"].get("items_override"))
    logs = []
    rows = importlib.import_module(mod).run(cfg, st, logs.append)
    return rows, st.manifest, logs


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        top = subprocess.check_output(["git", "-C", here, "rev-parse", "--show-toplevel"], text=True).strip()
        c = subprocess.check_output(["git", "-C", top, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", top, "status", "--porcelain", "--", "eaglevlm", "configs"], text=True).strip())
        return c, dirty
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--results-root", default=None); a = ap.parse_args()
    cfg = load_cfg(a.config); mod = TYPES[cfg["type"]]
    root = a.results_root or cfg["paths"]["results"]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); out = os.path.join(root, cfg["name"], ts); os.makedirs(out)
    yaml.safe_dump(cfg, open(f"{out}/config.yaml", "w"), sort_keys=False, allow_unicode=True)
    commit, dirty = git_info()
    import numpy, scipy, sklearn
    env = dict(git_commit=commit, git_dirty=dirty, python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
               sklearn=sklearn.__version__, host=socket.gethostname(), command=" ".join(sys.argv), start=datetime.datetime.now().isoformat())
    log = open(f"{out}/run.log", "w")
    # units: (dataset, judge) for certify / conventional; dataset (all judges, needed for the jury) for router
    if cfg["type"] == "router":
        units = [dict(cfg, datasets=[d]) for d in cfg["datasets"]]
    else:
        units = [dict(cfg, datasets=[d], judges=[j]) for d in cfg["datasets"] for j in cfg["judges"]]
    workers = a.workers or cfg.get("workers", 1)
    rows, manifest = [], {}
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for r, m, lg in ex.map(_unit, [(mod, u) for u in units]):
            rows += r; manifest.update(m)
            for line in lg:
                log.write(line + "\n"); print(line, flush=True)
    M = importlib.import_module(mod)
    order = {d: k for k, d in enumerate(cfg["datasets"])}; jord = {j: k for k, j in enumerate(cfg["judges"])}
    rows.sort(key=lambda r: (order.get(r["dataset"], 99), jord.get(r["judge"], 99)))
    if rows and "split" in rows[0]:
        write_csv(f"{out}/splits.csv.gz", rows)
    summ = M.summarize(rows); summ.sort(key=lambda r: (order.get(r["dataset"], 99), jord.get(r["judge"], 99)))
    write_csv(f"{out}/summary.csv", summ)
    with open(f"{out}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "path", "lines", "md5"]); w.writeheader()
        for v in sorted(manifest.values(), key=lambda v: (v["kind"], v["path"])):
            w.writerow(v)
    env["end"] = datetime.datetime.now().isoformat(); env["rows"] = len(rows); env["summary_rows"] = len(summ)
    json.dump(env, open(f"{out}/env.json", "w"), indent=1)
    link = os.path.join(root, cfg["name"], "latest")
    if os.path.islink(link):
        os.remove(link)
    os.symlink(ts, link)
    print("results:", out)


if __name__ == "__main__":
    main()
