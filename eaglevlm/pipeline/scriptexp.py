"""'script' experiments: run an existing analysis script from scripts/ with the arguments given in the YAML, writing its
output into the results directory, then convert every markdown table it printed into CSV (one file per table, with the
section heading it belongs to), so that all results - including the section-2 mechanism analyses that were not ported
into the pipeline - come out of one command with the same provenance files (config, env, script md5)."""
import csv, hashlib, os, re, subprocess, sys


def md_tables_to_csv(md_path, out_dir):
    stem = os.path.splitext(os.path.basename(md_path))[0]; heads = []; tables = []; cur = None
    for line in open(md_path):
        line = line.rstrip("\n")
        if line.startswith("#"):
            lvl = len(line) - len(line.lstrip("#")); heads = heads[: lvl - 1] + [line.lstrip("#").strip()]
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = dict(heading=" / ".join(heads), header=cells, rows=[])
            elif not all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                cur["rows"].append(cells)
        elif cur is not None:
            tables.append(cur); cur = None
    if cur is not None:
        tables.append(cur)
    index = []
    for k, t in enumerate(tables):
        p = os.path.join(out_dir, f"{stem}__t{k:02d}.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["section"] + t["header"])
            for r in t["rows"]:
                w.writerow([t["heading"]] + r)
        index.append(dict(file=os.path.basename(p), section=t["heading"], columns=len(t["header"]), rows=len(t["rows"])))
    with open(os.path.join(out_dir, f"{stem}__tables.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "section", "columns", "rows"]); w.writeheader(); [w.writerow(r) for r in index]
    return len(tables)


def run(cfg, out_dir, log):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = os.path.join(repo, cfg["script"]); md5 = hashlib.md5(open(script, "rb").read()).hexdigest()
    outputs = []
    for job in cfg["jobs"]:
        out = os.path.join(out_dir, job["out"])
        cmd = [sys.executable, script] + [str(x) for x in job.get("args", [])] + [cfg.get("out_arg", "--out"), out]
        log("$ " + " ".join(cmd))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(repo, "scripts"), os.path.join(repo, "eaglevlm"), repo]))
        r = subprocess.run(cmd, cwd=cfg["cwd"], env=env, capture_output=True, text=True)
        open(out + ".stdout.txt", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd} failed ({r.returncode}); see {out}.stdout.txt")
        n = md_tables_to_csv(out, out_dir) if out.endswith(".md") else 0
        log(f"{job['out']}: {n} tables -> csv"); outputs.append(job["out"])
    return dict(script=cfg["script"], script_md5=md5, outputs=outputs)
