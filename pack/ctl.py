#!/usr/bin/env python
"""EAGLE-V multi-GPU run controller. Everything is driven by config.yaml in the pack root.

  ./run.sh check                    environment, data and model check (no GPU work)
  ./run.sh models [PHASE ...]       download the models the phases need (hf download; set HF_ENDPOINT for a mirror)
  ./run.sh plan PHASE               print the task list of a phase (dry run)
  ./run.sh start PHASE [PHASE ...]  start a background scheduler that runs the phases in order
  ./run.sh status [-w SEC]          progress table (per task: done/total, rate, ETA), GPU usage
  ./run.sh stop                     stop the scheduler and every task it started (explicit PIDs)
  ./run.sh collect [PHASE ...]      merge shards and pack runs into return/<name>.tar.gz
  ./run.sh logs TASK                tail the log of one task (e.g. scale/qwen3vl2b/s0)

Tasks: a phase x model is split into S shards (S = #GPUs // gpus-per-model unless set); each task is one process
that loads the model once and walks all datasets of its shard. Tasks are resumable (finished rows are skipped), so
stop/start, a crash or a retry never repeats work. The scheduler hands GPUs to tasks in FIFO order.
"""
import argparse, glob, json, os, shutil, signal, subprocess, sys, tarfile, time
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- config
def load_cfg(path):
    C = yaml.safe_load(open(path))
    root = os.path.dirname(os.path.abspath(path))
    P = C.setdefault("paths", {})
    for k, d in (("data", "data"), ("models", "models"), ("runs", "runs"), ("code", "code"), ("return", "return")):
        P[k] = os.path.abspath(os.path.join(root, P.get(k, d)))
    C["_root"] = root
    return C


def gpu_list(C):
    g = C.get("gpus", "all")
    if g == "all":
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True).stdout
            return [int(x) for x in out.split()]
        except FileNotFoundError:
            return []
    return [int(x) for x in (g if isinstance(g, list) else str(g).split(","))]


def model_path(C, m):
    spec = C["models"][m]
    return spec.get("path") or os.path.join(C["paths"]["models"], m)


def phase_datasets(C, ph):
    d = ph.get("datasets", "all")
    return C["datasets"] if d == "all" else d


def tasks_for(C, name):
    """Deterministic task list for a phase. Shard counts are frozen in runs/<phase>/manifest.json on first start."""
    ph = C["phases"][name]; P = C["paths"]; py = C.get("python", sys.executable)
    rdir = os.path.join(P["runs"], name); ngpu = max(1, len(gpu_list(C)))
    man_p = os.path.join(rdir, "manifest.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {}
    code = os.path.join(P["code"], "eaglevlm")
    T = []
    for m in ph["models"]:
        spec = C["models"][m]; kind = ph.get("type", "judge")
        g = int(ph.get("gpus_per_task", spec.get("gpus", 1)))
        if kind == "train":
            folds = ph.get("folds", [0, 1])
            for f in folds:
                tid = f"{name}/{m}/f{f}"
                cmd = [py, "-m", "torch.distributed.run", f"--nproc_per_node={g}", f"--master_port={29500 + len(T)}",
                       os.path.join(code, "train_pref.py"), "--model", m, "--model-path", model_path(C, m),
                       "--out", os.path.join(rdir, "raw"), "--fold", str(f),
                       "--datasets", ",".join(phase_datasets(C, ph)),
                       "--progress", os.path.join(rdir, "state", f"{m}.f{f}.progress.json")]
                for k in ("epochs", "lr", "lora_r", "bs", "grad_accum", "eps", "bothbad_tie", "max_train", "max_eval"):
                    if k in ph:
                        cmd += ["--" + k.replace("_", "-"), str(ph[k])]
                T.append(dict(id=tid, model=m, gpus=g, cmd=cmd, log=os.path.join(rdir, "logs", f"{m}.f{f}.log"),
                              progress=os.path.join(rdir, "state", f"{m}.f{f}.progress.json")))
            continue
        S = man.get(m) or int(ph.get("shards", 0)) or max(1, ngpu // g)
        man[m] = S
        for i in range(S):
            tid = f"{name}/{m}/s{i}"
            plan = [{"items": os.path.join(P["data"], "items", f"{d}.jsonl"),
                     "out": os.path.join(rdir, "raw", f"{d}__{m}.s{i}of{S}.jsonl")} for d in phase_datasets(C, ph)]
            plan_p = os.path.join(rdir, "plans", f"{m}.s{i}.json")
            script = "describe.py" if kind == "describe" else "judge_bench.py"
            cmd = [py, os.path.join(code, script), "--model", m, "--model-path", model_path(C, m), "--plan", plan_p,
                   "--bs", str(ph.get("bs", spec.get("bs", C.get("defaults", {}).get("bs", 8)))),
                   "--shard", str(i), "--nshard", str(S),
                   "--progress", os.path.join(rdir, "state", f"{m}.s{i}.progress.json")]
            if g > 1:
                cmd += ["--device-map", "auto"]
            if ph.get("limit"):
                cmd += ["--limit", str(ph["limit"])]
            if kind == "describe":
                cmd += ["--max-new-tokens", str(ph.get("max_new_tokens", 256))]
            else:
                cmd += ["--conds", ph["conds"]]
                if ph.get("hidden"):
                    cmd += ["--hidden", str(ph["hidden"])]
                if ph.get("tiling") and "internvl" in m:
                    cmd += ["--tiling"]
                if ph.get("desc_from"):
                    src = ph.get("describer") or m
                    cmd += ["--desc", os.path.join(P["runs"], ph["desc_from"], "raw", f"*__{src}.s*.jsonl")]
            T.append(dict(id=tid, model=m, gpus=g, cmd=cmd, plan=plan, plan_path=plan_p,
                          log=os.path.join(rdir, "logs", f"{m}.s{i}.log"),
                          progress=os.path.join(rdir, "state", f"{m}.s{i}.progress.json")))
    return T, man


# ----------------------------------------------------------------------------- scheduler daemon
def daemon(C, phases):
    P = C["paths"]; os.makedirs(P["runs"], exist_ok=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    env0 = dict(os.environ, EAGLEVLM_DATA=P["data"], PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                PYTHONPATH=os.path.join(P["code"], "eaglevlm"))
    gpus = gpu_list(C); retries = int(C.get("retries", 2))
    for name in phases:
        rdir = os.path.join(P["runs"], name)
        for sub in ("raw", "plans", "state", "logs"):
            os.makedirs(os.path.join(rdir, sub), exist_ok=True)
        T, man = tasks_for(C, name)
        json.dump(man, open(os.path.join(rdir, "manifest.json"), "w"))
        st_p = os.path.join(rdir, "tasks.json")
        old = {t["id"]: t for t in json.load(open(st_p))} if os.path.exists(st_p) else {}
        for t in T:
            if "plan" in t:
                json.dump(t["plan"], open(t["plan_path"], "w"))
            o = old.get(t["id"], {})
            t["state"] = "done" if o.get("state") == "done" else "pending"
            t["tries"] = 0; t["pid"] = None; t["devices"] = []
        free = list(gpus); running = {}

        def save():
            tmp = st_p + ".tmp"
            json.dump([{k: v for k, v in t.items() if k not in ("plan", "proc")} for t in T], open(tmp, "w"), indent=1)
            os.replace(tmp, st_p)

        save()
        while not stop["flag"]:
            for pid, t in list(running.items()):
                rc = t["proc"].poll()
                if rc is None:
                    continue
                del running[pid]; free += t["devices"]; t.pop("proc")
                t["ended"] = time.strftime("%Y-%m-%d %H:%M:%S"); t["rc"] = rc
                if rc == 0:
                    t["state"] = "done"
                else:
                    t["tries"] += 1
                    t["state"] = "pending" if t["tries"] <= retries else "failed"
                save()
            pend = [t for t in T if t["state"] == "pending"]
            if pend and len(free) >= pend[0]["gpus"]:                     # strict FIFO: big tasks are not starved
                t = pend[0]; dev = sorted(free)[: t["gpus"]]
                for d in dev:
                    free.remove(d)
                env = dict(env0, CUDA_VISIBLE_DEVICES=",".join(map(str, dev)))
                lf = open(t["log"], "a"); lf.write(f"\n=== {time.strftime('%F %T')} start on GPU {dev}\n"); lf.flush()
                p = subprocess.Popen(t["cmd"], env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=P["data"],
                                     start_new_session=True)
                t.update(state="running", pid=p.pid, devices=dev, started=time.strftime("%Y-%m-%d %H:%M:%S"), proc=p)
                running[p.pid] = t; save()
                continue
            if not pend and not running:
                break
            time.sleep(5)
        if stop["flag"]:
            for pid, t in running.items():
                try:
                    os.killpg(pid, signal.SIGTERM)     # the task's own process group (pgid == its recorded pid)
                except ProcessLookupError:
                    pass
            time.sleep(10)
            for pid, t in running.items():
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                t["state"] = "pending"; t["devices"] = []; t.pop("proc", None)
            save(); return


# ----------------------------------------------------------------------------- commands
def cmd_start(C, a):
    for ph in a.phases:
        if ph not in C["phases"]:
            sys.exit(f"unknown phase {ph}; known: {list(C['phases'])}")
    pid_p = os.path.join(C["paths"]["runs"], "scheduler.pid")
    if os.path.exists(pid_p):
        pid = int(open(pid_p).read().split()[0])
        if os.path.exists(f"/proc/{pid}"):
            sys.exit(f"scheduler already running (pid {pid}); ./run.sh stop first")
    os.makedirs(C["paths"]["runs"], exist_ok=True)
    log = open(os.path.join(C["paths"]["runs"], "scheduler.log"), "a")
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--config", a.config, "_daemon", *a.phases],
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    open(pid_p, "w").write(f"{p.pid} {' '.join(a.phases)}\n")
    print(f"scheduler pid {p.pid}: phases {a.phases} on GPUs {gpu_list(C)}\n  watch:  ./run.sh status -w 30\n  stop:   ./run.sh stop")


def cmd_stop(C, a):
    pid_p = os.path.join(C["paths"]["runs"], "scheduler.pid")
    if not os.path.exists(pid_p):
        print("no scheduler pid file"); return
    pid = int(open(pid_p).read().split()[0])
    try:
        os.kill(pid, signal.SIGTERM); print(f"SIGTERM -> scheduler {pid}; it stops its tasks (about 10 s)")
        for _ in range(40):
            if not os.path.exists(f"/proc/{pid}"):
                break
            time.sleep(1)
    except ProcessLookupError:
        print("scheduler not running")
    os.remove(pid_p)
    # tasks left over from a scheduler that died: kill the recorded task pids
    for st in glob.glob(os.path.join(C["paths"]["runs"], "*", "tasks.json")):
        for t in json.load(open(st)):
            if t.get("state") == "running" and t.get("pid") and os.path.exists(f"/proc/{t['pid']}"):
                try:
                    os.killpg(t["pid"], signal.SIGKILL); print("killed orphan task", t["id"], t["pid"])
                except ProcessLookupError:
                    pass


def fmt_s(s):
    if s is None:
        return "-"
    s = int(s); return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def status_once(C):
    P = C["paths"]; lines = []
    pid_p = os.path.join(P["runs"], "scheduler.pid")
    if os.path.exists(pid_p):
        pid, *ph = open(pid_p).read().split()
        alive = os.path.exists(f"/proc/{pid}")
        lines.append(f"scheduler pid {pid} ({'running' if alive else 'DEAD'}) phases {ph}")
    else:
        lines.append("scheduler: not running")
    for st in sorted(glob.glob(os.path.join(P["runs"], "*", "tasks.json"))):
        T = json.load(open(st)); name = os.path.basename(os.path.dirname(st))
        cnt = {s: sum(t["state"] == s for t in T) for s in ("done", "running", "pending", "failed")}
        lines.append(f"\n[{name}] " + " ".join(f"{k}={v}" for k, v in cnt.items()))
        lines.append(f"  {'task':34s} {'gpu':7s} {'state':8s} {'done/total':>17s} {'%':>6s} {'rate/s':>7s} {'ETA':>7s}")
        tot_d = tot_t = 0
        for t in T:
            pr = json.load(open(t["progress"])) if os.path.exists(t["progress"]) else {}
            d, n = pr.get("done", 0), pr.get("total", 0); tot_d += d; tot_t += n
            pct = f"{100 * d / n:5.1f}" if n else "  -"
            eta = fmt_s(pr.get("eta_s")) if t["state"] == "running" else ""
            lines.append(f"  {t['id']:34s} {','.join(map(str, t.get('devices') or [])):7s} {t['state']:8s} "
                         f"{d:>8}/{n:<8} {pct:>6} {pr.get('rate', ''):>7} {eta:>7}"
                         + (f"  rc={t.get('rc')} see {t['log']}" if t["state"] == "failed" else ""))
        if tot_t:
            lines.append(f"  {'total (tasks started so far)':34s} {'':16s} {tot_d:>8}/{tot_t:<8} {100 * tot_d / tot_t:5.1f}")
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
        lines.append("\nGPU  util%  mem(GB)")
        lines += [f"  {x.split(',')[0]:>2} {x.split(',')[1]:>5}  {int(x.split(',')[2]) / 1024:5.1f}/{int(x.split(',')[3]) / 1024:.0f}" for x in g]
    except FileNotFoundError:
        pass
    return "\n".join(lines)


def cmd_status(C, a):
    if not a.w:
        print(status_once(C)); return
    try:
        while True:
            s = status_once(C); os.system("clear"); print(time.strftime("%F %T"), "\n" + s); time.sleep(a.w)
    except KeyboardInterrupt:
        pass


def cmd_logs(C, a):
    ph, m, s = a.task.split("/")
    f = os.path.join(C["paths"]["runs"], ph, "logs", f"{m}.{s}.log")
    os.execvp("tail", ["tail", "-n", str(a.n), "-f", f] if a.f else ["tail", "-n", str(a.n), f])


def merge_jsonl(files, out, key):
    rows = {}
    for f in sorted(files):
        for l in open(f):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue                         # a line cut by a kill
            k = key(d)
            if k not in rows or ("lp" in d and "lp" not in rows[k]) or ("ctx" in d):
                rows[k] = d
    with open(out, "w") as fo:
        for d in rows.values():
            fo.write(json.dumps(d) + "\n")
    return len(rows)


def cmd_collect(C, a):
    P = C["paths"]; phases = a.phases or [os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(P["runs"], "*", "tasks.json"))]
    stamp = time.strftime("%Y%m%d_%H%M"); os.makedirs(P["return"], exist_ok=True)
    tar_p = os.path.join(P["return"], f"eaglev_{'-'.join(phases)}_{stamp}.tar.gz")
    with tarfile.open(tar_p, "w:gz") as tar:
        for name in phases:
            rdir = os.path.join(P["runs"], name); mdir = os.path.join(rdir, "merged"); os.makedirs(mdir, exist_ok=True)
            groups = {}
            for f in glob.glob(os.path.join(rdir, "raw", "*.jsonl")):
                b = os.path.basename(f)
                base = b.split(".s")[0] if ".s" in b and "of" in b else b.split(".f")[0] if "_ft.f" in b else b[:-6]
                groups.setdefault(base, []).append(f)
            n = 0
            for base, fs in groups.items():
                key = (lambda d: d["id"]) if C["phases"][name].get("type") == "describe" else (lambda d: (d["id"], d["cond"], d["order"]))
                n += merge_jsonl(fs, os.path.join(mdir, base + ".jsonl"), key)
            print(f"[{name}] merged {len(groups)} files, {n} rows -> {mdir}")
            for sub in ("merged", "logs", "state"):
                if os.path.isdir(os.path.join(rdir, sub)):
                    tar.add(os.path.join(rdir, sub), arcname=f"{name}/{sub}")
            for f in ("tasks.json", "manifest.json"):
                if os.path.exists(os.path.join(rdir, f)):
                    tar.add(os.path.join(rdir, f), arcname=f"{name}/{f}")
            if a.hidden:
                for h in glob.glob(os.path.join(rdir, "raw", "*.hid")):
                    tar.add(h, arcname=f"{name}/hidden/{os.path.basename(h)}")
            if a.lora:
                for h in glob.glob(os.path.join(rdir, "raw", "lora_*")):
                    tar.add(h, arcname=f"{name}/lora/{os.path.basename(h)}")
        tar.add(a.config, arcname="config.yaml")
    sz = os.path.getsize(tar_p) / 1e6
    print(f"\nreturn package: {tar_p} ({sz:.1f} MB)\n  fetch from your machine:  rsync -avP <user>@<host>:{tar_p} .")


def cmd_plan(C, a):
    T, man = tasks_for(C, a.phase)
    print(f"phase {a.phase}: {len(T)} tasks, GPUs {gpu_list(C)}")
    for t in T:
        print(f"  {t['id']:34s} gpus={t['gpus']}\n    " + " ".join(t["cmd"]))


def needed_models(C, phases):
    ms = []
    for p in phases or C["phases"]:
        for m in C["phases"][p]["models"]:
            if m not in ms:
                ms.append(m)
    return ms


def cmd_models(C, a):
    os.makedirs(C["paths"]["models"], exist_ok=True)
    exe = shutil.which("hf") or shutil.which("huggingface-cli")
    if not exe:
        sys.exit("need the Hugging Face CLI: pip install -U 'huggingface_hub[cli]'")
    for m in needed_models(C, a.phases):
        dst = model_path(C, m)
        if os.path.exists(os.path.join(dst, "config.json")) and not glob.glob(os.path.join(dst, "*.incomplete")):
            print(f"[ok] {m} at {dst}"); continue
        repo = C["models"][m]["repo"]; print(f"[get] {m} <- {repo}", flush=True)
        cmd = [exe, "download", repo, "--local-dir", dst] + (["--max-workers", "8"] if exe.endswith("hf") else [])
        env = dict(os.environ); env.pop("HF_HUB_OFFLINE", None)
        if subprocess.run(cmd, env=env).returncode != 0:
            print(f"[FAIL] {m}")


def cmd_check(C, a):
    ok = True; py = C.get("python", sys.executable)
    r = subprocess.run([py, "-c", "import torch,transformers,PIL,yaml;print(torch.__version__,transformers.__version__,"
                        "torch.cuda.device_count())"], capture_output=True, text=True)
    print("python:", py, "->", r.stdout.strip() or r.stderr.strip()[-300:]); ok &= r.returncode == 0
    r = subprocess.run([py, "-c", "import peft;print(peft.__version__)"], capture_output=True, text=True)
    print("peft (phase D only):", r.stdout.strip() or "MISSING")
    print("GPUs:", gpu_list(C))
    D = C["paths"]["data"]; miss = 0; nimg = 0
    for d in C["datasets"]:
        f = os.path.join(D, "items", f"{d}.jsonl")
        if not os.path.exists(f):
            print("  MISSING items", f); ok = False; continue
        n = 0
        for l in open(f):
            r_ = json.loads(l); n += 1
            paths = list(r_["context"]) + (list(r_["cand"]) if r_["kind"] == "img" else [])
            for p in paths:
                nimg += 1
                if not os.path.exists(p if os.path.isabs(p) else os.path.join(D, p)):
                    miss += 1
        print(f"  {d:14s} {n:6d} pairs")
    print(f"images referenced {nimg}, missing {miss}"); ok &= miss == 0
    for m in needed_models(C, a.phases):
        p = model_path(C, m); have = os.path.exists(os.path.join(p, "config.json"))
        print(f"  model {m:16s} {'ok ' if have else 'MISSING'} {p}")
    print("\nCHECK", "PASSED" if ok else "FAILED (see above)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("start"); s.add_argument("phases", nargs="+")
    s = sp.add_parser("status"); s.add_argument("-w", type=int, default=0, help="refresh every W seconds")
    sp.add_parser("stop")
    s = sp.add_parser("collect"); s.add_argument("phases", nargs="*"); s.add_argument("--hidden", action="store_true")
    s.add_argument("--lora", action="store_true")
    s = sp.add_parser("plan"); s.add_argument("phase")
    s = sp.add_parser("models"); s.add_argument("phases", nargs="*")
    s = sp.add_parser("check"); s.add_argument("phases", nargs="*")
    s = sp.add_parser("logs"); s.add_argument("task"); s.add_argument("-n", type=int, default=30); s.add_argument("-f", action="store_true")
    s = sp.add_parser("_daemon"); s.add_argument("phases", nargs="+")
    a = ap.parse_args(); a.config = os.path.abspath(a.config); C = load_cfg(a.config)
    if a.cmd == "_daemon":
        daemon(C, a.phases)
    else:
        globals()["cmd_" + a.cmd](C, a)


if __name__ == "__main__":
    main()
