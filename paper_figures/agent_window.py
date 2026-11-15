"""Window figure, web-agent points at three targets (descriptive, 2026-11-15).  Same certified ship-or-reject pipeline as
arb_analysis_v3.run_cert, per source benchmark, alpha in {.1, .2, .3}.  Gain = correct certified decisions per 100 tasks with
pointwise scores + tie probability minus pointwise scores alone; mean over the four judges; 95% task bootstrap.
usage: python agent_window.py <items_dir> <raw_dir> <out.json>"""
import json, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/media/shawn/C/papers/VLM/agent_reward_bench")
import arb_analysis_v3 as A
if __name__ == "__main__":
    items, raw, outp = sys.argv[1:4]; JUD = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen3vl32b"]; BENCH = ["webarena", "visualwebarena", "workarena", "assistantbench"]; AL = [.1, .2, .3]
    jobs = []; meta = []
    for j in JUD:
        D, n_all, n_err = A.load(items, raw, j); print(j, "pairs", len(D["id"]), "of", n_all, "error rows", n_err, flush=True)
        for b in BENCH:
            jobs.append((D, D["bench"] == b, 100, AL, b)); meta.append((j, b))
    with Pool(16) as pool:
        O = pool.map(A.run_cert, jobs)
    rng = np.random.default_rng(0); out = {}
    for b in BENCH:
        for al in AL:
            diffs, base, pi = [], [], None
            for (j, bb), o in zip(meta, O):
                if bb != b: continue
                T = A.summarize(o, rng); pi = o["pi"]; diffs.append(np.asarray(T[f"B+T|sr@{al}"]["correct"]) - np.asarray(T[f"pw|sr@{al}"]["correct"])); base.append(float(np.mean(T[f"pw|sr@{al}"]["correct"])))
            same = len({len(d) for d in diffs}) == 1
            m, lo, hi = A.boot(np.mean(diffs, 0), rng) if same else (float(np.mean([d.mean() for d in diffs])), float("nan"), float("nan"))
            out[f"{b}@{al}"] = dict(pi=float(pi), gain=float(m), lo=float(lo), hi=float(hi), base=float(np.mean(base)), per_judge=[float(d.mean()) for d in diffs], tasks=int(len(diffs[0])), n_cal=float(o["n_cal"]))
            print(f"{b:15s} alpha {al} pi {pi:.2f} base {np.mean(base):5.1f} gain {m:+.1f} [{lo:+.1f}, {hi:+.1f}] per judge {np.round([d.mean() for d in diffs], 1)}", flush=True)
    json.dump(out, open(outp, "w"), indent=1)
