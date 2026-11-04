"""Reproduce tie-dropping evaluation / certification protocols and measure what happens when the human ties are kept.
Per set x judge (order-averaged; bin = forced A/B prompt, tri = A/B/Tie prompt, sa = 4 simulated-annotator personas):
  Reported-accuracy protocols
    acc w/o tie      forced choice on decided pairs only (VL-RewardBench, MMRB2, Multimodal RewardBench, Diffusion-DPO)
    acc w/ tie       forced choice on all pairs, a human tie counts as wrong (MJ-Bench "w/ tie")
    acc 3-way        argmax of the tie prompt vs the 3-way label (GenAI-Bench)
    acc tie-aware    PickScore metric: 1 if equal, 0.5 if exactly one of (label, prediction) is a tie, else 0
    judge tie rate   share of tie-prompt argmax == T, next to the human tie share
  Certification protocols, calibrated as published (ties dropped) and deployed on all pairs (tie = error for 'k better')
    SCOPE            BPE score, linearised threshold                         (alpha .2)
    ToE              Trust or Escalate (Jung et al. ICLR'25): simulated-annotator ensemble; confidence = max of the
                     persona-averaged A/B probability; fixed-sequence binomial test (delta .1)  (where tri_sa runs exist)
    -> promised alpha vs realised selective risk, violation rate, coverage; and the same rules calibrated WITH ties
  Davidson check     Spearman rho(P(T), closeness 1-2|p-.5|): a Bradley-Terry-with-ties judge has P(T) peak at p = .5
usage (judgebench dir): python repro_tiedrop.py --out paper_tables/repro_tiedrop.md"""
import argparse, collections, os, sys
import numpy as np
from scipy.stats import binom, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tie_mech import items, load, H
from label_noise_lib import per_judge, thr, eval_items

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper_tables/repro_tiedrop.md")
ap.add_argument("--splits", type=int, default=200)
ap.add_argument("--alpha", type=float, default=0.2)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--min-share", type=float, default=0.95)
a = ap.parse_args()
SETS = [("mtbench_human", None), ("arena55k", None), ("mllm_judge", None), ("genai_arena", "generation"),
        ("genai_arena", "edition"), ("pickapic_u", None), ("imagerewarddb", None)]
JUD = ["qwen3vl8b", "internvl35_8b", "qwen35_9b", "qwen25vl7b", "qwen3vl32b", "qwen3_8b", "qwen25_7b"]
SA = ["tri_sa1", "tri_sa2", "tri_sa3", "tri_sa4"]
G = np.linspace(0, 1, 201)[1:]


def sa_conf(ds, j, ids):
    """Trust-or-Escalate style ensemble over the 4 personas: order-aligned A/B probability per persona, averaged"""
    R = load(ds, j, conds=set(SA)); out = {}
    for i in ids:
        r = R.get(i, {})
        if not all((c, o) in r for c in SA for o in ("AB", "BA")):
            continue
        ps = []
        for c in SA:
            t1, t2 = r[(c, "AB")], r[(c, "BA")]
            pa = (t1["A"] + t2["B"]) / 2; pb = (t1["B"] + t2["A"]) / 2; ps.append(pa / (pa + pb))
        out[i] = float(np.mean(ps))
    return out


def fst(s_cal, e_cal, al, delta):
    """fixed-sequence binomial testing from the most confident threshold outward (ToE's calibration)"""
    best = -np.inf
    for g in G:
        m = s_cal <= g
        if m.sum() == 0:
            continue
        if binom.cdf(int(e_cal[m].sum()), int(m.sum()), al) <= delta:
            best = g
        else:
            break
    return best


def certify(s, eb, y, rule):
    """published calibration (non-tie pairs) and tie-kept calibration, deployed on all pairs; returns cells"""
    n = len(y); rng = np.random.default_rng(0); R = collections.defaultdict(list)
    et = np.where(y == 2, 1.0, eb)                       # deployed error: tie = error for 'k better'
    for _ in range(a.splits):
        pm = rng.permutation(n); cal, te = pm[: n // 2], pm[n // 2:]; nt = cal[y[cal] != 2]
        for tag, cc, ec in (("drop", nt, eb[nt]), ("keep", cal, et[cal])):
            lam = thr(s[cc], ec, a.alpha) if rule == "scope" else fst(s[cc], ec, a.alpha, a.delta)
            acc = s[te] <= lam
            if acc.any():
                r = et[te][acc].mean(); R[tag].append((acc.mean(), r, float(r > a.alpha)))
            else:
                R[tag].append((0.0, np.nan, np.nan))
    f = lambda Z: "{:.2f} / {:.3f} / {:.2f}".format(np.mean([z[0] for z in Z]), np.nanmean([z[1] for z in Z]) if any(
        not np.isnan(z[1]) for z in Z) else np.nan, np.nanmean([z[2] for z in Z]) if any(not np.isnan(z[2]) for z in Z) else np.nan)
    return f(R["drop"]), f(R["keep"])


L = ["# Reproducing tie-dropping protocols, then keeping the ties", "",
     f"Certification cells = coverage / realised selective risk / violation rate at promised alpha = {a.alpha} "
     f"({a.splits} splits, 50% cal / 50% test); 'drop' = calibrated without ties as published, 'keep' = calibrated "
     "with ties counted as errors; both deployed on all pairs.  Davidson rho = Spearman(P(T), 1-2|p-.5|).", "",
     "| set | judge | n | human tie | acc w/o tie | acc w/ tie | acc 3-way | acc tie-aware | judge tie rate | "
     "Davidson rho | SCOPE drop | SCOPE keep | ToE drop | ToE keep |", "|" + "---|" * 14]
for ds, sub in SETS:
    I = eval_items(ds, items(ds)); I = {i: x for i, x in I.items() if sub is None or x["subset"] == sub}
    for j in JUD:
        P = per_judge(ds, j, I); ids = sorted(P)
        if len(ids) < 200 or len(ids) < a.min_share * len(I):
            continue
        y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids])
        p = np.array([P[i][0] for i in ids]); Q = np.array([P[i][1] for i in ids])
        fc = np.where(p > .5, 0, 1); nt = y != 2; v3 = Q.argmax(1)
        tie_aware = np.where(v3 == y, 1.0, np.where((v3 == 2) ^ (y == 2), 0.5, 0.0))
        rho = spearmanr(Q[:, 2], 1 - 2 * np.abs(p - .5)).correlation
        eb = (fc != y).astype(float)
        sd, sk = certify(H(p), eb, y, "scope")
        S = sa_conf(ds, j, ids)
        if len(S) >= 200:                 # persona runs exist on the capped 1500-pair subsets only: own (smaller) n
            k = np.array([i in S for i in ids]); ps = np.array([S[i] for i in np.array(ids)[k]])
            fs = np.where(ps > .5, 0, 1); td, tk = certify(1 - np.maximum(ps, 1 - ps), (fs != y[k]).astype(float), y[k], "fst")
            td, tk = td + f" (n={k.sum()})", tk + f" (n={k.sum()})"
        else:
            td = tk = "-"
        L.append(f"| {ds}{'/' + sub if sub else ''} | {j} | {len(ids)} | {nt.size and (1 - nt.mean()):.2f} | "
                 f"{(fc[nt] == y[nt]).mean():.3f} | {(fc == y).mean():.3f} | {(v3 == y).mean():.3f} | {tie_aware.mean():.3f} | "
                 f"{(v3 == 2).mean():.2f} | {rho:+.2f} | {sd} | {sk} | {td} | {tk} |")
        print(L[-1], flush=True)
open(a.out, "w").write("\n".join(L) + "\n")
