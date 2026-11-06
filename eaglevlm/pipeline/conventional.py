"""Conventional judge metrics ('conventional'): accuracy, calibration and error detection of the order-averaged forced
choice, in the published setting (human ties dropped) and with ties kept (a tie is an error for any forced choice).
Ported unchanged from scripts/conv_metrics.py; one CSV row per dataset x judge (no random splits)."""
import numpy as np
from scipy import stats

H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))


def auroc(score, positive):
    score = np.asarray(score, float); positive = np.asarray(positive, int)
    if positive.min() == positive.max():
        return float("nan")
    return float(stats.mannwhitneyu(score[positive == 1], score[positive == 0]).statistic / ((positive == 1).sum() * (positive == 0).sum()))


def aurc(conf, err):
    o = np.argsort(-conf, kind="stable"); e = err[o]; return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))


def selacc(conf, err, cov=.5):
    o = np.argsort(-conf, kind="stable")[: max(1, int(cov * len(conf)))]; return float(1 - err[o].mean())


def ece(conf, correct, bins=10):
    e = np.linspace(.5, 1, bins + 1); out = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf >= lo) & ((conf < hi) | (hi == 1))
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def auprc(score, pos):
    o = np.argsort(-score, kind="stable"); y = pos[o].astype(float)
    if y.sum() == 0:
        return float("nan")
    prec = np.cumsum(y) / np.arange(1, len(y) + 1); return float((prec * y).sum() / y.sum())


def run(cfg, store, log):
    from eaglevlm.pipeline.io import dataset_items
    out = []
    for spec in cfg["datasets"]:
        ds, I = dataset_items(store, spec)
        for j in cfg["judges"]:
            R = store.load(ds, j, conds={"bin", "tri"}); S = {}
            for i, r in R.items():
                if i not in I or not all((c, o) in r for c in ("bin", "tri") for o in ("AB", "BA")):
                    continue
                b1, b2, t1, t2 = r[("bin", "AB")], r[("bin", "BA")], r[("tri", "AB")], r[("tri", "BA")]
                p = (b1["A"] / (b1["A"] + b1["B"]) + b2["B"] / (b2["A"] + b2["B"])) / 2
                q = np.array([(t1["A"] + t2["B"]) / 2, (t1["B"] + t2["A"]) / 2, (t1["T"] + t2["T"]) / 2]); S[i] = (p, q / q.sum())
            ids = sorted(S)
            if len(ids) < 200 or len(ids) < cfg.get("min_coverage", .95) * len(I):
                log(f"skip {spec} {j}: {len(ids)} / {len(I)}"); continue
            y = np.array([{"0": 0, "1": 1, "tie": 2}[str(I[i]["label"])] for i in ids]); t = y == 2; d = ~t
            p = np.array([S[i][0] for i in ids]); Q = np.array([S[i][1] for i in ids])
            fc = np.where(p > .5, 0, 1); err_d = (fc[d] != y[d]).astype(float); err_all = np.where(t, 1.0, (fc != y).astype(float))
            conf = np.abs(p - .5); am = Q.argmax(1)
            tp = ((am == 2) & t).sum(); prec = tp / max((am == 2).sum(), 1); rec = tp / max(t.sum(), 1)
            out.append(dict(dataset=spec, judge=j, n=len(ids), human_tie=float(t.mean()),
                            acc_ties_dropped=float(1 - err_d.mean()), acc_ties_wrong=float(1 - err_all.mean()),
                            acc_half_credit=float(np.where(t, .5, 1 - err_all).mean()), acc_3way=float((am == y).mean()),
                            judge_tie_share=float((am == 2).mean()), tie_f1=float(2 * prec * rec / (prec + rec) if prec + rec else 0.0),
                            ece_ties_dropped=ece(.5 + conf[d], 1 - err_d), ece_ties_kept=ece(.5 + conf, 1 - err_all),
                            err_auroc_ties_dropped=(auroc(H(p[d]), err_d) if 0 < err_d.mean() < 1 else float("nan")),
                            err_auroc_ties_kept=auroc(H(p), err_all), err_auprc_ties_dropped=auprc(H(p[d]), err_d),
                            err_auprc_ties_kept=auprc(H(p), err_all), aurc_ties_dropped=aurc(conf[d], err_d), aurc_ties_kept=aurc(conf, err_all),
                            selacc50_ties_dropped=selacc(conf[d], err_d), selacc50_ties_kept=selacc(conf, err_all)))
            log(f"done {spec} {j}")
    return out


def summarize(rows):
    return rows
