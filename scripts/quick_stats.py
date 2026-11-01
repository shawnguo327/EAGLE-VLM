"""Quick EAGLE-V stats for one or more datasets on whatever results exist so far.
usage (from $EAGLEVLM_DATA): python quick_stats.py pp vlrb ...   [--alphas .2,.25,.3]"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eaglevlm"))
from make_paper_tables import diag_row
from eaglev_stats import load_records, selective_eval, ladder_eval

ap = argparse.ArgumentParser(); ap.add_argument("ds", nargs="+"); ap.add_argument("--alphas", default=".15,.2,.25,.3")
a = ap.parse_args()
for ds in a.ds:
    for f in sorted(glob.glob(f"results/{ds}__*.jsonl")):
        m = os.path.basename(f)[:-6].split("__")[1]
        recs = load_records(f"items/{ds}.jsonl", f)
        if len([r for r in recs if r["y"] is not None]) < 20:
            continue
        d, S, err = diag_row(recs)
        print(f"{ds:13s} {m:11s} " + " ".join(f"{k}={float(v):.3f}" if k != "n" else f"n={v}" for k, v in d.items()))
        for al in map(float, a.alphas.split(",")):
            lad = ladder_eval(recs, al)
            print(f"   alpha={al:.2f} cov(dp)={selective_eval(S['dp'], err, al)['coverage']:.3f} "
                  f"cov(lo)={selective_eval(S['abs_logodds'], err, al)['coverage']:.3f}"
                  + (f" ladder auto={lad['auto']:.3f} (pair {lad['pair_share']:.3f}, point {lad['point_share']:.3f}) risk={lad['risk']:.3f}" if lad else ""))
