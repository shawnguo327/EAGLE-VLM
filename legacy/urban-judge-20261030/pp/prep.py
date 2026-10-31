import gzip,json,random,collections,csv,math,pathlib
P=pathlib.Path("/media/shawn/C/papers/VLM/place-pulse-2.0"); W=pathlib.Path("/media/shawn/C/papers/VLM/urban-judge-20261030/pp")
q={}
with open(P/"raw/tables/qscores.tsv") as f:
    for r in csv.DictReader(f,delimiter="\t"):
        try:q[(r["location_id"],r["study_id"])]=(float(r["trueskill.score"]),int(r["num_votes"]))
        except: pass
votes=[];pairc=collections.Counter()
for fn,lab in [("pairs_binary.jsonl.gz",None),("pairs_ties.jsonl.gz","equal")]:
    with gzip.open(P/"annotations"/fn,"rt") as f:
        for l in f:
            r=json.loads(l); 
            if lab: r["choice"]="equal"
            votes.append(r); pairc[(r["study_id"],)+tuple(sorted([r["left_id"],r["right_id"]]))]+=1
print("votes",len(votes))
rep=collections.Counter(v for v in pairc.values()); print("pair multiplicity",sorted(rep.items())[:8])
random.seed(20261031)
used=set(json.loads(l)["a_id"] for l in open(P/"experiments/vlm-pilot-20261030/pilot.jsonl")) if (P/"experiments/vlm-pilot-20261030/pilot.jsonl").exists() else set()
bycat=collections.defaultdict(list)
for r in votes:
    ka=(r["left_id"],r["study_id"]);kb=(r["right_id"],r["study_id"])
    if ka not in q or kb not in q: continue
    r["gap"]=q[ka][0]-q[kb][0]; bycat[(r["category"],r["choice"]=="equal")].append(r)
out=[];imgs=set()
for (cat,tie),L in sorted(bycat.items()):
    random.shuffle(L); n=250 if not tie else 60; k=0
    for r in L:
        if r["left_id"] in imgs or r["right_id"] in imgs: continue
        imgs.update([r["left_id"],r["right_id"]]); out.append(r); k+=1
        if k>=n: break
    print(cat,tie,len(L),k)
for i,r in enumerate(out): r["id"]=i
with open(W/"sample.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
print("sample",len(out),"images",len(imgs))
