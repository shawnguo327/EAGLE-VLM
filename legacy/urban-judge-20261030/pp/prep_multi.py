import gzip,json,collections,pathlib,random
P=pathlib.Path("/media/shawn/C/papers/VLM/place-pulse-2.0"); W=pathlib.Path(".")
agg=collections.defaultdict(lambda:{"votes":collections.Counter()})
for fn in ["pairs_binary.jsonl.gz","pairs_ties.jsonl.gz"]:
    with gzip.open(P/"annotations"/fn,"rt") as f:
        for l in f:
            r=json.loads(l); a,b=sorted([r["left_id"],r["right_id"]]); k=(r["study_id"],a,b)
            d=agg[k]; d.update(study_id=r["study_id"],category=r["category"],question=r["question"],a_id=a,b_id=b,a_image=f"images/{a}.jpg",b_image=f"images/{b}.jpg")
            c=r.get("choice","equal") if "ties" not in fn else "equal"
            if c=="left": c="a" if r["left_id"]==a else "b"
            elif c=="right": c="a" if r["right_id"]==a else "b"
            d["votes"][c]+=1
multi=[d for d in agg.values() if sum(d["votes"].values())>=4]
print("pairs>=4 votes",len(multi),collections.Counter(d["category"] for d in multi))
random.seed(7); random.shuffle(multi)
bycat=collections.defaultdict(list)
for d in multi:
    if len(bycat[d["category"]])<120: d["votes"]=dict(d["votes"]); bycat[d["category"]].append(d)
out=[d for L in bycat.values() for d in L]
for i,d in enumerate(out): d["id"]=i
with open("multi.jsonl","w") as f:
    for d in out: f.write(json.dumps(d)+"\n")
import statistics
print("n",len(out),"mean votes",statistics.mean(sum(d["votes"].values()) for d in out), "tie share",sum(d["votes"].get("equal",0) for d in out)/sum(sum(d["votes"].values()) for d in out))
