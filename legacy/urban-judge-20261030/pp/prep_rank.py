import csv,json,random,pathlib
P=pathlib.Path("/media/shawn/C/papers/VLM/place-pulse-2.0")
cats={c["_id"]:c for c in json.load(open(P/"annotations/categories.json"))}
imgs=set(p.stem for p in (P/"images").glob("*.jpg"))
random.seed(11); rows=[]; meta={}
for sid,c in cats.items():
    if c["category"] not in ("safety","wealthy","lively"): continue
    L=[]
    for r in csv.DictReader(open(P/"raw/tables/qscores.tsv"),delimiter="\t"):
        if r["study_id"]==sid and r["location_id"] in imgs and int(r["num_votes"])>=8: L.append((float(r["trueskill.stds.-1"]),r["location_id"],float(r["trueskill.score"])))
    L.sort(); pool=L[:2000]; random.shuffle(pool); sel=pool[:150]
    meta[c["category"]]={x[1]:x[2] for x in sel}
    ids=[x[1] for x in sel]; pairs=set()
    while len(pairs)<1200:
        a,b=random.sample(ids,2); pairs.add(tuple(sorted((a,b))))
    for a,b in sorted(pairs):
        if random.random()<0.5: a,b=b,a
        rows.append(dict(id=len(rows),category=c["category"],question=c["vlm_prompt"],left_id=a,right_id=b,left_image=f"images/{a}.jpg",right_image=f"images/{b}.jpg"))
    print(c["category"],len(L),len(sel))
json.dump(meta,open("rank_truth.json","w"))
with open("rank_pairs.jsonl","w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
with open("rank_images.jsonl","w") as f:
    k=0
    for cat,d in meta.items():
        q=[r for r in rows if r["category"]==cat][0]["question"]
        for i in d: f.write(json.dumps(dict(id=k,category=cat,question=q,left_id=i,right_id=i,left_image=f"images/{i}.jpg",right_image=f"images/{i}.jpg"))+"\n"); k+=1
print(len(rows))
