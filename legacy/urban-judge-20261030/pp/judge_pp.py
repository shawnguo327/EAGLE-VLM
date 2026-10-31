"""Place Pulse VLM judge: logit-level pairwise (2-way, 3-way), low-res pairwise, pointwise expected score, and per-image fact extraction."""
import argparse,json,os,pathlib,time,torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
from PIL import Image
P=pathlib.Path("/media/shawn/C/papers/VLM/place-pulse-2.0")
MODELS={"qwen3vl8b":P/"models/Qwen3-VL-8B-Instruct","qwen35_9b":P/"models/Qwen3.5-9B","qwen25vl7b":pathlib.Path("/media/shawn/C/nyserver-files/mntstorage/models/Qwen2.5-VL-7B-Instruct")}
SCALES={"safety":("safe","very unsafe","very safe"),"lively":("lively","not lively at all","very lively"),
 "boring":("boring","not boring at all","very boring"),"wealthy":("wealthy","very poor","very wealthy"),
 "depressing":("depressing","not depressing at all","very depressing"),"beautiful":("beautiful","very ugly","very beautiful")}
BIN="Judge only the visual appearance of these two urban places. {q} Answer with exactly one letter: A for the first image, or B for the second image."
TRI="Judge only the visual appearance of these two urban places. {q} Answer with exactly one letter: A for the first image, B for the second image, or C if they look about the same."
SCORE="Judge only the visual appearance of this urban place. How {attr} does it look, on an integer scale from 1 to 9, where 1 means {low} and 9 means {high}? Answer with exactly one digit."
FACTS=("Look at this street-level photo and fill in this JSON with your best estimates. Scales are 0-10 unless noted. "
 "{\"greenery\":,\"sky_open\":,\"cars\":<count>,\"people\":<count>,\"building_upkeep\":,\"litter_graffiti\":,\"road_width\":,"
 "\"shops_signs\":,\"building_height_floors\":<count>,\"sidewalk_quality\":,\"fences_walls\":,\"vacant_or_derelict\":} Output only the JSON.")
def load(path,size=None):
    im=Image.open(P/path).convert("RGB")
    if size: w,h=im.size; im=im.resize(size,Image.BILINEAR).resize((w,h),Image.BILINEAR)
    return im
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",default="qwen3vl8b"); ap.add_argument("--input"); ap.add_argument("--out")
    ap.add_argument("--conds",default="bin,tri,low,score,facts"); ap.add_argument("--bs",type=int,default=16); ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()
    from transformers import AutoProcessor,AutoModelForImageTextToText
    mp=MODELS[a.model]; proc=AutoProcessor.from_pretrained(mp); proc.tokenizer.padding_side="left"
    model=AutoModelForImageTextToText.from_pretrained(mp,dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa").eval()
    tok=proc.tokenizer; ids={c:tok.encode(c,add_special_tokens=False)[0] for c in list("ABC")+[str(i) for i in range(1,10)]}
    print({c:tok.decode([i]) for c,i in ids.items()},flush=True)
    rows=[json.loads(l) for l in open(a.input)]
    if a.limit: rows=rows[:a.limit]
    for r in rows:  # canonical A/B
        if "left_id" in r: r["A"],r["B"]=r["left_image"],r["right_image"]
        else: r["A"],r["B"]=r["a_image"],r["b_image"]
    jobs=[]; conds=a.conds.split(",")
    for r in rows:
        q=r["question"]
        for c in conds:
            if c in("bin","tri","low"):
                for o in("AB","BA"): jobs.append((r,c,o))
            elif c in("score","facts"):
                for o in("A","B"): jobs.append((r,c,o))
    done=set(); outp=pathlib.Path(a.out)
    if outp.exists():
        for l in open(outp): d=json.loads(l); done.add((d["id"],d["cond"],d["order"]))
    jobs=[j for j in jobs if (j[0]["id"],j[1],j[2]) not in done]
    gen_jobs=[j for j in jobs if j[1]=="facts"]; fw_jobs=[j for j in jobs if j[1]!="facts"]
    print("jobs",len(fw_jobs),len(gen_jobs),flush=True); t0=time.time()
    def build(r,c,o):
        content=[]
        if c in("bin","tri","low"):
            imgs=[r[o[0]],r[o[1]]]
            for i,p in enumerate(imgs):
                content+= [{"type":"text","text":f"Image {chr(65+i)}:"},{"type":"image","image":load(p,(48,36) if c=="low" else None)}]
            content.append({"type":"text","text":(TRI if c=="tri" else BIN).format(q=r["question"])})
        else:
            content.append({"type":"image","image":load(r[o])})
            attr,low,high=SCALES[r["category"]]
            content.append({"type":"text","text":SCORE.format(attr=attr,low=low,high=high) if c=="score" else FACTS})
        return [{"role":"user","content":content}]
    with open(outp,"a",buffering=1) as f:
        for jl,gen in ((fw_jobs,False),(gen_jobs,True)):
            bs=a.bs if not gen else a.bs*2
            for s in range(0,len(jl),bs):
                B=jl[s:s+bs]; msgs=[build(*j) for j in B]
                inp=proc.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt",padding=True,enable_thinking=False).to("cuda:0")
                with torch.inference_mode():
                    if not gen:
                        lg=model(**inp).logits[:,-1,:].float(); lp=torch.log_softmax(lg,-1)
                        for (r,c,o),row in zip(B,lp):
                            keys=["A","B"] if c in("bin","low") else (["A","B","C"] if c=="tri" else [str(i) for i in range(1,10)])
                            raw={k:row[ids[k]].item() for k in keys}
                            f.write(json.dumps({"id":r["id"],"cond":c,"order":o,"lp":raw,"mass":sum(torch.exp(torch.tensor(list(raw.values())))).item()})+"\n")
                    else:
                        g=model.generate(**inp,max_new_tokens=150,do_sample=False)
                        txt=proc.batch_decode(g[:,inp.input_ids.shape[1]:],skip_special_tokens=True)
                        for (r,c,o),t in zip(B,txt): f.write(json.dumps({"id":r["id"],"cond":c,"order":o,"text":t})+"\n")
                if (s//bs)%20==0: print(("gen" if gen else "fw"),s,len(jl),round(time.time()-t0),flush=True)
    print("DONE",round(time.time()-t0),flush=True)
if __name__=="__main__": main()
