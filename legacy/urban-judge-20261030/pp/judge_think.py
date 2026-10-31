import json,os,pathlib,re,time,torch,random
os.environ.setdefault("HF_HUB_OFFLINE","1")
from PIL import Image
from judge_pp import P,MODELS,BIN
from transformers import AutoProcessor,AutoModelForImageTextToText
mp=MODELS["qwen35_9b"]; proc=AutoProcessor.from_pretrained(mp); proc.tokenizer.padding_side="left"
model=AutoModelForImageTextToText.from_pretrained(mp,dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa").eval()
rows=[json.loads(l) for l in open("sample.jsonl")]
sel=[];cnt={}
for r in rows:
    if r["choice"]=="equal": continue
    if cnt.get(r["category"],0)<50: sel.append(r); cnt[r["category"]]=cnt.get(r["category"],0)+1
out=pathlib.Path("raw_think_qwen35_9b.jsonl"); done=set()
if out.exists(): done={(json.loads(l)["id"],json.loads(l)["order"]) for l in open(out)}
jobs=[(r,o) for r in sel for o in ("AB","BA") if (r["id"],o) not in done]; t0=time.time()
Q=BIN+" Think briefly, then end your answer with \"Answer: A\" or \"Answer: B\"."
with open(out,"a",buffering=1) as f:
    for s in range(0,len(jobs),24):
        B=jobs[s:s+24]; msgs=[]
        for r,o in B:
            imgs=[r["left_image"],r["right_image"]] if o=="AB" else [r["right_image"],r["left_image"]]
            c=[]
            for i,p in enumerate(imgs): c+=[{"type":"text","text":f"Image {chr(65+i)}:"},{"type":"image","image":Image.open(P/p).convert("RGB")}]
            c.append({"type":"text","text":Q.format(q=r["question"])}); msgs.append([{"role":"user","content":c}])
        inp=proc.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt",padding=True,enable_thinking=True).to("cuda:0")
        with torch.inference_mode(): g=model.generate(**inp,max_new_tokens=1200,do_sample=False)
        txt=proc.batch_decode(g[:,inp.input_ids.shape[1]:],skip_special_tokens=True)
        for (r,o),t in zip(B,txt):
            m=re.findall(r"Answer:\s*\**\s*([AB])",t); f.write(json.dumps({"id":r["id"],"order":o,"ans":m[-1] if m else None,"len":len(t),"tail":t[-300:]})+"\n")
        print(s,len(jobs),round(time.time()-t0),flush=True)
print("DONE")
