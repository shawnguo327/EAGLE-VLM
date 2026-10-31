"""Deterministic zero-shot VLM pilot; labels are never included in prompts."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

SCALES = {
    'safety': ('safety', 'very unsafe', 'very safe'),
    'lively': ('liveliness', 'not lively at all', 'very lively'),
    'boring': ('boredom', 'not boring at all', 'very boring'),
    'wealthy': ('wealth', 'very poor', 'very wealthy'),
    'depressing': ('depressiveness', 'not depressing at all', 'very depressing'),
    'beautiful': ('beauty', 'very ugly', 'very beautiful'),
}
PAIR_INSTRUCTION = 'Judge only the visual appearance of these two urban places. {question} Answer with exactly one letter: A for the first image, or B for the second image. Do not explain.'
SCORE_INSTRUCTION = 'Judge only the visual appearance of this urban place. Rate its perceived {attribute} on an integer scale from 1 to 10, where 1 means it looks {low} and 10 means it looks {high}. Answer with exactly one integer from 1 to 10. Do not explain.'

def log(msg):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), msg, flush=True)

def parse_prediction(raw, task):
    s = raw.strip()
    if task.startswith('pair_'):
        m = re.fullmatch(r'([AB])[.!]?', s)
        return m.group(1) if m else None
    m = re.fullmatch(r'(10|[1-9])[.]?', s)
    return int(m.group(1)) if m else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, default=pathlib.Path('/media/shawn/C/place-pulse-2.0'))
    ap.add_argument('--experiment', type=pathlib.Path, required=True)
    ap.add_argument('--model', required=True, choices=['Qwen3-VL-8B-Instruct','Qwen3.5-9B'])
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--limit-pairs', type=int)
    args=ap.parse_args()
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration
    torch.manual_seed(20261030)
    torch.set_num_threads(8)
    model_dir=args.root/'models'/args.model
    assert json.loads((model_dir/'DOWNLOAD_COMPLETE.json').read_text())['status']=='complete'
    sample_file=args.experiment/'pilot.jsonl'
    rows=[json.loads(s) for s in sample_file.read_text().splitlines()]
    if args.limit_pairs: rows=rows[:args.limit_pairs]
    out=args.experiment/'results'/args.model
    out.mkdir(parents=True,exist_ok=True)
    prediction_file=out/'predictions.jsonl'
    signature={
        'model':args.model,'sample_sha256':hashlib.sha256(sample_file.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'model_manifest_sha256':hashlib.sha256((model_dir/'download_manifest.json').read_bytes()).hexdigest(),
        'dtype':'bfloat16','attention':'sdpa','thinking':False,'batch_size':args.batch_size,
        'max_new_tokens':16,'do_sample':False,
        'torch':torch.__version__,'transformers':transformers.__version__,
    }
    config_path=out/'run_config.json'
    if config_path.exists():
        prior=json.loads(config_path.read_text())
        if prior.get('signature')!=signature:
            raise ValueError('Resume protocol changed; use a new experiment/results directory.')
    elif prediction_file.exists() and prediction_file.stat().st_size:
        raise ValueError('Existing predictions have no provenance configuration.')
    done=set()
    if prediction_file.exists():
        for line in prediction_file.read_text().splitlines():
            r=json.loads(line)
            key=(r['id'],r['task'])
            assert key not in done
            done.add(key)
    jobs=[(r,t) for t in ['pair_ab','pair_ba','score_a','score_b'] for r in rows if (r['id'],t) not in done]
    completion_path=out/('SMOKE_COMPLETE.json' if args.limit_pairs else 'COMPLETE.json')
    if not jobs:
        completion_path.write_text(json.dumps({'status':'complete','model':args.model,
            'pairs':len(rows),'expected_jobs_verified':len(rows)*4,'restored_from_predictions':True})+'\n')
        log('No pending jobs')
        return
    completion_path.unlink(missing_ok=True)
    log(f'LOAD {args.model}; pending={len(jobs)}')
    start=time.monotonic()
    processor=AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.tokenizer.padding_side='left'
    cls=Qwen3_5ForConditionalGeneration if args.model=='Qwen3.5-9B' else Qwen3VLForConditionalGeneration
    model=cls.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={'':'cuda:0'},
                             attn_implementation='sdpa', local_files_only=True).eval()
    torch.cuda.reset_peak_memory_stats()
    gen_config=transformers.GenerationConfig(max_new_tokens=16,do_sample=False,
                 use_cache=True, eos_token_id=model.generation_config.eos_token_id,
                 pad_token_id=processor.tokenizer.pad_token_id,
                 bos_token_id=model.generation_config.bos_token_id)
    metadata={
        'signature':signature,
        'model':args.model,'model_path':str(model_dir),'dtype':'bfloat16','quantized':False,
        'attention':'sdpa','thinking':False,'device':torch.cuda.get_device_name(0),
        'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','torchvision','Pillow','accelerate','safetensors']},
        'sample_sha256':hashlib.sha256(sample_file.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'generation_config':gen_config.to_dict(),'batch_size':args.batch_size,
        'pair_prompt_template':PAIR_INSTRUCTION,'score_prompt_template':SCORE_INSTRUCTION,
        'scale_endpoints':SCALES,'processor_image_config':processor.image_processor.to_dict(),
        'model_manifest_sha256':hashlib.sha256((model_dir/'download_manifest.json').read_bytes()).hexdigest(),
        'seed':20261030,'load_seconds':time.monotonic()-start,
    }
    if not config_path.exists():
        config_path.write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    with prediction_file.open('a',buffering=1) as f:
        for offset in range(0,len(jobs),args.batch_size):
            batch=jobs[offset:offset+args.batch_size]
            messages=[]
            extra=[]
            for row,task in batch:
                if task.startswith('pair_'):
                    keys=['a_image','b_image'] if task=='pair_ab' else ['b_image','a_image']
                    prompt=PAIR_INSTRUCTION.format(question=row['question'])
                else:
                    keys=['a_image' if task=='score_a' else 'b_image']
                    attribute,low,high=SCALES[row['category']]
                    prompt=SCORE_INSTRUCTION.format(attribute=attribute,low=low,high=high)
                content=[]
                for i,key in enumerate(keys):
                    if len(keys)==2: content.append({'type':'text','text':f'Image {chr(65+i)}:'})
                    with Image.open(args.root/row[key]) as im:
                        content.append({'type':'image','image':im.convert('RGB')})
                content.append({'type':'text','text':prompt})
                messages.append([{'role':'user','content':content}])
                extra.append({'prompt_text':prompt,'image_paths':[row[k] for k in keys]})
            t=time.monotonic()
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                enable_thinking=False,return_dict=True,return_tensors='pt',padding=True)
            if offset==0:
                text=processor.apply_chat_template(messages[0],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                (out/'example_rendered_prompt.txt').write_text(text)
                log(f'INPUT {tuple(inputs.input_ids.shape)} image_grid={inputs.image_grid_thw.tolist()}')
            inputs=inputs.to('cuda:0')
            with torch.inference_mode():
                generated=model.generate(**inputs,generation_config=gen_config)
            output=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            torch.cuda.synchronize()
            elapsed=time.monotonic()-t
            for (row,task),raw,other in zip(batch,output,extra):
                prediction=parse_prediction(raw,task)
                result={'id':row['id'],'category':row['category'],'task':task,
                        'raw_output':raw,'prediction':prediction,
                        'elapsed_seconds':elapsed/len(batch),'batch_elapsed_seconds':elapsed,
                        'batch_index':offset//args.batch_size,**other}
                f.write(json.dumps(result,ensure_ascii=False)+'\n')
            log(f'PROGRESS {args.model} {offset+len(batch)}/{len(jobs)} task={batch[-1][1]} batch_seconds={elapsed:.2f} outputs={output[:2]}')
    complete={'status':'complete','model':args.model,'pairs_this_invocation':len(rows),
              'jobs_this_invocation':len(jobs),'elapsed_seconds':time.monotonic()-start,
              'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
              'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
              'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    completion_path.write_text(json.dumps(complete,indent=2)+'\n')
    with (out/'invocations.jsonl').open('a') as f:
        f.write(json.dumps(complete)+'\n')
    log(f'COMPLETE {complete}')

if __name__=='__main__':
    assert parse_prediction(' B\n','pair_ab')=='B'
    assert parse_prediction('10','score_a')==10
    assert parse_prediction('1 or 10','score_a') is None
    main()
