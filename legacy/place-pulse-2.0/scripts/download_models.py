import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path('/media/shawn/C/place-pulse-2.0/models')
MODELS = ['Qwen3-VL-8B-Instruct', 'Qwen3.5-9B']

def log(s):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), s, flush=True)

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def get_file(model, f):
    name = f['Path']
    assert pathlib.PurePosixPath(name).name == name
    dest = ROOT / model / name
    partial = dest.with_name(dest.name + '.partial')
    if dest.exists() and dest.stat().st_size == f['Size'] and digest(dest) == f['Sha256']:
        log(f'REUSE {model}/{name}')
        return
    url = f'https://modelscope.cn/api/v1/models/Qwen/{model}/repo?' + urllib.parse.urlencode({'Revision':f['Revision'], 'FilePath':name})
    for attempt in range(8):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > f['Size']:
                partial.unlink()
                offset = 0
            if offset < f['Size']:
                req = urllib.request.Request(url, headers={'User-Agent':'place-pulse-research-setup', **({'Range':f'bytes={offset}-'} if offset else {})})
                with urllib.request.urlopen(req, timeout=120) as response:
                    if offset and response.status != 206:
                        offset = 0
                    if offset:
                        assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                    with partial.open('ab' if offset else 'wb') as out:
                        while True:
                            data = response.read(8*1024*1024)
                            if not data: break
                            out.write(data)
            assert partial.stat().st_size == f['Size'], (partial.stat().st_size, f['Size'])
            actual = digest(partial)
            if actual != f['Sha256']:
                partial.unlink()
                raise ValueError('SHA-256 mismatch')
            partial.replace(dest)
            log(f'VERIFIED {model}/{name} {f["Size"]:,} bytes')
            return
        except Exception as e:
            log(f'RETRY {model}/{name} attempt={attempt+1} {e}')
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f'Download failed: {model}/{name}')

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        folder = ROOT / model
        folder.mkdir(exist_ok=True)
        metadata_path = folder / 'download_manifest.json'
        if metadata_path.exists():
            manifest = json.loads(metadata_path.read_text())
        else:
            url = f'https://modelscope.cn/api/v1/models/Qwen/{model}/repo/files?Revision=master&Recursive=true'
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = json.load(response)
            assert raw['Success']
            files = [f for f in raw['Data']['Files'] if f['Type']=='blob']
            manifest = {'repository':f'Qwen/{model}', 'source':'https://modelscope.cn/models/Qwen/'+model,
                        'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                        'files':files, 'note':'Each file is pinned by its ModelScope commit revision and verified against upstream SHA256.'}
            metadata_path.write_text(json.dumps(manifest, indent=2)+'\n')
        log(f'START {model}: {sum(f["Size"] for f in manifest["files"]):,} bytes')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda f:get_file(model,f), manifest['files']))
        (folder / 'DOWNLOAD_COMPLETE.json').write_text(json.dumps({'status':'complete','files':len(manifest['files']),'verified_sha256':True})+'\n')
        log(f'MODEL READY {model}')
    log('ALL MODELS READY')

if __name__ == '__main__':
    main()
