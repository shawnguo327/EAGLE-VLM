import concurrent.futures, hashlib, json, pathlib, time, urllib.request, urllib.parse

ROOT=pathlib.Path('/media/shawn/C/place-pulse-2.0/models/MiniCPM-V-4.6')
REPO='OpenBMB/MiniCPM-V-4.6'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def download(f):
    name=f['Path']; assert pathlib.PurePosixPath(name).name==name
    dst=ROOT/name; tmp=ROOT/(name+'.partial')
    if dst.exists() and dst.stat().st_size==f['Size'] and digest(dst)==f['Sha256']: return
    url=f'https://modelscope.cn/api/v1/models/{REPO}/repo?'+urllib.parse.urlencode({'Revision':f['Revision'],'FilePath':name})
    for attempt in range(6):
        try:
            n=tmp.stat().st_size if tmp.exists() else 0
            if n<f['Size']:
                req=urllib.request.Request(url,headers={'Range':f'bytes={n}-'} if n else {})
                with urllib.request.urlopen(req,timeout=120) as r:
                    if n and r.status!=206: n=0
                    if n: assert r.headers.get('Content-Range','').startswith(f'bytes {n}-')
                    with tmp.open('ab' if n else 'wb') as out:
                        for b in iter(lambda:r.read(8*1024*1024),b''): out.write(b)
            assert tmp.stat().st_size==f['Size']
            if digest(tmp)!=f['Sha256']:
                tmp.unlink(); raise ValueError('SHA mismatch')
            tmp.replace(dst)
            print('VERIFIED',name,f['Size'],flush=True); return
        except Exception as e:
            print('RETRY',name,str(e),flush=True);time.sleep(3+attempt*3)
    raise RuntimeError(name)

ROOT.mkdir(parents=True,exist_ok=True)
mp=ROOT/'download_manifest.json'
if mp.exists(): manifest=json.loads(mp.read_text())
else:
    with urllib.request.urlopen(f'https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true',timeout=30) as r: data=json.load(r)
    assert data['Success']
    manifest={'repository':REPO,'source':f'https://modelscope.cn/models/{REPO}',
              'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':[x for x in data['Data']['Files'] if x['Type']=='blob']}
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(download,manifest['files']))
(ROOT/'DOWNLOAD_COMPLETE.json').write_text(json.dumps({'status':'complete','verified_sha256':True,'files':len(manifest['files'])})+'\n')
print('MINICPM READY',flush=True)
