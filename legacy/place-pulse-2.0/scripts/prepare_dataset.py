#!/usr/bin/env python3
"""Verify the author's archive and build image-ID aligned pairwise indexes."""
import collections, csv, gzip, hashlib, json, os, pathlib, re, shutil, struct, sys, time, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / 'raw'
ANN = ROOT / 'annotations'
IMAGE_DIR = ROOT / 'images'
for directory in (RAW, ANN, IMAGE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), message, flush=True)

def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

manifest = json.loads((RAW / 'transfer_manifest.json').read_text())
(ROOT / 'READY.json').unlink(missing_ok=True)
archive = RAW / 'place-pulse-2.0.zip'
if not archive.exists():
    tmp = RAW / 'place-pulse-2.0.zip.part'
    archive_hash = hashlib.sha256()
    with tmp.open('wb') as out:
        for part in manifest['parts']:
            source = RAW / 'parts' / part['name']
            if source.stat().st_size != part['size'] or digest_file(source) != part['sha256']:
                raise RuntimeError(f'Part checksum mismatch: {source}')
            with source.open('rb') as inp:
                for block in iter(lambda: inp.read(8 * 1024 * 1024), b''):
                    out.write(block)
                    archive_hash.update(block)
            log(f'Verified and assembled {source.name}')
    if tmp.stat().st_size != manifest['size']:
        raise RuntimeError('Archive size mismatch')
    tmp.replace(archive)
    archive_sha = archive_hash.hexdigest()
else:
    if archive.stat().st_size != manifest['size']:
        raise RuntimeError('Existing archive has wrong size')
    archive_sha = digest_file(archive)
(RAW / 'SHA256SUMS').write_text(f'{archive_sha}  place-pulse-2.0.zip\n')

tables = RAW / 'tables'
tables.mkdir(exist_ok=True)
image_manifest = {}
table_names = []
try:
    from PIL import Image
except ImportError:
    Image = None
corrupt_images = []
dimensions = collections.Counter()
with zipfile.ZipFile(archive) as z:
    log(f'Archive entries: {len(z.infolist())}; verifying CRC for all members')
    bad = z.testzip()
    if bad is not None:
        raise RuntimeError(f'ZIP CRC failure: {bad}')
    log('All ZIP members passed CRC verification')
    for member in z.infolist():
        name = pathlib.PurePosixPath(member.filename)
        if name.is_absolute() or '..' in name.parts:
            raise RuntimeError(f'Unsafe archive path: {name}')
        if member.is_dir() or '__MACOSX' in name.parts or name.name.startswith('._'):
            continue
        if name.suffix.lower() in ('.tsv', '.csv', '.txt') and 'images' not in name.parts:
            dest = tables / name.name
            with z.open(member) as src, dest.open('wb') as out:
                shutil.copyfileobj(src, out)
            table_names.append(name.name)
        elif name.suffix.lower() in ('.jpg', '.jpeg', '.png') and 'images' in name.parts:
            pieces = name.stem.split('_', 3)
            if len(pieces) != 4 or not re.fullmatch(r'[a-fA-F0-9]{24}', pieces[2]):
                raise RuntimeError(f'Unexpected image filename: {name}')
            lat, lon, image_id, city = pieces
            if image_id in image_manifest:
                raise RuntimeError(f'Duplicate image ID: {image_id}')
            dest = IMAGE_DIR / (image_id + name.suffix.lower())
            with z.open(member) as src, dest.open('wb') as out:
                shutil.copyfileobj(src, out)
            size = None
            if Image is not None:
                try:
                    with Image.open(dest) as im:
                        size = im.size
                        im.verify()
                    dimensions['x'.join(map(str, size))] += 1
                except Exception as exc:
                    corrupt_images.append({'image_id': image_id, 'error': str(exc)})
            image_manifest[image_id] = {'image_id': image_id, 'image': str(dest.relative_to(ROOT)), 'original_member': member.filename, 'city_from_filename': city, 'latitude_from_filename': float(lat), 'longitude_from_filename': float(lon), 'bytes': member.file_size, 'width': size[0] if size else None, 'height': size[1] if size else None}
            if len(image_manifest) % 10000 == 0:
                log(f'Extracted {len(image_manifest)} original images')

with (ANN / 'images.jsonl').open('w') as out:
    for image_id in sorted(image_manifest):
        out.write(json.dumps(image_manifest[image_id], ensure_ascii=False) + '\n')
if corrupt_images:
    write_json(ANN / 'corrupt_images.json', corrupt_images)
    raise RuntimeError(f'{len(corrupt_images)} unreadable images; see annotations/corrupt_images.json')

studies_path = tables / 'studies.tsv'
with studies_path.open(newline='') as f:
    study_rows = list(csv.DictReader(f, delimiter='\t'))
log('Study records: ' + json.dumps(study_rows, ensure_ascii=False))
known_categories = {
    '50a68a51fdc9f05596000002': ('safety', 'Which place looks safer?'),
    '50f62c41a84ea7c5fdd2e454': ('lively', 'Which place looks livelier?'),
    '50f62c68a84ea7c5fdd2e456': ('boring', 'Which place looks more boring?'),
    '50f62cb7a84ea7c5fdd2e458': ('wealthy', 'Which place looks wealthier?'),
    '50f62ccfa84ea7c5fdd2e459': ('depressing', 'Which place looks more depressing?'),
    '5217c351ad93a7d3e7b07a64': ('beautiful', 'Which place looks more beautiful?'),
}
study_ids = {row.get('_id', '') for row in study_rows}
if study_ids != set(known_categories):
    raise RuntimeError(f'Unexpected studies table: {study_rows}')
for row in study_rows:
    category, prompt = known_categories[row['_id']]
    source_question = row.get('study_question', '').strip().lower()
    expected_question = {'safety': 'safer', 'lively': 'livelier', 'boring': 'more boring', 'wealthy': 'wealthier', 'depressing': 'more depressing', 'beautiful': 'more beautiful'}[category]
    if source_question != expected_question:
        raise RuntimeError(f'Study ID/question mismatch: {row}')
    row['category'] = category
    row['vlm_prompt'] = prompt
write_json(ANN / 'categories.json', study_rows)

counts = collections.Counter()
choice_histogram = collections.Counter()
by_category = collections.defaultdict(collections.Counter)
missing_ids = collections.Counter()
voted_ids = set()
seen_votes = set()
timestamps = []
min_timestamp = None
max_timestamp = None
with (tables / 'votes.tsv').open(newline='') as f, gzip.open(ANN / 'pairs_binary.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as binary, gzip.open(ANN / 'pairs_ties.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as ties, gzip.open(ANN / 'votes_missing_images.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as missing, gzip.open(ANN / 'votes_invalid.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as invalid:
    rows = csv.DictReader(f, delimiter='\t')
    fields = rows.fieldnames
    for row in rows:
        counts['raw_votes'] += 1
        choice = row['choice']
        choice_histogram[choice] += 1
        study_id = row['study_id']
        stamp = row.get('timestamp', '')
        if stamp:
            min_timestamp = min(min_timestamp, stamp) if min_timestamp else stamp
            max_timestamp = max(max_timestamp, stamp) if max_timestamp else stamp
        vote_id = row['_id']
        if vote_id in seen_votes:
            counts['duplicate_vote_ids'] += 1
        seen_votes.add(vote_id)
        left, right = row['left'], row['right']
        voted_ids.update((left, right))
        if choice not in ('left', 'right', 'equal') or study_id not in known_categories:
            counts['invalid_votes'] += 1
            reason = 'invalid_choice' if choice not in ('left', 'right', 'equal') else 'unknown_study'
            invalid.write(json.dumps({'reason': reason, 'source_row_number': counts['raw_votes'], 'record': row}, ensure_ascii=False) + '\n')
            continue
        category, question = known_categories[study_id]
        counts['raw_' + choice] += 1
        by_category[category]['raw_' + choice] += 1
        absent = [image_id for image_id in (left, right) if image_id not in image_manifest]
        if absent:
            counts['votes_missing_images'] += 1
            by_category[category]['missing_images'] += 1
            missing_ids.update(absent)
            missing.write(json.dumps(row, ensure_ascii=False) + '\n')
            continue
        if left == right:
            counts['self_comparisons'] += 1
        record = {'vote_id': vote_id, 'left_id': left, 'right_id': right, 'left_image': image_manifest[left]['image'], 'right_image': image_manifest[right]['image'], 'study_id': study_id, 'category': category, 'question': question, 'choice': choice, 'timestamp': stamp}
        if choice == 'equal':
            ties.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            counts['usable_ties'] += 1
            by_category[category]['usable_ties'] += 1
        else:
            binary.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            counts['usable_binary'] += 1
            by_category[category]['usable_binary'] += 1
        if counts['raw_votes'] % 250000 == 0:
            log(f'Indexed {counts["raw_votes"]} votes')

info = {'source_page': manifest['source'], 'download_url': manifest['download_url'], 'archive_bytes': archive.stat().st_size, 'archive_sha256': archive_sha, 'archive_crc_all_members_passed': True, 'original_tables': sorted(table_names), 'images': len(image_manifest), 'image_validation': 'Pillow Image.verify for every image' if Image else 'archive CRC only; Pillow unavailable', 'dimensions': dict(dimensions), 'unique_image_ids_in_votes': len(voted_ids), 'missing_image_ids': dict(missing_ids), 'counts': dict(counts), 'by_category': {k: dict(v) for k,v in sorted(by_category.items())}, 'timestamp_min': min_timestamp, 'timestamp_max': max_timestamp, 'votes_fields': fields, 'category_mapping_source': 'Official archive studies.tsv; fixed ID map cross-checked against author-released Figshare vote records', 'notes': ['Original archive and original TSV tables are retained unchanged.', 'Image bytes are extracted unchanged; only filenames are normalized to location ID.', 'Every human vote is kept as a separate record. Repeated image pairs are not aggregated.', 'Equal votes are in pairs_ties.jsonl.gz and excluded from pairs_binary.jsonl.gz.', 'Votes with absent image IDs are preserved separately rather than silently dropped.', 'qscores.tsv is not used to create pairwise target labels.', 'No train/validation/test split has been imposed.']}
if not image_manifest or not counts['raw_votes']:
    raise RuntimeError('Empty dataset')
info['raw_choice_histogram'] = dict(choice_histogram)
info['notes'].append('Invalid choice/study records are preserved in votes_invalid.jsonl.gz and excluded from usable indexes without guessing labels.')
if counts['usable_binary'] + counts['usable_ties'] + counts['votes_missing_images'] + counts['invalid_votes'] != counts['raw_votes']:
    raise RuntimeError('Vote accounting does not balance')
write_json(ANN / 'dataset_info.json', info)
for filename in ('pairs_binary.jsonl.gz', 'pairs_ties.jsonl.gz', 'votes_missing_images.jsonl.gz', 'votes_invalid.jsonl.gz'):
    n = 0
    with gzip.open(ANN / filename, 'rt') as f:
        for line in f:
            if not isinstance(json.loads(line), dict):
                raise RuntimeError('Invalid JSONL record')
            n += 1
    expected_count = {'pairs_binary.jsonl.gz': counts['usable_binary'], 'pairs_ties.jsonl.gz': counts['usable_ties'], 'votes_missing_images.jsonl.gz': counts['votes_missing_images'], 'votes_invalid.jsonl.gz': counts['invalid_votes']}[filename]
    if n != expected_count:
        raise RuntimeError(f'JSONL row count mismatch: {filename}: {n} != {expected_count}')
    log(f'Verified JSONL/GZIP: {filename}: {n} records')

readme = f'''# Place Pulse 2.0 — original images and human pairwise votes

Source: {manifest['source']}
Official archive URL: {manifest['download_url']}

This directory was prepared for image-pair VLM evaluation. The current author-hosted archive is preserved; it must not be assumed to match the exact 2016 paper snapshot.

## Verified contents

- Archive: {archive.stat().st_size:,} bytes; SHA-256 `{archive_sha}`; every ZIP member passed CRC verification.
- Original images: {len(image_manifest):,}; no cropping, resizing or resampling.
- Raw vote records: {counts['raw_votes']:,}.
- Usable strict binary votes: {counts['usable_binary']:,}.
- Usable equality votes: {counts['usable_ties']:,} (kept separately).
- Votes missing one or both images: {counts['votes_missing_images']:,}; exact IDs/counts in `annotations/dataset_info.json`.
- Invalid label/study records: {counts['invalid_votes']:,}; preserved separately, no inferred corrections.
- Recorded timestamp range: {min_timestamp} to {max_timestamp} (source strings, not converted).
- Image validation: {info['image_validation']}.

## Files

- `raw/place-pulse-2.0.zip`: unchanged author archive.
- `raw/SHA256SUMS`: locally computed checksum (not an upstream published checksum).
- `raw/tables/`: unchanged votes.tsv, studies.tsv, locations.tsv, places.tsv, qscores.tsv and any accompanying text.
- `images/<location_id>.jpg`: original image bytes, ID-based names.
- `annotations/images.jsonl`: ID, path, original filename, city, coordinates from filename, and dimensions.
- `annotations/categories.json`: original studies table plus canonical category/prompt.
- `annotations/pairs_binary.jsonl.gz`: one human vote per row; `choice` is left/right.
- `annotations/pairs_ties.jsonl.gz`: corresponding equal votes.
- `annotations/votes_missing_images.jsonl.gz`: original rows excluded only because image files are absent.
- `annotations/votes_invalid.jsonl.gz`: nonstandard label/study records with reasons and original source rows.
- `annotations/dataset_info.json`: complete counts, category breakdown, validation and provenance.
- `scripts/prepare_dataset.py`: reproducible preparation/verification script.

## Loading a binary pair

```python
import gzip, json
from pathlib import Path
root = Path('/media/shawn/C/place-pulse-2.0')
with gzip.open(root / 'annotations/pairs_binary.jsonl.gz', 'rt') as f:
    row = json.loads(next(f))
print(row['question'], root / row['left_image'], root / row['right_image'], row['choice'])
```

The six questions concern safety, liveliness, boredom, wealth, depression and beauty. A winner for `boring` or `depressing` means more boring/depressing, not more desirable. These are perceived attributes, not objective measures of safety or wealth. `qscores.tsv` is not used as human pairwise ground truth. Original vote repetitions and disagreements remain; no split or majority-vote aggregation was imposed. When later creating evaluation splits, account for repeated image IDs/pairs and geographic overlap.

Citation: Dubey, A., Naik, N., Parikh, D., Raskar, R., & Hidalgo, C. A. (2016). Deep Learning the City: Quantifying Urban Perception at a Global Scale. ECCV, 196–212. Dataset/image reuse terms remain those of the original sources; no new license is assigned by this preparation.
'''
(ROOT / 'README.md').write_text(readme)
write_json(ROOT / 'READY.json', {'status': 'complete', 'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'archive_sha256': archive_sha, 'images': len(image_manifest), **dict(counts)})
log('DATASET READY: ' + json.dumps(info['counts']))
