# Place Pulse 2.0 — original images and human pairwise votes

Source: https://centerforcollectivelearning.org/urbanperception
Official archive URL: https://www.dropbox.com/scl/fi/j19aw939mvgz2prdfo722/place-pulse-2.0.zip?rlkey=dpo90zmmi35f26ahejo4mb4m5&dl=1

This directory was prepared for image-pair VLM evaluation. The current author-hosted archive is preserved; it must not be assumed to match the exact 2016 paper snapshot.

## Verified contents

- Archive: 2,640,932,754 bytes; SHA-256 `aeebc8673b7daaa240dc5dda3aaf5c8ae50b12bd3e4c739c5721a63dea125b8c`; every ZIP member passed CRC verification.
- Original images: 110,988; no cropping, resizing or resampling.
- Raw vote records: 1,555,561.
- Usable strict binary votes: 1,342,992.
- Usable equality votes: 203,017 (kept separately).
- Votes missing one or both images: 9,529; exact IDs/counts in `annotations/dataset_info.json`.
- Invalid label/study records: 23; preserved separately, no inferred corrections.
- Recorded timestamp range: 2013-05-28 15:27:06.914000 to 2018-08-27 01:04:21.474000 (source strings, not converted).
- Image validation: Pillow Image.verify for every image.

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
