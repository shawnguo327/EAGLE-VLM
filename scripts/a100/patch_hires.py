"""Add full-resolution conditions to judge_bench.py: tri_hires (A/B/T) and score_hires (pointwise), long side <= HIRES (1536)
instead of MAXSIDE (768).  Existing conditions are unchanged (score keeps MAXSIDE)."""
import sys
f = sys.argv[1]; s = open(f).read()
if "tri_hires" in s:
    print("already patched"); sys.exit()
rep = [
 ('    bin_hires  A/B with images at HIRES px (more visual evidence; use --tiling for InternVL)\n',
  '    bin_hires  A/B with images at HIRES px (more visual evidence; use --tiling for InternVL)\n'
  '    tri_hires  A/B/T with images at HIRES px; score_hires: pointwise 1..9 with the candidate at HIRES px (resolution ablation)\n'),
 ('PAIR = {"bin", "tri", "str7", "bin_blind", "bin_lowres", "bin_hires", ',
  'PAIR = {"bin", "tri", "str7", "bin_blind", "bin_lowres", "bin_hires", "tri_hires", '),
 ('IMG_ONLY = {"tri_jpeg", "tri_crop", "bin_lowres", "bin_hires", "bin_desc"}',
  'IMG_ONLY = {"tri_jpeg", "tri_crop", "bin_lowres", "bin_hires", "bin_desc", "tri_hires", "score_hires"}'),
 ('"score": [str(i) for i in range(1, 10)]}.get(c, ["A", "B"])',
  '"score": [str(i) for i in range(1, 10)], "score_hires": [str(i) for i in range(1, 10)]}.get(c, ["A", "B"])'),
 ('    ms = {"bin_lowres": LOWRES, "bin_hires": HIRES}.get(cond, MAXSIDE)',
  '    ms = {"bin_lowres": LOWRES, "bin_hires": HIRES, "tri_hires": HIRES, "score_hires": HIRES}.get(cond, MAXSIDE)'),
 ('{"type": "text", "text": "Candidate image:"}, {"type": "image", "image": load_img(p)},',
  '{"type": "text", "text": "Candidate image:"}, {"type": "image", "image": load(p)},'),
]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(f, "w").write(s); print("patched", f)
