"""Small helper for paper_figures.ipynb (headless use; the notebook itself stays the source).
  nbtool.py list                      cell ids and their first line
  nbtool.py run c14 c20               run the setup cell (c02) and the given cells without Jupyter; figures go to OUT
  nbtool.py patch edits.json          apply [{"cell": "c14", "old": "...", "new": "..."}, ...]; every old text must occur exactly once
Environment: FIG_OUT / FIG_TABLE redirect the outputs (see the setup cell)."""
import json, os, sys
NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figures.ipynb")
nb = json.load(open(NB)); cell = {c["id"]: c for c in nb["cells"]}; src = lambda i: "".join(cell[i]["source"])
cmd = sys.argv[1]
if cmd == "list":
    for c in nb["cells"]:
        print(c["id"], c["cell_type"][:4], "|", "".join(c["source"]).strip().splitlines()[0][:90])
elif cmd == "run":
    import matplotlib; matplotlib.use("Agg"); g = {"__name__": "__main__"}
    for i in ["c02"] + sys.argv[2:]:
        exec(compile(src(i), i, "exec"), g)
elif cmd == "patch":
    for e in json.load(open(sys.argv[2])):
        s = src(e["cell"]); assert s.count(e["old"]) == 1, (e["cell"], e["old"][:60], s.count(e["old"]))
        cell[e["cell"]]["source"] = s.replace(e["old"], e["new"]).splitlines(True)
    json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False); print("patched", NB)
