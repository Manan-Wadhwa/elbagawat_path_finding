import os
import json

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Task1_FullSite.ipynb")
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

src2 = "".join(nb["cells"][2]["source"])
idx = src2.find("def bayesian_ensemble")
# Print the pairs section
seg = src2[idx:idx+2000]
print(repr(seg[600:1300]))
