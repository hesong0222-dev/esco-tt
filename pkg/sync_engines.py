"""Copy the research engines (esco/tt, esco/dense, esco/approx) into src/escott/_engines/.

The package vendors these files unchanged so that the numbers it prints are the numbers in the paper.
Run after any change to the research code:  python sync_engines.py
"""
import hashlib
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ESCO = os.path.dirname(HERE)
DST = os.path.join(HERE, "src", "escott", "_engines")
FILES = [("tt/esco_tt.py", "esco_tt.py"), ("tt/ttcore.py", "ttcore.py"),
         ("dense/esco_dense.py", "esco_dense.py"), ("approx/mascot.py", "mascot.py")]

lines = []
for src, dst in FILES:
    shutil.copyfile(os.path.join(ESCO, src), os.path.join(DST, dst))
    h = hashlib.sha256(open(os.path.join(DST, dst), "rb").read()).hexdigest()[:16]
    lines.append(f"{dst}  <-  esco/{src}  sha256:{h}")
open(os.path.join(DST, "SOURCES.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
