"""Lightweight twotensors-equivalent passes (no research repo).
  local_tools.py downscale <in> <out_dir> <max_long_side>   # twotensors "1024px downscale"
  local_tools.py metadata  <in> <out_dir>                   # strip EXIF/XMP/C2PA (its "Metadata & C2PA Remover")
"""
import shutil, subprocess, sys
from pathlib import Path
from PIL import Image

op, src, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
dst = out_dir / f"{src.stem}.png"

if op == "downscale":
    m = int(sys.argv[4]); img = Image.open(src).convert("RGB")
    w, h = img.size; s = min(1.0, m / max(w, h))
    img.resize((max(1, round(w*s)), max(1, round(h*s))), Image.LANCZOS).save(dst)
    print(f"wrote {dst} ({img.size} -> {Image.open(dst).size})")
else:  # metadata: re-encode pixels only (drops EXIF/XMP), then exiftool/c2patool belt-and-suspenders
    Image.open(src).convert("RGB").save(dst)  # PIL re-save carries no metadata
    for tool, args in (("exiftool", ["-all=", "-overwrite_original", str(dst)]),
                       ("c2patool", [str(dst), "--remove-manifest", "--output", str(dst), "--force"])):
        if shutil.which(tool):
            subprocess.run([tool, *args], capture_output=True)
    print(f"wrote {dst} (EXIF/XMP/C2PA stripped)")
