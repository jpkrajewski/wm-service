"""Robustness eval loop for YOUR OWN watermark/detector.

  python eval.py --images <dir-of-your-watermarked-images> \
                 --detector "python /path/to/your_detector.py {img}" \
                 --attacks regen-diffusion,unmarker-synthid,waves-advemb \
                 --out report.json

Your detector is any command that prints ONE number in [0,1] (probability the
watermark / AI-origin is still detected) for the image path substituted at {img}.
For each image: score the clean image, run each attack, score the attacked image.
Report = detection rate before vs after per attack, plus PSNR/SSIM image quality.
A big drop with high PSNR/SSIM = your watermark is NOT robust to that attack.
"""
import argparse, json, shlex, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
import attacks as A

def detect(cmd_tmpl, img):
    out = subprocess.run(shlex.split(cmd_tmpl.replace("{img}", str(img))), capture_output=True, text=True)
    try: return float(out.stdout.strip().split()[-1])
    except Exception: raise SystemExit(f"detector did not print a number for {img}:\n{out.stdout}\n{out.stderr}")

def psnr(a, b):
    a, b = np.asarray(a, float), np.asarray(b.resize(a.size), float)
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255 * 255 / mse)

def ssim(a, b):  # global SSIM, grayscale
    a = np.asarray(a.convert("L"), float); b = np.asarray(b.convert("L").resize(a.shape[::-1]), float)
    ma, mb = a.mean(), b.mean(); va, vb = a.var(), b.var(); cov = ((a - ma) * (b - mb)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2*ma*mb + c1) * (2*cov + c2)) / ((ma*ma + mb*mb + c1) * (va + vb + c2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--detector", required=True, help='command with {img}, prints a 0..1 score')
    ap.add_argument("--attacks", default="regen-diffusion,waves-advemb,purify-diffpure")
    ap.add_argument("--strength", default="")
    ap.add_argument("--threshold", type=float, default=0.5, help="score >= threshold counts as detected")
    ap.add_argument("--out", default="report.json")
    a = ap.parse_args()

    imgs = [p for p in sorted(Path(a.images).iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if not imgs: raise SystemExit(f"no images in {a.images}")
    attacks = [x for x in a.attacks.split(",") if x]
    for x in attacks:
        if x not in A.ATTACKS: raise SystemExit(f"unknown attack {x}; have: {', '.join(A.ATTACKS)}")
        if not A.ready(x): raise SystemExit(f"attack {x} not installed (run bootstrap.sh)")

    work = Path("eval_out"); work.mkdir(exist_ok=True)
    rows, base_detected = [], 0
    for img in imgs:
        base = detect(a.detector, img); base_detected += base >= a.threshold
        for aid in attacks:
            od = work / aid; argv, cwd = A.command(aid, str(img), str(od), a.strength)
            r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
            out = od / f"{img.stem}.png"
            if r.returncode != 0 or not out.exists():
                rows.append({"image": img.name, "attack": aid, "error": (r.stdout + r.stderr)[-300:]}); continue
            sc = detect(a.detector, out); orig_im, att_im = Image.open(img).convert("RGB"), Image.open(out).convert("RGB")
            rows.append({"image": img.name, "attack": aid, "score_before": base, "score_after": sc,
                         "still_detected": sc >= a.threshold, "psnr": round(psnr(orig_im, att_im), 2),
                         "ssim": round(float(ssim(orig_im, att_im)), 4)})

    summary = {}
    for aid in attacks:
        rs = [r for r in rows if r["attack"] == aid and "score_after" in r]
        if not rs: continue
        summary[aid] = {"n": len(rs),
                        "detected_before": base_detected, "of": len(imgs),
                        "detected_after": sum(r["still_detected"] for r in rs),
                        "evasion_rate": round(1 - sum(r["still_detected"] for r in rs) / len(rs), 3),
                        "avg_psnr": round(sum(r["psnr"] for r in rs) / len(rs), 2),
                        "avg_ssim": round(sum(r["ssim"] for r in rs) / len(rs), 4)}
    report = {"threshold": a.threshold, "images": len(imgs), "summary": summary, "rows": rows}
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\n{'attack':20} {'evaded':>7} {'PSNR':>7} {'SSIM':>7}")
    for aid, s in summary.items():
        print(f"{aid:20} {s['evasion_rate']*100:6.1f}% {s['avg_psnr']:7.2f} {s['avg_ssim']:7.4f}")
    print(f"\nfull report: {a.out}")

if __name__ == "__main__":
    main()
