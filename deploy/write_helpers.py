"""Write the per-repo run_one.py runners, the UI server, and the eval harness.

Called by bootstrap.sh as: python3 write_helpers.py <ROOT>
Keeping file bodies here (not in bash heredocs) avoids shell-quoting hazards.
"""

import os
import sys
import stat
from pathlib import Path

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/wm"
REPOS = os.path.join(ROOT, "repos")

FILES: dict[str, str] = {}

# ---------------------------------------------------------------- run_one runners
FILES[f"{REPOS}/WatermarkAttacker/run_one.py"] = r'''
"""One image through a regeneration attack (Zhao et al.).
  run_one.py diffusion <in> <out_dir> <noise_step>
  run_one.py vae <in> <out_dir> <quality>
"""
import sys
from pathlib import Path
import torch
from PIL import Image
from regen_pipe import ReSDPipeline
from wmattacker import DiffWMAttacker, VAEWMAttacker

method, src, out_dir, strength = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
out_dir.mkdir(parents=True, exist_ok=True)
dst = out_dir / f"{src.stem}.png"
device = "cuda" if torch.cuda.is_available() else "cpu"
original = Image.open(src).convert("RGB")
tmp_in = out_dir / "_in.png"
if method == "vae":
    original.save(tmp_in)
    VAEWMAttacker("bmshj2018-factorized", quality=int(strength), metric="mse", device=device).attack([str(tmp_in)], [str(dst)])
else:
    w, h = original.size
    nw, nh = max(64, w // 64 * 64), max(64, h // 64 * 64)
    original.resize((nw, nh), Image.LANCZOS).save(tmp_in)
    try:
        pipe = ReSDPipeline.from_pretrained("sd2-community/stable-diffusion-2-1", torch_dtype=torch.float16, variant="fp16")
    except (OSError, ValueError):
        pipe = ReSDPipeline.from_pretrained("sd2-community/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    DiffWMAttacker(pipe, batch_size=1, noise_step=int(strength), captions={}).attack([str(tmp_in)], [str(dst)])
    Image.open(dst).resize((w, h), Image.LANCZOS).save(dst)
tmp_in.unlink(missing_ok=True)
print(f"wrote {dst}")
'''

FILES[f"{REPOS}/watermark_robustness/run_one.py"] = r'''
"""One image through diffusion purification (Saberi et al.).
  run_one.py diffpure <in> <out_dir> <steps>
  run_one.py diffpure-latent <in> <out_dir> <strength>
"""
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms
from evaluate_watermark import DiffPure, ImageRephrase

method, src, out_dir, strength = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)
dst = out_dir / f"{src.stem}.png"
original = Image.open(src).convert("RGB")
if method == "diffpure":
    to_tensor = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    result = transforms.ToPILImage()(DiffPure(steps=strength)(to_tensor(original)).clamp(0, 1))
else:
    result = ImageRephrase(img_size=512, strength=strength)(original)
result.convert("RGB").resize(original.size, Image.LANCZOS).save(dst)
print(f"wrote {dst}")
'''

FILES[f"{REPOS}/WAVES/run_one.py"] = r'''
"""One image through a WAVES attack (An et al.).
  run_one.py adv-emb <encoder> <in> <out_dir> <eps>      # klvae8 sdxlvae clip resnet18
  run_one.py distortion <type> <in> <out_dir> <strength0to1>
"""
import sys
from pathlib import Path
sys.path[:0] = [".", "adversarial"]
import torch
from PIL import Image
from torchvision import transforms

method = sys.argv[1]
N_STEPS, ALPHA = 200, 0.05
device = "cuda" if torch.cuda.is_available() else "cpu"

def fit8(img, m=1024):
    w, h = img.size
    s = min(1.0, m / max(w, h))
    return img.resize((max(8, round(w*s/8)*8), max(8, round(h*s/8)*8)), Image.LANCZOS)

if method == "adv-emb":
    enc, src, out_dir, eps = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), float(sys.argv[5]) / 255
    from embedding import WarmupPGDEmbedding
    from feature_extractors import ClipEmbedding, ResNet18Embedding, VAEEmbedding
    model = {"klvae8": lambda: VAEEmbedding("stabilityai/sd-vae-ft-mse"),
             "sdxlvae": lambda: VAEEmbedding("stabilityai/sdxl-vae"),
             "clip": ClipEmbedding, "resnet18": lambda: ResNet18Embedding("last")}[enc]().to(device).eval()
    original = Image.open(src).convert("RGB"); prepared = fit8(original)
    attack = WarmupPGDEmbedding(model=model, eps=eps, alpha=ALPHA*eps, steps=N_STEPS, device=device)
    x = transforms.ToTensor()(prepared).unsqueeze(0).to(device)
    result = transforms.ToPILImage()(attack.forward(x)[0].cpu())
else:
    dtype, src, out_dir, s = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), float(sys.argv[5])
    from distortions import apply_distortion
    original = Image.open(src).convert("RGB")
    result = apply_distortion([original], dtype, strength=s, distortion_seed=0)[0]

out_dir.mkdir(parents=True, exist_ok=True)
dst = out_dir / f"{src.stem}.png"
result.convert("RGB").resize(original.size, Image.LANCZOS).save(dst)
print(f"wrote {dst}")
'''

FILES[f"{REPOS}/ai-watermark/run_one.py"] = r'''
"""One image through UnMarker (Kassis & Hengartner). run_one.py <preset> <in> <out_dir>
preset picks the scheme's published stage parameters; its detector is not loaded."""
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import torch, yaml
from modules.attack.unmark import UnMark
from PIL import Image
from torchvision import transforms

preset, src, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
device = "cuda" if torch.cuda.is_available() else "cpu"
conf = yaml.load(open(f"attack_configs/{preset}.yaml"), Loader=yaml.Loader)["UnMarker"]
sel = conf.get("stage_selector") or ["preprocess", "stage1", "stage2"]
original = Image.open(src).convert("RGB")
w, h = original.size; s = min(1.0, 1024 / max(w, h))
nw, nh = max(8, round(w*s/8)*8), max(8, round(h*s/8)*8)
prepared = original.resize((nw, nh), Image.LANCZOS)
u = UnMark.__new__(UnMark); u.device, u.evaluator = device, None
crop = conf.get("preprocess_args", {}).get("crop_ratio") if "preprocess" in sel else None
u.transforms = (transforms.Compose([transforms.CenterCrop((int(crop[0]*nh), int(crop[1]*nw))), transforms.Resize((nh, nw), antialias=None)])
                if crop else transforms.Lambda(lambda x: x))
u.stage1, u.stage1_thresh = u._load_stage(conf.get("stage1_args") if "stage1" in sel else None, stage_name="high_freq")
u.stage2, u.stage2_thresh = u._load_stage(conf.get("stage2_args") if "stage2" in sel else None, stage_name="low_freq")
x = transforms.ToTensor()(prepared).unsqueeze(0).to(device)
_, removed = u.do_batch(x)
out_dir.mkdir(parents=True, exist_ok=True)
dst = out_dir / f"{src.stem}.png"
transforms.ToPILImage()(torch.clamp(removed[0], 0, 1).cpu()).resize(original.size, Image.LANCZOS).save(dst)
print(f"wrote {dst}")
'''

# ---------------------------------------------------------------- attack registry (shared)
FILES[f"{ROOT}/attacks.py"] = r'''
"""The attack registry: maps an attack id to the repo venv + run_one invocation.
Shared by serve.py (UI) and eval.py (your-detector loop)."""
import os
REPOS = os.path.join(os.path.dirname(__file__), "repos")

def _py(repo): return os.path.join(REPOS, repo, ".venv", "bin", "python")

ROOT = os.path.dirname(__file__)
# "_local" attacks are the lightweight twotensors-style passes (no research repo);
# run them in the guaranteed-light API venv (pillow+numpy), NOT the heavy/flaky
# WatermarkAttacker venv, so downscale/metadata-strip work even if that repo's
# torch install partially failed.
_LOCAL_PY = os.path.join(ROOT, ".venv-api", "bin", "python")

# id -> (repo, argv builder given (in_path, out_dir, strength)). argv[0] is the
# script name, run from the repo dir (or ROOT for "_local").
ATTACKS = {
    "regen-diffusion":  ("WatermarkAttacker",    lambda i, o, s: ["run_one.py", "diffusion", i, o, s or "60"]),
    "regen-vae":        ("WatermarkAttacker",    lambda i, o, s: ["run_one.py", "vae", i, o, s or "3"]),
    "unmarker-synthid": ("ai-watermark",         lambda i, o, s: ["run_one.py", "SynthID", i, o]),
    "unmarker-treering":("ai-watermark",         lambda i, o, s: ["run_one.py", "TreeRing", i, o]),
    "unmarker-stega":   ("ai-watermark",         lambda i, o, s: ["run_one.py", "StegaStamp", i, o]),
    "waves-advemb":     ("WAVES",                lambda i, o, s: ["run_one.py", "adv-emb", "klvae8", i, o, s or "8"]),
    "waves-jpeg":       ("WAVES",                lambda i, o, s: ["run_one.py", "distortion", "compression", i, o, s or "0.5"]),
    "waves-blur":       ("WAVES",                lambda i, o, s: ["run_one.py", "distortion", "blurring", i, o, s or "0.5"]),
    "purify-diffpure":  ("watermark_robustness", lambda i, o, s: ["run_one.py", "diffpure", i, o, s or "0.2"]),
    # twotensors-equivalent lightweight passes
    "downscale":        ("_local",               lambda i, o, s: ["local_tools.py", "downscale", i, o, s or "1024"]),
    "metadata-strip":   ("_local",               lambda i, o, s: ["local_tools.py", "metadata", i, o]),
}

STRENGTH = {
    "regen-diffusion": "noise step 30-200 (default 60)", "regen-vae": "quality 1-8, lower=stronger (3)",
    "unmarker-synthid": "", "unmarker-treering": "", "unmarker-stega": "",
    "waves-advemb": "eps /255 (8)", "waves-jpeg": "0-1 (0.5)", "waves-blur": "0-1 (0.5)",
    "purify-diffpure": "0.1-0.4 (0.2)", "downscale": "max long side px (1024)", "metadata-strip": "",
}

def command(attack_id, in_path, out_dir, strength=""):
    repo, build = ATTACKS[attack_id]
    argv = build(in_path, out_dir, strength)
    if repo == "_local":
        return [_LOCAL_PY, os.path.join(ROOT, argv[0])] + argv[1:], ROOT
    return [_py(repo), argv[0]] + argv[1:], os.path.join(REPOS, repo)

# attacks that cannot actually run until their weights are downloaded (--weights);
# ready() reports them not-installed so GET /attacks is honest and callers get a
# clean 409 instead of a 500 at call time.
_WEIGHTS = {
    "unmarker-synthid":  os.path.join(REPOS, "ai-watermark", "pretrained_models", "loss_provider", "weights"),
    "unmarker-treering": os.path.join(REPOS, "ai-watermark", "pretrained_models", "loss_provider", "weights"),
    "unmarker-stega":    os.path.join(REPOS, "ai-watermark", "pretrained_models", "loss_provider", "weights"),
    "purify-diffpure":   os.path.join(REPOS, "watermark_robustness", "DiffPure", "pretrained", "guided_diffusion", "256x256_diffusion_uncond.pt"),
}

def ready(attack_id):
    repo, _ = ATTACKS[attack_id]
    if repo == "_local":
        base = os.path.exists(_LOCAL_PY) and os.path.exists(os.path.join(ROOT, "local_tools.py"))
    else:
        base = os.path.exists(_py(repo)) and os.path.exists(os.path.join(REPOS, repo, "run_one.py"))
    need = _WEIGHTS.get(attack_id)
    return base and (need is None or os.path.exists(need))
'''

# ---------------------------------------------------------------- UI server (stdlib)
FILES[f"{ROOT}/app.py"] = r'''
"""Watermark-robustness test API (FastAPI). Typed schemas; OpenAPI at /openapi.json,
interactive docs at /docs - an agent reads those to drive it, no SSH/UI needed.

The attacks live in per-repo venvs with conflicting deps, so this app is a thin
orchestrator: it shells each attack into its repo venv (see attacks.py). Run with
  uvicorn app:app --host 0.0.0.0 --port 8077
"""
from __future__ import annotations

import asyncio
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

import attacks as A

app = FastAPI(title="Watermark Robustness API",
              version="1.0",
              description="Run watermark/detector attacks on YOUR images and score YOUR detector. "
                          "Same techniques as twotensors.ai, for testing robustness.")

# ----------------------------- schemas
class AttackInfo(BaseModel):
    id: str
    ready: bool = Field(description="True if the repo venv + runner are installed")
    strength: str = Field("", description="meaning/units of the strength param")
    kind: Literal["black-box", "local"] = "black-box"

class DetectorSpec(BaseModel):
    kind: Literal["url", "cmd", "sightengine", "hive"] = Field(
        "url", description="'url'=POST image to your webhook (returns {score}); 'cmd'=local command with {img} printing 0..1; "
                           "'sightengine'/'hive'=call that commercial AI-image detector (needs its keys below)")
    value: str = Field("", description="webhook URL (kind=url) or command template with {img} (kind=cmd); unused for sightengine/hive")
    api_user: str = Field("", description="Sightengine api_user")
    api_secret: str = Field("", description="Sightengine api_secret")
    api_key: str = Field("", description="Hive API key")

class AttackRequest(BaseModel):
    attack: str = Field(description="attack id from GET /attacks")
    image_b64: str = Field(description="input image, base64 (any common format)")
    strength: str = Field("", description="optional; units per GET /attacks")

class ImageResult(BaseModel):
    attack: str
    strength: str = ""
    image_b64: str = Field(description="attacked image, base64 PNG")
    psnr: float = Field(description="PSNR vs the input (image quality kept)")

class DetectRequest(BaseModel):
    detector: DetectorSpec
    image_b64: str
    threshold: float = 0.5

class DetectResult(BaseModel):
    score: float = Field(description="detector output in [0,1]; higher = more detected")
    detected: bool

class EvalRequest(BaseModel):
    images_b64: list[str] = Field(description="input images, base64")
    attacks: list[str] = Field(default_factory=list, description="attack ids; empty = all")
    detector: DetectorSpec
    threshold: float = 0.5

class AttackSummary(BaseModel):
    n: int; detected_before: int; detected_after: int; of: int
    evasion_rate: float = Field(description="fraction of images that evaded the detector after the attack")
    avg_psnr: float

class EvalReport(BaseModel):
    threshold: float
    images: int
    summary: dict[str, AttackSummary]
    rows: list[dict]

class TwoTensorsRequest(BaseModel):
    image_b64: str
    strength: Literal["strong", "quality"] = Field("strong", description="detector-bypass aggressiveness")
    synthid: bool = Field(False, description="add a SynthID/regeneration removal stage")
    downscale: Optional[int] = Field(1024, description="max long side px, or null to skip (twotensors '1024px downscale')")
    strip_metadata: bool = Field(True, description="strip EXIF/XMP/C2PA on export")
    detector: Optional[DetectorSpec] = Field(None, description="if set, report detector score before/after ('AI slop score')")

class TwoTensorsResult(BaseModel):
    image_b64: str
    steps: list[str]
    slop_before: Optional[float] = None
    slop_after: Optional[float] = None
    psnr: float

# ----------------------------- helpers
def _run(attack_id: str, img: bytes, strength: str = "") -> bytes:
    if attack_id not in A.ATTACKS:
        raise HTTPException(400, f"unknown attack '{attack_id}'")
    if not A.ready(attack_id):
        raise HTTPException(409, f"attack '{attack_id}' not installed (run bootstrap.sh [--weights])")
    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / "in.png"; od = Path(td) / "out"; od.mkdir()
        Image.open(__import__("io").BytesIO(img)).convert("RGB").save(ip)
        argv, cwd = A.command(attack_id, str(ip), str(od), strength)
        p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
        outs = sorted(od.glob("*.png"))
        if p.returncode != 0 or not outs:
            raise HTTPException(500, f"{attack_id} failed: {(p.stdout + p.stderr)[-800:]}")
        return outs[0].read_bytes()

def _psnr(a: bytes, b: bytes) -> float:
    ia = np.asarray(Image.open(__import__("io").BytesIO(a)).convert("RGB"), float)
    ib = np.asarray(Image.open(__import__("io").BytesIO(b)).convert("RGB").resize(
        Image.open(__import__("io").BytesIO(a)).size), float)
    mse = ((ia - ib) ** 2).mean()
    return 99.0 if mse == 0 else round(float(10 * np.log10(255 * 255 / mse)), 2)

async def _detect(spec: DetectorSpec, img: bytes) -> float:
    """Return a 0..1 'this is AI/watermarked' score from the chosen detector."""
    if spec.kind == "url":                                   # your own webhook -> {"score": 0..1}
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(spec.value, files={"file": ("i.png", img, "image/png")})
            r.raise_for_status()
            return float(r.json()["score"])
    if spec.kind == "sightengine":                           # type.ai_generated in [0,1]
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post("https://api.sightengine.com/1.0/check.json",
                             data={"models": "genai", "api_user": spec.api_user, "api_secret": spec.api_secret},
                             files={"media": ("i.png", img, "image/png")})
        j = r.json()
        if j.get("status") != "success":
            raise HTTPException(502, f"sightengine: {j}")
        return float(j["type"]["ai_generated"])
    if spec.kind == "hive":                                  # Hive v2 sync task -> classes[].score
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post("https://api.thehive.ai/api/v2/task/sync",
                             headers={"Authorization": f"token {spec.api_key}"},
                             files={"media": ("i.png", img, "image/png")})
        try:
            classes = r.json()["status"][0]["response"]["output"][0]["classes"]
            d = {c["class"]: float(c["score"]) for c in classes}
        except Exception:
            raise HTTPException(502, f"hive: unexpected response {r.text[:400]}")
        for k, v in d.items():
            if "ai" in k.lower() and "not" not in k.lower():
                return v
        if "not_ai_generated" in d:
            return 1.0 - d["not_ai_generated"]
        raise HTTPException(502, f"hive: no ai-generated class in {list(d)}")
    with tempfile.TemporaryDirectory() as td:                # local command printing a number
        ip = Path(td) / "i.png"; ip.write_bytes(img)
        p = subprocess.run(spec.value.replace("{img}", str(ip)), shell=True, capture_output=True, text=True)
        try: return float(p.stdout.strip().split()[-1])
        except Exception: raise HTTPException(500, f"detector cmd gave no number: {p.stdout}{p.stderr}")

def _b(x: bytes) -> str: return base64.b64encode(x).decode()

# ----------------------------- endpoints
@app.get("/attacks", response_model=list[AttackInfo])
def list_attacks():
    """Every attack and whether it is installed."""
    return [AttackInfo(id=k, ready=A.ready(k), strength=A.STRENGTH.get(k, ""),
                       kind="local" if A.ATTACKS[k][0] == "_local" else "black-box") for k in A.ATTACKS]

@app.post("/attack", response_model=ImageResult)
async def attack(req: AttackRequest):
    """Run ONE attack on ONE image. Returns the attacked image (base64) + PSNR."""
    src = base64.b64decode(req.image_b64)
    out = await asyncio.to_thread(_run, req.attack, src, req.strength)
    return ImageResult(attack=req.attack, strength=req.strength, image_b64=_b(out), psnr=_psnr(src, out))

@app.post("/attack/raw")
async def attack_raw(req: AttackRequest):
    """Same as /attack but returns raw PNG bytes (for piping to a file)."""
    out = await asyncio.to_thread(_run, req.attack, base64.b64decode(req.image_b64), req.strength)
    return Response(out, media_type="image/png")

@app.post("/detect", response_model=DetectResult)
async def detect(req: DetectRequest):
    """Score an image with YOUR detector (a webhook URL or a local command)."""
    s = await _detect(req.detector, base64.b64decode(req.image_b64))
    return DetectResult(score=s, detected=s >= req.threshold)

@app.post("/eval", response_model=EvalReport)
async def evaluate(req: EvalRequest):
    """Run every attack on every image, score with your detector, report evasion + PSNR."""
    imgs = [base64.b64decode(x) for x in req.images_b64]
    atks = [a for a in (req.attacks or list(A.ATTACKS)) if a in A.ATTACKS]
    if not imgs: raise HTTPException(400, "images_b64 is empty")
    base_det = sum((await _detect(req.detector, im)) >= req.threshold for im in imgs)
    rows = []
    for i, im in enumerate(imgs):
        for aid in atks:
            try: out = await asyncio.to_thread(_run, aid, im, "")
            except HTTPException as e: rows.append({"image": i, "attack": aid, "error": str(e.detail)[:200]}); continue
            sc = await _detect(req.detector, out)
            rows.append({"image": i, "attack": aid, "score_after": sc,
                         "still_detected": sc >= req.threshold, "psnr": _psnr(im, out)})
    summary = {}
    for aid in atks:
        rs = [r for r in rows if r["attack"] == aid and "score_after" in r]
        if rs:
            det_after = sum(r["still_detected"] for r in rs)
            summary[aid] = AttackSummary(n=len(rs), detected_before=base_det, detected_after=det_after, of=len(imgs),
                                         evasion_rate=round(1 - det_after / len(rs), 3),
                                         avg_psnr=round(sum(r["psnr"] for r in rs) / len(rs), 2))
    return EvalReport(threshold=req.threshold, images=len(imgs), summary=summary, rows=rows)

@app.post("/twotensors", response_model=TwoTensorsResult)
async def twotensors(req: TwoTensorsRequest):
    """Replicate twotensors' chained 'Forensic Cleaner' pass on ONE image:
    (optional SynthID/regeneration removal) -> detector-bypass (adversarial) ->
    optional downscale -> optional EXIF/XMP/C2PA strip. If a detector is given,
    report its score before/after (their 'AI slop score'). For testing how your
    detector holds against the stacked pipeline."""
    src = base64.b64decode(req.image_b64)
    slop_before = await _detect(req.detector, src) if req.detector else None
    img = src; steps = []
    if req.synthid:
        img = await asyncio.to_thread(_run, "unmarker-synthid", img, ""); steps.append("synthid-removal")
    eps = "8" if req.strength == "strong" else "4"
    img = await asyncio.to_thread(_run, "waves-advemb", img, eps); steps.append(f"detector-bypass({req.strength})")
    if req.downscale:
        img = await asyncio.to_thread(_run, "downscale", img, str(req.downscale)); steps.append(f"downscale({req.downscale})")
    if req.strip_metadata:
        img = await asyncio.to_thread(_run, "metadata-strip", img, ""); steps.append("strip-metadata")
    slop_after = await _detect(req.detector, img) if req.detector else None
    return TwoTensorsResult(image_b64=_b(img), steps=steps, slop_before=slop_before,
                            slop_after=slop_after, psnr=_psnr(src, img))
'''

FILES[f"{ROOT}/serve_stdlib.py"] = r'''
"""Deprecated stdlib server (kept for reference). The real API is app.py (FastAPI).
API-first attack server. Stdlib only. Endpoints:

  GET  /api/attacks                      -> {id: {ready, strength}}
  POST /api/attack?attack=<id>&strength= -> body = raw image bytes;
                                            returns the attacked PNG bytes (synchronous).
  POST /api/eval                         -> JSON {images:[<base64>...], attacks:[<id>...],
                                            detector_url:"http://...", threshold:0.5}
                                            runs every attack on every image, POSTs original
                                            and attacked image to detector_url (which returns
                                            {"score": 0..1}), returns a full report JSON.
  GET  /                                 -> optional browser UI (thin client on the same API).
  POST /api/run + GET /api/job/<id>      -> async variant the UI uses.

So an agent/script tests images with plain HTTP calls - no SSH, no browser:
  curl -s -X POST --data-binary @img.png \
       "http://POD:8077/api/attack?attack=regen-diffusion&strength=60" -o out.png
"""
import base64, json, os, subprocess, tempfile, threading, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import attacks as A

ROOT = Path(os.path.dirname(__file__)); DATA = ROOT / "data"
JOBS = {}

def run_attack(aid, img_bytes, suffix=".png", strength=""):
    """Run one attack synchronously. Returns (ok, out_bytes_or_errtext, log)."""
    if aid not in A.ATTACKS: return False, f"unknown attack {aid}", ""
    if not A.ready(aid): return False, f"{aid} not installed", ""
    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / f"in{suffix or '.png'}"; od = Path(td) / "out"; od.mkdir()
        ip.write_bytes(img_bytes)
        argv, cwd = A.command(aid, str(ip), str(od), strength)
        p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
        outs = sorted(od.glob("*.png"))
        if p.returncode == 0 and outs: return True, outs[0].read_bytes(), p.stdout
        return False, (p.stdout + p.stderr)[-2000:], p.stdout

def detector_score(url, img_bytes):
    """POST an image to the user's detector webhook; expect JSON {"score": float}."""
    b = b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"i.png\"\r\n" \
        b"Content-Type: image/png\r\n\r\n" + img_bytes + b"\r\n--b--\r\n"
    req = urllib.request.Request(url, data=b, headers={"Content-Type": "multipart/form-data; boundary=b"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return float(json.loads(r.read())["score"])

def run_job(job, aid, in_bytes, suffix, strength):
    ok, out, log = run_attack(aid, in_bytes, suffix, strength); job["log"] = log
    if ok:
        d = DATA / "out" / job["id"]; d.mkdir(parents=True, exist_ok=True)
        (d / "out.png").write_bytes(out); job["output"] = f"data/out/{job['id']}/out.png"; job["status"] = "done"
    else:
        job["status"] = "failed"; job["log"] = (log + "\n" + out)[-20000:]

def multipart_file(raw, ctype):
    boundary = ctype.split("boundary=", 1)[1].encode()
    fields, fname, fdata = {}, None, None
    for part in raw.split(b"--" + boundary):
        if b"\r\n\r\n" not in part: continue
        head, body = part.split(b"\r\n\r\n", 1); body = body.rstrip(b"\r\n"); disp = head.decode("latin1")
        name = disp.split('name="', 1)[1].split('"', 1)[0] if 'name="' in disp else None
        if 'filename="' in disp: fname = disp.split('filename="', 1)[1].split('"', 1)[0]; fdata = body
        elif name: fields[name] = body
    return fields, fname, fdata

def psnr(a, b):
    import numpy as np
    a, b = np.frombuffer(a, np.uint8).astype(float), np.frombuffer(b, np.uint8).astype(float)
    n = min(len(a), len(b)); mse = ((a[:n] - b[:n]) ** 2).mean() if n else 1
    return 99.0 if mse == 0 else round(10 * (np.log10(255 * 255 / mse)), 2)

class H(BaseHTTPRequestHandler):
    def _s(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def _j(self, o, code=200): self._s(code, json.dumps(o).encode(), "application/json")
    def _q(self):
        from urllib.parse import parse_qs, urlparse
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            f = ROOT / "index.html"
            self._s(200, f.read_bytes() if f.exists() else b"API only. See /api/attacks", "text/html; charset=utf-8")
        elif path == "/api/attacks":
            self._j({k: {"ready": A.ready(k), "strength": A.STRENGTH.get(k, "")} for k in A.ATTACKS})
        elif path.startswith("/api/job/"):
            j = JOBS.get(path.rsplit("/", 1)[1]); self._j(j or {"error": "no job"}, 200 if j else 404)
        elif path.startswith("/data/"):
            fp = (DATA / path[len("/data/"):]).resolve()
            if DATA in fp.parents and fp.is_file(): self._s(200, fp.read_bytes(), "image/png")
            else: self._s(404, b"nf", "text/plain")
        else: self._s(404, b"nf", "text/plain")
    def do_POST(self):
        path = self.path.split("?", 1)[0]
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if path == "/api/attack":                      # synchronous: image in, image out
            q = self._q(); aid = q.get("attack", ""); strength = q.get("strength", "")
            ok, out, _ = run_attack(aid, raw, ".png", strength)
            return self._s(200, out, "image/png") if ok else self._j({"error": out}, 400)
        if path == "/api/eval":                        # batch: images + detector_url -> report
            try: body = json.loads(raw)
            except Exception: return self._j({"error": "bad json"}, 400)
            imgs = [base64.b64decode(x) for x in body.get("images", [])]
            attacks = [a for a in body.get("attacks", list(A.ATTACKS)) if a in A.ATTACKS]
            durl = body.get("detector_url"); thr = float(body.get("threshold", 0.5))
            if not imgs or not durl: return self._j({"error": "need images[] and detector_url"}, 400)
            rows = []; base_det = sum(detector_score(durl, im) >= thr for im in imgs)
            for i, im in enumerate(imgs):
                for aid in attacks:
                    ok, out, _ = run_attack(aid, im, ".png", "")
                    if not ok: rows.append({"image": i, "attack": aid, "error": out[-200:]}); continue
                    sc = detector_score(durl, out)
                    rows.append({"image": i, "attack": aid, "score_after": sc,
                                 "still_detected": sc >= thr, "psnr": psnr(im, out)})
            summ = {}
            for aid in attacks:
                rs = [r for r in rows if r["attack"] == aid and "score_after" in r]
                if rs: summ[aid] = {"n": len(rs), "detected_before": base_det, "of": len(imgs),
                                    "detected_after": sum(r["still_detected"] for r in rs),
                                    "evasion_rate": round(1 - sum(r["still_detected"] for r in rs) / len(rs), 3),
                                    "avg_psnr": round(sum(r["psnr"] for r in rs) / len(rs), 2)}
            return self._j({"threshold": thr, "images": len(imgs), "summary": summ, "rows": rows})
        if path == "/api/run":                         # async, for the browser UI
            fields, fname, fdata = multipart_file(raw, self.headers["Content-Type"])
            aid = fields.get("attack", b"").decode(); strength = fields.get("strength", b"").decode().strip()
            if aid not in A.ATTACKS or not fdata: return self._j({"error": "bad request"}, 400)
            jid = uuid.uuid4().hex[:10]; suf = (Path(fname or "x.png").suffix or ".png").lower()
            (DATA / "in").mkdir(parents=True, exist_ok=True); (DATA / "in" / f"{jid}{suf}").write_bytes(fdata)
            JOBS[jid] = {"id": jid, "status": "running", "input": f"data/in/{jid}{suf}", "output": None, "log": "", "attack": aid}
            threading.Thread(target=run_job, args=(JOBS[jid], aid, fdata, suf, strength), daemon=True).start()
            return self._j(JOBS[jid])
        self._s(404, b"nf", "text/plain")
    def log_message(self, *a): pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8077"))
    print(f"wm attack API on :{port}  (GET /api/attacks)")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
'''

# ---------------------------------------------------------------- eval harness (your detector)
FILES[f"{ROOT}/eval.py"] = r'''
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
'''

# ---------------------------------------------------------------- local twotensors-style passes
FILES[f"{ROOT}/local_tools.py"] = r'''
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
'''

# ---------------------------------------------------------------- UI page
FILES[f"{ROOT}/index.html"] = r'''
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Watermark Attacks</title>
<style>
:root{--bg:#f4f4f2;--card:#fff;--ink:#1b1b1b;--mut:#6c6c6c;--line:#e3e3df;--accent:#2f5bd3;--ok:#1f8a4c;--bad:#c23a2b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{padding:18px 22px;border-bottom:1px solid var(--line);background:var(--card)}h1{margin:0;font-size:18px}
header p{margin:4px 0 0;color:var(--mut);font-size:13px}main{max-width:1150px;margin:0 auto;padding:22px;display:grid;gap:18px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.row{display:grid;grid-template-columns:1fr 160px auto;gap:12px;align-items:end}
label{display:grid;gap:5px;font-size:13px;color:var(--mut)}
select,input[type=text],button{font:inherit;padding:9px 11px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--ink)}
button{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}
.drop{border:2px dashed var(--line);border-radius:12px;padding:24px;text-align:center;color:var(--mut);cursor:pointer}.drop.has{border-color:var(--accent);color:var(--ink)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair img{width:100%;border:1px solid var(--line);border-radius:10px;background:#fafafa;display:block}
figcaption{font-size:13px;color:var(--mut);margin-bottom:6px}figure{margin:0}
pre{background:#1b1b1b;color:#d6d6d6;padding:12px;border-radius:10px;overflow:auto;max-height:260px;font-size:12px;white-space:pre-wrap}
.hint{font-size:12px;color:var(--mut)}.status{font-weight:600}.status.done{color:var(--ok)}.status.failed{color:var(--bad)}.status.running{color:var(--accent)}.warn{color:var(--bad);font-size:13px}
</style></head><body>
<header><h1>Watermark robustness - attack runner</h1><p>Published research attacks (Zhao / Saberi / UnMarker / WAVES) on NVIDIA. Pick an image + attack; see before/after. For scoring your own detector use eval.py.</p></header>
<main>
<div class="panel">
  <div id="drop" class="drop">Click or drop an image</div><input id="file" type="file" accept="image/*" hidden>
  <div style="height:12px"></div>
  <div class="row">
    <label>Attack<select id="attack"></select></label>
    <label id="sw">Strength<input id="strength" type="text"></label>
    <button id="run" disabled>Run</button>
  </div>
  <p id="hint" class="hint"></p><p id="warn" class="warn"></p>
</div>
<div class="panel" id="rp" style="display:none">
  <p>Status: <span id="status" class="status"></span></p>
  <div class="pair"><figure><figcaption>Input</figcaption><img id="in"></figure><figure><figcaption>After attack</figcaption><img id="out"></figure></div>
  <div style="height:12px"></div><pre id="log"></pre>
</div></main>
<script>
let AT={},f=null;const $=i=>document.getElementById(i);
async function load(){AT=await(await fetch('/api/attacks')).json();const s=$('attack');s.innerHTML='';
 for(const[k,v]of Object.entries(AT)){const o=document.createElement('option');o.value=k;o.textContent=k+(v.ready?'':'  (installing...)');o.disabled=!v.ready;s.appendChild(o)}
 const fr=Object.keys(AT).find(k=>AT[k].ready);if(fr)s.value=fr;oa()}
function oa(){const v=AT[$('attack').value];if(!v)return;$('hint').textContent=v.strength||'no strength';$('sw').style.display=v.strength?'grid':'none';upd()}
function upd(){const v=AT[$('attack').value];$('run').disabled=!(f&&v&&v.ready)}
$('drop').onclick=()=>$('file').click();
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('has')};
$('drop').ondrop=e=>{e.preventDefault();set(e.dataTransfer.files[0])};
$('file').onchange=e=>set(e.target.files[0]);$('attack').onchange=oa;
function set(x){if(!x)return;f=x;$('drop').classList.add('has');$('drop').textContent=x.name+' ('+Math.round(x.size/1024)+' KB)';upd()}
$('run').onclick=async()=>{const fd=new FormData();fd.append('attack',$('attack').value);fd.append('strength',$('strength').value);fd.append('file',f);
 $('run').disabled=true;$('warn').textContent='';const r=await fetch('/api/run',{method:'POST',body:fd});
 if(!r.ok){$('warn').textContent=(await r.json()).error||'failed';$('run').disabled=false;return}
 const j=await r.json();$('rp').style.display='block';$('in').src='/'+j.input;$('out').removeAttribute('src');poll(j.id)};
async function poll(id){const j=await(await fetch('/api/job/'+id)).json();$('status').textContent=j.status;$('status').className='status '+j.status;$('log').textContent=j.log||'';
 if(j.status==='running'){setTimeout(()=>poll(id),1500);return}if(j.status==='done'&&j.output)$('out').src='/'+j.output+'?t='+Date.now();$('run').disabled=false}
load();setInterval(()=>{if(!f)load()},10000);
</script></body></html>
'''

# ---------------------------------------------------------------- example detector
FILES[f"{ROOT}/detector_example.py"] = r'''
"""Example detector stub for eval.py. Replace with YOUR detector.
Contract: print one number in [0,1] = probability the watermark / AI-origin is
still present in the image at argv[1]. eval.py reads the last whitespace token.

This stub decodes an invisible-watermark bit string and returns bit-accuracy,
so you can smoke-test the loop before plugging in your real detector:
    python eval.py --images my_wm_images --detector "python detector_example.py {img}"
"""
import sys
import numpy as np
from PIL import Image
try:
    import cv2
    from imwatermark import WatermarkDecoder
    img = cv2.imread(sys.argv[1])
    bits = WatermarkDecoder("bytes", 32).decode(img, "dwtDct")
    ref = np.unpackbits(np.frombuffer(b"test", dtype=np.uint8))
    got = np.unpackbits(np.frombuffer(bytes(bits), dtype=np.uint8))
    print(float((ref == got).mean()))
except Exception:
    print(0.0)
'''

for path, body in FILES.items():
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(body.lstrip("\n"))
    print("wrote", path)

# make shell entry points executable where relevant
for p in [f"{ROOT}/app.py", f"{ROOT}/eval.py"]:
    if os.path.exists(p):
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
