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
