# wm-service

FastAPI service exposing watermark / AI-detector robustness attacks on **your own** images,
plus a scoring harness. Same technique family as twotensors.ai's "Forensic Cleaner" — for
testing how robust a detector/watermark is. Run against images and detectors you own.

## Endpoints (OpenAPI at `/openapi.json`, docs at `/docs`)

- `GET  /attacks`  — list wired attacks + whether each is `ready` (some gated on weights)
- `POST /attack`   — run one attack on an image (base64 or multipart); returns attacked image
- `POST /attack/raw` — same, returns raw bytes
- `POST /detect`   — score an image against a detector (url | cmd | sightengine | hive)
- `POST /eval`     — attack then score, before/after in one call
- `POST /twotensors` — a **configurable chain** (currently: advemb → downscale → strip-metadata).
  NOTE: this approximates twotensors with the bundled tools; it is **not** a reverse-engineered
  copy of their pipeline. See "Known limitation" below.

## Architecture

Each attack lives in its own research repo with conflicting deps, so `app.py` is a thin
orchestrator that shells each attack into its own venv (`attacks.py` holds the registry +
the `_WEIGHTS` gate). `local_tools.py` has the dependency-free ops (downscale, JPEG, blur,
metadata strip). `waves_run_one.py` is the WAVES driver the advemb/regen/surrogate attacks call.

```
service/
  app.py              FastAPI app, all endpoints, Pydantic schemas
  attacks.py          attack registry; gates weight-dependent attacks via ready()
  local_tools.py      downscale / jpeg / blur / metadata-strip (no heavy deps)
  waves_run_one.py    WAVES advemb / regen-vae / diffusion / clip16-surrogate driver
  eval.py             attack+score harness
  detector_example.py minimal local detector stub for /detect kind=cmd
  index.html          minimal upload UI
deploy/
  bootstrap.sh        on-pod installer: clone repos, per-repo venvs, CUDA torch, launch uvicorn :8077
  write_helpers.py    generator that emits the service/ files onto the pod
```

## Deploy (RunPod or any CUDA box)

```bash
# on the GPU host:
bash deploy/bootstrap.sh          # clones repos, builds venvs, writes service files, starts uvicorn :8077
```
`bootstrap.sh` takes no secrets. Weight-gated attacks (UnMarker, DiffPure) download large
weights on first use; `/attacks` reports `ready:false` for them until present.

## Known limitation (measured, 2026-10-24)

Against Sightengine's `genai` classifier, **every bundled attack scored 0.97–0.99** on a
real AI image (original 0.99). The real twotensors output scored **0.01**. The bundled repos
attack *watermarks* (SynthID / Tree-Ring / stega decoders) and do image regeneration; they do
**not** perturb against a genai *classifier*, which is a different target. Matching the 0.01
result requires swapping the attack's target model to a genai-classifier surrogate (e.g. a
local detector) and running PGD against that — then it transfers to the commercial API.
The attack loop here is correct; the target model is the missing piece.
