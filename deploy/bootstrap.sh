#!/usr/bin/env bash
# One-shot setup for the watermark-robustness repos on a RunPod CUDA pod.
#
#   bash bootstrap.sh            # clone + per-repo venv + UI (HF models pull on first use)
#   bash bootstrap.sh --weights  # also fetch the heavy pretrained weights (UnMarker ~30GB, Saberi ~2.5GB)
#
# Recommended pod: a RunPod "PyTorch 2.x / CUDA 12.1" template, >=1x 24GB GPU,
# >=60GB container disk (>=120GB if --weights). Expose HTTP port 8077.
# After it finishes, open the pod's 8077 proxy URL: upload an image, pick an
# attack, see before/after. For the eval loop (your own detector) see eval.py.
set -euo pipefail

ROOT="${WM_ROOT:-/workspace/wm}"
WEIGHTS=0
[ "${1:-}" = "--weights" ] && WEIGHTS=1
mkdir -p "$ROOT"/{repos,data/in,data/out}
cd "$ROOT"

# big CUDA wheels (torch ~2.5GB) time out on a slow pod at uv's 30s default; give
# generous timeouts + retries and fewer parallel downloads so none stalls out.
export UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=4 PIP_DEFAULT_TIMEOUT=900 UV_VENV_CLEAR=1

log(){ echo -e "\n=== $* ($(date +%H:%M:%S)) ==="; }

# exiftool for the metadata-strip attack (C2PA stripping also tries c2patool if present)
command -v exiftool >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq libimage-exiftool-perl; } || echo "exiftool install skipped (metadata-strip falls back to PIL re-save)"
# system libs cv2/compressai need on a bare CUDA image (full opencv links libGL/libglib;
# build-essential lets compressai build from sdist if no wheel resolves)
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 build-essential || echo "system libs install skipped (cv2 import may fail on a bare image; compressai may need a toolchain)"

# uv makes the per-repo installs fast; fall back to venv+pip if absent.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Each repo pins conflicting versions (numpy 1.24 vs 2.x, diffusers 0.19 vs 0.33),
# so each gets its own venv. These are the repos' ORIGINAL CUDA requirements -
# on an NVIDIA pod they install as the authors intended, no ROCm swaps.
clone(){ # clone <dir> <url> <branch-or-empty>
  [ -d "repos/$1" ] || git clone --depth 1 ${3:+-b $3} "$2" "repos/$1"
}
venv(){ # venv <dir> <python>
  uv venv --python "${2:-3.11}" "repos/$1/.venv"
}
pipin(){ # pipin <dir> <args...>
  ( cd "repos/$1" && VIRTUAL_ENV="$PWD/.venv" uv pip install "${@:2}" )
}

log "clone repos"
clone WatermarkAttacker   https://github.com/XuandongZhao/WatermarkAttacker
clone watermark_robustness https://github.com/mehrdadsaberi/watermark_robustness
clone ai-watermark        https://github.com/andrekassis/ai-watermark
clone WAVES               https://github.com/umd-huang-lab/WAVES
clone FPBA                https://github.com/nxZhai/FPBA
clone DuFIA               https://github.com/multimediaFor/DuFIA

# stabilityai/stable-diffusion-2-1(-base) were pulled from HF; use the community mirror.
{ grep -rl 'stabilityai/stable-diffusion-2-1' repos --include='*.py' --include='*.yaml' 2>/dev/null \
  | xargs -r sed -i 's#stabilityai/stable-diffusion-2-1#sd2-community/stable-diffusion-2-1#g'; } || true

CU=https://download.pytorch.org/whl/cu121

# --- WatermarkAttacker (Zhao): regeneration ---
log "WatermarkAttacker"
venv WatermarkAttacker 3.11
# bm3d (no cp311 wheel, needs a toolchain) is unused by run_one -> drop it from the
# install and make its import lazy so `import wmattacker` doesn't need the package.
sed -i 's/^from bm3d import bm3d_rgb$/pass  # bm3d imported lazily in BM3DAttacker (unused by run_one)/' repos/WatermarkAttacker/wmattacker.py
sed -i 's/^\( *\)y_est = bm3d_rgb(/\1from bm3d import bm3d_rgb\n\1y_est = bm3d_rgb(/' repos/WatermarkAttacker/wmattacker.py
pipin WatermarkAttacker torch torchvision --index-url "$CU"
pipin WatermarkAttacker diffusers==0.27.2 transformers==4.41.2 huggingface_hub==0.25.2 \
      accelerate==0.30.1 "numpy<2" opencv-python invisible-watermark==0.2.0 scikit-image \
      matplotlib compressai==1.2.6 torch_fidelity onnxruntime

# --- watermark_robustness (Saberi): diffusion purification ---
log "watermark_robustness"
venv watermark_robustness 3.11
# pin to the repo's own torch (2.5.x breaks the tensorflow==2.13 typing-extensions constraint)
pipin watermark_robustness torch==2.1.1 torchvision==0.16.1 --index-url "$CU"
pipin watermark_robustness opencv-python==4.7.0.72 diffusers==0.19.3 transformers==4.34.0 \
      huggingface_hub==0.17.3 timm==0.4.12 fairscale==0.4.4 invisible-watermark==0.1.5 \
      kornia==0.7.1 scikit-learn==1.4.0 ftfy==6.1.1 datasets==2.4.0 accelerate==0.23.0 \
      "tensorflow==2.13.0" matplotlib==3.7.1 seaborn==0.12.2 openai-clip==1.0.1 "setuptools<70"
# bchlib==0.7 dropped: won't compile on py3.11 (lvalue errors) and only StegaStamp
# decode uses it (lazy import), which our purify attacks never call.

# --- WAVES (An): adversarial + distortions ---
log "WAVES"
venv WAVES 3.11
pipin WAVES torch torchvision --index-url "$CU"
pipin WAVES diffusers==0.23.1 transformers==4.35.2 "huggingface_hub<0.26" accelerate==0.24.1 \
      "numpy<2" open-clip-torch==2.23.0 opencv-python scikit-image scikit-learn scipy \
      timm==0.9.12 torchattacks==3.5.1 torchmetrics==1.2.0 matplotlib omegaconf==2.1.1 einops==0.3.0 \
      "git+https://github.com/openai/CLIP.git" "setuptools<70" ipython==8.16.1 imageio

# --- ai-watermark (UnMarker): spectral adversarial ---
log "ai-watermark (UnMarker)"
venv ai-watermark 3.11
pipin ai-watermark torch==2.2.2 torchvision==0.17.2 --index-url "$CU"
pipin ai-watermark numpy==1.24.1 timm==0.9.16 transformers==4.40.1 diffusers==0.21.3 \
      accelerate==0.28.0 omegaconf einops==0.3.2 torch_dct==0.1.6 lpips==0.1.4 kornia==0.7.0 \
      open_clip_torch==2.7.0 pytorch_msssim==1.0.0 compressai==1.2.4 "tensorflow==2.13.0" \
      torcheval piqa==1.3.2 ninja huggingface_hub==0.25.0 ldpc galois pycryptodome==3.21.0 peft==0.11.1 \
      matplotlib pytorch_forecasting==1.0.0 "setuptools<70"
# the repo's four local packages (authors install these editable)
( cd repos/ai-watermark && VIRTUAL_ENV="$PWD/.venv" uv pip install -e systems \
    -e systems/watermarkers/networks/ptw -e systems/watermarkers/networks/stable_sig \
    -e modules/attack/unmark )
# UnMark builds via __new__ with evaluator=None and never calls BaseAttack.__init__,
# but base_attack.py eagerly imports the whole watermarker zoo (ldm/lightning/wandb/tf).
# Defer that import to its only use site so importing UnMark stays light.
sed -i -e 's/^from watermarkers import init_watermarker$//' \
       -e 's/^\( *\)self.evaluator = init_watermarker(/\1from watermarkers import init_watermarker\n\1self.evaluator = init_watermarker(/' \
    repos/ai-watermark/modules/attack/base_attack.py

# --- FPBA / DuFIA (white-box adversarial attacks ON a detector) ---
# These attack a specific detector architecture + its checkpoint over a 0_real/1_fake
# folder; they are NOT detector-agnostic per-image attacks. Installed + documented
# (see README "white-box"); not added to the per-image runner/UI.
log "FPBA (white-box)"
venv FPBA 3.11
pipin FPBA torch==2.5.1 torchvision==0.20.1 --index-url "$CU"
pipin FPBA numpy Pillow opencv-python scipy scikit-learn tqdm typed-argument-parser==1.10.1 "huggingface_hub>=0.27.0"

log "DuFIA (white-box)"
venv DuFIA 3.11
pipin DuFIA torch==2.2.2 torchvision==0.17.2 --index-url "$CU"
# DuFIA vendors its own clip (relative import), so skip the openai/CLIP git build; its
# vendored clip needs pkg_resources.packaging -> setuptools<70.
pipin DuFIA numpy Pillow opencv-python scipy scikit-learn tqdm ftfy regex "setuptools<70"

# per-image runners + FastAPI app + eval harness, written next to this script's helpers
log "write helpers"
python3 "$(dirname "$0")/write_helpers.py" "$ROOT"

# the FastAPI orchestrator runs in its own light venv (it only shells into the
# attack venvs, so it needs no torch).
log "API venv (FastAPI)"
uv venv --python 3.11 "$ROOT/.venv-api"
VIRTUAL_ENV="$ROOT/.venv-api" uv pip install "fastapi>=0.111" "uvicorn[standard]>=0.30" "pydantic>=2.7" httpx pillow numpy python-multipart

if [ "$WEIGHTS" = 1 ]; then
  log "UnMarker weights (~30GB)"; ( cd repos/ai-watermark && bash download_data_and_models.sh ) || echo "UnMarker weights failed - rerun later"
  log "Saberi weights (~2.5GB)";  ( cd repos/watermark_robustness && bash _bash_download_models.sh ) || echo "Saberi weights failed - rerun later"
fi

log "launch API on :8077"
cd "$ROOT"
nohup "$ROOT/.venv-api/bin/uvicorn" app:app --host 0.0.0.0 --port 8077 > api.log 2>&1 &
sleep 3
echo "API up on :8077. Schema: http://POD:8077/openapi.json   Docs: http://POD:8077/docs"
echo "Endpoints: GET /attacks  POST /attack  POST /detect  POST /eval  POST /twotensors"
echo "Logs: $ROOT/api.log"
