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
