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
