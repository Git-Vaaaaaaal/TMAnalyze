import os

# Hugging Face token
hf_token = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# ---------------------------------------------------------------------------
# Cache Hugging Face
# ---------------------------------------------------------------------------

JEAN_ZAY_HF_HOME = "/lustre/fsn1/projects/rech/ehe/udq27fb/cache/huggingface"

HF_HOME      = os.environ.get("HF_HOME", JEAN_ZAY_HF_HOME)
HF_HUB_CACHE = os.path.join(HF_HOME, "hub") if HF_HOME else None
