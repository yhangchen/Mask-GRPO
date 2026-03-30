"""Download all HuggingFace models needed by Mask-GRPO to local disk."""

import os
from huggingface_hub import snapshot_download

LOCAL_DIR = "/data/banyuanhao/Mask-GRPO/models"
os.makedirs(LOCAL_DIR, exist_ok=True)

# HF token (optional, set if needed for gated models)
token = os.environ.get("HF_TOKEN", None)

MODELS = [
    # --- required for from_pretrained / save showo.pth ---
    # "showlab/show-o-512x512",
    # "showlab/magvitv2",
    # "microsoft/phi-1_5",

    # --- reward models (download whichever you plan to use) ---
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",       # reward_type=clip
    "THUDM/ImageReward",                             # reward_type=ima (ImageReward-v1.0)
    "CodeGoat24/UnifiedReward-qwen-7b",            # reward_type=uni
]

for repo_id in MODELS:
    local_name = repo_id.replace("/", "--")
    local_path = os.path.join(LOCAL_DIR, local_name)
    print(f"\n[INFO] Downloading {repo_id} -> {local_path}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=token,
    )
    print(f"[INFO] Done: {repo_id}")

print("\n========== All downloads complete ==========")
print("Now update configs/Mask_GRPO_train_512x512.yaml:")
print(f'  vq_model_name: "{LOCAL_DIR}/showlab--magvitv2"')
print(f'  pretrained_model_path: "{LOCAL_DIR}/showlab--show-o-512x512"')
print(f'  llm_model_path: "{LOCAL_DIR}/microsoft--phi-1_5"')
