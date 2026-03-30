"""Generate images for TIIF-Bench evaluation using Mask-GRPO / Show-O.

Supports multi-GPU parallel inference via torchrun.
Output directory structure matches TIIF-Bench expected format:
  save_dir/{attr_type}/{model_name}/{short_description|long_description}/{idx}.png
"""

import argparse
import json
import os
import glob
import random

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", type=str, required=True,
                        help="Directory containing TIIF-Bench prompt jsonl files")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Root directory to save generated images")
    parser.add_argument("--model_name", type=str, default="mask_grpo",
                        help="Model name for directory structure")
    parser.add_argument("--showo_model_path", type=str, default="showo.pth")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to trained checkpoint (state_dict). Empty = base model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--specific_file", type=str, default=None,
                        help="Only process this specific jsonl file")
    args, _ = parser.parse_known_args()
    return args


torch.set_grad_enabled(False)


def generate_image(prompt, model, vq_model, uni_prompting, mask_token_id, mask_schedule, config, device, guidance_scale, timesteps):
    """Generate a single image from a prompt."""
    prompts = [prompt]
    image_tokens = torch.ones((1, config.model.showo.num_vq_tokens),
                              dtype=torch.long, device=device) * mask_token_id
    input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')

    if guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''], image_tokens), 't2i_gen')
        attention_mask = create_attention_mask_predict_next(
            torch.cat([input_ids, uncond_input_ids], dim=0),
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True)
    else:
        uncond_input_ids = None
        attention_mask = create_attention_mask_predict_next(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True)

    gen_token_ids, _ = model.t2i_generate(
        input_ids=input_ids,
        uncond_input_ids=uncond_input_ids,
        attention_mask=attention_mask,
        guidance_scale=guidance_scale,
        temperature=config.training.get("generation_temperature", 1.0),
        timesteps=timesteps,
        noise_schedule=mask_schedule,
        noise_type=config.training.get("noise_type", "mask"),
        seq_len=config.model.showo.num_vq_tokens,
        uni_prompting=uni_prompting,
        config=config,
    )

    gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return Image.fromarray(images[0])


def main():
    args = parse_args()
    config = get_config()

    # --- Distributed setup ---
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    seed_everything(args.seed + rank)

    # --- Tokenizer & prompting ---
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100, cond_dropout_prob=0.0,
    )

    # --- Load VQ model ---
    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.requires_grad_(False)
    vq_model.eval()
    vq_model = vq_model.to(device)

    # --- Load Show-O model ---
    model = torch.load(args.showo_model_path, map_location='cpu', weights_only=False)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        if rank == 0:
            print(f"[INFO] Loaded checkpoint: {args.checkpoint}")
    model.eval()
    model = model.to(device)
    mask_token_id = model.config.mask_token_id

    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    # --- Collect all tasks from all jsonl files ---
    if args.specific_file:
        jsonl_files = [os.path.join(args.jsonl_dir, args.specific_file)]
    else:
        jsonl_files = sorted(glob.glob(os.path.join(args.jsonl_dir, "*.jsonl")))

    all_tasks = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r') as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
        for idx, item in enumerate(lines):
            for desc_key in ['short_description', 'long_description']:
                out_dir = os.path.join(args.save_dir, item['type'], args.model_name, desc_key)
                out_path = os.path.join(out_dir, f"{idx}.png")
                all_tasks.append((item, idx, desc_key, out_dir, out_path))

    # --- Shard tasks across GPUs ---
    my_tasks = all_tasks[rank::world_size]

    skipped = sum(1 for _, _, _, _, p in my_tasks if os.path.exists(p))
    to_generate = len(my_tasks) - skipped
    if rank == 0:
        print(f"[INFO] Total: {len(all_tasks)} images across {world_size} GPUs")
    print(f"[INFO] Rank {rank}: {len(my_tasks)} tasks, skipping {skipped}, generating {to_generate}")

    generated = 0
    pbar = tqdm(my_tasks, desc=f"GPU {rank}", disable=(rank != 0))
    for item, idx, desc_key, out_dir, out_path in pbar:
        if os.path.exists(out_path):
            continue

        os.makedirs(out_dir, exist_ok=True)
        prompt = item[desc_key]
        pil_image = generate_image(
            prompt, model, vq_model, uni_prompting,
            mask_token_id, mask_schedule, config, device,
            args.guidance_scale, args.timesteps)
        pil_image.save(out_path)
        generated += 1

    print(f"[INFO] Rank {rank} done. {generated} images generated.")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"[INFO] All GPUs done. Images saved to {args.save_dir}")


if __name__ == "__main__":
    main()
