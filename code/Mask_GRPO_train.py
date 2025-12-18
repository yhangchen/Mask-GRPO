import os
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import time
import torch
import wandb
from typing import Dict, Tuple
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel, CLIPConfig
from diffusers.optimization import (
    Union,
    SchedulerType,
    Optional,
    Optimizer,
    TYPE_TO_SCHEDULER_FUNCTION,
)
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from pathlib import Path
from accelerate import Accelerator
import open_clip
import ImageReward as RM
import random
from pathlib import Path
from dataset.grpo_dataset import GRPODataset

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
        ]

def get_optimizer(model, weight_decay, learning_rate, betas): # beta: Tuple[float, float]) 
    optim_groups = add_weight_decay(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    # Manually set 'initial_lr' for each parameter group (assuming a base learning rate)
    for param_group in optimizer.param_groups:
        if "initial_lr" not in param_group:
            param_group["initial_lr"] = param_group["lr"]  # or set a specific initial learning rate

    return optimizer

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs,
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )

@torch.no_grad()
def get_group_reward(prompts, images, device, clip_processor, clip_model, eval = False):
    if eval:
        assert len(prompts) == len(images)
        length = len(prompts)
        
        for i in range(length):
            clip_inputs = clip_processor(text = [prompts[i]], images=[images[i]], return_tensors="pt", padding=True)
            clip_inputs = clip_inputs.to(device)
            clip_outputs = clip_model(**clip_inputs)
            logits_per_image = clip_outputs.logits_per_image # this is the image-text similarity score
            logits = logits_per_image.squeeze(-1).float()
            if i == 0:
                logits_sum = logits
            else:
                logits_sum = logits_sum + logits
        del clip_inputs, clip_outputs,logits_per_image
        return logits_sum / length
    clip_inputs = clip_processor(text = prompts, images=images, return_tensors="pt", padding=True)
    clip_inputs = clip_inputs.to(device)
    clip_outputs = clip_model(**clip_inputs)
    logits_per_image = clip_outputs.logits_per_image # this is the image-text similarity score
    logits = logits_per_image.squeeze(-1).float()
    mean = torch.mean(logits)
    std = torch.std(logits) + 1e-4
    # print('shape of logits is:', logits.shape)
    normalized_logits = (logits - mean) / std
    group_reward = normalized_logits
    del clip_inputs, clip_outputs,logits_per_image
    # torch.cuda.empty_cache()
    return group_reward, mean, std

@torch.no_grad()
def get_group_reward_uni(prompts, pil_images, device, reward_model, processor, eval = False):
    rewards = []
    if eval:
        bsz = len(pil_images)
        for i in range(bsz):
            messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_images[i]},
                    {
                        "type": "text",
                        "text": f'You are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompts[i]}]'
                    },
                ],
            }
        ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                generated_ids = reward_model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            match = re.search(r"Final Score:\s*(-?\d+\.\d+|\d+)", output_text)
            # print(output_text)
            # print(float(match.group(1)))
            if match:
                rewards.append(float(match.group(1)))
            else:
                return 0, False
        try:
            rewards = torch.tensor(rewards, dtype=torch.float32)
            rewards = rewards.to(device)
            mean = torch.mean(rewards)
            del text, image_inputs, video_inputs, inputs, generated_ids, generated_ids_trimmed
            return mean, True
        except Exception as e:
            return 0, False
            
    
    bsz = len(pil_images)
    for image in pil_images:
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f'You are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompts[0]}]'
                },
            ],
        }
    ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            generated_ids = reward_model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        match = re.search(r"Final Score:\s*(-?\d+\.\d+|\d+)", output_text)
        # print(output_text)
        # print(float(match.group(1)))
        if match:
            rewards.append(float(match.group(1)))
        else:
            return 0 ,0, 0, False
    try:
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = rewards.to(device)
        mean = torch.mean(rewards)
        std = torch.std(rewards) + 1e-5
        normalized_logits = (rewards - mean) / std
        group_reward = normalized_logits
        del text, image_inputs, video_inputs, inputs, generated_ids, generated_ids_trimmed
        return group_reward, mean, std, True
    except Exception as e:
        return 0, 0, 0, False

@torch.no_grad()
def get_group_reward_ima(prompts, pil_images, device, reward_model, eval = False):
    rewards = []
    if eval:
        bsz = len(pil_images)
        for i in range(bsz):
            reward = reward_model.score(prompts[i], [pil_images[i]])
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = rewards.to(device)
        mean = torch.mean(rewards)
        return mean
        
    rewards = reward_model.score(prompts[0], pil_images)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = rewards.to(device)
    mean = torch.mean(rewards)
    std = torch.std(rewards) + 1e-5
    normalized_logits = (rewards - mean) / std
    group_reward = normalized_logits
    return group_reward, mean, std


def token_preprocess(prompts, image_tokens, uni_prompting, device, config, cfg = False):  
    input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')
    input_ids = input_ids.to(device)
    # input_ids is the full token input of text_token+image_mask_token
    if config.training.guidance_scale > 0 or cfg: 
        uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen') 
        attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True)
        # print('attention_mask shape is:', attention_mask.shape) 
    else: 
        attention_mask = create_attention_mask_predict_next(input_ids,
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True)
        uncond_input_ids = None
    attention_mask = attention_mask.to(device)
    return input_ids, uncond_input_ids, attention_mask

use_uni = False # for unified reward
use_ima = False # for image reward
if __name__ == '__main__':
    config = get_config()
    set_seed(config.seed)
    if config.training.debug:
        config.training.batch_size = 1
        shuffle = False
    else:
        shuffle = True

    gradient_accumulation_steps = config.training.generation_timesteps
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision="fp16")
    if accelerator.is_main_process:
        wandb.init(
            project="GRPO_try",
            name=config.name,
        )
    
    train_dataset = GRPODataset(data_path=config.dataset.path)
    val_dataset = GRPODataset(data_path=config.dataset.val_path)
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=shuffle, num_workers=config.training.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    
    if accelerator.is_main_process:
        print("train_dataset:", len(train_dataset), "train_dataloader:", len(train_dataloader))
        print("val_dataset:", len(val_dataset), "val_dataloader:", len(val_dataloader))

    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                    special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                    ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.requires_grad_(False)
    vq_model.eval()

    ## how to save and load
    
    # save start
    # model = Showo.from_pretrained(config.model.showo.pretrained_model_path) # use the old version of transformers
    # # torch.save(model, 'path/to/showo.pth')
    # exit()
    # save end

    # load start
    model = torch.load('path/to/showo.pth', map_location='cpu')
    # load end
    
    mask_token_id = model.config.mask_token_id # 58497
    # print(f"mask_token_id: {mask_token_id}")

    # Load reward model
    if use_ima:
        reward_model = RM.load("ImageReward-v1.0")
    elif use_uni:
        model_path = 'CodeGoat24/UnifiedReward-qwen-7b'
        reward_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", attn_implementation="flash_attention_2", device_map={"": accelerator.device}
        )
        processor = AutoProcessor.from_pretrained(model_path)
    else:
        clip_model = CLIPModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        clip_processor = CLIPProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        clip_model.requires_grad_(False)
        clip_model.eval()  

    optimizer = get_optimizer(model, weight_decay=config.training.weight_decay, learning_rate=config.training.learning_rate, betas=config.training.betas)
    # num_training_steps = (len(train_dataloader) * config.training.num_epochs) + 1
    num_training_steps = config.training.number_training_steps
    lr_scheduler = get_scheduler(
        config.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.num_warmup_steps,
        num_training_steps=num_training_steps,
    )   
    
    if use_uni or use_ima:
        # prepare for distributed training 
        (
            train_dataloader,
            # val_dataloader,
            vq_model,
            model,
            # clip_model,
            reward_model,
            optimizer,
            lr_scheduler,
        ) = accelerator.prepare(
            train_dataloader,
            # val_dataloader,
            vq_model,
            model,
            # clip_model,
            reward_model,
            optimizer,
            lr_scheduler,
        )
    else:
        # prepare for distributed training 
        (
            train_dataloader,
            # val_dataloader,
            vq_model,
            model,
            clip_model,
            # reward_model,
            optimizer,
            lr_scheduler,
        ) = accelerator.prepare(
            train_dataloader,
            # val_dataloader,
            vq_model,
            model,
            clip_model,
            # reward_model,
            optimizer,
            lr_scheduler,
        )
    device = model.device
    if accelerator.is_main_process:
        print("train dataset:", len(train_dataset), "train dataloader after split:", len(train_dataloader))
    
    if config.training.debug:
        num_training_steps = 2  
        config.saving.saving_step = 1
        config.logging.logging_step = 1
        config.training.group_size = 2

    if config.mode == 't2i': # Only text-to-image mission in Mask-GRPO
        '''
        Make dir
        We don't use all of them because we log the pictures through wandb.
        However, you can use these dirs to save the pictures locally.
        '''
        exp_dir = config.generation.out_dir + '/' + config.name
        train_dir = exp_dir + '/train'
        train_pic_dir = train_dir + '/pic'
        eval_on_train_dir = train_dir + '/eval_on_train'
        save_dir = train_dir + '/save'
        if accelerator.is_main_process:
            for dir in [exp_dir, train_dir, train_pic_dir, eval_on_train_dir, save_dir]:
                os.makedirs(dir, exist_ok=True)
        exp_dir = Path(exp_dir)
        train_dir = Path(train_dir)
        train_pic_dir = Path(train_pic_dir)
        eval_on_train_dir = Path(eval_on_train_dir)
        save_dir = Path(save_dir)
        accelerator.wait_for_everyone()
        
        # Training Start!!!
        if accelerator.is_main_process:  
            print("Start Training!")
        step = 0
        for epoch in range(config.training.num_epochs): 
            t0 = time.time()
            with tqdm(train_dataloader, desc = f'Training epoch {epoch}', leave=False, disable=not accelerator.is_main_process) as tepoch:
                last_step = False
                reward_list = []
                for batch_idx, batch in enumerate(tepoch): # batch_size = 1 because of cuda memory limit 
                    t0 = time.time() - t0
                    if accelerator.is_main_process:
                        print(f"time_cost per batch is: {t0:.4f}")
                    t0 = time.time()
                    prompts = batch
                    # print('prompts is:', prompts[0])
                    image_tokens = torch.ones((len(prompts), config.model.showo.num_vq_tokens), 
                                    dtype=torch.long, device=device) * mask_token_id
                    image_tokens = image_tokens.to(device)
                    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
                    '''
                    def cosine_schedule(t):
                        return torch.cos(t * math.pi * 0.5)
                    '''
                    gen_token_ids_list = [] # save the generated token ids for each group
                    selected_indices_list_list = [] # save the indices of newly unmasked tokens (each unmasked step) for each group
                    selected_ids_list_list = [] # save the token_ids of newly unmasked tokens(each unmasked step) for each group
                    min_probs_list_list = [] 
                    # save the min_probs (each unmasked step) for each group, this min_probs equals to min(cs_t) in the paper
                    masking_list_list = [] # save the indices of newly re-masked tokens (each unmasked step) for each group
                    if accelerator.is_main_process:  
                        print("----------------------------------------------------------------------")
                        print('-----------------------First Generation a group!----------------------')
                        print("----------------------------------------------------------------------")
                    with torch.inference_mode():
                        mod = accelerator.unwrap_model(model)
                        mod.eval()
                        for generate_idx in range(config.training.group_size):
                            # prepare input
                            input_ids, uncond_input_ids, attention_mask = token_preprocess(prompts, image_tokens, uni_prompting, device, config)
                            gen_token_ids, selected_indices_list, selected_ids_list, min_probs_list, masking_list = mod.t2i_generate_grpo(
                                input_ids=input_ids,
                                uncond_input_ids=uncond_input_ids,
                                attention_mask=attention_mask,
                                guidance_scale=config.training.guidance_scale, # 0 here. During training, we always set cfg=0 because of the cuda memory limit
                                temperature=config.training.get("generation_temperature", 1.0),
                                timesteps=config.training.generation_timesteps,
                                noise_schedule=mask_schedule,
                                noise_type=config.training.get("noise_type", "mask"),
                                seq_len=config.model.showo.num_vq_tokens,
                                uni_prompting=uni_prompting,
                                config=config,
                                )
                            gen_token_ids_list.append(gen_token_ids)
                            selected_indices_list_list.append(selected_indices_list)
                            selected_ids_list_list.append(selected_ids_list)
                            min_probs_list_list.append(min_probs_list)
                            masking_list_list.append(masking_list)
                            # import pdb; pdb.set_trace()
                            
                        # process a group to compute Advantages
                        # print('gen_token_ids_list len is:', len(gen_token_ids_list)) 
                        gen_token_ids = torch.cat(gen_token_ids_list, dim=0).detach()
                        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
                        # codebook_size = 8192, clamp just to make sure the value is in the range of [0, 8191]
                        images = vq_model.decode_code(gen_token_ids)
                        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                        images *= 255.0
                        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                        pil_images = [Image.fromarray(image) for image in images] 
                        assert len(pil_images) == config.training.group_size * len(prompts) and len(prompts) == config.training.batch_size 
                        if use_ima:
                            mod = accelerator.unwrap_model(reward_model)
                            group_reward, mean_reward, std = get_group_reward_ima(prompts, pil_images, device, mod) 
                        elif use_uni:
                            mod = accelerator.unwrap_model(reward_model)
                            group_reward, mean_reward, std, noerror = get_group_reward_uni(prompts, pil_images, device, mod, processor) 
                            if not noerror:
                                print('reward error')
                                continue
                            # print('group_reward is:', group_reward)
                            # print('mean_reward is:', mean_reward)
                            # print('std is:', std)
                        else:
                            group_reward, mean_reward, std = get_group_reward(prompts, pil_images, device, clip_processor, clip_model) 
                        # print('group_reward is:', group_reward)
                        # print('mean_reward is:', mean_reward)
                        # print('std is:', std)
                        
                        # if accelerator.is_main_process:
                        #     wandb_images = [wandb.Image(image, caption=f"Prompt: {prompts[0]}\nReward: {group_reward[i]:.2f}") for i, image in enumerate(pil_images)] 
                        #     wandb.log({f"epoch_{epoch}_group_train_images": wandb_images}, step=step) 
                        gather_reward = accelerator.gather(mean_reward)
                        if accelerator.is_main_process:
                            gather_reward = gather_reward.mean().item()
                            with open(os.path.join(train_dir, 'reward.txt'), 'a') as f: 
                                f.write(f"{gather_reward}\n")
                            wandb.log({"mean_train_reward_step": gather_reward}, step=step)

                        reward_list.append(mean_reward.cpu().item())
                        del gen_token_ids, gen_token_ids_list, pil_images, images
                        torch.cuda.empty_cache()
                    
                    
                    if accelerator.is_main_process:  
                        print("--------------------------------------------------------------------------")
                        print('--------------Finish Group Generation and Start GRPO Loss-----------------')
                        print("--------------------------------------------------------------------------")
                    accelerator.wait_for_everyone()
                    mod = accelerator.unwrap_model(model)
                    mod.train()
                    
                    
                    input_ids, uncond_input_ids, attention_mask = token_preprocess(prompts, image_tokens, uni_prompting, device, config)
                    input_ids = input_ids.repeat(config.training.group_size, 1) 
                    for timestep in range(config.training.generation_timesteps): 
                        # you can change to range(25) and also change the timesteps below to see the Computational reduction strategy mentioned in the paper
                        with accelerator.accumulate(model):
                            loss, new_input_ids = mod.t2i_generate_grpo_loss(
                                    input_ids=input_ids,
                                    uncond_input_ids=uncond_input_ids,
                                    attention_mask=attention_mask,
                                    guidance_scale=config.training.guidance_scale, # 0 here. During training, we always set cfg=0 because of the cuda memory limit
                                    temperature=config.training.get("generation_temperature", 1.0),
                                    timesteps=config.training.generation_timesteps, #also change to 25 here to see the Computational reduction strategy mentioned in the paper
                                    noise_schedule=mask_schedule,
                                    noise_type=config.training.get("noise_type", "mask"),
                                    seq_len=config.model.showo.num_vq_tokens,
                                    uni_prompting=uni_prompting,
                                    config=config,
                                    selected_indices_list_list=selected_indices_list_list,
                                    selected_ids_list_list=selected_ids_list_list,
                                    timestep=timestep,
                                    group_reward = group_reward.clone().detach(),
                                    min_probs_list_list = min_probs_list_list,
                                    masking_list_list = masking_list_list,
                                    )
                            tepoch.set_postfix(loss=loss.cpu().item, refresh=False)
                            accelerator.backward(loss)
                            # del input_ids
                            input_ids = new_input_ids
                            if accelerator.sync_gradients:
                                optimizer.step()
                                optimizer.zero_grad()
                                lr_scheduler.step()
                                
                    del input_ids, new_input_ids
                    torch.cuda.empty_cache()
                    accelerator.wait_for_everyone()


                    if accelerator.is_main_process:
                        print("----------------------------------------------------------------------")
                        print('-------------------------------End GRPO-------------------------------')
                        print("----------------------------------------------------------------------")

                        if step % len(train_dataloader) == len(train_dataloader) - 1:
                            last_step = True
                        step_log = {
                                    "loss": loss.cpu().item(),
                                    "step": step,
                                    "lr": lr_scheduler.get_last_lr()[0],
                                    }
                        wandb.log(step_log, step=step)
                        
                        if step % config.logging.logging_step == 0 and step != 0: # log group reward during training
                            if len(reward_list) > 0:
                                step_log = {
                                            "reward_train": sum(reward_list) / len(reward_list),
                                            }
                                reward_list = []
                                wandb.log(step_log, step=step)
                        
                        if step % (config.logging.logging_step * 6) == 0:
                            print("----------------------------------------------------------------------")
                            print('---------------------------EVALUATING MODEL---------------------------')
                            print("----------------------------------------------------------------------")
                            
                            mod = accelerator.unwrap_model(model)
                            mod.eval()
                            with torch.inference_mode():
                                for batch_idx_val, batch_val in enumerate(val_dataloader):
                                    prompt_val = batch_val
                                    image_tokens_val = torch.ones((len(prompt_val), config.model.showo.num_vq_tokens), 
                                    dtype=torch.long, device=device) * mask_token_id
                                    input_ids_val, uncond_input_ids_val, attention_mask_val = token_preprocess(prompt_val, image_tokens_val, uni_prompting, device, config, cfg = True)
                                    gen_token_ids_val, _ = mod.t2i_generate(
                                        input_ids=input_ids_val,
                                        uncond_input_ids=uncond_input_ids_val,
                                        attention_mask=attention_mask_val,
                                        guidance_scale=5, # During evaluation, we fix cfg=5
                                        temperature=config.training.get("generation_temperature", 1.0),
                                        timesteps=config.training.eval_generation_timesteps,
                                        noise_schedule=mask_schedule,
                                        noise_type=config.training.get("noise_type", "mask"),
                                        seq_len=config.model.showo.num_vq_tokens,
                                        uni_prompting=uni_prompting,
                                        config=config,
                                        )
                                    gen_token_ids_val = torch.clamp(gen_token_ids_val, max=config.model.showo.codebook_size - 1, min=0)
                                    images_val = vq_model.decode_code(gen_token_ids_val)
                                    images_val = torch.clamp((images_val + 1.0) / 2.0, min=0.0, max=1.0)
                                    images_val *= 255.0
                                    images_val = images_val.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                                    pil_images_val = [Image.fromarray(image) for image in images_val]
                                    # wandb_images_val = [wandb.Image(image, caption=prompt_val[j]) for j, image in enumerate(pil_images_val)] # batch_size = 1 here
                                    # wandb.log({"eval_images": wandb_images_val}, step=step)
                                    if use_ima:
                                        mod = accelerator.unwrap_model(reward_model)
                                        reward = get_group_reward_ima(prompt_val, pil_images_val, device, mod, eval = True)
                                    elif use_uni:
                                        mod = accelerator.unwrap_model(reward_model)
                                        reward, noerror = get_group_reward_uni(prompt_val, pil_images_val, device, mod, processor, eval = True)
                                        if not noerror:
                                            print('eval reward error')
                                    else:
                                        reward = get_group_reward(prompt_val, pil_images_val, device, clip_processor, clip_model, eval = True)
                                    print('eval reward is:', reward)
                                    # log reward during evaluation
                                    step_log = {
                                            "reward_eval": torch.mean(reward).cpu().item(),
                                            }
                                    wandb.log(step_log, step=step)
                                    del pil_images_val, images_val, gen_token_ids_val
                                    torch.cuda.empty_cache()
                                
                        if step % config.saving.saving_step == 0 or last_step:
                            print("----------------------------------------------------------------------")
                            print('-----------------------------SAVING MODEL-----------------------------')
                            print("----------------------------------------------------------------------")
                            mod = accelerator.unwrap_model(model)
                            mod.eval()
                            torch.save(mod.state_dict(), save_dir / f"model_{step}.pth")
                    
                    if step % config.saving.saving_step == 0 or last_step or step % config.logging.logging_step == 0:
                        accelerator.wait_for_everyone()
                    step = step + 1

                    if step >= num_training_steps // accelerator.num_processes + 1:
                        break
                accelerator.wait_for_everyone()
        accelerator.wait_for_everyone()
        accelerator.end_training()
                            

''' Usage!
accelerate launch --num_processes=16 --num_machines=${NNODES} \
--machine_rank=${NODE_RANK} \
--main_process_ip=${MASTER_ADDR} \
--main_process_port=${MASTER_PORT} Mask_GRPO_train.py config=configs/Mask_GRPO_train_512x512.yaml \
mode='t2i'

or

accelerate launch --num_processes=1 Mask_GRPO_train.py config=configs/Mask_GRPO_train_512x512.yaml \
mode='t2i'
'''