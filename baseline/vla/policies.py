"""One interface for the VLA models we can plug in: SmolVLA (default), π0 and π0.5 (lerobot 0.4.4).

    policy = load_policy(chunk, device, pretrained, img_hw)   # lerobot/smolvla_base | lerobot/pi0_base | lerobot/pi05_base | a fine-tuned dir
    tokens = tokenize(policy, task)                           # fixed instruction → token ids / mask
    batch  = make_batch(policy, img_u8, tokens, device)       # one camera, zero state, actions already in [-1, 1]
    chunk  = policy.predict_action_chunk(batch, noise=...)    # (B, chunk, 2)

Differences handled here
  SmolVLA : SmolVLM2 tokenizer, keys observation.image / observation.state (2) / observation.language.*
  π0      : PaliGemma tokenizer (48 tokens, right padded), key observation.images.cam, state padded to 32
  π0.5    : same, but the (zero) state is discretised into the prompt  "Task: ..., State: 127 ...;\\nAction: "
            and the model takes no separate state input.
The PaliGemma tokenizer is gated on the Hub; we use an ungated copy (identical files) unless
PALIGEMMA_TOKENIZER points elsewhere. Only the action expert / projections are trained for every model.
"""
import os

import numpy as np
import torch

PALIGEMMA_TOK = os.environ.get('PALIGEMMA_TOKENIZER', 'leo009/paligemma-3b-pt-224')
IMG_KEY = {'smolvla': 'observation.image', 'pi0': 'observation.images.cam', 'pi05': 'observation.images.cam'}


def policy_type(policy_or_cfg):
    cfg = getattr(policy_or_cfg, 'config', policy_or_cfg)
    return cfg.type                                     # 'smolvla' | 'pi0' | 'pi05'


def load_policy(chunk, device, pretrained='lerobot/smolvla_base', img_hw=(640, 480), bf16=None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import get_policy_class
    cfg = PreTrainedConfig.from_pretrained(pretrained)
    t = cfg.type
    if not os.path.isdir(pretrained):   # adapting a base model: replace the I/O spec
        cfg.input_features = {
            IMG_KEY[t]: PolicyFeature(FeatureType.VISUAL, (3, *img_hw)),
            'observation.state': PolicyFeature(FeatureType.STATE, (2,)),
        }
        cfg.output_features = {'action': PolicyFeature(FeatureType.ACTION, (2,))}
        cfg.chunk_size = chunk
        cfg.n_action_steps = chunk
        if t in ('pi0', 'pi05'):
            cfg.train_expert_only = True          # freeze PaliGemma (VLM + SigLIP), train the 300M action expert
            cfg.freeze_vision_encoder = True
            cfg.dtype = 'bfloat16' if bf16 is None or bf16 else 'float32'
            cfg.empty_cameras = 0
    cfg.device = device
    policy = get_policy_class(t).from_pretrained(pretrained, config=cfg).to(device)
    if t in ('pi0', 'pi05'):
        if cfg.dtype == 'bfloat16':
            policy = policy.to(torch.bfloat16)
        for n, q in policy.named_parameters():        # the expert's lm_head is unused for actions (263M params)
            if 'gemma_expert.lm_head' in n:
                q.requires_grad_(False)
    return policy


def _paligemma_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(PALIGEMMA_TOK)


def tokenize(policy, task):
    t = policy_type(policy)
    task = task.rstrip('\n')
    if t == 'smolvla':
        tok = policy.model.vlm_with_expert.processor.tokenizer
        enc = tok([task + '\n'], return_tensors='pt', padding='longest')
    else:
        tok = _paligemma_tokenizer()
        if t == 'pi05':   # lerobot Pi05PrepareStateTokenizerProcessorStep with a zero state (32 dims → bin 127)
            state_str = ' '.join(['127'] * policy.config.max_state_dim)
            task = f'Task: {task}, State: {state_str};\nAction: '
        else:
            task = task + '\n'
        enc = tok([task], return_tensors='pt', padding='max_length', max_length=policy.config.tokenizer_max_length,
                  truncation=True, padding_side='right')
    return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}


def make_batch(policy, img_u8, tokens, device, state_dim=2):
    """SmolVLA/π0 batch dict built directly (bypassing LeRobot processors): one camera, no proprioception,
    actions already normalised to [-1, 1] by robots.<robot>.act_scale."""
    B = img_u8.shape[0]
    t = policy_type(policy)
    return {
        IMG_KEY[t]: img_u8.to(device, non_blocking=True).float() / 255.0,
        'observation.state': torch.zeros(B, state_dim, device=device),
        'observation.language.tokens': tokens['input_ids'].to(device).expand(B, -1),
        'observation.language.attention_mask': tokens['attention_mask'].to(device).bool().expand(B, -1),
    }


def load_finetuned(ckpt_dir, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(ckpt_dir, config=cfg).to(device)
    if cfg.type in ('pi0', 'pi05') and getattr(cfg, 'dtype', 'float32') == 'bfloat16':
        policy = policy.to(torch.bfloat16)
    return policy


def trainable_params(policy):
    return [p for p in policy.parameters() if p.requires_grad]
