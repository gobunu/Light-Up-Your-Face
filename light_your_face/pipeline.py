"""One-step FiLitDiff inference with the original SD 2.1 and LoRA weights."""
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from safetensors.torch import load_file

from .lighting import load_lighting_encoder


def merge_lora(unet, path, alpha=16.0):
    """Merge the checkpoint's PEFT-format LoRA, including convolution layers."""
    weights = load_file(str(path), device="cpu")
    state = unet.state_dict()
    consumed = set()
    for key, down in weights.items():
        if key.endswith('.lora_A.weight'):
            up_key = key.replace('.lora_A.weight', '.lora_B.weight')
            target = key.removeprefix('unet.').replace('.lora_A.weight', '.weight')
        elif key.endswith('.lora.down.weight'):
            up_key = key.replace('.lora.down.weight', '.lora.up.weight')
            target = key.removeprefix('unet.').replace('.lora.down.weight', '.weight')
        else:
            continue
        if up_key not in weights or target not in state:
            raise ValueError(f'Incompatible LoRA layer: {key}')
        up = weights[up_key]
        if down.ndim not in (2, 4) or (up.ndim == 4 and up.shape[2:] != (1, 1)):
            raise ValueError(f'Unsupported LoRA shape: {key}')
        delta = up.flatten(1) @ down.flatten(1)
        if delta.numel() != state[target].numel():
            raise ValueError(f'LoRA shape mismatch: {key}')
        state[target] = state[target] + (alpha / down.shape[0]) * delta.reshape_as(state[target])
        consumed.update((key, up_key))
    if not consumed or consumed != set(weights):
        raise ValueError(f'Unrecognized LoRA weights: {sorted(set(weights) - consumed)}')
    unet.load_state_dict(state, strict=True)


def predict_clean_latent(latent, noise, alphas, timestep):
    """Original epsilon-to-x0 conversion, evaluated in double precision."""
    alpha = alphas[timestep].double().reshape(-1, 1, 1, 1)
    return ((latent.double() - (1 - alpha).sqrt() * noise.double()) / alpha.sqrt()).float()


class FiLitDiff:
    def __init__(self, base_model, checkpoint, device='cuda', precision='fp32',
                 timestep=199, lora_alpha=16.0, vae_tiling=False):
        checkpoint = Path(checkpoint)
        for name in ('img_encoder_weights.pth', 'pytorch_lora_weights.safetensors'):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        self.device = torch.device(device)
        self.dtype = torch.float16 if precision == 'fp16' else torch.float32
        if self.device.type == 'cpu' and self.dtype == torch.float16:
            raise ValueError('Use fp32 for CPU inference.')
        scheduler = DDIMScheduler.from_pretrained(base_model, subfolder='scheduler')
        if not 0 <= timestep < len(scheduler.alphas_cumprod):
            raise ValueError('Timestep is outside the base scheduler range.')
        self.timestep = timestep
        self.alphas = scheduler.alphas_cumprod.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(base_model, subfolder='vae')
        self.unet = UNet2DConditionModel.from_pretrained(base_model, subfolder='unet')
        merge_lora(self.unet, checkpoint / 'pytorch_lora_weights.safetensors', lora_alpha)
        self.lighting = load_lighting_encoder(checkpoint / 'img_encoder_weights.pth')
        for module in (self.vae, self.unet, self.lighting):
            module.eval().requires_grad_(False).to(self.device, dtype=self.dtype)
        if vae_tiling:
            self.vae.enable_tiling()

    @torch.inference_mode()
    def __call__(self, image, condition, seed=114, guidance_scale=1.0):
        """image: RGB tensor [1,3,H,W] in [0,1]; condition: normalized six-vector."""
        if image.ndim != 4 or image.shape[:2] != (1, 3):
            raise ValueError('Expected a single RGB image shaped [1,3,H,W].')
        if len(condition) != 6 or guidance_scale < 1:
            raise ValueError('Provide six lighting values and guidance_scale >= 1.')
        height, width = image.shape[-2:]
        image = image.to(self.device, dtype=self.dtype)
        # Padding keeps the original field of view and supports non-multiple-of-8 inputs.
        pad_h, pad_w = (-height) % 8, (-width) % 8
        normalized = F.pad(image * 2 - 1, (0, pad_w, 0, pad_h), mode='replicate')
        cond = torch.tensor(condition, device=self.device, dtype=self.dtype).reshape(1, 6)
        embedding = self.lighting(cond)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent = self.vae.encode(normalized).latent_dist.sample(generator=generator)
        latent = latent * self.vae.config.scaling_factor
        timesteps = torch.tensor([self.timestep], device=self.device, dtype=torch.long)
        if guidance_scale > 1:
            noise = self.unet(torch.cat((latent, latent)), timesteps.repeat(2),
                              encoder_hidden_states=torch.cat((torch.zeros_like(embedding), embedding))).sample
            uncond, cond_noise = noise.chunk(2)
            noise = uncond + guidance_scale * (cond_noise - uncond)
        else:
            noise = self.unet(latent, timesteps, encoder_hidden_states=embedding).sample
        clean = predict_clean_latent(latent, noise, self.alphas, timesteps)
        result = self.vae.decode(clean.to(self.dtype) / self.vae.config.scaling_factor).sample
        return (result * 0.5 + 0.5)[..., :height, :width].float()
