"""Enhance images with manual lighting control or per-record LYF metadata."""
import argparse
import math
from pathlib import Path

from light_your_face.conditions import normalize_light_params
from light_your_face.inference_inputs import image_jobs, metadata_jobs


def six_values(value):
    try:
        values = tuple(float(x) for x in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use six comma-separated numbers.') from exc
    if len(values) != 6 or not all(math.isfinite(x) for x in values):
        raise argparse.ArgumentTypeError('Use six finite comma-separated numbers.')
    return values


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path, help='Image or directory of images.')
    inputs.add_argument('--metadata', type=Path, help='metadata.jsonl from an extracted LYF archive.')
    parser.add_argument('--input-dir', type=Path, help='Metadata inputs directory; defaults to the metadata directory.')
    parser.add_argument('--input-map', type=Path, help='JSON object mapping source IDs to image paths for metadata inference.')
    parser.add_argument('--output', type=Path, required=True, help='Output directory.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Directory containing both model weights.')
    parser.add_argument('--base-model', required=True, help='SD 2.1 base directory or Hugging Face model ID.')
    conditions = parser.add_mutually_exclusive_group()
    conditions.add_argument('--cond6d', type=six_values, help='Normalized lighting parameters (default: 0,0,0,0,0,0.3).')
    conditions.add_argument('--light-params', type=six_values, help='Physical parameters: diameter,distance,x,y,angle,temperature.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'fp16'), default='fp32')
    parser.add_argument('--seed', type=int, default=114)
    parser.add_argument('--timestep', type=int, default=199)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--lora-alpha', type=float, default=16.0)
    parser.add_argument('--strength', type=float, default=1.0, help='Wavelet fill-light adjustment; 1 keeps the raw prediction.')
    parser.add_argument('--vae-tiling', action='store_true', help='Use Diffusers VAE tiling for larger inputs.')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def prepare_jobs(args):
    if args.metadata is not None:
        if args.cond6d is not None or args.light_params is not None:
            raise ValueError('--metadata uses per-record lighting parameters; omit --cond6d and --light-params.')
        return metadata_jobs(args.metadata, args.output, args.input_dir, args.input_map)
    if args.input_dir is not None or args.input_map is not None:
        raise ValueError('--input-dir and --input-map require --metadata.')
    condition = args.cond6d if args.cond6d is not None else (0, 0, 0, 0, 0, 0.3)
    if args.light_params is not None:
        condition = normalize_light_params(args.light_params)
    return image_jobs(args.input, args.output, condition)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.strength) or args.strength < 0:
        parser.error('--strength must be finite and nonnegative.')
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
        parser.error('--guidance-scale must be finite and >= 1.')
    if not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0:
        parser.error('--lora-alpha must be finite and positive.')
    try:
        jobs = prepare_jobs(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    jobs = [job for job in jobs if args.overwrite or not job.target.exists()]
    if not jobs:
        print('All outputs already exist; use --overwrite to regenerate.')
        return
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from light_your_face.pipeline import FiLitDiff

    model = FiLitDiff(args.base_model, args.checkpoint, args.device, args.precision,
                     args.timestep, args.lora_alpha, args.vae_tiling)
    args.output.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        source, target = job.source, job.target
        with Image.open(source) as opened:
            array = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'), dtype=np.float32) / 255
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
        result = model(tensor, job.condition, args.seed, args.guidance_scale)[0].permute(1, 2, 0).cpu().numpy()
        if args.strength != 1:
            from light_your_face.postprocess import adjust_strength
            result = adjust_strength(result, array, args.strength)
        if not np.isfinite(result).all():
            raise RuntimeError(f'Nonfinite model output for {source}')
        Image.fromarray((np.clip(result, 0, 1) * 255).round().astype(np.uint8)).save(target)
        print(f'Saved {target}')


if __name__ == '__main__':
    main()
