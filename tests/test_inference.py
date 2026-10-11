"""Regression checks for checkpoint conversion and inference contracts."""
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn

from inference import main, six_values
from light_your_face.pipeline import merge_lora, predict_clean_latent


class InferenceTests(unittest.TestCase):
    def test_linear_and_convolution_lora_scaling(self):
        model = nn.Module()
        model.linear = nn.Linear(3, 4, bias=False)
        model.conv = nn.Conv2d(3, 4, 3, bias=False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        weights = {
            'unet.linear.lora.down.weight': torch.randn(2, 3),
            'unet.linear.lora.up.weight': torch.randn(4, 2),
            'unet.conv.lora_A.weight': torch.randn(2, 3, 3, 3),
            'unet.conv.lora_B.weight': torch.randn(4, 2, 1, 1),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lora.safetensors'
            save_file(weights, str(path))
            merge_lora(model, path, alpha=4)
            for layer, down, up in (
                ('linear', weights['unet.linear.lora.down.weight'], weights['unet.linear.lora.up.weight']),
                ('conv', weights['unet.conv.lora_A.weight'], weights['unet.conv.lora_B.weight']),
            ):
                expected = before[layer + '.weight'] + 2 * (up.flatten(1) @ down.flatten(1)).reshape_as(before[layer + '.weight'])
                torch.testing.assert_close(model.state_dict()[layer + '.weight'], expected)
            weights['unexpected.weight'] = torch.zeros(1)
            save_file(weights, str(path))
            with self.assertRaises(ValueError):
                merge_lora(model, path)

    def test_epsilon_inversion_recovers_clean_latent(self):
        clean, noise = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        alphas = torch.tensor([0.9, 0.6])
        latent = alphas[1].sqrt() * clean + (1 - alphas[1]).sqrt() * noise
        result = predict_clean_latent(latent, noise, alphas, torch.tensor([1]))
        torch.testing.assert_close(result, clean, atol=1e-6, rtol=1e-5)

    def test_strength_preserves_odd_image_dimensions(self):
        import numpy as np
        from light_your_face.postprocess import adjust_strength
        source = np.full((17, 19, 3), 0.2, dtype=np.float32)
        result = adjust_strength(source + 0.1, source, 0.5)
        self.assertEqual(result.shape, source.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 0).all() and (result <= 1).all())

    def test_condition_rejects_invalid_values(self):
        import argparse
        for value in ('0,0', '0,0,0,0,0,nan', '0,0,0,0,0,inf'):
            with self.assertRaises(argparse.ArgumentTypeError):
                six_values(value)

    def test_metadata_cli_passes_each_condition_and_resumes_by_variant(self):
        import numpy as np
        from PIL import Image
        calls = []

        class RecordingModel:
            def __call__(self, image, condition, seed, guidance):
                calls.append((condition, seed, guidance, tuple(image.shape)))
                return torch.full_like(image, condition[-1] + 0.5)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.fromarray(np.full((5, 7, 3), 64, dtype=np.uint8)).save(root / '00000007.png')
            params = dict(disk_diameter_px=200, plane_distance_px=1000, lamp_offset_x=0,
                          lamp_offset_y=0, half_peak_angle_deg=40)
            rows = [dict(source_id='00000007', variant=variant, input='00000007.png',
                         light_params=dict(params, color_temp_K=temperature))
                    for variant, temperature in [('warm', 4000), ('cool', 12000)]]
            metadata = root / 'metadata.jsonl'
            metadata.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            argv = ['inference.py', '--metadata', str(metadata), '--output', str(root / 'out'),
                    '--checkpoint', 'unused', '--base-model', 'unused', '--seed', '123']
            with patch('sys.argv', argv), patch('light_your_face.pipeline.FiLitDiff', return_value=RecordingModel()):
                main()
                self.assertEqual([call[0][-1] for call in calls], [-0.5, 0.5])
                self.assertEqual([call[1:] for call in calls], [(123, 1.0, (1, 3, 5, 7))] * 2)
                for variant, value in [('warm', 0), ('cool', 255)]:
                    with Image.open(root / 'out' / f'00000007_{variant}.png') as prediction:
                        self.assertTrue((np.asarray(prediction) == value).all())
                (root / 'out' / '00000007_cool.png').unlink()
                calls.clear()
                main()
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0][-1], 0.5)


if __name__ == '__main__':
    unittest.main()
