"""Check released metadata routing before loading large model weights."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from inference import build_parser, prepare_jobs
from light_your_face.inference_inputs import metadata_jobs


class MetadataInferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.metadata = self.root / 'metadata.jsonl'
        self.output = self.root / 'outputs'
        self.params = dict(disk_diameter_px=200, plane_distance_px=1000,
                           lamp_offset_x=0, lamp_offset_y=0,
                           half_peak_angle_deg=40, color_temp_K=8000)
        self.rows = [dict(source_id='00000007', variant=variant,
                          component=f'00000007_{variant}.png',
                          input='00000007.png',
                          light_params=dict(self.params, color_temp_K=temperature))
                     for variant, temperature in [('warm', 4000), ('cool', 12000)]]
        (self.root / '00000007.png').touch()
        self.write_rows()

    def write_rows(self):
        self.metadata.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def arguments(self, *flags):
        return build_parser().parse_args([
            '--output', str(self.output), '--checkpoint', 'unused',
            '--base-model', 'unused', *flags])

    def test_editval_uses_embedded_input_and_per_row_condition(self):
        jobs = prepare_jobs(self.arguments('--metadata', str(self.metadata)))
        self.assertEqual([job.source for job in jobs], [self.root / '00000007.png'] * 2)
        self.assertEqual([job.target.name for job in jobs], ['00000007_warm.png', '00000007_cool.png'])
        self.assertEqual([job.condition for job in jobs], [(0, 0, 0, 0, 0, -0.5), (0, 0, 0, 0, 0, 0.5)])
        # Inference never requires the component PNGs or a reconstructed target.
        self.assertFalse((self.root / self.rows[0]['component']).exists())

    def test_val_inputs_from_external_directory(self):
        for row in self.rows:
            row.pop('input')
        self.write_rows()
        portraits = self.root / 'portraits'
        portraits.mkdir()
        (portraits / '00000007.png').touch()
        jobs = prepare_jobs(self.arguments('--metadata', str(self.metadata), '--input-dir', str(portraits)))
        self.assertEqual(jobs[0].source, portraits / '00000007.png')

    def test_external_source_id_mapping(self):
        portraits = self.root / 'portraits'
        portraits.mkdir()
        portrait = portraits / 'original.jpg'
        portrait.touch()
        mapping = self.root / 'inputs.json'
        mapping.write_text(json.dumps({'00000007': 'original.jpg'}))
        jobs = metadata_jobs(self.metadata, self.output, portraits, mapping)
        self.assertEqual(jobs[0].source, portrait)
        mapping.write_text(json.dumps({'00000007': str(portrait)}))
        self.assertEqual(metadata_jobs(self.metadata, self.output, input_map=mapping)[0].source, portrait)
        mapping.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'metadata.jsonl:1'):
            metadata_jobs(self.metadata, self.output, input_map=mapping)

    def test_conflicting_modes_rejected(self):
        for flag in ('--cond6d', '--light-params'):
            with self.assertRaisesRegex(ValueError, 'per-record'):
                prepare_jobs(self.arguments('--metadata', str(self.metadata), flag, '0,0,0,0,0,0'))
        with self.assertRaisesRegex(ValueError, 'require --metadata'):
            prepare_jobs(self.arguments('--input', str(self.root), '--input-dir', str(self.root)))

    def test_image_mode_retains_defaults_and_filename_suffix(self):
        jobs = prepare_jobs(self.arguments('--input', str(self.root / '00000007.png')))
        self.assertEqual(jobs[0].condition, (0, 0, 0, 0, 0, 0.3))
        self.assertEqual(jobs[0].target.name, '00000007.png.png')
        physical = prepare_jobs(self.arguments('--input', str(self.root / '00000007.png'),
                                               '--light-params', '200,1000,0,0,40,10400'))
        for actual, expected in zip(physical[0].condition, jobs[0].condition):
            self.assertAlmostEqual(actual, expected)

    def test_corrupt_missing_and_unsafe_records_fail_with_line_number(self):
        for key, value in [('input', '../outside.png'), ('variant', '../../warm'),
                           ('source_id', 7), ('light_params', dict(self.params, color_temp_K=float('nan')))]:
            with self.subTest(key=key):
                self.metadata.write_text(json.dumps(dict(self.rows[0], **{key: value})) + '\n')
                with self.assertRaisesRegex(ValueError, 'metadata.jsonl:1'):
                    metadata_jobs(self.metadata, self.output)
        self.metadata.write_text('{broken json\n')
        with self.assertRaisesRegex(ValueError, 'metadata.jsonl:1'):
            metadata_jobs(self.metadata, self.output)
        self.write_rows()
        (self.root / '00000007.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Input image missing'):
            metadata_jobs(self.metadata, self.output)

    def test_duplicate_targets_and_dataset_overwrite_rejected(self):
        self.rows.append(self.rows[0])
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate output'):
            metadata_jobs(self.metadata, self.output)
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            metadata_jobs(self.metadata, self.root)

    def test_resume_skips_finished_outputs_without_loading_model(self):
        self.output.mkdir()
        for row in self.rows:
            (self.output / f'{row["source_id"]}_{row["variant"]}.png').touch()
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'inference.py'),
                   '--metadata', str(self.metadata), '--output', str(self.output),
                   '--checkpoint', 'missing-checkpoint', '--base-model', 'missing-model']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('All outputs already exist', result.stdout)


if __name__ == '__main__':
    unittest.main()
