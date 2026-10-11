"""Plan inference from images or the extracted LYF release metadata."""
import json
from dataclasses import dataclass
from pathlib import Path

from .conditions import PARAMETER_KEYS, normalize_light_params


@dataclass(frozen=True)
class InferenceJob:
    source: Path
    target: Path
    condition: tuple


def _basename(value):
    if (not isinstance(value, str) or not value or value in ('.', '..')
            or '/' in value or '\\' in value):
        raise ValueError(f'Expected a flat filename or ID, got {value!r}')
    return value


def _check_outputs(jobs, protected):
    targets = set()
    for job in jobs:
        target = job.target.resolve()
        if target in protected:
            raise ValueError(f'Output would overwrite an input or dataset file: {job.target}')
        if target in targets:
            raise ValueError(f'Duplicate output filename: {job.target.name}')
        targets.add(target)
    return jobs


def image_jobs(input_path, output_dir, condition):
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
    if input_path.is_file():
        inputs = [input_path]
    elif input_path.is_dir():
        inputs = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    else:
        raise ValueError(f'Input does not exist: {input_path}')
    if not inputs:
        raise ValueError('No input images found.')
    # Retain the suffix to distinguish face.jpg from face.png.
    jobs = [InferenceJob(p, output_dir / (p.name + '.png'), condition) for p in inputs]
    return _check_outputs(jobs, {p.resolve() for p in inputs})


def metadata_jobs(metadata, output_dir, input_dir=None, input_map=None):
    """Resolve each row's input and 6D condition without loading components.

    Inputs default to the row's `input` basename, or `{source_id}.png`.
    A JSON mapping from source IDs to paths supports externally named portraits.
    Relative mapped paths use input_dir (or the metadata directory) as root.
    """
    root = input_dir if input_dir is not None else metadata.parent
    mapping = None
    protected = {metadata.resolve()}
    if input_map is not None:
        with input_map.open(encoding='utf-8') as stream:
            mapping = json.load(stream)
        if not isinstance(mapping, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) or not v
                for k, v in mapping.items()):
            raise ValueError('--input-map must contain a JSON object mapping source IDs to image paths.')
        protected.add(input_map.resolve())
    jobs = []
    with metadata.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                source_id, variant = _basename(row['source_id']), _basename(row['variant'])
                condition = normalize_light_params(row['light_params'][key] for key in PARAMETER_KEYS)
                if 'component' in row:
                    protected.add((metadata.parent / _basename(row['component'])).resolve())
                if 'input' in row:
                    protected.add((metadata.parent / _basename(row['input'])).resolve())
                if mapping is not None:
                    source = root / mapping[source_id]
                else:
                    source = root / _basename(row.get('input', source_id + '.png'))
                if not source.is_file():
                    raise ValueError(f'Input image missing: {source}. Supply --input-dir or --input-map with matching portraits.')
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{metadata}:{line_number}: {exc}') from exc
            protected.add(source.resolve())
            jobs.append(InferenceJob(source, output_dir / f'{source_id}_{variant}.png', condition))
    if not jobs:
        raise ValueError(f'No records found in {metadata}')
    return _check_outputs(jobs, protected)
