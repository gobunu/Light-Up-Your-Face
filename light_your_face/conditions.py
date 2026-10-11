"""Physical lighting parameters shared by image and metadata inference."""
import math


PARAMETER_KEYS = (
    'disk_diameter_px', 'plane_distance_px', 'lamp_offset_x',
    'lamp_offset_y', 'half_peak_angle_deg', 'color_temp_K',
)


def normalize_light_params(values):
    """Normalize diameter, distance, x, y, angle, and temperature for PALP."""
    values = tuple(float(value) for value in values)
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError('Use six finite physical lighting parameters.')
    diameter, distance, x, y, angle, temperature = values
    if diameter <= 0 or distance <= 0 or not 0 < angle < 90 or temperature <= 0:
        raise ValueError('Diameter, distance, and temperature must be positive; angle must be between 0 and 90.')
    return (diameter / 200 - 1, distance / 1000 - 1, x / 3000,
            y / 3000, angle / 40 - 1, temperature / 8000 - 1)
