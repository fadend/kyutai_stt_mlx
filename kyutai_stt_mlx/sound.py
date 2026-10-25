import numpy as np
import resampy
import soundfile


def read_sound(path: str, sample_rate: int) -> np.array:
    data, original_rate = soundfile.read(path, always_2d=True)
    # Convert stereo to mono
    data = np.mean(data.astype(np.float32), axis=1)
    if original_rate != sample_rate:
        data = resampy.resample(data, sr_orig=original_rate, sr_new=sample_rate)
    return np.reshape(data, (1, -1))
