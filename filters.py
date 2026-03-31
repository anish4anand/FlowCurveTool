import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt


def apply_filter(y, mode="none"):
    y = np.asarray(y)

    if mode == "none":
        return y

    if mode == "mild":
        return savgol_filter(y, window_length=11, polyorder=2)

    if mode == "strong":
        # Butterworth low-pass
        # cutoff chosen empirically for smooth flow curves
        b, a = butter(N=3, Wn=0.08, btype="low", analog=False)
        return filtfilt(b, a, y)

    raise ValueError(f"Unknown filter mode: {mode}")
