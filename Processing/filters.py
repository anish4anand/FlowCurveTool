from scipy.signal import savgol_filter, butter, filtfilt
import numpy as np

def apply_filter(y, mode="none"):
    if mode == "none":
        return y.copy()
    elif mode == "light":
        window = min(11, len(y)//2*2 + 1)
        return savgol_filter(y, window_length=window, polyorder=3)
    elif mode == "strong":
        b, a = butter(3, 0.1, btype="low")
        return filtfilt(b, a, y)
    else:
        raise ValueError(f"Unknown filter mode: {mode}")
