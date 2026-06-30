import numpy as np
import math
import pandas as pd
from scipy import interpolate as interp
from scipy.signal import savgol_filter

from utils import apply_offset_to_flow, detect_end_index

def safe_savgol(y_data, default_window, polyorder=2):
    """
    Safely applies the Savitzky-Golay filter.
    Dynamically shrinks the window if the dataset is too small to prevent crashes.
    """
    if len(y_data) <= polyorder + 1:
        return y_data
    
    w_len = min(default_window, len(y_data))
    if w_len % 2 == 0:
        w_len -= 1
        
    return savgol_filter(y_data, window_length=w_len, polyorder=polyorder)

def preprocess_data(data, diameter, height, apply_offset=False, strain_increment=0.005):
    """
    Preprocess raw compression-test data into resampled flow curves.
    """
    dx = float(strain_increment)
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("strain_increment must be a positive finite number.")

    def _grid_by_dx(x_max: float, dx: float) -> np.ndarray:
        xs = np.arange(0.0, x_max + 0.5 * dx, dx)
        if xs.size < 2:
            raise ValueError("strain_increment too large for the available strain range.")
        if xs[-1] > x_max + 1e-12:
            xs[-1] = x_max
        elif xs[-1] < x_max - 1e-12:
            xs = np.append(xs, x_max)
        return xs

    data = data.copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data.dropna(inplace=True)

    required = {"Jaw", "Force"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    diameter = float(diameter)
    height = float(height)
    if diameter <= 0 or height <= 0:
        raise ValueError("diameter and height must be positive.")

    # -------------------- SIGN & COMPLIANCE CORRECTION --------------------
    data["Jaw"] = -data["Jaw"]
    data["Force"] = -data["Force"]
    data["Jaw_corr"] = data["Jaw"] - data["Force"] * 0.0029

    # -------------------- STARTUP / JAW SANITY TRIM --------------------
    jaw = data["Jaw_corr"].values
    valid = np.isfinite(jaw) & (jaw < 0.98 * height)

    if not np.any(valid):
        raise ValueError("No valid Jaw_corr samples found.")

    N_CONSEC = 5
    ok = valid.astype(int)
    consec = np.convolve(ok, np.ones(N_CONSEC, dtype=int), mode="valid") == N_CONSEC
    if np.any(consec):
        start_idx = int(np.where(consec)[0][0])
        data = data.iloc[start_idx:].copy()
    else:
        start_idx = int(np.where(valid)[0][0])
        data = data.iloc[start_idx:].copy()

    data.reset_index(drop=True, inplace=True)

    # -------------------- NEW: SENSOR SETTLING TRIM --------------------
    # Bypasses anomalous sensor jumps (like Jaw starting at 7.25 then settling to 0)
    # Finds the true start of the test by locating the minimum Jaw displacement before peak force
    force_vals = data["Force"].values
    jaw_vals = data["Jaw_corr"].values
    peak_force_idx = int(np.argmax(force_vals))

    if peak_force_idx > 0:
        true_start_idx = int(np.argmin(jaw_vals[:peak_force_idx + 1]))
        if true_start_idx > 0:
            data = data.iloc[true_start_idx:].reset_index(drop=True)

    # -------------------- TRUE STRAIN & STRESS --------------------
    denom = (height - data["Jaw_corr"])
    # Prevent divide-by-zero or negative logs if correction overshoots
    denom = np.clip(denom, 0.001, None) 

    data["x"] = -np.log(denom / height)
    data["y"] = (
        (data["Force"] * 1000.0)
        / ((math.pi / 4.0) * diameter**2 * (height / denom))
    )

    if "TC1" in data.columns:
        data["temperature"] = pd.to_numeric(data["TC1"], errors="coerce")

    x_raw = data["x"].values.astype(float)
    y_raw = data["y"].values.astype(float)
    temp_raw = data["temperature"].values.astype(float) if "temperature" in data else None

    # -------------------- 1. CHRONOLOGICAL END CUT --------------------
    jaw_corr_raw = data["Jaw_corr"].values.astype(float)
    cut = detect_end_index(x_raw, y_raw, jaw_corr=jaw_corr_raw, height=height)

    if cut is not None and cut > 10:
        x_raw = x_raw[:cut]
        y_raw = y_raw[:cut]
        if temp_raw is not None:
            temp_raw = temp_raw[:cut]

    # -------------------- 2. MONOTONIC STRAIN FILTER --------------------
    # Drops backwards noise points safely instead of scrambling the timeline
    running_max = np.maximum.accumulate(x_raw)
    keep = x_raw >= running_max
    
    x_raw = x_raw[keep]
    y_raw = y_raw[keep]
    if temp_raw is not None:
        temp_raw = temp_raw[keep]

    if x_raw.size < 5:
        raise ValueError("Not enough valid samples after trimming.")
        
    # -------------------- 3. APPLY OPTIONAL OFFSET --------------------
    x_stress = x_raw
    y_stress = y_raw

    if apply_offset:
        df_xy = pd.DataFrame({"x": x_stress, "y": y_stress})
        df_xy = apply_offset_to_flow(df_xy)
        x_stress = df_xy["x"].values.astype(float)
        y_stress = df_xy["y"].values.astype(float)

    if x_stress.size < 5:
        raise ValueError("Not enough samples after offset removal.")

    max_x_stress = float(np.max(x_stress))
    if not np.isfinite(max_x_stress) or max_x_stress <= 0:
        raise ValueError("Invalid strain range after preprocessing.")

    # -------------------- 4. SAFE INTERPOLATION --------------------
    new_x = _grid_by_dx(max_x_stress, dx)
    
    # fill_value boundaries prevent the line from extrapolating into negative values
    fY = interp.interp1d(
        x_stress,
        y_stress,
        bounds_error=False,
        fill_value=(y_stress[0], y_stress[-1]),
    )
    y_none = fY(new_x)

    # -------------------- 5. FINAL CLIP & EXPORT --------------------
    # np.clip mathematically guarantees no negative stress values (rings) from the filter
    y_none_safe = np.clip(y_none, 0, None)
    
    result = pd.DataFrame(
        {
            "x": new_x,
            "y_none": y_none_safe,
            "y_mild": np.clip(safe_savgol(y_none_safe, default_window=51), 0, None),
            "y_strong": np.clip(safe_savgol(y_none_safe, default_window=101), 0, None),
        }
    )

    # Temperature interpolation
    if temp_raw is not None:
        fT = interp.interp1d(
            x_raw,
            temp_raw,
            bounds_error=False,
            fill_value=(float(temp_raw[0]), float(temp_raw[-1])),
        )
        result["temperature"] = fT(new_x)

    return result