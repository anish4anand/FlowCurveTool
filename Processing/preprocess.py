import numpy as np
import math
import pandas as pd
from scipy import interpolate as interp
from .filters import apply_filter

def preprocess_data(data, diameter, height, filter_mode="none"):
    
    # -------------------- CLEANUP --------------------
    data = data.copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data.dropna(inplace=True)

    # Sign correction
    data["Jaw"] = -data["Jaw"]
    data["Force"] = -data["Force"]

    # Compliance correction
    data["Jaw_corr"] = data["Jaw"] - (data["Force"] * 0.0029)

    # True strain & stress
    data["true_strain"] = -np.log((height - data["Jaw_corr"]) / height)
    data["flow_stress"] = (
        (data["Force"] * 1000)
        / ((math.pi / 4) * diameter**2 * (height / (height - data["Jaw_corr"])))
    )

    # Optional temperature
    if "TC1" in data.columns:
        data["temperature"] = pd.to_numeric(data["TC1"], errors="coerce")
    else:
        data["temperature"] = None

    # -------------------- RAW FAILURE CUTOFF --------------------
    x_raw = data["true_strain"].values
    y_raw = data["flow_stress"].values
    temp_raw = data["temperature"].values if data["temperature"] is not None else None

    peak_idx = np.argmax(y_raw)
    peak_stress = y_raw[peak_idx]

    drop_ratio = 0.85
    raw_fail = np.where(y_raw < drop_ratio * peak_stress)[0]
    raw_fail = raw_fail[raw_fail > peak_idx]

    if raw_fail.size > 0:
        cut_idx = raw_fail[0]
        x_raw = x_raw[:cut_idx]
        y_raw = y_raw[:cut_idx]
        if temp_raw is not None:
            temp_raw = temp_raw[:cut_idx]

    # -------------------- INTERPOLATION --------------------
    max_raw_strain = x_raw.max()
    new_x = np.linspace(0, max_raw_strain, 209)

    f_flow = interp.interp1d(
        x_raw, y_raw,
        bounds_error=False,
        fill_value=(y_raw[0], y_raw[-1])   # clamp to ends
    )
    new_y = f_flow(new_x)

    # Temperature interpolation
    if temp_raw is not None:
        f_temp = interp.interp1d(
            x_raw, temp_raw,
            bounds_error=False,
            fill_value=(temp_raw[0], temp_raw[-1])
        )
        new_temp = f_temp(new_x)
    else:
        new_temp = None

    # -------------------- FILTER --------------------
    from .filters import apply_filter
    y_filt = apply_filter(new_y, filter_mode)

    # -------------------- SLOPE-BASED BACKUP CUTOFF --------------------
    dy = np.gradient(y_filt, new_x)
    med = np.median(dy)
    mad = np.median(np.abs(dy - med)) + 1e-12
    thr = med - 6 * mad

    below = dy < thr
    N = 3
    consecutive = np.convolve(below.astype(int), np.ones(N), "valid") == N
    drop_candidates = np.where(consecutive)[0]

    slope_fail = drop_candidates[drop_candidates > peak_idx]

    if slope_fail.size > 0:
        cut_idx = slope_fail[0]
        new_x = new_x[:cut_idx]
        y_filt = y_filt[:cut_idx]
        if new_temp is not None:
            new_temp = new_temp[:cut_idx]

    # -------------------- FINAL RESULT --------------------
    result = pd.DataFrame({"x": new_x, "y": y_filt})
    if new_temp is not None:
        result["temperature"] = new_temp

    return result




