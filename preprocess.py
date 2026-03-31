import numpy as np
import math
import pandas as pd
from scipy import interpolate as interp

from filters import apply_filter
from utils import apply_offset_to_flow, detect_end_index


def preprocess_data(data, diameter, height, apply_offset=False, strain_increment=0.005):
    """
    Preprocess raw compression-test data into resampled flow curves.

    Parameters
    ----------
    data : pd.DataFrame
        Raw dataframe containing at least 'Jaw' and 'Force'. Optional 'TC1'.
    diameter : float
        Specimen diameter in mm.
    height : float
        Specimen height in mm.
    apply_offset : bool
        If True, apply elastic-region removal / re-zero to the (x,y) flow curve only.
    strain_increment : float
        Strain step size Δε used to build the resampled x-axis: 0, Δε, 2Δε, ..., x_max.
        Example: 0.005.

    Returns
    -------
    pd.DataFrame
        Columns: x, y_none, y_mild, y_strong, and optionally temperature_x, temperature.
    """
    dx = float(strain_increment)
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("strain_increment must be a positive finite number.")

    def _grid_by_dx(x_max: float, dx: float) -> np.ndarray:
        xs = np.arange(0.0, x_max + 0.5 * dx, dx)
        if xs.size < 2:
            raise ValueError("strain_increment too large for the available strain range.")
        # Ensure last point is exactly x_max (nice for exports/plots)
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

    # Sign & compliance correction
    data["Jaw"] = -data["Jaw"]
    data["Force"] = -data["Force"]
    data["Jaw_corr"] = data["Jaw"] - data["Force"] * 0.0029

    # -------------------- STARTUP / JAW SANITY TRIM --------------------
    jaw = data["Jaw_corr"].values
    valid = np.isfinite(jaw) & (jaw < 0.98 * height)

    if not np.any(valid):
        raise ValueError(
            "No valid Jaw_corr samples found. "
            "Check sign flip/compliance or specimen height."
        )

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

    # True strain & flow stress
    denom = (height - data["Jaw_corr"])
    if (denom <= 0).any():
        raise ValueError("Invalid compression state: (height - Jaw_corr) <= 0 encountered.")

    data["x"] = -np.log(denom / height)
    data["y"] = (
        (data["Force"] * 1000.0)
        / ((math.pi / 4.0) * diameter**2 * (height / denom))
    )

    if "TC1" in data.columns:
        data["temperature"] = pd.to_numeric(data["TC1"], errors="coerce")

    # Raw failure cutoff
    x_raw = data["x"].values.astype(float)
    y_raw = data["y"].values.astype(float)
    temp_raw = data["temperature"].values.astype(float) if "temperature" in data else None

    # enforce strictly increasing strain for safe interpolation
    order = np.argsort(x_raw)
    x_raw = x_raw[order]
    y_raw = y_raw[order]
    if temp_raw is not None:
        temp_raw = temp_raw[order]

    # remove duplicate strain values
    keep = np.ones_like(x_raw, dtype=bool)
    keep[1:] = x_raw[1:] > x_raw[:-1]
    x_raw = x_raw[keep]
    y_raw = y_raw[keep]
    if temp_raw is not None:
        temp_raw = temp_raw[keep]

    if x_raw.size < 5:
        raise ValueError("Not enough valid samples after trimming.")

    jaw_corr_raw = data["Jaw_corr"].values.astype(float)

    cut = detect_end_index(
        x_raw,
        y_raw,
        jaw_corr=jaw_corr_raw,
        height=height,
        min_after_peak=20,
        near_zero_frac=0.05,
        vol_window=15,
        vol_factor=6.0,
        n_consec=8,
    )

    if cut is not None and cut > 10:
        x_raw = x_raw[:cut]
        y_raw = y_raw[:cut]
        if temp_raw is not None:
            temp_raw = temp_raw[:cut]


    # ---------- Offset applies ONLY to stress curve ----------
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
        raise ValueError("Invalid strain range after preprocessing (max strain <= 0).")

    # ---------- Stress interpolation + filters ----------
    new_x = _grid_by_dx(max_x_stress, dx)
    fY = interp.interp1d(
        x_stress,
        y_stress,
        bounds_error=False,
        fill_value="extrapolate",
    )
    y_none = fY(new_x)

    result = pd.DataFrame(
        {
            "x": new_x,
            "y_none": y_none,
            "y_mild": apply_filter(y_none, "mild"),
            "y_strong": apply_filter(y_none, "strong"),
        }
    )

# Temperature interpolation onto same strain grid
    if temp_raw is not None:
        fT = interp.interp1d(
            x_raw,
            temp_raw,
            bounds_error=False,
            fill_value=(float(temp_raw[0]), float(temp_raw[-1])),
        )
        result["temperature"] = fT(new_x)


    return result
