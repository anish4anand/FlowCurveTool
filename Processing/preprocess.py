import numpy as np
import math
import pandas as pd
from scipy import interpolate as interp
from .filters import apply_filter

def preprocess_data(data, diameter, height, filter_mode="none"):
    data = data.copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data.dropna(inplace=True)

    if "Jaw" in data.columns:
        data["Jaw"] = -data["Jaw"]
    if "Force" in data.columns:
        data["Force"] = -data["Force"]

    if "Jaw" in data.columns and "Force" in data.columns:
        data["Jaw_corr"] = data["Jaw"] - (data["Force"] * 0.0029)
    else:
        raise ValueError("CSV must contain 'Jaw' and 'Force'!")

    data["true_strain"] = -np.log((height - data["Jaw_corr"]) / height)
    data["flow_stress"] = (
        (data["Force"] * 1000)
        / ((math.pi / 4) * diameter**2 * height / (height - data["Jaw_corr"]))
    )

    x1, y1 = data["true_strain"].values, data["flow_stress"].values
    new_x = np.linspace(0, 1.04, 209)
    f_flow = interp.interp1d(x1, y1, fill_value="extrapolate")
    new_y = f_flow(new_x)

    y_filt = apply_filter(new_y, filter_mode)

    dy = np.gradient(y_filt, new_x)
    med, mad = np.median(dy), np.median(np.abs(dy - np.median(dy))) + 1e-12
    thr = med - 6 * mad
    drop = np.where(dy < thr)[0]
    if drop.size:
        new_x, y_filt = new_x[:drop[0]], y_filt[:drop[0]]

    return pd.DataFrame({"x": new_x, "y": y_filt})
