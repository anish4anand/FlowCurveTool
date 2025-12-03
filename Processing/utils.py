import numpy as np
import pandas as pd
def apply_offset(df, window=10, slope_jump_factor=5):
    """
    Remove elastic region and re-zero the strain axis.
    Applies the same shift to ALL columns (x, y, temperature, etc.)
    """

    df = df.copy()

    x = df.iloc[:, 0].values.astype(float)  # strain
    y = df.iloc[:, 1].values.astype(float)  # stress

    # --- Compute slope
    dy = np.gradient(y, x)

    slope_var = np.zeros_like(dy)
    for i in range(len(dy)):
        start = max(0, i - window)
        slope_var[i] = np.var(dy[start:i+1])

    baseline_var = np.median(slope_var[:20])
    plastic_pts = np.where(slope_var > slope_jump_factor * baseline_var)[0]
    idx_yield = plastic_pts[0] if len(plastic_pts) > 0 else 0

    # --- Cut elastic part
    df_cut = df.iloc[idx_yield:].copy()

    # --- SHIFT X AXIS
    strain0 = df_cut.iloc[0, 0]
    df_cut.iloc[:, 0] = df_cut.iloc[:, 0] - strain0

    # --- SHIFT Y AXIS
    stress0 = df_cut.iloc[0, 1]
    df_cut.iloc[:, 1] = df_cut.iloc[:, 1] - stress0

    # --- NO SHIFT FOR TEMPERATURE
    # (temperature begins at ~constant 200°C — this is correct physically)
    # But temperature must remain aligned with shifted strain!
    # And since we only cut rows, alignment is preserved.

    return df_cut

def temperature_compensation(df):
    df = df.copy()
    df.iloc[:, 1] *= 0.95
    return df
