import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def detect_end_index(x: np.ndarray, y: np.ndarray, near_zero_frac=0.05, neg_slope_thresh=0.0, vol_factor=6.0, post_peak_drop_frac=0.40, **kwargs) -> int:
    """
    Finds where the test actually ends by detecting jaw reversal, 
    specimen fracture, or machine stop (post-peak drop).
    """
    if len(y) == 0:
        return 0
        
    peak_idx = np.argmax(y)
    peak_stress = y[peak_idx]
    
    # 1. Post-Peak Drop (Now tolerates up to 40% high-temp flow softening)
    for i in range(peak_idx, len(y)):
        if y[i] < (peak_stress * (1.0 - post_peak_drop_frac)):
            print(f"End cut triggered: Post-peak drop at index {i}")
            return i

    # 2. Jaw Reversal / Machine Stop (Negative strain slope)
    dx = np.diff(x)
    reversals = np.where(dx <= neg_slope_thresh)[0]
    reversals = [i for i in reversals if i > peak_idx]
    if len(reversals) > 0:
        print(f"End cut triggered: Jaw reversal at index {reversals[0]}")
        return int(reversals[0])

    # 3. Near-zero collapse (for brittle fracture)
    nz = np.where(y < near_zero_frac * peak_stress)[0]
    nz = [i for i in nz if i > peak_idx]
    if len(nz) > 0:
        print(f"End cut triggered: Near-zero collapse at index {nz[0]}")
        return int(nz[0])

    # 4. Volatility / Noise (Cracking)
    dy = np.diff(y)
    if len(dy) > 20:
        pre = dy[:max(10, peak_idx - 5)]
        baseline = np.median(np.abs(pre - np.median(pre))) if len(pre) > 0 else 1.0
        baseline = max(baseline, 1e-6)
        
        rolling_mad = np.zeros_like(dy)
        window = 10
        for i in range(len(dy)):
            start = max(0, i - window)
            chunk = dy[start:i+1]
            rolling_mad[i] = np.median(np.abs(chunk - np.median(chunk)))
            
        spike_idx = np.where(rolling_mad > vol_factor * baseline)[0]
        spike_idx = [i for i in spike_idx if i > peak_idx]
        if len(spike_idx) > 0:
            print(f"End cut triggered: Volatility spike at index {spike_idx[0]}")
            return int(spike_idx[0])
            
    return len(y) - 1

def deform_qform_table(df: pd.DataFrame, y_col: str = "y_none", n_points: int = 25, cluster_factor: float = 50.0) -> pd.DataFrame:
    """
    Downsamples a flow curve into exactly `n_points` using geometric spacing.
    Automatically adapts to the total strain length of the dataset to capture the yield knee perfectly.
    """
    # 1. Fallback if the requested filter column doesn't exist
    if y_col not in df.columns:
        y_col = "y" if "y" in df.columns else df.columns[1]
        
    x = df["x"].values.astype(float)
    y = df[y_col].values.astype(float)
    
    x_min = np.min(x)
    x_max = np.max(x)
    
    # 2. Generate geometrically spaced target X values (normalized 0 to 1)
    norm_space = (np.geomspace(1, cluster_factor, n_points) - 1) / (cluster_factor - 1)
    
    # 3. Scale the normalized points to fit exactly between our actual x_min and x_max
    x_target = x_min + (x_max - x_min) * norm_space
    
    # 4. Interpolate the actual Y values at these exact target X locations
    y_target = np.interp(x_target, x, y)
    
    # 5. Prevent physically impossible negative stress artifacts
    y_target = np.clip(y_target, 0, None)
    
    # 6. Create the final output DataFrame
    out = pd.DataFrame({
        "Strain": x_target,
        "Flow_Stress": y_target
    })
    
    # Optional: If your original data tracked temperature, carry it over
    if "Temperature" in df.columns:
        out["Temperature"] = df["Temperature"].iloc[0]
        
    return out

def force_displacement_curve_from_raw(
    raw: pd.DataFrame,
    diameter: float,
    height: float,
    compliance: float = 0.0029,
    failure_drop_frac: float = 0.85,
    jaw_upper_frac: float = 0.98,
    n_consec: int = 5,
) -> pd.DataFrame:
    """
    Build a solver/plot-friendly FORCE–DISPLACEMENT curve from raw data, using the
    SAME trimming + failure cutoff logic as preprocess_data, but outputting:
      - displacement_mm (Jaw_corr, mm)
      - force_kN (Force, kN)

    Assumes raw has columns: 'Jaw', 'Force' (and may contain others).
    """
    if raw is None or raw.empty:
        raise ValueError("Raw data is empty.")

    data = raw.copy()
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

    # Match preprocess_data sign flip + compliance correction
    data["Jaw"] = -data["Jaw"]
    data["Force"] = -data["Force"]
    data["Jaw_corr"] = data["Jaw"] - data["Force"] * float(compliance)

    # -------------------- STARTUP / JAW SANITY TRIM --------------------
    jaw = data["Jaw_corr"].values.astype(float)
    valid = np.isfinite(jaw) & (jaw < float(jaw_upper_frac) * height)

    if not np.any(valid):
        raise ValueError("No valid Jaw_corr samples found after sign/compliance correction.")

    ok = valid.astype(int)
    consec = np.convolve(ok, np.ones(int(n_consec), dtype=int), mode="valid") == int(n_consec)
    if np.any(consec):
        start_idx = int(np.where(consec)[0][0])
        data = data.iloc[start_idx:].copy()
    else:
        start_idx = int(np.where(valid)[0][0])
        data = data.iloc[start_idx:].copy()

    data.reset_index(drop=True, inplace=True)

    # Recompute x,y only to reproduce the same failure cutoff
    denom = (height - data["Jaw_corr"])
    if (denom <= 0).any():
        raise ValueError("Invalid compression state: (height - Jaw_corr) <= 0 encountered.")

    x = -np.log(denom / height)
    y = (data["Force"] * 1000.0) / ((np.pi / 4.0) * diameter**2 * (height / denom))

    y_vals = np.asarray(y.values, dtype=float)
    peak_idx = int(np.argmax(y_vals))
    drop = np.where(y_vals < float(failure_drop_frac) * y_vals[peak_idx])[0]
    drop = drop[drop > peak_idx]

    if drop.size:
        cut = int(drop[0])
        data = data.iloc[:cut].copy()

    out = pd.DataFrame(
        {
            "displacement_mm": data["Jaw_corr"].to_numpy(dtype=float),
            "force_kN": data["Force"].to_numpy(dtype=float),
        }
    )

    # Clean up
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out["displacement_mm"] >= 0.0].copy()
    if out.empty:
        raise ValueError("Force–displacement curve is empty after trimming/cut.")

    return out

# EXTRAPOLATION MODELS
# =====================================================================

def hollomon(epsilon, K, n):
    return K * epsilon**n

def ludwik(epsilon, sigma_0, K, n):
    return sigma_0 + K * epsilon**n

def swift(epsilon, epsilon_0, K, n):
    epsilon_total = epsilon_0 + epsilon
    epsilon_total = np.maximum(epsilon_total, 1e-8)
    return K * epsilon_total**n

def voce(epsilon, sigma_inf, sigma_0, m):
    return sigma_inf + (sigma_0 - sigma_inf) * np.exp(-m * epsilon)

def hockett_sherby(epsilon, sigma_inf, sigma_0, m, p):
    return sigma_inf - (sigma_inf - sigma_0) * np.exp(-m * epsilon**p)

def fit_extrapolation_models(epsilon, sigma):
    epsilon = np.maximum(epsilon, 1e-8)
    results = {}
    
    # Safe initial guesses based on data
    sigma_max = max(np.max(sigma) * 1.2, 1.0)
    sigma_0 = max(sigma[0], 1.0)
    
    # Hollomon
    try:
        popt, _ = curve_fit(hollomon, epsilon, sigma, p0=[500, 0.3], bounds=([0, 0], [np.inf, np.inf]), maxfev=20000)
        results['Hollomon'] = popt
    except Exception as e:
        print(f"Hollomon fit failed: {e}")
        results['Hollomon'] = None
        
    # Ludwik
    try:
        popt, _ = curve_fit(ludwik, epsilon, sigma, p0=[sigma_0, 50, 0.25], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=20000)
        results['Ludwik'] = popt
    except Exception as e:
        print(f"Ludwik fit failed: {e}")
        results['Ludwik'] = None
        
    # Swift
    try:
        popt, _ = curve_fit(swift, epsilon, sigma, p0=[0.01, 50, 0.2], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=20000)
        results['Swift'] = popt
    except Exception as e:
        print(f"Swift fit failed: {e}")
        results['Swift'] = None
        
    # Voce
    try:
        popt, _ = curve_fit(voce, epsilon, sigma, p0=[sigma_max, sigma_0, 10], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=20000)
        results['Voce'] = popt
    except Exception as e:
        print(f"Voce fit failed: {e}")
        results['Voce'] = None
        
    # Hockett-Sherby
    try:
        popt, _ = curve_fit(hockett_sherby, epsilon, sigma, p0=[sigma_max, sigma_0, 5, 1.0], bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]), maxfev=20000)
        results['Hockett-Sherby'] = popt
    except Exception as e:
        print(f"Hockett-Sherby fit failed: {e}")
        results['Hockett-Sherby'] = None

    # Swift+Voce Combination
    if results['Swift'] is not None and results['Voce'] is not None:
        def swift_voce(eps, a):
            return a * swift(eps, *results['Swift']) + (1 - a) * voce(eps, *results['Voce'])
        try:
            popt, _ = curve_fit(swift_voce, epsilon, sigma, p0=[0.5], bounds=([0], [1]), maxfev=20000)
            results['Swift+Voce'] = popt
        except Exception as e:
            print(f"Swift+Voce fit failed: {e}")
            results['Swift+Voce'] = None
    else:
        results['Swift+Voce'] = None
        
    return results

def evaluate_model(model_name, epsilon, params, swift_params=None, voce_params=None):
    if params is None:
        return np.full_like(epsilon, np.nan)
    
    if model_name == 'Hollomon':
        return hollomon(epsilon, *params)
    elif model_name == 'Ludwik':
        return ludwik(epsilon, *params)
    elif model_name == 'Swift':
        return swift(epsilon, *params)
    elif model_name == 'Voce':
        return voce(epsilon, *params)
    elif model_name == 'Hockett-Sherby':
        return hockett_sherby(epsilon, *params)
    elif model_name == 'Swift+Voce':
        a = params[0]
        return a * swift(epsilon, *swift_params) + (1 - a) * voce(epsilon, *voce_params)
    return np.full_like(epsilon, np.nan)

