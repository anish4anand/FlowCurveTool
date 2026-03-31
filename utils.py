import numpy as np
import pandas as pd


def apply_offset_to_flow(df, window=10, slope_jump_factor=5):
    """
    Remove elastic region and re-zero strain & stress.
    Expects columns 'x' and 'y'. Preserves temperature if present.
    """
    df = df.copy()

    x = df["x"].values.astype(float)
    y = df["y"].values.astype(float)

    dy = np.gradient(y, x)

    slope_var = np.zeros_like(dy)
    for i in range(len(dy)):
        start = max(0, i - window)
        slope_var[i] = np.var(dy[start:i + 1])

    baseline = np.median(slope_var[:20])
    plastic_pts = np.where(slope_var > slope_jump_factor * baseline)[0]
    idx_yield = plastic_pts[0] if plastic_pts.size else 0

    df_cut = df.iloc[idx_yield:].copy()

    x0 = df_cut["x"].iloc[0]
    y0 = df_cut["y"].iloc[0]

    df_cut["x"] -= x0
    df_cut["y"] -= y0

    return df_cut


def detect_end_index(
    x: np.ndarray,
    y: np.ndarray,
    jaw_corr: np.ndarray | None = None,
    height: float | None = None,
    min_after_peak: int = 20,
    near_zero_frac: float = 0.05,
    vol_window: int = 15,
    vol_factor: float = 6.0,
    neg_slope_thresh: float = 0.0,
    n_consec: int = 8,
) -> int | None:
    """
    Returns cut index (exclusive) where valid curve should end, or None if no cut.
    Robust across: hardening/softening/plateau metals.
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 50:
        return None

    peak = int(np.argmax(y))
    start = min(n - 1, peak + min_after_peak)
    if start >= n - 5:
        return None

    y_peak = float(y[peak])
    if not np.isfinite(y_peak) or y_peak <= 0:
        return None

    # ---------- (A) Near-zero collapse ----------
    nz = np.where(y < near_zero_frac * y_peak)[0]
    nz = nz[nz > start]
    if nz.size:
        return int(nz[0])

    # ---------- (B) Jaw reversal / stall ----------
    if jaw_corr is not None:
        jc = np.asarray(jaw_corr, float)
        dj = np.diff(jc)
        # reversal or stall (<= 0) after peak, sustained
        bad = (dj <= neg_slope_thresh)
        if bad.size:
            conv = np.convolve(bad.astype(int), np.ones(n_consec, int), mode="valid") >= n_consec
            idx = np.where(conv)[0]
            idx = idx[idx > start]
            if idx.size:
                return int(idx[0])

        # geometric crush limit: height - jaw_corr close to 0
        if height is not None and np.isfinite(height) and height > 0:
            remain = (float(height) - jc)
            crush = np.where(remain <= 0.01 * float(height))[0]   # last 1% height
            crush = crush[crush > start]
            if crush.size:
                return int(crush[0])

    # ---------- (C) Volatility / spike region ----------
    # Use rolling MAD of first difference as spike detector
    dy = np.diff(y)
    # baseline volatility from "clean" region before peak
    pre = dy[:max(10, peak - 5)]
    if pre.size < 10:
        pre = dy[:max(10, int(0.2 * len(dy)))]
    baseline = np.median(np.abs(pre - np.median(pre))) + 1e-12  # MAD

    # rolling MAD on dy
    w = int(vol_window)
    if w < 5:
        w = 5
    spike = np.zeros_like(dy, dtype=bool)
    for i in range(w, len(dy)):
        seg = dy[i - w:i]
        mad = np.median(np.abs(seg - np.median(seg))) + 1e-12
        spike[i] = mad > vol_factor * baseline

    # require sustained spikes
    conv = np.convolve(spike.astype(int), np.ones(n_consec, int), mode="valid") >= n_consec
    idx = np.where(conv)[0]
    idx = idx[idx > start]
    if idx.size:
        return int(idx[0])

    return None

def deform_qform_table(
    df: pd.DataFrame,
    y_col: str = "y_strong",
    n_early: int = 15,
    n_late: int = 10,
    split_frac: float = 0.20,
    enforce_monotonic: bool = True,
    clip_nonnegative: bool = True,
    tail_drop_frac: float = 0.05,     # "near-zero" = 5% of peak
    tail_vol_window: int = 15,        # window for volatility
    tail_vol_factor: float = 6.0,     # how many times noisier than baseline counts as artifact
    tail_consec: int = 8,             # require condition for 8 consecutive points
) -> pd.DataFrame:
    """
    Build a <=25-point solver-friendly flow curve table for DEFORM/QForm.

    New: trims end-of-test artifacts BEFORE sampling points, so export is stable across metals.
    """

    if "x" not in df.columns or y_col not in df.columns:
        raise ValueError(f"Need columns: 'x' and '{y_col}'")

    n_early = int(n_early)
    n_late = int(n_late)
    if n_early < 2 or n_late < 2:
        raise ValueError("n_early and n_late must be >= 2")
    if n_early + n_late > 25:
        raise ValueError("Total points must be <= 25")

    x = np.asarray(df["x"].values, dtype=float)
    y = np.asarray(df[y_col].values, dtype=float)

    # sort + unique x
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    keep = np.ones_like(x, dtype=bool)
    keep[1:] = x[1:] > x[:-1]
    x = x[keep]
    y = y[keep]

    if x.size < 10:
        raise ValueError("Not enough points to export.")

    # -------------------- NEW: tail trimming --------------------
    peak_idx = int(np.argmax(y))
    peak_y = float(y[peak_idx])
    if not np.isfinite(peak_y) or peak_y <= 0:
        raise ValueError("Invalid peak stress for export.")

    start = min(x.size - 1, peak_idx + 20)  # don't cut immediately after peak

    # A) sustained near-zero stress (collapse / lost contact / machine tail)
    thr = float(tail_drop_frac) * peak_y
    if start < y.size - tail_consec:
        below = (y < thr).astype(int)
        consec = np.convolve(below, np.ones(int(tail_consec), dtype=int), mode="valid") >= int(tail_consec)
        cand = np.where(consec)[0]
        cand = cand[cand > start]
        if cand.size:
            cut = int(cand[0])
            x = x[:cut]
            y = y[:cut]

    if x.size < 10:
        raise ValueError("Curve becomes too short after tail trim (near-zero).")

    # B) sustained volatility/spikes (artifact zone)
    dy = np.diff(y)
    if dy.size >= 30:
        pre = dy[:max(10, peak_idx - 5)]
        if pre.size < 10:
            pre = dy[:max(10, int(0.2 * dy.size))]

        baseline_mad = np.median(np.abs(pre - np.median(pre))) + 1e-12

        w = max(5, int(tail_vol_window))
        spike = np.zeros_like(dy, dtype=bool)
        for i in range(w, len(dy)):
            seg = dy[i - w:i]
            mad = np.median(np.abs(seg - np.median(seg))) + 1e-12
            spike[i] = mad > float(tail_vol_factor) * baseline_mad

        if spike.size > tail_consec:
            consec = np.convolve(spike.astype(int), np.ones(int(tail_consec), dtype=int), mode="valid") >= int(tail_consec)
            cand = np.where(consec)[0]
            cand = cand[cand > start]
            if cand.size:
                cut = int(cand[0])
                x = x[:cut]
                y = y[:cut]

    if x.size < 10:
        raise ValueError("Curve becomes too short after tail trim (volatility).")

    # -------------------- sampling (as before) --------------------
    x_max = float(np.max(x))
    if not np.isfinite(x_max) or x_max <= 0:
        raise ValueError("Invalid strain range (max x <= 0).")

    split_frac = float(split_frac)
    if not (0.01 <= split_frac <= 0.99):
        raise ValueError("split_frac must be in [0.01, 0.99].")
    x_split = split_frac * x_max

    x_early = np.linspace(0.0, x_split, n_early, endpoint=True)
    x_late = np.linspace(x_split, x_max, n_late, endpoint=True)
    x_target = np.unique(np.concatenate([x_early, x_late]))
    if x_target.size > 25:
        x_target = x_target[:25]

    y_target = np.interp(x_target, x, y)

    out = pd.DataFrame({"strain": x_target, "flow_stress": y_target})

    # Temperature (optional) — interpolate onto the SAME x_target
    if "temperature" in df.columns:
        tx = np.asarray(
            df["temperature_x"].values if "temperature_x" in df.columns else df["x"].values,
            dtype=float,
        )
        t = np.asarray(df["temperature"].values, dtype=float)

        torder = np.argsort(tx)
        tx = tx[torder]
        t = t[torder]
        tkeep = np.ones_like(tx, dtype=bool)
        tkeep[1:] = tx[1:] > tx[:-1]
        tx = tx[tkeep]
        t = t[tkeep]

        out["temperature"] = np.interp(x_target, tx, t)

    if enforce_monotonic:
        ys = out["flow_stress"].to_numpy(dtype=float)
        out["flow_stress"] = np.maximum.accumulate(ys)

    if clip_nonnegative:
        out = out[out["strain"] >= 0.0].copy()
        out["strain"] = out["strain"].clip(lower=0.0)
        out["flow_stress"] = out["flow_stress"].clip(lower=0.0)
        if "temperature" in out.columns:
            out["temperature"] = out["temperature"].clip(lower=0.0)

    if out.empty:
        raise ValueError("DEFORM/QForm table is empty after applying constraints.")

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

