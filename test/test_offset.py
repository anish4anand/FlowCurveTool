import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from preprocess import preprocess_data

# ==========================================
# 1. TUNABLE PARAMETERS (Change these to find the common logic)
# ==========================================
FILE_PATH = r"C:\Users\anish\Desktop\IFU\Flow Curve\Anish Working\FlowCurveTool\001-T200-SR1-1050.csv"  # Change to your steel or zinc file to test
DIAMETER = 7.8
HEIGHT = 8.3

# Yield Detection Parameters
WINDOW = 15                # Rolling variance window (higher = smoother variance)
SLOPE_JUMP_FACTOR = 2      # Multiplier for the baseline (lower = more sensitive)
IGNORE_START_PTS = 10      # How many noisy points at the very start to completely ignore

# Filter Parameters (Applied ONLY for yield detection, does not alter final exported data)
SG_WINDOW = 21             # Savitzky-Golay window size (must be an ODD number)
SG_POLY = 2                # Polynomial order for the filter
# ==========================================

def test_offset_logic(x, y):
    """
    The proposed new offset logic, isolated for testing.
    Returns the detected index, plus all internal variables for plotting.
    """
    # 1. Temporary smoothing filter
    if len(y) > SG_WINDOW:
        y_smooth = savgol_filter(y, window_length=SG_WINDOW, polyorder=SG_POLY)
    else:
        y_smooth = y

    # 2. Gradient on the SMOOTHED data
    dy = np.gradient(y_smooth, x)

    # 3. Rolling variance
    slope_var = np.zeros_like(dy)
    for i in range(len(dy)):
        start = max(0, i - WINDOW)
        slope_var[i] = np.var(dy[start:i + 1])

    # 4. Calculate baseline (skipping the initial startup noise)
    if len(slope_var) > IGNORE_START_PTS + 20:
        baseline = np.median(slope_var[IGNORE_START_PTS:IGNORE_START_PTS + 20])
    else:
        baseline = np.median(slope_var)
        
    baseline = max(baseline, 1e-12) # Prevent divide-by-zero

    # 5. Threshold and Trigger
    threshold = SLOPE_JUMP_FACTOR * baseline
    
    # Only allow the algorithm to trigger AFTER the ignored startup region
    plastic_pts = np.where(slope_var > threshold)[0]
    valid_pts = [p for p in plastic_pts if p >= IGNORE_START_PTS]

    idx_yield = valid_pts[0] if valid_pts else 0

    return idx_yield, slope_var, baseline, threshold, y_smooth


def main():
    # Load and preprocess (without offset)
    try:
        raw_df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find {FILE_PATH}. Check the filename!")
        return

    df_proc = preprocess_data(raw_df, diameter=DIAMETER, height=HEIGHT, apply_offset=False)
    
    x = df_proc["x"].values.astype(float)
    y_raw = df_proc["y_none"].values.astype(float)

    # Run the test logic
    idx_yield, slope_var, baseline, threshold, y_smooth = test_offset_logic(x, y_raw)

    # Print results
    print("\n--- TUNING RESULTS ---")
    if idx_yield == 0:
        print("RESULT: FAILED. The variance never crossed the threshold.")
    else:
        print(f"RESULT: SUCCESS. Yield detected at index {idx_yield} (Strain: {x[idx_yield]:.4f})")
    print(f"Baseline:  {baseline:.4f}")
    print(f"Threshold: {threshold:.4f}")

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Plot: Stress-Strain
    ax1.plot(x, y_raw, label="Raw Curve", color="tab:blue", alpha=0.5)
    ax1.plot(x, y_smooth, label="Smoothed Curve (Algorithm's view)", color="black", linestyle="--")
    if idx_yield > 0:
        ax1.axvline(x[idx_yield], color="red", linestyle="-", linewidth=2, label=f"Detected Yield (idx {idx_yield})")
    ax1.axvspan(0, x[IGNORE_START_PTS], color='gray', alpha=0.3, label="Ignored Startup Region")
    
    ax1.set_ylabel("True Stress [MPa]")
    ax1.set_title(f"Yield Detection Tuning: {FILE_PATH}")
    ax1.legend()
    ax1.grid(True)

    # Bottom Plot: Variance Math
    ax2.plot(x, slope_var, label="Rolling Variance", color="tab:orange")
    ax2.axhline(threshold, color="red", linestyle="-", linewidth=2, label="Trigger Threshold")
    ax2.axhline(baseline, color="green", linestyle=":", linewidth=2, label="Baseline Noise")
    if idx_yield > 0:
        ax2.axvline(x[idx_yield], color="red", linestyle="-", linewidth=2)
    ax2.axvspan(0, x[IGNORE_START_PTS], color='gray', alpha=0.3)

    ax2.set_xlabel("True Strain [-]")
    ax2.set_ylabel("Variance")
    
    # Zoom Y axis intelligently so we can see the threshold crossing
    ymax = max(threshold * 1.5, np.median(slope_var[IGNORE_START_PTS:]) * 5)
    ax2.set_ylim(0, ymax)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()