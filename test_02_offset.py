import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from preprocess import preprocess_data

# ==========================================
# 1. PARAMETERS
# ==========================================
FILE_PATH = r"C:\Users\anish\Desktop\IFU\Flow Curve\From Toni\FlowCurveOpt\Rohdaten_ENAW1050\001-RT-SR1-30MnV6.csv"  # Test on your raw Al, Zn, or Steel CSV
DIAMETER = 7.8
HEIGHT = 8.3

# Yield Detection Parameters
STRAIN_OFFSET = 0.002        # 0.2% standard offset
IGNORE_START_PTS = 5         # Skip initial machine settling noise
STRAIN_SEARCH_LIMIT = 0.02   # Only look for Young's Modulus in the first 2% strain
# ==========================================

def calculate_robust_E(x, y):
    """
    Finds Young's Modulus (E) and intercept by searching for the maximum 
    stable slope, ignoring startup noise and end-of-test artifacts.
    """
    # Smooth data slightly to calculate a stable derivative
    if len(y) > 15:
        y_smooth = savgol_filter(y, window_length=15, polyorder=2)
    else:
        y_smooth = y
        
    # Calculate slope (dy/dx)
    dx = np.gradient(x)
    dx[dx == 0] = 1e-12  # Prevent divide-by-zero
    dy_dx = np.gradient(y_smooth) / dx
    
    # Restrict search to physically realistic elastic region
    search_indices = np.where((x < STRAIN_SEARCH_LIMIT) & (np.arange(len(x)) >= IGNORE_START_PTS))[0]
    
    # Fallback if the test is extremely short
    if len(search_indices) == 0:
        search_indices = np.arange(IGNORE_START_PTS, min(len(x), IGNORE_START_PTS + 10))
         
    # Find the steepest slope in the valid region
    early_slopes = dy_dx[search_indices]
    max_slope_idx = search_indices[np.argmax(early_slopes)]
    
    # Refine E by fitting a line around the steepest point
    fit_window = 3
    start_fit = max(0, max_slope_idx - fit_window)
    end_fit = min(len(x), max_slope_idx + fit_window + 1)
    
    E, intercept = np.polyfit(x[start_fit:end_fit], y_smooth[start_fit:end_fit], 1)
    return E, intercept, max_slope_idx


def find_exact_intersection(strain, stress, E, intercept, offset=0.002):
    """
    Finds the precise fractional intersection point using linear interpolation.
    """
    # The offset line must account for the intercept of the real curve
    stress_offset = E * (strain - offset) + intercept
    diff = stress - stress_offset
    
    # Find where the actual curve drops below the offset line
    sign_change = np.where(diff[:-1] * diff[1:] < 0)[0]
    
    if len(sign_change) == 0:
        if np.all(diff[strain >= offset] >= 0):
            i = np.argmax(strain >= offset)
            return strain[i], stress[i]
        return np.nan, np.nan
    
    # We want the FIRST time it crosses the line
    i = sign_change[0]
    
    x1, x2 = strain[i], strain[i+1]
    y1c, y2c = stress[i], stress[i+1]
    y1o, y2o = stress_offset[i], stress_offset[i+1]
    
    # Linear interpolation for sub-grid precision
    t = (y1o - y1c) / ((y2c - y1c) - (y2o - y1o) + 1e-15)
    t = np.clip(t, 0, 1)
    
    eps_y = x1 + t * (x2 - x1)
    sigma_y = y1c + t * (y2c - y1c)
    
    return eps_y, sigma_y


def main():
    try:
        raw_df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find {FILE_PATH}.")
        return

    # 1. Preprocess without any yield offset so we have the raw True Stress/Strain
    df_proc = preprocess_data(raw_df, diameter=DIAMETER, height=HEIGHT, apply_offset=False)
    
    x = df_proc["x"].values.astype(float)
    y = df_proc["y_none"].values.astype(float)

    # 2. Find robust Young's Modulus (E)
    E, intercept, max_slope_idx = calculate_robust_E(x, y)

    # 3. Find exact intersection
    eps_y, Rp02 = find_exact_intersection(x, y, E, intercept, offset=STRAIN_OFFSET)

    # Print Text Summary
    print("\n" + "="*60)
    print(" 0.2% OFFSET YIELD DETECTION RESULTS ")
    print("="*60)
    print(f"File:           {FILE_PATH}")
    print(f"E (Modulus):    {E / 1000:.2f} GPa")
    if np.isnan(Rp02):
        print("Rp0.2:          FAILED to find intersection.")
    else:
        print(f"Rp0.2 (Stress): {Rp02:.2f} MPa")
        print(f"Strain @ Yield: {eps_y:.6f}")
    print("="*60)

    # 4. SIDE-BY-SIDE PLOTTING (Full View & Zoomed View)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"0.2% Offset Yield Detection: {FILE_PATH}\nE = {E/1000:.2f} GPa → Rp0.2 = {Rp02:.1f} MPa", 
                 fontsize=14, fontweight='bold')

    y_elastic = E * x + intercept
    
    mask = x >= STRAIN_OFFSET
    x_offset = x[mask]
    y_offset = E * (x_offset - STRAIN_OFFSET) + intercept

    # --- Plot 1: Full Curve ---
    ax1.plot(x, y, label="Raw Flow Curve", color="tab:blue", linewidth=2)
    ax1.plot(x, y_elastic, label="Elastic Line", color="green", linestyle=":")
    ax1.plot(x_offset, y_offset, label="0.2% Offset Line", color="red", linestyle="--")
    
    # Highlight the region used to calculate E
    ax1.axvspan(x[max(0, max_slope_idx-3)], x[min(len(x)-1, max_slope_idx+3)], 
                color='green', alpha=0.2, label="Region used for E")

    if not np.isnan(Rp02):
        ax1.plot(eps_y, Rp02, 'ko', markersize=8, label="Detected Yield Point")
        
    ax1.set_xlim(-0.01, max(x) * 1.05)
    ax1.set_ylim(0, max(y) * 1.1)
    ax1.set_xlabel("True Strain [-]", fontsize=12)
    ax1.set_ylabel("True Stress [MPa]", fontsize=12)
    ax1.set_title("Full Flow Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # --- Plot 2: Zoomed-in View ---
    ax2.plot(x, y, label="Raw Flow Curve", color="tab:blue", linewidth=2.5)
    ax2.plot(x, y_elastic, label="Elastic Line", color="green", linestyle=":")
    ax2.plot(x_offset, y_offset, label="0.2% Offset Line", color="red", linestyle="--")
    
    if not np.isnan(Rp02):
        ax2.plot(eps_y, Rp02, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
        
        # Zoom dynamically around the yield point
        margin_x = 0.01
        margin_y = Rp02 * 0.2
        ax2.set_xlim(max(0, eps_y - margin_x), eps_y + margin_x * 2)
        ax2.set_ylim(max(0, Rp02 - margin_y), Rp02 + margin_y)
    
    ax2.set_xlabel("True Strain [-]", fontsize=12)
    ax2.set_title("Zoomed View at Yield Point")
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()