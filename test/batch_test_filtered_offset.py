import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from preprocess import preprocess_data

# ==========================================
# BATCH TEST SETTINGS (Tune these as needed)
# ==========================================
FOLDER_PATH = r"C:\Users\anish\Desktop\IFU\Flow Curve\From Toni\FlowCurveOpt\Rohdaten_ENAW1050"  
# Specimen Defaults (Used if not explicitly handling varying sizes)
DIAMETER = 7.8
HEIGHT = 8.3

# Yield Detection Parameters
WINDOW = 15                # Rolling variance window
SLOPE_JUMP_FACTOR = 2.5      # Multiplier for the baseline
IGNORE_START_PTS = 10      # How many noisy points at the very start to ignore

# Filter Parameters (Applied ONLY for yield detection)
SG_WINDOW = 21             # Savitzky-Golay window size (must be ODD)
SG_POLY = 2                # Polynomial order
# ==========================================

def run_batch_test():
    csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in: {FOLDER_PATH}")
        return

    print(f"Found {len(csv_files)} files. Starting filtered batch test...\n")
    print(f"Settings -> Window: {WINDOW} | Jump Factor: {SLOPE_JUMP_FACTOR} | Ignored Pts: {IGNORE_START_PTS}")
    print("-" * 75)
    print(f"{'Filename':<30} | {'Status':<10} | {'Yield Index':<12} | {'Strain':<10}")
    print("-" * 75)

    successes = []
    failures = []
    errors = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        try:
            # 1. Load data and preprocess WITHOUT offset
            raw_df = pd.read_csv(file_path)
            df_proc = preprocess_data(
                raw_df, 
                diameter=DIAMETER, 
                height=HEIGHT, 
                apply_offset=False
            )
            
            x = df_proc["x"].values.astype(float)
            y_raw = df_proc["y_none"].values.astype(float)

            # 2. THE NEW FILTERED OFFSET LOGIC
            # Temporary smoothing filter
            if len(y_raw) > SG_WINDOW:
                y_smooth = savgol_filter(y_raw, window_length=SG_WINDOW, polyorder=SG_POLY)
            else:
                y_smooth = y_raw

            # Gradient on the SMOOTHED data
            dy = np.gradient(y_smooth, x)

            # Rolling variance
            slope_var = np.zeros_like(dy)
            for i in range(len(dy)):
                start = max(0, i - WINDOW)
                slope_var[i] = np.var(dy[start:i + 1])

            # Calculate baseline (skipping the initial startup noise)
            if len(slope_var) > IGNORE_START_PTS + 20:
                baseline = np.median(slope_var[IGNORE_START_PTS:IGNORE_START_PTS + 20])
            else:
                baseline = np.median(slope_var)
                
            baseline = max(baseline, 1e-12) # Prevent divide-by-zero

            # Threshold and Trigger
            threshold = SLOPE_JUMP_FACTOR * baseline
            plastic_pts = np.where(slope_var > threshold)[0]
            
            # Only allow triggers AFTER the ignored startup region
            valid_pts = [p for p in plastic_pts if p >= IGNORE_START_PTS]
            idx_yield = valid_pts[0] if valid_pts else 0

            # 3. Log the result
            if idx_yield == 0:
                print(f"{filename:<30} | FAILED     | {'0':<12} | -")
                failures.append(filename)
            else:
                strain_val = f"{x[idx_yield]:.4f}"
                print(f"{filename:<30} | SUCCESS    | {idx_yield:<12} | {strain_val:<10}")
                successes.append(filename)

        except Exception as e:
            print(f"{filename:<30} | ERROR      | {str(e)[:25]}...")
            errors.append(filename)

    # Print Final Summary
    print("-" * 75)
    print("BATCH TEST COMPLETE")
    print(f"Total Files Tested: {len(csv_files)}")
    print(f"Successes: {len(successes)} (Found a valid yield point)")
    print(f"Failures:  {len(failures)} (Defaulted to index 0)")
    print(f"Errors:    {len(errors)} (Failed to load or preprocess)")

if __name__ == "__main__":
    run_batch_test()