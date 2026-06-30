import os
import glob
import pandas as pd
import numpy as np
from preprocess import preprocess_data

# ==========================================
# TEST SETTINGS (Update these as needed)
# ==========================================
FOLDER_PATH = r"C:\Users\anish\Desktop\IFU\Flow Curve\From Toni\FlowCurveOpt\Rohdaten_ENAW1050"  # Folder containing your CSVs
WINDOW = 10                                  # Rolling variance window
SLOPE_JUMP_FACTOR = 2.5                       # Your new lower jump factor
DIAMETER = 8.3                              # Specimen diameter (mm)
HEIGHT = 7.8                                # Specimen height (mm)
# ==========================================

def batch_test():
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {FOLDER_PATH}")
        return

    print(f"Found {len(csv_files)} files. Starting batch test...\n")
    print("-" * 60)
    print(f"{'Filename':<30} | {'Status':<10} | {'Yield Index':<12}")
    print("-" * 60)

    successes = []
    failures = []
    errors = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        try:
            # 1. Load raw data
            raw_df = pd.read_csv(file_path)
            
            # 2. Get x and y data WITHOUT applying the offset (so we can test it manually)
            df_proc = preprocess_data(
                raw_df, 
                diameter=DIAMETER, 
                height=HEIGHT, 
                apply_offset=False
            )
            
            x = df_proc["x"].values.astype(float)
            y = df_proc["y_none"].values.astype(float)

            # 3. Apply the offset math
            dy = np.gradient(y, x)
            slope_var = np.zeros_like(dy)
            for i in range(len(dy)):
                start = max(0, i - WINDOW)
                slope_var[i] = np.var(dy[start:i + 1])

            baseline = np.median(slope_var[:20])
            threshold = SLOPE_JUMP_FACTOR * baseline

            plastic_pts = np.where(slope_var > threshold)[0]
            idx_yield = plastic_pts[0] if plastic_pts.size else 0

            # 4. Log the result
            if idx_yield == 0:
                print(f"{filename:<30} | FAILED     | 0")
                failures.append(filename)
            else:
                print(f"{filename:<30} | SUCCESS    | {idx_yield} (Strain: {x[idx_yield]:.4f})")
                successes.append(filename)

        except Exception as e:
            print(f"{filename:<30} | ERROR      | {str(e)[:15]}...")
            errors.append(filename)

    # Print Final Summary
    print("-" * 60)
    print("BATCH TEST COMPLETE")
    print(f"Total Files: {len(csv_files)}")
    print(f"Successes:   {len(successes)} (Found a valid yield point)")
    print(f"Failures:    {len(failures)} (Defaulted to index 0)")
    print(f"Errors:      {len(errors)} (Failed to load or preprocess)")

if __name__ == "__main__":
    batch_test()