import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def debug_end_index(x, y):
    """Verbose version of detect_end_index that prints exactly what it is thinking."""
    print("\n--- [STEP 3] RUNNING END DETECTION ---")
    if len(y) < 20:
        print("   -> Array too small (<20 points), skipping cut.")
        return len(y) - 1
        
    peak_idx = int(np.argmax(y))
    peak_stress = y[peak_idx]
    print(f"   -> Peak Stress found at index {peak_idx}: {peak_stress:.2f} MPa")
    
    for i in range(peak_idx + 5, len(y)):
        # RULE 1: The 40% Drop Cliff
        if y[i] < (peak_stress * 0.60):
            print(f"   -> FATAL CUT TRIGGERED: 40% stress drop hit at index {i}!")
            print(f"      Stress dropped to {y[i]:.2f} MPa (Limit was {peak_stress * 0.60:.2f} MPa)")
            return i
            
        # RULE 2: Sustained Jaw Reversal
        if i >= 3:
            if (x[i] < x[i-1]) and (x[i-1] < x[i-2]) and (x[i-2] < x[i-3]):
                print(f"   -> FATAL CUT TRIGGERED: Sustained jaw reversal at index {i}!")
                print(f"      Strains: {x[i-3]:.5f} -> {x[i-2]:.5f} -> {x[i-1]:.5f} -> {x[i]:.5f}")
                return i
                
    print("   -> No fatal triggers hit. Test ran to completion.")
    return len(y) - 1

def run_diagnostics(file_path, diameter=10.0, height=15.0):
    print(f"\n========== FORENSIC DIAGNOSTIC REPORT ==========")
    print(f"File: {file_path}")
    
    # 1. Load Data
    try:
        df = pd.read_csv(file_path).apply(pd.to_numeric, errors="coerce").dropna()
        print(f"\n[STEP 1] Raw data loaded. Total points: {len(df)}")
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # 2. Base Math
    df["Jaw"] = -df["Jaw"]
    df["Force"] = -df["Force"]
    df["Jaw_corr"] = df["Jaw"] - df["Force"] * 0.0029
    
    # Startup Trim (skip to where Jaw > 0)
    jaw = df["Jaw_corr"].values
    valid = np.isfinite(jaw) & (jaw < 0.98 * height)
    if not np.any(valid):
        print("ERROR: No valid Jaw_corr points.")
        return
        
    start_idx = int(np.where(valid)[0][0])
    df = df.iloc[start_idx:].reset_index(drop=True)
    print(f"[STEP 2] Startup slack trimmed. Points remaining: {len(df)}")
    
    # True Strain & Stress
    denom = height - df["Jaw_corr"]
    denom = np.clip(denom, 0.001, None)
    
    x_raw = -np.log(denom / height).values
    y_raw = ((df["Force"] * 1000.0) / ((math.pi / 4.0) * diameter**2 * (height / denom))).values
    
    # 3. RUN THE CUT LOGIC
    cut_idx = debug_end_index(x_raw, y_raw)
    
    x_cut = x_raw[:cut_idx]
    y_cut = y_raw[:cut_idx]
    print(f"\n[STEP 4] Array sliced at index {cut_idx}. Points remaining: {len(x_cut)}")
    
    # 4. MONOTONIC FILTER
    print("\n--- [STEP 5] MONOTONIC STRAIN FILTER ---")
    running_max = np.maximum.accumulate(x_cut)
    keep = x_cut >= running_max
    
    points_dropped = sum(~keep)
    x_final = x_cut[keep]
    y_final = y_cut[keep]
    
    print(f"   -> Points dropped due to backwards strain noise: {points_dropped}")
    print(f"   -> Final points ready for offset/interpolation: {len(x_final)}")
    
    print("================================================\n")
    
    # 5. VISUAL DIAGNOSTIC PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(x_raw, y_raw, color='lightgray', label='Raw Scrambled Data (Ignored)', linestyle='--')
    plt.plot(x_final, y_final, color='tab:orange', linewidth=2, label='Surviving Processed Data')
    
    # Mark the exact spot the algorithm cut the data
    if cut_idx < len(x_raw) - 1:
        plt.scatter(x_raw[cut_idx], y_raw[cut_idx], color='red', s=100, zorder=5, label=f'Cut Triggered Here (Idx: {cut_idx})')
        
    plt.title(f"Diagnostic View: {file_path}")
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [MPa]")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # ---> CHANGE THIS TO YOUR ACTUAL CSV FILE PATH <---
    FILE_PATH = r"C:\Users\anish\Desktop\IFU\Flow Curve\From Toni\Test Files\012-T400-SR0.01.csv" 
    
    # Put your actual specimen diameter and height here
    run_diagnostics(FILE_PATH, diameter=10.0, height=15.0)