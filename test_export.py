import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_data

# ==========================================
# TEST PARAMETERS
# ==========================================
FILE_PATH = r"C:\Users\anish\Desktop\IFU\Flow Curve\From Toni\Test Files\001-RT-SR1-30MnV6.csv"  # Test on your Al1050, 30MnV6, or Zn20
DIAMETER = 7.8
HEIGHT = 8.3
EXPORT_POINTS = 25
CLUSTER_FACTOR = 50  # Higher = more points packed at the yield point
# ==========================================

def smart_export_points(x, y, n_points=25, cluster_factor=50):
    """
    Downsamples a flow curve into exactly `n_points` using geometric spacing.
    Automatically adapts to the total strain length of the dataset.
    """
    x_min = np.min(x)
    x_max = np.max(x)
    
    # 1. Generate geometrically spaced target X values (normalized 0 to 1)
    norm_space = (np.geomspace(1, cluster_factor, n_points) - 1) / (cluster_factor - 1)
    
    # 2. Scale the normalized points to fit exactly between our actual x_min and x_max
    x_target = x_min + (x_max - x_min) * norm_space
    
    # 3. Interpolate the actual Y values at these exact target X locations
    y_target = np.interp(x_target, x, y)
    
    return x_target, y_target

def main():
    print(f"Loading {FILE_PATH}...")
    try:
        raw_df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find {FILE_PATH}.")
        return

    # 1. Preprocess the real data
    # We set apply_offset=True so it uses your newly integrated 0.2% yield offset!
    # This means the x-axis (Plastic Strain) will perfectly start at 0 for DEFORM.
    df_proc = preprocess_data(raw_df, diameter=DIAMETER, height=HEIGHT, apply_offset=True)
    
    x_raw = df_proc["x"].values.astype(float)
    
    # You can change "y_none" to "y_mild" or "y_strong" if you want to export filtered data
    y_raw = df_proc["y_none"].values.astype(float) 

    # 2. Run the smart export logic on YOUR data
    x_export, y_export = smart_export_points(x_raw, y_raw, 
                                             n_points=EXPORT_POINTS, 
                                             cluster_factor=CLUSTER_FACTOR)
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"DEFORM Export Fit: {FILE_PATH}", fontsize=14, fontweight='bold')
    
    # Plot 1: The Visual Overlay (Comparison)
    ax1.plot(x_raw, y_raw, color='tab:blue', alpha=0.6, linewidth=3, 
             label="Actual Flow Curve (Preprocessed)")
    ax1.plot(x_export, y_export, color='red', marker='o', markersize=8, linestyle='-', 
             linewidth=1.5, markeredgecolor='black', label=f"Exported DEFORM Table ({EXPORT_POINTS} pts)")
    
    ax1.set_xlabel("True Plastic Strain [-]", fontsize=12)
    ax1.set_ylabel("True Stress [MPa]", fontsize=12)
    ax1.set_title("Flow Curve vs. Exported Points")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Point Density Visualization
    point_distances = np.diff(x_export)
    ax2.bar(range(1, len(point_distances) + 1), point_distances, color='tab:orange', edgecolor='black')
    
    ax2.set_xlabel("Export Point Segment", fontsize=12)
    ax2.set_ylabel("Strain Gap to Next Point (\u0394x)", fontsize=12)
    ax2.set_title("Distance Between Target Points")
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Print the DEFORM table output format to terminal
    print("\n" + "="*35)
    print(f" DEFORM EXPORT TABLE ({EXPORT_POINTS} Pts) ")
    print("="*35)
    print(f"{'Plastic Strain':<15} | {'Flow Stress (MPa)':<15}")
    print("-" * 35)
    for sx, sy in zip(x_export, y_export):
        print(f"{sx:<15.5f} | {sy:<15.2f}")
    print("="*35)

if __name__ == "__main__":
    main()