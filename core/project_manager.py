import numpy as np
import pandas as pd

class ProjectManager:
    """Centralized data state and dataset manipulation."""
    def __init__(self):
        self.datasets = []

    def get_processed_datasets(self):
        return [d for d in self.datasets if d.get("preprocessed", False) and d.get("proc_data") is not None]

    def run_averaging(self) -> int:
        processed = self.get_processed_datasets()
        if not processed: return 0

        groups = {}
        for ds in processed:
            if str(ds["filename"]).startswith("AVG_"):
                continue
            
            raw_name = ds["filename"].replace(".csv", "")
            parts = raw_name.split("-")
            condition = "-".join(parts[1:]) if len(parts) > 1 else raw_name 
                
            if condition not in groups:
                groups[condition] = []
            groups[condition].append(ds)

        new_datasets = []
        averaged_count = 0

        for condition, group in groups.items():
            if len(group) < 2: continue
            min_len = min(len(ds["proc_data"]["x"]) for ds in group)

            avg_df = pd.DataFrame()
            avg_df["x"] = group[0]["proc_data"]["x"].values[:min_len]

            cols_to_avg = ["y_none", "y_mild", "y_strong", "temperature"]
            for col in cols_to_avg:
                if col in group[0]["proc_data"].columns:
                    stacked = np.vstack([ds["proc_data"][col].values[:min_len] for ds in group])
                    avg_df[col] = np.mean(stacked, axis=0)

            avg_ds = {
                "filename": f"AVG_{condition}.csv",
                "raw_data": avg_df, 
                "proc_data": avg_df,
                "preprocessed": True,
                "diameter": group[0]["diameter"],
                "height": group[0]["height"],
                "strain_increment": group[0].get("strain_increment", 0.005),
            }
            new_datasets.append(avg_ds)
            averaged_count += 1

        for nds in new_datasets:
            existing_names = [d["filename"] for d in self.datasets]
            if nds["filename"] in existing_names:
                self.datasets[existing_names.index(nds["filename"])] = nds
            else:
                self.datasets.append(nds)
                
        return averaged_count

    def apply_manual_cut_all(self, target_x: float):
        for ds in self.datasets:
            df = ds.get("proc_data")
            if df is None or "x" not in df.columns: continue
            
            if "proc_data_orig" not in ds:
                ds["proc_data_orig"] = df.copy()
            
            x_vals = df["x"].values
            closest_idx = (np.abs(x_vals - target_x)).argmin()
            df_cut = df.iloc[closest_idx:].copy()
            
            if df_cut.empty: continue
            
            x0 = df_cut["x"].iloc[0]
            df_cut["x"] = df_cut["x"] - x0
            
            if "temperature_x" in df_cut.columns:
                df_cut["temperature_x"] = df_cut["temperature_x"] - x0
                
            ds["proc_data"] = df_cut.reset_index(drop=True)

    def reset_manual_cut_all(self):
        """Restores the data to its original pre-cut state."""
        restored = False
        for ds in self.datasets:
            if "proc_data_orig" in ds:
                ds["proc_data"] = ds["proc_data_orig"].copy()
                del ds["proc_data_orig"]  # Clear backup so it can be cleanly created again later
                restored = True
        return restored        