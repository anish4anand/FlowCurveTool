import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openpyxl.chart import ScatterChart, Reference, Series

from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QFileDialog, QMessageBox, QLabel, QListWidget, QTableWidget,
    QTableWidgetItem, QCheckBox, QComboBox, QDialog
)

from dialogs import GeometryDialog, PreprocessOptionsDialog
from preprocess import preprocess_data
from utils import deform_qform_table, force_displacement_curve_from_raw

class DataOptimizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flow Curve Tool")
        self.setGeometry(200, 200, 1300, 650)

        self.datasets = []
        self.current_index = None

        # ---------- Layout ----------
        main_layout = QHBoxLayout()

        # Preview table (left)
        self.preview_table = QTableWidget()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        main_layout.addWidget(self.preview_table, 3)

        # Sidebar (right)
        sidebar = QVBoxLayout()

        self.dataset_list = QListWidget()
        self.dataset_list.currentRowChanged.connect(self.change_dataset)
        self.dataset_list.currentRowChanged.connect(self.update_buttons)
        sidebar.addWidget(QLabel("Loaded Datasets:"))
        sidebar.addWidget(self.dataset_list)

        self.load_btn = QPushButton("Add CSV + Parameters")
        self.load_btn.clicked.connect(self.load_dataset)
        sidebar.addWidget(self.load_btn)

        self.edit_params_button = QPushButton("Edit Specimen Parameters")
        self.edit_params_button.clicked.connect(self.edit_parameters)
        self.edit_params_button.setEnabled(False)
        sidebar.addWidget(self.edit_params_button)

        self.remove_button = QPushButton("Remove Dataset from List")
        self.remove_button.clicked.connect(self.remove_dataset)
        self.remove_button.setEnabled(False)
        sidebar.addWidget(self.remove_button)

        self.clear_all_btn = QPushButton("Clear All Datasets")
        self.clear_all_btn.clicked.connect(self.clear_all_datasets)
        self.clear_all_btn.setEnabled(False)
        sidebar.addWidget(self.clear_all_btn)

        self.preprocess_btn = QPushButton("Process Raw Data (Selected)")
        self.preprocess_btn.clicked.connect(self.run_preprocess)
        self.preprocess_btn.setEnabled(False)
        sidebar.addWidget(self.preprocess_btn)
        
        self.chk_process_all = QCheckBox("Process all loaded datasets")
        self.chk_process_all.setChecked(False)
        sidebar.addWidget(self.chk_process_all)

        # Plot filter checkboxes
        sidebar.addWidget(QLabel("Curves to show in plot:"))
        self.chk_none = QCheckBox("Unfiltered (reference)")
        self.chk_mild = QCheckBox("Mild filter (Savgol)")
        self.chk_strong = QCheckBox("Strong filter (Butterworth)")
        self.chk_none.setChecked(True)
        self.chk_mild.setChecked(True)
        self.chk_strong.setChecked(False)
        sidebar.addWidget(self.chk_none)
        sidebar.addWidget(self.chk_mild)
        sidebar.addWidget(self.chk_strong)

        self.plot_btn = QPushButton("Plots")
        self.plot_btn.clicked.connect(self.show_plots)
        self.plot_btn.setEnabled(False)
        sidebar.addWidget(self.plot_btn)

        # -------- Export --------
        sidebar.addWidget(QLabel("Export filter:"))
        self.export_filter_combo = QComboBox()
        self.export_filter_combo.addItems(["none", "mild", "strong"])
        sidebar.addWidget(self.export_filter_combo)

        self.export_btn = QPushButton("Export (Excel)")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        sidebar.addWidget(self.export_btn)
        
        self.deform_export_btn = QPushButton("DEFORM / QForm Export")
        self.deform_export_btn.clicked.connect(self.export_deform_qform)
        self.deform_export_btn.setEnabled(False)
        sidebar.addWidget(self.deform_export_btn)
        
        self.force_disp_export_btn = QPushButton("Export Force–Displacement (Selected)")
        self.force_disp_export_btn.clicked.connect(self.export_force_displacement_selected)
        self.force_disp_export_btn.setEnabled(False)
        sidebar.addWidget(self.force_disp_export_btn)

        sidebar.addStretch()
        main_layout.addLayout(sidebar, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # ---------- Helpers ----------

    def update_list_entry(self, idx: int):
        ds = self.datasets[idx]
        off = ds.get("apply_offset", None)
        off_txt = "Offset=?" if off is None else f"Offset={'Yes' if off else 'No'}"

        dx = ds.get("strain_increment", None)
        dx_txt = "Δε=?" if dx is None else f"Δε={dx:g}"

        entry = (
            f"{ds['filename']}  |  D={ds['diameter']} mm, H={ds['height']} mm  |  "
            f"{off_txt}  |  {dx_txt}"
        )

        if idx < self.dataset_list.count():
            self.dataset_list.item(idx).setText(entry)
        else:
            self.dataset_list.addItem(entry)

    def show_preview(self):
        if self.current_index is None or self.current_index < 0 or self.current_index >= len(self.datasets):
            self.preview_table.clear()
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return

        ds = self.datasets[self.current_index]

        if ds.get("preprocessed", False) and ds.get("proc_data") is not None:
            df = ds["proc_data"]
        else:
            df = ds["raw_data"]

        preview = df.head(20)

        self.preview_table.setRowCount(len(preview))
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setHorizontalHeaderLabels(preview.columns)

        for i in range(len(preview)):
            for j in range(len(preview.columns)):
                item = QTableWidgetItem(str(preview.iat[i, j]))
                self.preview_table.setItem(i, j, item)

    def refresh_ui_state(self):
        has_selection = self.current_index is not None and 0 <= self.current_index < len(self.datasets)
        any_processed = any(d.get("preprocessed", False) and d.get("proc_data") is not None for d in self.datasets)

        self.clear_all_btn.setEnabled(len(self.datasets) > 0)
        self.preprocess_btn.setEnabled(has_selection)
        
        self.plot_btn.setEnabled(any_processed)
        self.export_btn.setEnabled(any_processed)
        
        has_proc_sel = (
            has_selection   
            and self.datasets[self.current_index].get("preprocessed", False)
            and self.datasets[self.current_index].get("proc_data") is not None
            )
        self.deform_export_btn.setEnabled(has_proc_sel)
        self.force_disp_export_btn.setEnabled(has_proc_sel)

    # -------------------- Button Enable/Disable --------------------
    def update_buttons(self, index):
        has_selection = index >= 0 and index < len(self.datasets)
        self.edit_params_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)

    # -------------------- Load CSV --------------------
    def load_dataset(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open CSV files (multi-select)",
            "",
            "CSV Files (*.csv)"
            )
        if not paths:
            return

        dlg = GeometryDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        diameter, height = dlg.get_values()
        if diameter is None or height is None:
            QMessageBox.warning(self, "Error", "Please enter valid numbers!")
            return

        loaded = 0
        failed = []

        for path in paths:
            try:
                raw = pd.read_csv(path)
                if raw is None or raw.empty:
                    raise ValueError("CSV is empty.")

                ds = {
                    "filename": os.path.basename(path),
                    "raw_data": raw,
                    "diameter": float(diameter),
                    "height": float(height),
                    "proc_data": None,
                    "preprocessed": False,
                    "apply_offset": False,
                }

                self.datasets.append(ds)
                self.update_list_entry(len(self.datasets) - 1)
                loaded += 1

            except Exception as e:
                failed.append(f"{os.path.basename(path)}: {e}")

        if loaded > 0:
            self.refresh_ui_state()
            QMessageBox.information(self, "Success", f"Loaded {loaded} file(s).")

        if failed:
            QMessageBox.warning(
                self,
                "Some files failed",
                "The following files could not be loaded:\n\n" + "\n".join(failed)
            )

    # -------------------- Switch Dataset --------------------
    def change_dataset(self, idx: int):
        self.current_index = idx
        self.refresh_ui_state()
        self.show_preview()

    # -------------------- Edit / Remove --------------------
    def edit_parameters(self):
        if self.current_index is None:
            return

        # NEW: Check if there are multiple datasets and ask for Probendaten
        if len(self.datasets) > 1:
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle("Edit Specimen Parameters")
            msgBox.setText("You have multiple datasets loaded.\nHow would you like to edit the geometry?")
            
            btn_manual = msgBox.addButton("Edit Selected Manually", QMessageBox.ActionRole)
            btn_auto = msgBox.addButton("Auto-Match All (Probendaten Excel)", QMessageBox.ActionRole)
            btn_cancel = msgBox.addButton("Cancel", QMessageBox.RejectRole)
            
            msgBox.exec()
            
            if msgBox.clickedButton() == btn_cancel:
                return
            elif msgBox.clickedButton() == btn_auto:
                self._import_probendaten()
                return

        # Standard manual edit
        ds = self.datasets[self.current_index]
        dialog = GeometryDialog(self, ds["diameter"], ds["height"])
        if dialog.exec() == QDialog.Accepted:
            diameter, height = dialog.get_values()
            if diameter is None or height is None:
                QMessageBox.warning(self, "Error", "Please enter valid numbers!")
                return
            ds["diameter"] = diameter
            ds["height"] = height
            self.update_list_entry(self.current_index)
            QMessageBox.information(self, "Info", "Parameters updated successfully.")
            if ds["preprocessed"]:
                QMessageBox.warning(self, "Warning", "Raw data already processed. Reprocess if needed.")

    def _import_probendaten(self):
        """Internal worker function to load and match the Probendaten Excel file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Probendaten File", "", "Excel Files (*.xlsx *.xls);;CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            if file_path.lower().endswith('.csv'):
                df_probe = pd.read_csv(file_path)
            else:
                df_probe = pd.read_excel(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file:\n{str(e)}")
            return

        v_col = None
        d_col = None
        h_col = None

        for c in df_probe.columns:
            c_str = str(c).strip().lower()
            
            # Match "Versuch" or "Versuchsnummer"
            if 'versuch' in c_str:
                v_col = c
            
            # Match Diameter: contains 'ø', 'durchmesser', or 'd0'
            elif 'ø' in c_str or 'durchmesser' in c_str or 'd0' in c_str:
                d_col = c
                
            # Match Height: contains 'h0', 'höh', or 'hoeh'
            elif 'h0' in c_str or 'höh' in c_str or 'hoeh' in c_str:
                h_col = c

        if not v_col or not d_col or not h_col:
            QMessageBox.warning(
                self, "Error", 
                f"Could not find the required columns in the file.\n\n"
                f"Found columns: {', '.join(str(c) for c in df_probe.columns)}\n"
                f"Looking for: 'Versuch', 'Ø' / 'd0' (Diameter), and 'h0' (Height)"
            )
            return

        matched_count = 0
        already_processed = False

        for i, ds in enumerate(self.datasets):
            fname = ds["filename"]
            base_fname = fname.lower().replace('.csv', '')
            
            # Extract prefix number from filename e.g. "015-T100.csv" -> 15
            prefix_match = re.match(r'^0*(\d+)', fname)
            prefix_int = int(prefix_match.group(1)) if prefix_match else None

            for idx, row in df_probe.iterrows():
                v_val = row[v_col]
                if pd.isna(v_val): continue
                v_str = str(v_val).strip()

                is_match = False
                # Method 1: Strict Integer prefix match (Safest)
                if prefix_int is not None:
                    try:
                        if int(float(v_str)) == prefix_int:
                            is_match = True
                    except ValueError:
                        pass
                
                # Method 2: String prefix match (Fallback)
                if not is_match:
                    if base_fname == v_str.lower() or base_fname.startswith(v_str.lower() + "-") or base_fname.startswith(v_str.lower() + "_"):
                        is_match = True

                if is_match:
                    d_val = row[d_col]
                    h_val = row[h_col]
                    if pd.notna(d_val) and pd.notna(h_val):
                        ds["diameter"] = float(d_val)
                        ds["height"] = float(h_val)
                        self.update_list_entry(i)
                        matched_count += 1
                        if ds.get("preprocessed"):
                            already_processed = True
                    break

        msg = f"Successfully matched and updated {matched_count} out of {len(self.datasets)} datasets."
        if already_processed:
            msg += "\n\nNote: Some updated datasets were already processed. You will need to re-process them for the new geometry to take effect."
        
        QMessageBox.information(self, "Probendaten Import", msg)

    def remove_dataset(self):
        row = self.dataset_list.currentRow()
        if row < 0 or row >= self.dataset_list.count():
            return

        if row >= len(self.datasets):
            self.datasets = self.datasets[: self.dataset_list.count()]
            if row >= len(self.datasets):
                item = self.dataset_list.takeItem(row)
                del item
                self.current_index = None
                self.show_preview()
                self.refresh_ui_state()
                return

        reply = QMessageBox.question(
            self, "Remove Dataset",
            f"Really remove '{self.datasets[row]['filename']}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        del self.datasets[row]
        item = self.dataset_list.takeItem(row)
        del item

        if len(self.datasets) == 0 or self.dataset_list.count() == 0:
            self.dataset_list.blockSignals(True)
            self.dataset_list.clear()
            self.dataset_list.setCurrentRow(-1)
            self.dataset_list.blockSignals(False)

            self.current_index = None
            self.show_preview()
            self.refresh_ui_state()
            return

        new_row = min(row, self.dataset_list.count() - 1)
        self.dataset_list.setCurrentRow(new_row)  
        self.refresh_ui_state()

    def clear_all_datasets(self):
        if not self.datasets:
            return

        reply = QMessageBox.question(
            self, "Clear All Datasets",
            "Are you sure you want to remove ALL loaded datasets? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.datasets.clear()
            self.current_index = None
            
            self.dataset_list.blockSignals(True)
            self.dataset_list.clear()
            self.dataset_list.blockSignals(False)
            
            self.show_preview()
            self.refresh_ui_state()
            self.update_buttons(-1)

    # -------------------- Processing --------------------
    def run_preprocess(self):
        if self.current_index is None or self.current_index < 0 or self.current_index >= len(self.datasets):
            QMessageBox.warning(self, "Error", "Select a dataset first.")
            return

        ds0 = self.datasets[self.current_index]

        dlg = PreprocessOptionsDialog(
            self,
            default_apply_offset=ds0.get("apply_offset", False),
            default_strain_increment=ds0.get("strain_increment", 0.005),
        )
        if dlg.exec() != QDialog.Accepted:
            return

        apply_offset = dlg.get_apply_offset()
        auto_average = dlg.get_auto_average() 
        dx = float(dlg.get_strain_increment())

        if self.chk_process_all.isChecked():
            targets = self.datasets
        else:
            targets = [self.datasets[self.current_index]]

        processed_count = 0

        for ds in targets:
            try:
                ds["apply_offset"] = apply_offset
                ds["strain_increment"] = dx

                ds["proc_data"] = preprocess_data(
                    ds["raw_data"],
                    ds["diameter"],
                    ds["height"],
                    apply_offset=apply_offset,
                    strain_increment=dx,
                )
                ds["preprocessed"] = True
                processed_count += 1

                self.update_list_entry(self.datasets.index(ds))

            except Exception as e:
                QMessageBox.critical(self, "Preprocessing error", f"{ds['filename']}:\n{str(e)}")
                return

        self.refresh_ui_state()
        
        msg = f"Processed {processed_count} dataset(s)."

        if auto_average and processed_count > 0:
            avg_count = self._run_averaging()
            if avg_count > 0:
                msg += f"\n\nAutomatically generated {avg_count} averaged dataset(s) at the bottom of the list."

        if processed_count > 0:
            QMessageBox.information(self, "Success", msg)

    def _run_averaging(self):
        processed = [d for d in self.datasets if d.get("preprocessed", False) and d.get("proc_data") is not None]
        if not processed:
            return 0

        groups = {}
        for ds in processed:
            if str(ds["filename"]).startswith("AVG_"):
                continue

            raw_name = ds["filename"].replace(".csv", "")
            parts = raw_name.split("-")
            
            if len(parts) > 1:
                condition = "-".join(parts[1:])
            else:
                condition = raw_name 
                
            if condition not in groups:
                groups[condition] = []
            groups[condition].append(ds)

        new_datasets = []
        averaged_count = 0

        for condition, group in groups.items():
            if len(group) < 2:
                continue

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
                "apply_offset": group[0].get("apply_offset", False),
                "strain_increment": group[0].get("strain_increment", 0.005),
            }
            new_datasets.append(avg_ds)
            averaged_count += 1

        for nds in new_datasets:
            existing_names = [d["filename"] for d in self.datasets]
            if nds["filename"] in existing_names:
                idx = existing_names.index(nds["filename"])
                self.datasets[idx] = nds
                self.update_list_entry(idx)
            else:
                self.datasets.append(nds)
                self.update_list_entry(len(self.datasets) - 1)

        self.refresh_ui_state()
        return averaged_count

    # -------------------- PLOTTING --------------------
    def _build_figure(self):
        processed = [
            d for d in self.datasets
            if d.get("preprocessed", False) and d.get("proc_data") is not None
        ]
        if not processed:
            QMessageBox.warning(self, "Error", "No processed datasets to plot.")
            return None

        selected = []
        if self.chk_none.isChecked():
            selected.append("none")
        if self.chk_mild.isChecked():
            selected.append("mild")
        if self.chk_strong.isChecked():
            selected.append("strong")

        if not selected:
            QMessageBox.warning(self, "Error", "Select at least one curve to plot.")
            return None

        FILTER_LINESTYLES = {
            "none": "-",
            "mild": "--",
            "strong": "-.",
        }

        num_items = len(processed)
        if num_items <= 15:
            leg_cols = 1
            plot_width = 0.75
            font_sz = 8
            fig_width = 9
        elif num_items <= 30:
            leg_cols = 2
            plot_width = 0.60
            font_sz = 7
            fig_width = 12
        else:
            leg_cols = 3
            plot_width = 0.45
            font_sz = 6
            fig_width = 15

        fig, axes = plt.subplots(2, 1, figsize=(fig_width, 10), sharex=False)
        plotted_any = False

        for ds in processed:
            df = ds["proc_data"]
            fname = ds["filename"]

            first_filt = next(
                (f for f in selected if f"y_{f}" in df.columns),
                None
            )
            if first_filt is None:
                continue

            line, = axes[0].plot(
                df["x"].values,
                df[f"y_{first_filt}"].values,
                linestyle=FILTER_LINESTYLES[first_filt],
                linewidth=2,
                label=fname,
            )
            dataset_color = line.get_color()
            plotted_any = True

            for filt in selected:
                if filt == first_filt:
                    continue
                col = f"y_{filt}"
                if col not in df.columns:
                    continue

                axes[0].plot(
                    df["x"].values,
                    df[col].values,
                    color=dataset_color,
                    linestyle=FILTER_LINESTYLES[filt],
                    linewidth=2 if filt != "none" else 1.2,
                    label=None,
                )

        axes[0].set_title("Flow Stress")
        axes[0].set_xlabel("Strain [-]")
        axes[0].set_ylabel("Flow Stress [MPa]")
        axes[0].grid(True)

        if not plotted_any:
            QMessageBox.warning(
                self,
                "Plotting error",
                "No flow-stress curves were plotted.\nCheck processed data columns."
            )
            plt.close(fig)
            return None

        has_temp = any("temperature" in d["proc_data"].columns for d in processed)
        if has_temp:
            for ds in processed:
                df = ds["proc_data"]
                if "temperature" not in df.columns:
                    continue

                tx = (
                    df["temperature_x"].values
                    if "temperature_x" in df.columns
                    else df["x"].values
                )

                axes[1].plot(tx, df["temperature"].values, label=ds["filename"])

            axes[1].set_title("Temperature (reference)")
            axes[1].set_xlabel("Strain [-]")
            axes[1].set_ylabel("Temperature [°C]")
            axes[1].set_ylim(0, 700)
            axes[1].grid(True)
        else:
            axes[1].set_visible(False)

        for ax in axes:
            if ax.get_visible():
                h, l = ax.get_legend_handles_labels()
                if h:
                    ax.legend(
                        h, l,
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0,
                        fontsize=font_sz,
                        ncol=leg_cols
                    )

        plt.tight_layout(rect=[0, 0, plot_width, 1], h_pad=3.0)
        return fig

    def show_plots(self):
        fig = self._build_figure()
        if fig:
            plt.show()

    # -------------------- EXPORT LOGIC --------------------
    def _safe_sheet_name(self, name: str, suffix: str) -> str:
        bad = [":", "\\", "/", "?", "*", "[", "]"]
        out = name
        for b in bad:
            out = out.replace(b, "_")
        out = f"{out}__{suffix}"
        return out[:31]
    
    def export_results(self):
        if not self.datasets:
            QMessageBox.warning(self, "Error", "No datasets loaded.")
            return

        choice = QMessageBox.question(
            self,
            "Export scope",
            "Export all processed datasets?\n\n"
            "Yes  = Export ALL processed datasets (one sheet per dataset)\n"
            "No   = Export SELECTED dataset only",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No,  
        )
        if choice == QMessageBox.Cancel:
            return

        export_all = (choice == QMessageBox.Yes)

        if export_all:
            targets = [d for d in self.datasets if d.get("preprocessed", False) and d.get("proc_data") is not None]
            if not targets:
                QMessageBox.warning(self, "Error", "No processed datasets to export.")
                return
        else:
            if self.current_index is None or self.current_index < 0 or self.current_index >= len(self.datasets):
                QMessageBox.warning(self, "Error", "Select a dataset first.")
                return
            ds = self.datasets[self.current_index]
            if not ds.get("preprocessed", False) or ds.get("proc_data") is None:
                QMessageBox.warning(self, "Error", "Selected dataset is not processed yet.")
                return
            targets = [ds]

        export_filter = self.export_filter_combo.currentText().strip().lower()
        col_map = {"none": "y_none", "mild": "y_mild", "strong": "y_strong"}
        y_col = col_map.get(export_filter)
        if y_col is None:
            QMessageBox.warning(self, "Error", f"Unknown export filter: {export_filter}")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "Excel File (*.xlsx)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".xlsx"):
            file_path += ".xlsx"

        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                for ds in targets:
                    df = ds["proc_data"]
                    if "x" not in df.columns or y_col not in df.columns:
                        raise ValueError(
                            f"{ds['filename']} missing required columns. Need 'x' and '{y_col}'."
                        )

                    out = pd.DataFrame({
                        "strain": df["x"].values,
                        "flow_stress": df[y_col].values,
                    })

                    if "temperature" in df.columns:
                        out["temperature"] = df["temperature"].values

                    sheet = self._safe_sheet_name(ds["filename"], f"filter_{export_filter}")
                    out.to_excel(writer, sheet_name=sheet, index=False)

                    # --- ALWAYS GENERATE EXCEL CHART ---
                    ws = writer.sheets[sheet]
                    chart = ScatterChart()
                    chart.title = f"Flow Curve: {ds['filename']}"
                    chart.style = 2  
                    
                    chart.x_axis.delete = False
                    chart.y_axis.delete = False
                    
                    chart.x_axis.tickLblPos = "low"
                    chart.y_axis.tickLblPos = "low"
                    
                    chart.x_axis.title = "Strain [-]"
                    chart.y_axis.title = "Flow Stress [MPa]"
                    chart.width = 16  
                    chart.height = 10 
                    chart.legend = None 

                    max_row = len(out) + 1
                    xvalues = Reference(ws, min_col=1, min_row=2, max_row=max_row)
                    values = Reference(ws, min_col=2, min_row=2, max_row=max_row)

                    series = Series(values, xvalues, title="Flow Stress")
                    chart.series.append(series)

                    ws.add_chart(chart, "E2")

            QMessageBox.information(
                self,
                "Success",
                f"Exported {len(targets)} dataset(s)\n"
                f"Scope: {'ALL processed' if export_all else 'Selected only'}\n"
                f"Filter: {export_filter}\n"
                f"File: {file_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def export_deform_qform(self):
        idx = self.current_index
        if idx is None or idx < 0 or idx >= len(self.datasets):
            QMessageBox.warning(self, "Error", "Select a dataset first.")
            return

        ds = self.datasets[idx]
        if not ds.get("preprocessed", False) or ds.get("proc_data") is None:
            QMessageBox.warning(self, "Error", "Process the dataset first.")
            return

        df = ds["proc_data"]

        try:
            filter_text = self.export_filter_combo.currentText().lower()
            if "mild" in filter_text:
                chosen_y_col = "y_mild"
            elif "strong" in filter_text:
                chosen_y_col = "y_strong"
            else:
                chosen_y_col = "y_none"

            out = deform_qform_table(df, y_col=chosen_y_col, n_points=25, cluster_factor=50.0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to build DEFORM/QForm table:\n{str(e)}")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export DEFORM/QForm Table", "", "Excel File (*.xlsx)")
        if not file_path:
            return
        if not file_path.lower().endswith(".xlsx"):
            file_path += ".xlsx"

        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                sheet = self._safe_sheet_name(ds["filename"], "deform_qform")
                out.to_excel(writer, sheet_name=sheet, index=False)

                # --- ALWAYS GENERATE DEFORM EXCEL CHART ---
                ws = writer.sheets[sheet]
                chart = ScatterChart()
                chart.title = f"DEFORM/QForm Curve: {ds['filename']}"
                chart.style = 2  
                
                chart.x_axis.delete = False
                chart.y_axis.delete = False
                
                chart.x_axis.tickLblPos = "low"
                chart.y_axis.tickLblPos = "low"
                
                chart.x_axis.title = "Strain [-]"
                chart.y_axis.title = "Flow Stress [MPa]"
                chart.width = 16  
                chart.height = 10 
                chart.legend = None 

                max_row = len(out) + 1
                xvalues = Reference(ws, min_col=1, min_row=2, max_row=max_row)
                values = Reference(ws, min_col=2, min_row=2, max_row=max_row)

                series = Series(values, xvalues, title="Flow Stress")
                chart.series.append(series)

                ws.add_chart(chart, "D2") 

            QMessageBox.information(self, "Success", f"Exported {len(out)} points to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def export_force_displacement_selected(self):
        idx = self.current_index
        if idx is None or idx < 0 or idx >= len(self.datasets):
            QMessageBox.warning(self, "Error", "Select a dataset first.")
            return

        ds = self.datasets[idx]
        if not ds.get("preprocessed", False) or ds.get("proc_data") is None:
            QMessageBox.warning(self, "Error", "Process the dataset first.")
            return

        raw = ds.get("raw_data", None)
        if raw is None or raw.empty:
            QMessageBox.warning(self, "Error", "No raw data available for this dataset.")
            return

        try:
            out = force_displacement_curve_from_raw(
                raw=raw,
                diameter=ds["diameter"],
                height=ds["height"],
                compliance=0.0029,
                failure_drop_frac=0.85,
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to build force–displacement curve:\n{str(e)}")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Force–Displacement (Selected)",
            "",
            "Excel File (*.xlsx)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".xlsx"):
            file_path += ".xlsx"

        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                sheet = self._safe_sheet_name(ds["filename"], "force_disp")
                out.to_excel(writer, sheet_name=sheet, index=False)

            QMessageBox.information(
                self,
                "Success",
                f"Exported force–displacement ({len(out)} rows)\nFile: {file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")