import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

        # Show processed if available, else raw
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

        # Ask geometry ONCE
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
            self.preprocess_btn.setEnabled(True)
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

    def remove_dataset(self):
        row = self.dataset_list.currentRow()
        if row < 0 or row >= self.dataset_list.count():
            return

        # Guard against model-view drift
        if row >= len(self.datasets):
            # Hard resync: safest behavior
            self.datasets = self.datasets[: self.dataset_list.count()]
            if row >= len(self.datasets):
                # nothing valid to delete from model; just clear the view row
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

        # Remove from model
        del self.datasets[row]

        # Remove from view (and actually delete the item)
        item = self.dataset_list.takeItem(row)
        del item

        # If nothing left, hard-clear list + selection
        if len(self.datasets) == 0 or self.dataset_list.count() == 0:
            self.dataset_list.blockSignals(True)
            self.dataset_list.clear()
            self.dataset_list.setCurrentRow(-1)
            self.dataset_list.blockSignals(False)

            self.current_index = None
            self.show_preview()
            self.refresh_ui_state()
            return

        # Otherwise, select next valid row
        new_row = min(row, self.dataset_list.count() - 1)
        self.dataset_list.setCurrentRow(new_row)  # triggers change_dataset -> preview
        self.refresh_ui_state()

 # -------------------- Processing --------------------
    def run_preprocess(self):
        if self.current_index is None or self.current_index < 0 or self.current_index >= len(self.datasets):
            QMessageBox.warning(self, "Error", "Select a dataset first.")
            return

        ds0 = self.datasets[self.current_index]

        # Dialog: offset + resample points
        dlg = PreprocessOptionsDialog(
            self,
            default_apply_offset=ds0.get("apply_offset", False),
            default_strain_increment=ds0.get("strain_increment", 0.005),
        )
        if dlg.exec() != QDialog.Accepted:
            return

        apply_offset = dlg.get_apply_offset()
        dx = float(dlg.get_strain_increment())

        # Decide which datasets to process
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

                # Keep list text in sync (shows offset state)
                self.update_list_entry(self.datasets.index(ds))

            except Exception as e:
                QMessageBox.critical(self, "Preprocessing error", f"{ds['filename']}:\n{str(e)}")
                return

        self.refresh_ui_state()

        if processed_count > 0:
            self.plot_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

        QMessageBox.information(self, "Success", f"Processed {processed_count} dataset(s).")


    def show_plots(self):
        processed = [
            d for d in self.datasets
            if d.get("preprocessed", False) and d.get("proc_data") is not None
        ]
        if not processed:
            QMessageBox.warning(self, "Error", "No processed datasets to plot.")
            return

        # Which filter curves to plot (based on checkboxes)
        selected = []
        if self.chk_none.isChecked():
            selected.append("none")
        if self.chk_mild.isChecked():
            selected.append("mild")
        if self.chk_strong.isChecked():
            selected.append("strong")

        if not selected:
            QMessageBox.warning(self, "Error", "Select at least one curve to plot.")
            return

        # Style map: different colours for different filters (same across datasets)
        FILTER_STYLES = {
            "none":   {"color": "black",    "linestyle": "-",  "label": "Unfiltered"},
            "mild":   {"color": "tab:blue", "linestyle": "--", "label": "Mild"},
            "strong": {"color": "tab:red",  "linestyle": "-.", "label": "Strong"},
        }

        fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharex=False)

       # ---------- FLOW STRESS ----------
        selected = []
        if self.chk_none.isChecked():
            selected.append("none")
        if self.chk_mild.isChecked():
            selected.append("mild")
        if self.chk_strong.isChecked():
            selected.append("strong")

        if not selected:
            QMessageBox.warning(self, "Error", "Select at least one curve to plot.")
            return

        FILTER_LINESTYLES = {
            "none": "-",
            "mild": "--",
            "strong": "-.",
        }

        plotted_any = False

        for ds in processed:
            df = ds["proc_data"]
            fname = ds["filename"]

            # pick first available filter to set dataset color
            first_filt = next(
                (f for f in selected if f"y_{f}" in df.columns),
                None
            )
            if first_filt is None:
                continue

            # plot first curve → one legend entry per dataset
            line, = axes[0].plot(
                df["x"].values,
                df[f"y_{first_filt}"].values,
                linestyle=FILTER_LINESTYLES[first_filt],
                linewidth=2,
                label=fname,
            )
            dataset_color = line.get_color()
            plotted_any = True

            # remaining filters: same color, no legend
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
                "No flow-stress curves were plotted.\n"
                "Check processed data columns."
            )
            return

                # ---------- TEMPERATURE (reference) ----------
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

# ---------- LEGENDS OUTSIDE ----------
        for ax in axes:
            if ax.get_visible():
                h, l = ax.get_legend_handles_labels()
                if h:
                    ax.legend(
                        h, l,
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0,
                        fontsize=8
                    )

        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.show()



    def _safe_sheet_name(self, name: str, suffix: str) -> str:
        """
        Excel sheet name constraints:
        - max 31 chars
        - cannot contain: : \ / ? * [ ]
        """
        bad = [":", "\\", "/", "?", "*", "[", "]"]
        out = name
        for b in bad:
            out = out.replace(b, "_")
        out = f"{out}__{suffix}"
        return out[:31]
    
    # --- Export Results to Excel ---

    def export_results(self):
        # 1) Ask scope AFTER clicking Export
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
            QMessageBox.No,  # default
        )
        if choice == QMessageBox.Cancel:
            return

        export_all = (choice == QMessageBox.Yes)

        # 2) Determine targets
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

        # 3) Choose curve to export
        export_filter = self.export_filter_combo.currentText().strip().lower()
        col_map = {"none": "y_none", "mild": "y_mild", "strong": "y_strong"}
        y_col = col_map.get(export_filter)
        if y_col is None:
            QMessageBox.warning(self, "Error", f"Unknown export filter: {export_filter}")
            return

        # 4) Choose file path
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

        # 5) Write Excel (one sheet per dataset)
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


# -------------------- Deform/QFORM Export --------------------
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

        # strong-by-default with fallback
        y_col = "y_strong" if "y_strong" in df.columns else ("y_mild" if "y_mild" in df.columns else "y_none")

        try:
            out = deform_qform_table(
                df,
                y_col=y_col,
                n_early=15,
                n_late=10,
                split_frac=0.20,
                enforce_monotonic=True,
                clip_nonnegative=True,
            )
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
