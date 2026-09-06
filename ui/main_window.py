import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from openpyxl.chart import ScatterChart, Reference, Series

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem, QDialog, QInputDialog
)

from ui.ui_dialogs import GeometryDialog, PreprocessOptionsDialog
from ui.ui_builder import UiBuilder
from ui.plot_widget import FlowPlotWidget
from core.project_manager import ProjectManager
from core.signal_processing import preprocess_data
from core.exporters import safe_sheet_name, export_extrapolated_curves, export_deform_keyword_file, export_deform_qform_table, export_force_displacement
from core.material_models import fit_extrapolation_models, evaluate_model, fit_hensel_spittel_global
from core.constants import DEFORM_MATERIALS

class DataOptimizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flow Curve Tool")
        self.setGeometry(200, 200, 1300, 650)
        
        self.data_manager = ProjectManager()
        self.current_index = None

        # The UiBuilder automatically creates self.plot_widget and embeds it in the tab
        UiBuilder.build(self)
        self._connect_signals()

    def _connect_signals(self):
        self.dataset_list.currentRowChanged.connect(self.change_dataset)
        self.dataset_list.currentRowChanged.connect(self.update_buttons)
        self.load_btn.clicked.connect(self.load_dataset)
        self.edit_params_button.clicked.connect(self.edit_parameters)
        self.remove_button.clicked.connect(self.remove_dataset)
        self.clear_all_btn.clicked.connect(self.clear_all_datasets)
        self.preprocess_btn.clicked.connect(self.run_preprocess)
        self.plot_btn.clicked.connect(self.show_plots)
        self.export_btn.clicked.connect(self.export_results)
        self.force_disp_export_btn.clicked.connect(self.export_force_displacement_selected)
        self.deform_export_btn.clicked.connect(self.export_deform_qform)
        self.extrapolate_btn.clicked.connect(self.run_extrapolation)
        self.deform_key_export_btn.clicked.connect(self.export_deform_key)
        self.hensel_btn.clicked.connect(self.run_hensel_spittel)

    def update_list_entry(self, idx: int):
        ds = self.data_manager.datasets[idx]
        dx = ds.get("strain_increment", None)
        dx_txt = "Δε=?" if dx is None else f"Δε={dx:g}"
        entry = f"{ds['filename']}  |  D={ds['diameter']} mm, H={ds['height']} mm  |  {dx_txt}"

        if idx < self.dataset_list.count(): self.dataset_list.item(idx).setText(entry)
        else: self.dataset_list.addItem(entry)

    def show_preview(self):
        if self.current_index is None or self.current_index < 0 or self.current_index >= len(self.data_manager.datasets):
            self.preview_table.clear()
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return

        ds = self.data_manager.datasets[self.current_index]
        df = ds["proc_data"] if ds.get("preprocessed", False) and ds.get("proc_data") is not None else ds["raw_data"]
        preview = df.head(20)

        self.preview_table.setRowCount(len(preview))
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setHorizontalHeaderLabels(preview.columns)

        for i in range(len(preview)):
            for j in range(len(preview.columns)):
                self.preview_table.setItem(i, j, QTableWidgetItem(str(preview.iat[i, j])))

    def refresh_ui_state(self):
        has_selection = self.current_index is not None and 0 <= self.current_index < len(self.data_manager.datasets)
        any_processed = len(self.data_manager.get_processed_datasets()) > 0

        self.clear_all_btn.setEnabled(len(self.data_manager.datasets) > 0)
        self.preprocess_btn.setEnabled(has_selection)
        self.plot_btn.setEnabled(any_processed)
        self.export_btn.setEnabled(any_processed)
        self.hensel_btn.setEnabled(any_processed)
        
        has_proc_sel = has_selection and self.data_manager.datasets[self.current_index].get("preprocessed", False)
        self.deform_export_btn.setEnabled(has_proc_sel)
        self.deform_key_export_btn.setEnabled(has_proc_sel)
        self.force_disp_export_btn.setEnabled(has_proc_sel)
        self.extrapolate_btn.setEnabled(has_proc_sel)

    def update_buttons(self, index):
        has_selection = index >= 0 and index < len(self.data_manager.datasets)
        self.edit_params_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)

    def load_dataset(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open CSV files (multi-select)", "", "CSV Files (*.csv)")
        if not paths: return

        dlg = GeometryDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted: return
        diameter, height = dlg.get_values()
        if diameter is None or height is None: 
            return self.display_message("Please enter valid numbers!", error=True)

        loaded, failed = 0, []
        for path in paths:
            try:
                raw = pd.read_csv(path)
                if raw is None or raw.empty: raise ValueError("CSV is empty.")
                ds = {
                    "filename": os.path.basename(path), "raw_data": raw,
                    "diameter": float(diameter), "height": float(height),
                    "proc_data": None, "preprocessed": False,
                }
                self.data_manager.datasets.append(ds)
                self.update_list_entry(len(self.data_manager.datasets) - 1)
                loaded += 1
            except Exception as e: failed.append(f"{os.path.basename(path)}: {e}")

        if loaded > 0:
            self.refresh_ui_state()
            self.display_message(f"Loaded {loaded} file(s).")
        if failed:
            self.display_message("Some files failed to load:\n" + "\n".join(failed), error=True)

    def change_dataset(self, idx: int):
        self.current_index = idx
        self.refresh_ui_state()
        self.show_preview()

    def edit_parameters(self):
        if self.current_index is None: return
        if len(self.data_manager.datasets) > 1:
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle("Edit Specimen Parameters")
            msgBox.setText("You have multiple datasets loaded.\nHow would you like to edit the geometry?")
            btn_manual = msgBox.addButton("Edit Selected Manually", QMessageBox.ActionRole)
            btn_auto = msgBox.addButton("Auto-Match All (Probendaten Excel)", QMessageBox.ActionRole)
            btn_cancel = msgBox.addButton("Cancel", QMessageBox.RejectRole)
            msgBox.exec()
            
            if msgBox.clickedButton() == btn_cancel: return
            elif msgBox.clickedButton() == btn_auto: return self._import_probendaten()

        ds = self.data_manager.datasets[self.current_index]
        dialog = GeometryDialog(self, ds["diameter"], ds["height"])
        if dialog.exec() == QDialog.Accepted:
            diameter, height = dialog.get_values()
            if diameter is None or height is None: 
                return self.display_message("Please enter valid numbers!", error=True)
            
            ds["diameter"], ds["height"] = diameter, height
            self.update_list_entry(self.current_index)
            
            if ds["preprocessed"]: 
                self.display_message("Parameters updated. Reprocess data for the new geometry to take effect.", error=True)
            else:
                self.display_message("Parameters updated successfully.")

    def _import_probendaten(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Probendaten File", "", "Excel Files (*.xlsx *.xls);;CSV Files (*.csv)")
        if not file_path: return
        try: df_probe = pd.read_csv(file_path) if file_path.lower().endswith('.csv') else pd.read_excel(file_path)
        except Exception as e: 
            return self.display_message(f"Could not read file: {str(e)}", error=True)

        v_col, d_col, h_col = None, None, None
        for c in df_probe.columns:
            c_str = str(c).strip().lower()
            if 'versuch' in c_str: v_col = c
            elif 'ø' in c_str or 'durchmesser' in c_str or 'd0' in c_str: d_col = c
            elif 'h0' in c_str or 'höh' in c_str or 'hoeh' in c_str: h_col = c

        if not v_col or not d_col or not h_col:
            return self.display_message(f"Could not find the required columns. Found: {', '.join(str(c) for c in df_probe.columns)}", error=True)

        matched_count, already_processed = 0, False
        for i, ds in enumerate(self.data_manager.datasets):
            fname = ds["filename"]
            base_fname = fname.lower().replace('.csv', '')
            prefix_match = re.match(r'^0*(\d+)', fname)
            prefix_int = int(prefix_match.group(1)) if prefix_match else None

            for idx, row in df_probe.iterrows():
                v_val = row[v_col]
                if pd.isna(v_val): continue
                v_str = str(v_val).strip()

                is_match = False
                if prefix_int is not None:
                    try: 
                        if int(float(v_str)) == prefix_int: is_match = True
                    except ValueError: pass
                
                if not is_match:
                    if base_fname == v_str.lower() or base_fname.startswith(v_str.lower() + "-") or base_fname.startswith(v_str.lower() + "_"):
                        is_match = True

                if is_match:
                    d_val, h_val = row[d_col], row[h_col]
                    if pd.notna(d_val) and pd.notna(h_val):
                        ds["diameter"], ds["height"] = float(d_val), float(h_val)
                        self.update_list_entry(i)
                        matched_count += 1
                        if ds.get("preprocessed"): already_processed = True
                    break

        msg = f"Successfully matched and updated {matched_count} out of {len(self.data_manager.datasets)} datasets."
        if already_processed: msg += " (Some updated datasets were already processed. Reprocess them.)"
        self.display_message(msg)

    def remove_dataset(self):
        row = self.dataset_list.currentRow()
        if row < 0 or row >= self.dataset_list.count(): return
        
        if row >= len(self.data_manager.datasets):
            self.data_manager.datasets = self.data_manager.datasets[: self.dataset_list.count()]
            if row >= len(self.data_manager.datasets):
                self.dataset_list.takeItem(row)
                self.current_index = None
                self.show_preview()
                self.refresh_ui_state()
                return

        reply = QMessageBox.question(self, "Remove Dataset", f"Really remove '{self.data_manager.datasets[row]['filename']}'?", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return
        
        del self.data_manager.datasets[row]
        self.dataset_list.takeItem(row)

        if len(self.data_manager.datasets) == 0 or self.dataset_list.count() == 0:
            self.dataset_list.blockSignals(True)
            self.dataset_list.clear()
            self.dataset_list.setCurrentRow(-1)
            self.dataset_list.blockSignals(False)
            self.current_index = None
            self.show_preview()
            self.refresh_ui_state()
            return
            
        self.dataset_list.setCurrentRow(min(row, self.dataset_list.count() - 1))  
        self.refresh_ui_state()

    def clear_all_datasets(self):
        if not self.data_manager.datasets: return
        
        reply = QMessageBox.question(self, "Clear All Datasets", "Are you sure you want to remove ALL loaded datasets?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.data_manager.datasets.clear()
            self.current_index = None
            self.dataset_list.blockSignals(True)
            self.dataset_list.clear()
            self.dataset_list.blockSignals(False)
            self.show_preview()
            self.refresh_ui_state()
            self.update_buttons(-1)

    def run_preprocess(self):
        if self.current_index is None or self.current_index < 0 or self.current_index >= len(self.data_manager.datasets):
            return self.display_message("Select a dataset first.", error=True)

        ds0 = self.data_manager.datasets[self.current_index]
        dlg = PreprocessOptionsDialog(self, default_strain_increment=ds0.get("strain_increment", 0.005))
        if dlg.exec() != QDialog.Accepted: return
        
        auto_average = dlg.get_auto_average() 
        dx = float(dlg.get_strain_increment())
        targets = self.data_manager.datasets if self.chk_process_all.isChecked() else [self.data_manager.datasets[self.current_index]]

        processed_count = 0
        for ds in targets:
            try:
                ds["strain_increment"] = dx
                ds["proc_data"] = preprocess_data(ds["raw_data"], ds["diameter"], ds["height"], strain_increment=dx)
                ds["preprocessed"] = True
                processed_count += 1
                self.update_list_entry(self.data_manager.datasets.index(ds))
            except Exception as e:
                return self.display_message(f"Preprocessing error for {ds['filename']}: {str(e)}", error=True)

        self.refresh_ui_state()
        msg = f"Processed {processed_count} dataset(s)."

        if auto_average and processed_count > 0:
            avg_count = self.data_manager.run_averaging()
            for i in range(len(self.data_manager.datasets)): self.update_list_entry(i)
            if avg_count > 0: msg += f" Auto-generated {avg_count} averaged dataset(s)."

        if processed_count > 0: 
            self.display_message(msg)

    def _build_figure(self):
        processed = self.data_manager.get_processed_datasets()
        if not processed:
            self.display_message("No processed datasets to plot.", error=True)
            return None

        selected = []
        if self.chk_none.isChecked(): selected.append("none")
        if self.chk_mild.isChecked(): selected.append("mild")
        if self.chk_strong.isChecked(): selected.append("strong")

        if not selected:
            self.display_message("Select at least one curve to plot.", error=True)
            return None

        FILTER_LINESTYLES = {"none": "-", "mild": "--", "strong": "-."}

        num_items = len(processed)
        if num_items <= 15:
            leg_cols, plot_width, font_sz = 1, 0.75, 8
        elif num_items <= 30:
            leg_cols, plot_width, font_sz = 2, 0.60, 7
        else:
            leg_cols, plot_width, font_sz = 3, 0.45, 6

        from matplotlib.figure import Figure
        fig = Figure(figsize=(9, 10))
        axes = fig.subplots(2, 1, sharex=False)
        plotted_any = False

        for ds in processed:
            df = ds["proc_data"]
            fname = ds["filename"]

            first_filt = next((f for f in selected if f"y_{f}" in df.columns), None)
            if first_filt is None: continue

            line, = axes[0].plot(df["x"].values, df[f"y_{first_filt}"].values, linestyle=FILTER_LINESTYLES[first_filt], linewidth=2, label=fname)
            dataset_color = line.get_color()
            plotted_any = True

            for filt in selected:
                if filt == first_filt: continue
                col = f"y_{filt}"
                if col not in df.columns: continue
                axes[0].plot(df["x"].values, df[col].values, color=dataset_color, linestyle=FILTER_LINESTYLES[filt], linewidth=2 if filt != "none" else 1.2, label=None)

        axes[0].set_title("Flow Stress")
        axes[0].set_xlabel("Strain [-]")
        axes[0].set_ylabel("Flow Stress [MPa]")
        axes[0].grid(True)

        if not plotted_any:
            self.display_message("No flow-stress curves were plotted.", error=True)
            return None

        has_temp = any("temperature" in d["proc_data"].columns for d in processed)
        if has_temp:
            for ds in processed:
                df = ds["proc_data"]
                if "temperature" not in df.columns: continue
                tx = df["temperature_x"].values if "temperature_x" in df.columns else df["x"].values
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
                if h: ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=font_sz, ncol=leg_cols)

        fig.tight_layout(rect=[0, 0, plot_width, 1], h_pad=3.0)
        fig.subplots_adjust(bottom=0.12)  
        return fig

    def show_plots(self):
        fig = self._build_figure()
        if not fig: return

        def handle_cut(x_val):
            self.data_manager.apply_manual_cut_all(x_val)
            self.show_preview()
            from PySide6.QtCore import QTimer
            QTimer.singleShot(50, self.show_plots)

        # Defines the reset action
        def handle_reset():
            if self.data_manager.reset_manual_cut_all():
                self.show_preview()
                from PySide6.QtCore import QTimer
                QTimer.singleShot(50, self.show_plots)

        # Passes BOTH callbacks to the widget so the reset button becomes visible
        self.plot_widget.plot_interactive_figure(fig, handle_cut, handle_reset)
        self.tabs.setCurrentIndex(1)

    def export_results(self):
        if not self.data_manager.datasets: 
            return self.display_message("No datasets loaded.", error=True)
            
        choice = QMessageBox.question(self, "Export scope", "Export all processed datasets?\n\nYes  = Export ALL processed datasets (one sheet per dataset)\nNo   = Export SELECTED dataset only", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.No)
        if choice == QMessageBox.Cancel: return
        export_all = (choice == QMessageBox.Yes)

        if export_all:
            targets = self.data_manager.get_processed_datasets()
            if not targets: 
                return self.display_message("No processed datasets to export.", error=True)
        else:
            if self.current_index is None: 
                return self.display_message("Select a dataset first.", error=True)
            ds = self.data_manager.datasets[self.current_index]
            if not ds.get("preprocessed", False) or ds.get("proc_data") is None: 
                return self.display_message("Selected dataset is not processed yet.", error=True)
            targets = [ds]

        export_filter = self.export_filter_combo.currentText().strip().lower()
        col_map = {"none": "y_none", "mild": "y_mild", "strong": "y_strong"}
        y_col = col_map.get(export_filter)

        file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "Excel File (*.xlsx)")
        if not file_path: return
        if not file_path.lower().endswith(".xlsx"): file_path += ".xlsx"

        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                for ds in targets:
                    df = ds["proc_data"]
                    if "x" not in df.columns or y_col not in df.columns: raise ValueError(f"{ds['filename']} missing required columns.")
                    out = pd.DataFrame({"strain": df["x"].values, "flow_stress": df[y_col].values})
                    if "temperature" in df.columns: out["temperature"] = df["temperature"].values
                    sheet = safe_sheet_name(ds["filename"], f"filter_{export_filter}")
                    out.to_excel(writer, sheet_name=sheet, index=False)

                    ws = writer.sheets[sheet]
                    chart = ScatterChart()
                    chart.title = f"Flow Curve: {ds['filename']}"
                    chart.style = 13  
                    chart.x_axis.title = "Strain [-]"
                    chart.y_axis.title = "Flow Stress [MPa]"
                    chart.width = 16  
                    chart.height = 10 
                    
                    max_row = len(out) + 1
                    xvalues = Reference(ws, min_col=1, min_row=2, max_row=max_row)
                    values = Reference(ws, min_col=2, min_row=2, max_row=max_row)
                    
                    # Explicitly name the series to prevent default labeling
                    series = Series(values, xvalues, title_from_data=False, title="Flow Stress")
                    chart.series.append(series)
                    
                    ws.add_chart(chart, "E2")

            self.display_message(f"Exported {len(targets)} dataset(s) to {file_path}")
        except Exception as e:
            self.display_message(f"Export failed: {str(e)}", error=True)

    def export_deform_qform(self):
        processed_ds = self.data_manager.get_processed_datasets()
        if not processed_ds: 
            return self.display_message("No processed datasets available.", error=True)

        n_points, ok = QInputDialog.getInt(self, "Strain Points", "Number of points:", 25)
        if not ok: return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export DEFORM/QForm Table", "", "Excel File (*.xlsx)")
        if not file_path: return
        if not file_path.lower().endswith(".csv"): file_path += ".xlsx"

        filter_text = self.export_filter_combo.currentText().lower()
        chosen_y_col = "y_mild" if "mild" in filter_text else "y_strong" if "strong" in filter_text else "y_none"

        try:
            export_deform_qform_table(processed_ds, chosen_y_col, n_points, 50.0, file_path)
            self.display_message(f"DEFORM/QForm table exported to: {file_path}")
        except Exception as e:
            self.display_message(f"Export failed: {str(e)}", error=True)

    def export_force_displacement_selected(self):
        if not self.data_manager.datasets: 
            return self.display_message("No datasets loaded.", error=True)

        d, ok1 = QInputDialog.getDouble(self, "Diameter", "Enter default diameter (mm):", 10.0)
        if not ok1: return
        
        h, ok2 = QInputDialog.getDouble(self, "Height", "Enter default height (mm):", 15.0)
        if not ok2: return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export Force-Displacement", "", "Excel File (*.xlsx)")
        if not file_path: return
        if not file_path.lower().endswith(".csv"): file_path += ".xlsx"

        try:
            export_force_displacement(self.data_manager.datasets, d, h, file_path)
            self.display_message(f"Force-Displacement data exported to: {file_path}")
        except Exception as e:
            self.display_message(f"Export failed: {str(e)}", error=True)

    def run_extrapolation(self):
        if self.current_index is None: 
            return self.display_message("Select a dataset first.", error=True)
        ds = self.data_manager.datasets[self.current_index]
        if not ds.get("preprocessed", False) or ds.get("proc_data") is None: 
            return self.display_message("Process the dataset first.", error=True)

        max_strain, ok = QInputDialog.getDouble(self, "Extrapolation Target", "Enter maximum strain for extrapolation:", 2.0, 0.1, 100.0, 2)
        if not ok: return

        df = ds["proc_data"]
        filter_text = self.export_filter_combo.currentText().strip().lower()
        y_col = "y_mild" if "mild" in filter_text else "y_strong" if "strong" in filter_text else "y_none"
        if y_col not in df.columns: 
            return self.display_message(f"Column {y_col} not found in processed data.", error=True)

        epsilon = df["x"].values.astype(float)
        sigma = df[y_col].values.astype(float)
        
        peak_idx = np.argmax(sigma)
        epsilon_fit = epsilon[:peak_idx + 1]
        sigma_fit = sigma[:peak_idx + 1]

        try: params_dict = fit_extrapolation_models(epsilon_fit, sigma_fit)
        except Exception as e: 
            return self.display_message(f"Model fitting failed: {str(e)}", error=True)
            
        interp_epsilon = np.linspace(max(0.01, np.min(epsilon)), max_strain, 200)
        out_data = {'epsilon': interp_epsilon}
        model_names = ['Hollomon', 'Ludwik', 'Swift', 'Voce', 'Hockett-Sherby', 'Swift+Voce']
        for model in model_names:
            safe_name = model.lower().replace('-', '_').replace('+', '_')
            out_data[safe_name] = evaluate_model(model, interp_epsilon, params_dict[model], params_dict.get('Swift'), params_dict.get('Voce'))
            
        interp_df = pd.DataFrame(out_data)
            
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(epsilon, sigma, label="Measured Data", color="black", s=20, alpha=0.8)
        if params_dict['Hollomon'] is not None: ax.plot(interp_epsilon, out_data['hollomon'], label="Hollomon", linestyle="--")
        if params_dict['Ludwik'] is not None: ax.plot(interp_epsilon, out_data['ludwik'], label="Ludwik", linestyle="-.")
        if params_dict['Swift'] is not None: ax.plot(interp_epsilon, out_data['swift'], label="Swift", linestyle=":")
        if params_dict['Voce'] is not None: ax.plot(interp_epsilon, out_data['voce'], label="Voce", linestyle="-")
        if params_dict['Hockett-Sherby'] is not None: ax.plot(interp_epsilon, out_data['hockett_sherby'], label="Hockett-Sherby", linestyle=(0, (3, 1, 1, 1)))
        if params_dict['Swift+Voce'] is not None: ax.plot(interp_epsilon, out_data['swift_voce'], label="Swift+Voce", linewidth=2)
        ax.set_xlabel("Strain [-]")
        ax.set_ylabel("Flow Stress [MPa]")
        ax.set_title(f"Extrapolation Models: {ds['filename']}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, borderaxespad=0.0)
        fig.subplots_adjust(bottom=0.25)    
        self.extra_plot_widget.plot_interactive_figure(fig, cut_callback=None)
        self.tabs.setCurrentIndex(3)

        reply = QMessageBox.question(self, "Export Results", "Would you like to export the extrapolated curve data to Excel?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            file_path, _ = QFileDialog.getSaveFileName(self, "Export Extrapolated Curves", safe_sheet_name(ds['filename'], 'extrapolated'), "Excel File (*.xlsx)")
            if file_path:
                if not file_path.lower().endswith(".xlsx"): file_path += ".xlsx"
                try:
                    export_extrapolated_curves(file_path, safe_sheet_name(ds['filename'], 'extrapolate'), interp_df, ds['filename'])
                    self.display_message(f"Extrapolated curves exported to: {file_path}")
                except Exception as e:
                    self.display_message(f"Failed to save Excel file: {str(e)}", error=True)

    def export_deform_key(self):
        processed_ds = self.data_manager.get_processed_datasets()
        if not processed_ds: 
            return self.display_message("No processed datasets available to export.", error=True)

        filter_text = self.export_filter_combo.currentText().strip().lower()
        y_col = "y_mild" if "mild" in filter_text else "y_strong" if "strong" in filter_text else "y_none"

        parsed_data, max_strain_global = [], 100.0
        for ds in processed_ds:
            match = re.search(r'T([\d\.]+).*?SR([\d\.]+)', ds['filename'], re.IGNORECASE)
            if not match: 
                return self.display_message("Could not extract Temperature and Strain Rate from filename.", error=True)
            
            temp, sr = float(match.group(1)), float(match.group(2))
            epsilon, sigma = ds['proc_data']['x'].values, ds['proc_data'][y_col].values
            max_strain_global = min(max_strain_global, epsilon[-1])
            parsed_data.append({'T': temp, 'SR': sr, 'eps': epsilon, 'sig': sigma})

        unique_T, unique_SR = sorted(list(set(d['T'] for d in parsed_data))), sorted(list(set(d['SR'] for d in parsed_data)))
        if len(parsed_data) != len(unique_T) * len(unique_SR):
            return self.display_message(f"DEFORM requires a full matrix. {len(unique_T)*len(unique_SR)} curves required.", error=True)

        selected_mat_name, ok = QInputDialog.getItem(self, "Select Base Material", "Choose material parameters to inherit:", list(DEFORM_MATERIALS.keys()), 0, False)
        if not ok or not selected_mat_name: return

        material_name, ok = QInputDialog.getText(self, "Material Name", "Enter Material Name for DEFORM:", text=selected_mat_name)
        if not ok or not material_name: return

        num_strains, ok = QInputDialog.getInt(self, "Strain Points", "How many strain points for the matrix?", 20, 5, 100)
        if not ok: return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save DEFORM Keyword File", f"{material_name}.key", "Keyword File (*.key);;Text File (*.txt)")
        if not file_path: return

        try:
            export_deform_keyword_file(processed_ds, y_col, selected_mat_name, material_name, num_strains, file_path, 50.0)
            self.display_message(f"DEFORM 3D Keyword file exported: {file_path}")
        except Exception as e:
            self.display_message(f"Failed to write file: {str(e)}", error=True)

    def run_hensel_spittel(self):
        processed_ds = self.data_manager.get_processed_datasets()
        if not processed_ds:
            return self.display_message("No processed datasets available.", error=True)

        filter_text = self.export_filter_combo.currentText().strip().lower()
        y_col = "y_mild" if "mild" in filter_text else "y_strong" if "strong" in filter_text else "y_none"

        all_eps, all_sig, all_T, all_SR = [], [], [], []
        curve_meta = []  

        for ds in processed_ds:
            match = re.search(r'(?:T([\d\.]+)|(RT)).*?SR([\d\.]+)', ds['filename'], re.IGNORECASE)
            
            if match:
                T_val = 20.0 if match.group(2) else float(match.group(1))
                SR_val = float(match.group(3))
            else:
                T_val, ok_t = QInputDialog.getDouble(self, "Missing Metadata", f"Enter Temperature (°C) for:\n{ds['filename']}", 20.0, -273.0, 3000.0, 1)
                if not ok_t: continue  
                
                SR_val, ok_sr = QInputDialog.getDouble(self, "Missing Metadata", f"Enter Strain Rate (1/s) for:\n{ds['filename']}", 1.0, 0.0001, 1000.0, 4)
                if not ok_sr: continue

            df = ds['proc_data']
            if y_col not in df.columns: continue

            eps = df['x'].values.astype(float)
            sig = df[y_col].values.astype(float)

            all_eps.extend(eps)
            all_sig.extend(sig)
            all_T.extend(np.full_like(eps, T_val))
            all_SR.extend(np.full_like(eps, SR_val))
            
            curve_meta.append((T_val, SR_val, eps, sig, ds['filename']))

        try:
            params = fit_hensel_spittel_global(all_eps, all_SR, all_T, all_sig)
            
            from matplotlib.figure import Figure
            import matplotlib.cm as cm
            from core.material_models import hensel_spittel_9param
            
            fig = Figure(figsize=(11, 8))
            ax = fig.subplots()
            
            colors = cm.get_cmap('tab10', len(curve_meta))
            
            for idx, (T_val, SR_val, eps, sig, fname) in enumerate(curve_meta):
                c = colors(idx)
                ax.scatter(eps, sig, color=c, s=15, alpha=0.5, label=f"Exp: {fname}")
                
                eps_smooth = np.linspace(max(0.01, min(eps)), max(eps) * 1.1, 100)
                T_smooth = np.full_like(eps_smooth, T_val)
                SR_smooth = np.full_like(eps_smooth, SR_val)
                
                sig_pred = hensel_spittel_9param((eps_smooth, SR_smooth, T_smooth), **params)
                ax.plot(eps_smooth, sig_pred, color=c, linewidth=2, label=f"HS: T={T_val}, SR={SR_val}")
                
            ax.set_xlabel("Strain [-]")
            ax.set_ylabel("Flow Stress [MPa]")
            ax.set_title("Hensel-Spittel Global Fit vs. Experimental Data")
            ax.grid(True, alpha=0.3)
            
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8)
            fig.subplots_adjust(right=0.75, bottom=0.15)
            
            # Pipe into the Hensel-Spittel Tab (Index 2)
            self.hs_plot_widget.plot_interactive_figure(fig, cut_callback=None)
            self.tabs.setCurrentIndex(2)
            
            # Display parameters in a popup for easy copying, but also notify via status bar
            self.display_message("Hensel-Spittel Global Fit Successful!")
            display_msg = "Fit Successful!\n\n"
            copy_text = ""
            for key, val in params.items(): 
                line = f"{key} = {val:.4f}\n"  # Forces standard decimal, max 4 places
                display_msg += line
                copy_text += line
                
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Hensel-Spittel Parameters")
            msg_box.setText(display_msg)
            
            copy_btn = msg_box.addButton("Copy for QForm", QMessageBox.ActionRole)
            msg_box.addButton(QMessageBox.Ok)
            msg_box.exec()
            
            if msg_box.clickedButton() == copy_btn:
                from PySide6.QtGui import QGuiApplication
                QGuiApplication.clipboard().setText(copy_text.strip())
                self.display_message("Parameters copied to clipboard!")
            
        except Exception as e:
            self.display_message(f"Fit Error: {str(e)}", error=True)

    def display_message(self, msg: str, error: bool = False):
        """Displays transient messages in the main window's status bar."""
        color = "#d32f2f" if error else "#2e7d32"
        self.statusBar().setStyleSheet(f"color: {color}; font-weight: bold;")
        
        flat_msg = msg.replace('\n', ' | ')
        prefix = "ERROR: " if error else "SUCCESS: "
        
        self.statusBar().showMessage(f"{prefix}{flat_msg}", 6000)