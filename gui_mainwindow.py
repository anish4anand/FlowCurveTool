import os
import pandas as pd
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QFileDialog, QMessageBox, QLabel, QListWidget, QTableWidget,
    QTableWidgetItem, QComboBox
)

from dialogs import GeometryDialog
from processing.preprocess import preprocess_data
from processing.utils import apply_offset, temperature_compensation


class DataOptimizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flow Curve Tool")
        self.setGeometry(200, 200, 1300, 600)

        self.datasets = []
        self.current_index = None

        main_layout = QHBoxLayout()
        self.preview_table = QTableWidget()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        main_layout.addWidget(self.preview_table, 3)

        sidebar_layout = QVBoxLayout()
        self.dataset_list = QListWidget()
        self.dataset_list.currentRowChanged.connect(self.change_dataset)
        self.dataset_list.currentRowChanged.connect(self.update_buttons)
        sidebar_layout.addWidget(QLabel("Loaded Datasets:"))
        sidebar_layout.addWidget(self.dataset_list)

        # Buttons
        self.load_button = QPushButton("Add CSV + Parameters")
        self.load_button.clicked.connect(self.load_dataset)
        sidebar_layout.addWidget(self.load_button)

        self.edit_params_button = QPushButton("Edit Specimen Parameters")
        self.edit_params_button.clicked.connect(self.edit_parameters)
        self.edit_params_button.setEnabled(False)
        sidebar_layout.addWidget(self.edit_params_button)

        self.remove_button = QPushButton("Remove Dataset from List")
        self.remove_button.clicked.connect(self.remove_dataset)
        self.remove_button.setEnabled(False)
        sidebar_layout.addWidget(self.remove_button)

        # --- Filter selection dropdown ---
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(QLabel("Noise Filter:"))
        self.filter_select = QComboBox()
        self.filter_select.addItems(["none", "light", "strong"])
        self.filter_select.setCurrentText("light")
        sidebar_layout.addWidget(self.filter_select)

        self.preprocess_button = QPushButton("Process Raw Data")
        self.preprocess_button.clicked.connect(self.run_preprocess)
        self.preprocess_button.setEnabled(False)
        sidebar_layout.addWidget(self.preprocess_button)

        self.offset_button = QPushButton("Apply Offset")
        self.offset_button.clicked.connect(self.run_offset)
        self.offset_button.setEnabled(False)
        sidebar_layout.addWidget(self.offset_button)

        self.tempcomp_button = QPushButton("Temperature Compensation")
        self.tempcomp_button.clicked.connect(self.run_tempcomp)
        self.tempcomp_button.setEnabled(False)
        sidebar_layout.addWidget(self.tempcomp_button)

        self.plot_button = QPushButton("Plots")
        self.plot_button.clicked.connect(self.show_plots)
        self.plot_button.setEnabled(False)
        sidebar_layout.addWidget(self.plot_button)

        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        sidebar_layout.addWidget(self.export_button)

        sidebar_layout.addStretch()
        main_layout.addLayout(sidebar_layout, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # -------------------- Button Enable/Disable --------------------
    def update_buttons(self, index):
        has_selection = index >= 0 and index < len(self.datasets)
        self.edit_params_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)

    # -------------------- Load CSV --------------------
    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            data = pd.read_csv(file_path)
            if data.shape[1] < 2:
                raise ValueError("CSV must have at least 2 columns (x, y).")

            dialog = GeometryDialog(self)
            if dialog.exec() == dialog.Accepted:
                diameter, height = dialog.get_values()
                if diameter is None or height is None:
                    QMessageBox.warning(self, "Error", "Please enter valid numbers!")
                    return
            else:
                return

            dataset = {
                "filename": os.path.basename(file_path),
                "raw_data": data,
                "proc_data": None,
                "diameter": diameter,
                "height": height,
                "preprocessed": False
            }
            self.datasets.append(dataset)
            self.update_list_entry(len(self.datasets) - 1)
            QMessageBox.information(self, "Success", f"File loaded: {dataset['filename']}")
            self.preprocess_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{str(e)}")

    def update_list_entry(self, index):
        ds = self.datasets[index]
        entry = f"{ds['filename']}  |  D={ds['diameter']} mm, H={ds['height']} mm"
        if index < self.dataset_list.count():
            self.dataset_list.item(index).setText(entry)
        else:
            self.dataset_list.addItem(entry)

    # -------------------- Switch Dataset --------------------
    def change_dataset(self, index):
        if index < 0 or index >= len(self.datasets):
            self.current_index = None
            self.edit_params_button.setEnabled(False)
            self.preprocess_button.setEnabled(False)
            self.offset_button.setEnabled(False)
            self.tempcomp_button.setEnabled(False)
            self.plot_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.preview_table.clear()
            return

        self.current_index = index
        self.show_preview()
        ds = self.datasets[index]
        self.edit_params_button.setEnabled(True)
        self.preprocess_button.setEnabled(True)
        self.offset_button.setEnabled(ds["preprocessed"])
        self.tempcomp_button.setEnabled(ds["preprocessed"])
        self.plot_button.setEnabled(ds["preprocessed"])
        self.export_button.setEnabled(any(ds["preprocessed"] for ds in self.datasets))

    # -------------------- Preview --------------------
    def show_preview(self):
        if not self.datasets or self.current_index is None:
            return
        ds = self.datasets[self.current_index]
        preview = ds["raw_data"].head(20)
        self.preview_table.setRowCount(len(preview))
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setHorizontalHeaderLabels(preview.columns)
        for i in range(len(preview)):
            for j in range(len(preview.columns)):
                item = QTableWidgetItem(str(preview.iat[i, j]))
                self.preview_table.setItem(i, j, item)

    # -------------------- Processing --------------------
    def run_preprocess(self):
        if not self.datasets:
            return
        filter_mode = self.filter_select.currentText()
        for ds in self.datasets:
            ds["proc_data"] = preprocess_data(ds["raw_data"], ds["diameter"], ds["height"], filter_mode)
            ds["preprocessed"] = True

        QMessageBox.information(self, "Info", f"Processed with '{filter_mode}' filter.")
        self.offset_button.setEnabled(True)
        self.tempcomp_button.setEnabled(True)
        self.plot_button.setEnabled(True)
        for i in range(len(self.datasets)):
            self.update_list_entry(i)

    def run_offset(self):
        if self.current_index is None:
            return
        ds = self.datasets[self.current_index]
        if ds["proc_data"] is None:
            QMessageBox.warning(self, "Error", "Data must be processed first!")
            return
        ds["proc_data"] = apply_offset(ds["proc_data"])
        QMessageBox.information(self, "Info", "Offset applied.")
        self.show_preview()

    def run_tempcomp(self):
        if self.current_index is None:
            return
        ds = self.datasets[self.current_index]
        if ds["proc_data"] is None:
            QMessageBox.warning(self, "Error", "Data must be processed first!")
            return
        ds["proc_data"] = temperature_compensation(ds["proc_data"])
        QMessageBox.information(self, "Info", "Temperature compensation applied.")
        self.show_preview()

    # -------------------- Edit / Remove --------------------
    def edit_parameters(self):
        if self.current_index is None:
            return
        ds = self.datasets[self.current_index]
        dialog = GeometryDialog(self, ds["diameter"], ds["height"])
        if dialog.exec() == dialog.Accepted:
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
        index = self.dataset_list.currentRow()
        if index < 0:
            return
        reply = QMessageBox.question(
            self, "Remove Dataset",
            f"Really remove '{self.datasets[index]['filename']}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            del self.datasets[index]
            self.dataset_list.takeItem(index)
            self.show_preview()

    # -------------------- Plots --------------------
    def show_plots(self):
        processed = [ds for ds in self.datasets if ds["preprocessed"]]
        if not processed:
            QMessageBox.warning(self, "Error", "No processed datasets available.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        for ds in processed:
            data = ds["proc_data"]
            axes[0].plot(data["x"], data["y"], label=f"{ds['filename']}")
        axes[0].set_ylabel("Flow Stress [MPa]")
        axes[0].legend()
        axes[0].grid(True)

        has_temp = any("temperature" in ds["proc_data"].columns for ds in processed)
        if has_temp:
            for ds in processed:
                data = ds["proc_data"]
                if "temperature" in data.columns:
                    axes[1].plot(data["x"], data["temperature"], label=ds["filename"])
            axes[1].set_ylabel("Temperature [°C]")
            axes[1].set_xlabel("Strain [-]")
            axes[1].legend()
            axes[1].grid(True)
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        plt.show()

    # -------------------- Export --------------------
    def export_results(self):
        processed = [ds for ds in self.datasets if ds["preprocessed"]]
        if not processed:
            QMessageBox.warning(self, "Error", "No processed datasets for export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "Excel File (*.xlsx)")
        if not file_path:
            return

        combined = pd.DataFrame()
        for ds in processed:
            df = ds["proc_data"].copy()
            df.columns = [f"{ds['filename']}_x", f"{ds['filename']}_y"]
            combined = pd.concat([combined, df], axis=1)

        combined.to_excel(file_path, index=False)
        QMessageBox.information(self, "Success", f"Results saved: {file_path}")
