from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox, QTabWidget,
    QTableWidget, QPushButton, QComboBox, QLabel, QListWidget, QCheckBox
)
from PySide6.QtCore import Qt
from ui.plot_widget import FlowPlotWidget

class UiBuilder:
    @staticmethod
    def build(main_window):
        central_widget = QWidget(main_window)
        main_window.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Panel (Tabbed Interface) ---
        main_window.tabs = QTabWidget()
        
        # Tab 1: Data Table
        tab_data = QWidget()
        tab_data_layout = QVBoxLayout(tab_data)
        tab_data_layout.setContentsMargins(0, 0, 0, 0)
        
        main_window.preview_table = QTableWidget()
        main_window.preview_table.setRowCount(0)
        main_window.preview_table.setColumnCount(0)
        tab_data_layout.addWidget(main_window.preview_table)
        main_window.tabs.addTab(tab_data, "Data Table")
        
        # Tab 2: Visualizations (Existing)
        tab_viz = QWidget()
        tab_viz_layout = QVBoxLayout(tab_viz)
        tab_viz_layout.setContentsMargins(0, 0, 0, 0)
        main_window.plot_widget = FlowPlotWidget()
        tab_viz_layout.addWidget(main_window.plot_widget)
        main_window.tabs.addTab(tab_viz, "Visualizations")

        # Tab 3: Hensel-Spittel (New)
        tab_hs = QWidget()
        tab_hs_layout = QVBoxLayout(tab_hs)
        tab_hs_layout.setContentsMargins(0, 0, 0, 0)
        main_window.hs_plot_widget = FlowPlotWidget()
        tab_hs_layout.addWidget(main_window.hs_plot_widget)
        main_window.tabs.addTab(tab_hs, "Hensel-Spittel")

        # Tab 4: Extrapolations (New)
        tab_extra = QWidget()
        tab_extra_layout = QVBoxLayout(tab_extra)
        tab_extra_layout.setContentsMargins(0, 0, 0, 0)
        main_window.extra_plot_widget = FlowPlotWidget()
        tab_extra_layout.addWidget(main_window.extra_plot_widget)
        main_window.tabs.addTab(tab_extra, "Extrapolations")
        
        splitter.addWidget(main_window.tabs)

        # --- Right Panel (Sidebar Controls) ---
        right_panel = QWidget()
        sidebar = QVBoxLayout(right_panel)
        sidebar.setContentsMargins(0, 0, 0, 0)

        # 1. Data Management
        data_group = QGroupBox("1. Data Management")
        data_layout = QVBoxLayout()
        main_window.dataset_list = QListWidget()
        data_layout.addWidget(main_window.dataset_list)
        main_window.load_btn = QPushButton("Add CSV + Parameters")
        data_layout.addWidget(main_window.load_btn)
        main_window.edit_params_button = QPushButton("Edit Specimen Parameters")
        main_window.edit_params_button.setEnabled(False)
        data_layout.addWidget(main_window.edit_params_button)
        main_window.remove_button = QPushButton("Remove Dataset from List")
        main_window.remove_button.setEnabled(False)
        data_layout.addWidget(main_window.remove_button)
        main_window.clear_all_btn = QPushButton("Clear All Datasets")
        main_window.clear_all_btn.setEnabled(False)
        data_layout.addWidget(main_window.clear_all_btn)
        data_group.setLayout(data_layout)
        sidebar.addWidget(data_group)

        # 2. Processing & Models
        proc_group = QGroupBox("2. Processing & Models")
        proc_layout = QVBoxLayout()
        main_window.chk_process_all = QCheckBox("Process all loaded datasets")
        main_window.chk_process_all.setChecked(False)
        proc_layout.addWidget(main_window.chk_process_all)
        main_window.preprocess_btn = QPushButton("Process Raw Data")
        main_window.preprocess_btn.setEnabled(False)
        proc_layout.addWidget(main_window.preprocess_btn)
        main_window.extrapolate_btn = QPushButton("Extrapolate Models")
        main_window.extrapolate_btn.setEnabled(False)
        proc_layout.addWidget(main_window.extrapolate_btn)
        main_window.hensel_btn = QPushButton("Fit Hensel-Spittel Model")
        main_window.hensel_btn.setEnabled(False)
        proc_layout.addWidget(main_window.hensel_btn)
        proc_group.setLayout(proc_layout)
        sidebar.addWidget(proc_group)

        # 3. Visualization
        plot_group = QGroupBox("3. Visualization")
        plot_layout = QVBoxLayout()
        filter_layout = QHBoxLayout()
        main_window.chk_none = QCheckBox("Unfiltered")
        main_window.chk_mild = QCheckBox("Mild")
        main_window.chk_strong = QCheckBox("Strong")
        main_window.chk_none.setChecked(True)
        main_window.chk_mild.setChecked(True)
        main_window.chk_strong.setChecked(False)
        filter_layout.addWidget(main_window.chk_none)
        filter_layout.addWidget(main_window.chk_mild)
        filter_layout.addWidget(main_window.chk_strong)
        plot_layout.addLayout(filter_layout)
        main_window.plot_btn = QPushButton("Render Plots")
        main_window.plot_btn.setEnabled(False)
        plot_layout.addWidget(main_window.plot_btn)
        plot_group.setLayout(plot_layout)
        sidebar.addWidget(plot_group)

        # 4. Exporting
        export_group = QGroupBox("4. Export Options")
        export_layout = QVBoxLayout()
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("Export filter:"))
        main_window.export_filter_combo = QComboBox()
        main_window.export_filter_combo.addItems(["none", "mild", "strong"])
        combo_layout.addWidget(main_window.export_filter_combo)
        export_layout.addLayout(combo_layout)
        main_window.export_btn = QPushButton("Export (Excel)")
        main_window.export_btn.setEnabled(False)
        export_layout.addWidget(main_window.export_btn)
        main_window.force_disp_export_btn = QPushButton("Export Force–Displacement")
        main_window.force_disp_export_btn.setEnabled(False)
        export_layout.addWidget(main_window.force_disp_export_btn)
        main_window.deform_export_btn = QPushButton("DEFORM / QForm Export")
        main_window.deform_export_btn.setEnabled(False)
        export_layout.addWidget(main_window.deform_export_btn)
        main_window.deform_key_export_btn = QPushButton("DEFORM 3D Keyword Export (.key)")
        main_window.deform_key_export_btn.setEnabled(False)
        export_layout.addWidget(main_window.deform_key_export_btn)
        export_group.setLayout(export_layout)
        sidebar.addWidget(export_group)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)