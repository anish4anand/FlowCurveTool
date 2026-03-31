from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QDialogButtonBox, QCheckBox, QDoubleSpinBox
)


class GeometryDialog(QDialog):
    def __init__(self, parent=None, diameter=None, height=None):
        super().__init__(parent)
        self.setWindowTitle("Specimen Geometry")

        self.diameter_input = QLineEdit()
        self.height_input = QLineEdit()

        if diameter is not None:
            self.diameter_input.setText(str(diameter))
        if height is not None:
            self.height_input.setText(str(height))

        form = QFormLayout()
        form.addRow("Diameter [mm]:", self.diameter_input)
        form.addRow("Height [mm]:", self.height_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_values(self):
        try:
            return float(self.diameter_input.text()), float(self.height_input.text())
        except ValueError:
            return None, None


class PreprocessOptionsDialog(QDialog):
    def __init__(self, parent=None, default_apply_offset=True, default_strain_increment=0.005):
        super().__init__(parent)
        self.setWindowTitle("Preprocess Options")

        form = QFormLayout()

        self.offset_checkbox = QCheckBox("Apply offset")
        self.offset_checkbox.setChecked(bool(default_apply_offset))
        form.addRow(self.offset_checkbox)

        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setDecimals(6)
        self.dx_spin.setRange(1e-6, 1.0)          # adjust upper bound if needed
        self.dx_spin.setSingleStep(0.001)
        self.dx_spin.setValue(float(default_strain_increment))
        self.dx_spin.setToolTip("Resample strain increment Δε (e.g., 0.005).")
        form.addRow("Strain increment Δε:", self.dx_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_apply_offset(self) -> bool:
        return self.offset_checkbox.isChecked()

    def get_strain_increment(self) -> float:
        return float(self.dx_spin.value())
