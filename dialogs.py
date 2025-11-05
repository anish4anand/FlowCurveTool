from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox

class GeometryDialog(QDialog):
    def __init__(self, parent=None, diameter=None, height=None):
        super().__init__(parent)
        self.setWindowTitle("Specimen Geometry")
        self.diameter_input = QLineEdit()
        self.height_input = QLineEdit()
        if diameter:
            self.diameter_input.setText(str(diameter))
        if height:
            self.height_input.setText(str(height))

        layout = QFormLayout()
        layout.addRow("Diameter [mm]:", self.diameter_input)
        layout.addRow("Height [mm]:", self.height_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_values(self):
        try:
            return float(self.diameter_input.text()), float(self.height_input.text())
        except ValueError:
            return None, None
