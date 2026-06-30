from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                               QLabel, QLineEdit, QCheckBox, 
                               QDialogButtonBox)

class GeometryDialog(QDialog):
    def __init__(self, parent=None, default_d=10.0, default_h=15.0):
        super().__init__(parent)
        self.setWindowTitle("Specimen Geometry")
        layout = QVBoxLayout(self)
        
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("Diameter (mm):"))
        self.le_d = QLineEdit(str(default_d))
        hbox1.addWidget(self.le_d)
        layout.addLayout(hbox1)
        
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("Height (mm):"))
        self.le_h = QLineEdit(str(default_h))
        hbox2.addWidget(self.le_h)
        layout.addLayout(hbox2)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        
    def get_values(self):
        try:
            return float(self.le_d.text()), float(self.le_h.text())
        except ValueError:
            return None, None


class PreprocessOptionsDialog(QDialog):
    def __init__(self, parent=None, default_apply_offset=False, default_strain_increment=0.005):
        super().__init__(parent)
        self.setWindowTitle("Preprocessing Options")
        layout = QVBoxLayout(self)
        
        self.chk_offset = QCheckBox("Apply elastic offset removal")
        self.chk_offset.setChecked(default_apply_offset)
        layout.addWidget(self.chk_offset)
        
        # --- NEW: AUTO-AVERAGE CHECKBOX ---
        self.chk_average = QCheckBox("Auto-average replicates after processing")
        self.chk_average.setChecked(False) # Defaults to unchecked
        layout.addWidget(self.chk_average)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Strain Increment:"))
        self.le_dx = QLineEdit(str(default_strain_increment))
        hbox.addWidget(self.le_dx)
        layout.addLayout(hbox)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        
    def get_apply_offset(self):
        return self.chk_offset.isChecked()

    def get_auto_average(self):
        return self.chk_average.isChecked()

    def get_strain_increment(self):
        try:
            return float(self.le_dx.text())
        except ValueError:
            return 0.005