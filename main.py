import sys
from PySide6.QtWidgets import QApplication
from gui_mainwindow import DataOptimizer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataOptimizer()
    window.show()
    sys.exit(app.exec())
