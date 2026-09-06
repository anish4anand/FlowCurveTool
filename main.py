import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import DataOptimizer

def main():
    app = QApplication(sys.argv)
    w = DataOptimizer()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()