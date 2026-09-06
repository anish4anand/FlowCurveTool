from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

class FlowPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        
        btn_layout = QVBoxLayout()
        self.cut_btn = QPushButton("Activate Manual Offset Mode (All Data)")
        self.cut_btn.setStyleSheet("background-color: #f0f0f0; font-weight: bold; padding: 6px;")
        
        # --- Make sure this Reset Button is declared ---
        self.reset_btn = QPushButton("Reset Offset (Undo)")
        self.reset_btn.setStyleSheet("background-color: #ffcccc; font-weight: bold; padding: 6px;")
        
        self.cut_btn.hide()
        self.reset_btn.hide()
        
        btn_layout.addWidget(self.cut_btn)
        btn_layout.addWidget(self.reset_btn)
        self._main_layout.addLayout(btn_layout)

        self.canvas = None
        self.toolbar = None
        self._is_cutting = False
        self._hover_line = None
        self.cut_callback = None
        self.reset_callback = None
        self.main_ax = None
        
        self.cut_btn.clicked.connect(self._activate_cutting_mode)
        
        # Safely connect the reset button if it's clicked
        self.reset_btn.clicked.connect(self._on_reset_click)

    def plot_interactive_figure(self, fig, cut_callback, reset_callback=None):
        self._clear_layout()
        self.cut_callback = cut_callback
        self.reset_callback = reset_callback
        
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self._main_layout.insertWidget(0, self.toolbar)
        self._main_layout.insertWidget(1, self.canvas)
        
        self.main_ax = fig.axes[0] if fig.axes else None
        
        # --- CONDITIONAL BUTTON VISIBILITY ---
        if self.cut_callback:
            self.cut_btn.show()
            self.cut_btn.setText("Activate Manual Offset Mode (All Data)")
        else:
            self.cut_btn.hide()
            
        if self.reset_callback:
            self.reset_btn.show()
        else:
            self.reset_btn.hide()
        # -------------------------------------
        
        self._is_cutting = False
        
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.canvas.draw_idle()

    def _activate_cutting_mode(self):
        self._is_cutting = True
        self.cut_btn.setText("Mode Active: Hover & Click on Plot Axis!")
        if self.main_ax:
            self.main_ax.set_title("Click to set new origin for ALL curves", color='red', fontweight='bold')
            self.canvas.draw_idle()

    def _on_mouse_move(self, event):
        if not self._is_cutting or not self.main_ax: return
        if event.inaxes != self.main_ax:
            if self._hover_line:
                self._hover_line.set_visible(False)
                self.canvas.draw_idle()
            return
        
        if self._hover_line is None:
            self._hover_line = self.main_ax.axvline(x=event.xdata, color='red', linestyle='--', linewidth=1.5)
        else:
            self._hover_line.set_xdata([event.xdata, event.xdata])
            self._hover_line.set_visible(True)
        self.canvas.draw_idle()

    def _on_plot_click(self, event):
        if not self._is_cutting or event.inaxes != self.main_ax: return
        x_val = event.xdata
        self._is_cutting = False
        self.cut_btn.setText("Applying Offset...")
        if self.cut_callback:
            self.cut_callback(x_val)
            
    def _on_reset_click(self):
        if self.reset_callback:
            self.reset_callback()

    def _clear_layout(self):
        if self.toolbar:
            self._main_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
        if self.canvas:
            self._main_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
        self._hover_line = None
        self.main_ax = None