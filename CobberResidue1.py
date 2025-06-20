# CobberResidue.py
# An application to visualize model performance by plotting
# predicted vs. actual values and their residuals.
#
# This is an updated version of the original MLDataVisualizer, rewritten
# in PyQt6 and Matplotlib for consistency with the CobberLearn suite.

import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QTextEdit, QSizePolicy
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PlotCanvas(FigureCanvas):
    """A helper class for a Matplotlib canvas."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)


class CobberResidue(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberResidue: Model Evaluation Explorer")
        self.setGeometry(100, 100, 1400, 800)

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # --- Top Controls ---
        top_layout = QHBoxLayout()
        self.load_button = QPushButton("Load CSV Dataset")
        self.load_button.clicked.connect(self.load_csv)
        self.info_label = QLabel("Please load a CSV dataset containing 'Actual' and 'Predicted' columns.")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.info_label, 1)
        self.layout.addLayout(top_layout)

        # --- Main Content Area (Plots and Metrics) ---
        main_content_layout = QHBoxLayout()
        self.layout.addLayout(main_content_layout)

        # Plots Layout
        plots_layout = QHBoxLayout()
        self.pv_a_plot = PlotCanvas(self, width=6, height=5)
        self.residuals_plot = PlotCanvas(self, width=6, height=5)
        plots_layout.addWidget(self.pv_a_plot)
        plots_layout.addWidget(self.residuals_plot)

        # Metrics Layout
        metrics_layout = QVBoxLayout()
        metrics_label = QLabel("<b>Model Evaluation Metrics:</b>")
        metrics_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumWidth(250)
        self.metrics_text.setMaximumWidth(350)
        metrics_layout.addWidget(metrics_label)
        metrics_layout.addWidget(self.metrics_text)

        main_content_layout.addLayout(plots_layout, 3)  # Give more space to plots
        main_content_layout.addLayout(metrics_layout, 1)

        # Initialize Data
        self.data = None
        self.clear_plots()  # Set initial placeholder text

    def load_csv(self):
        """Opens a file dialog to load and process a CSV file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV Dataset", "", "CSV Files (*.csv);;All Files (*)")
        if file_name:
            try:
                self.data = pd.read_csv(file_name)
                if 'Actual' not in self.data.columns or 'Predicted' not in self.data.columns:
                    self.info_label.setText(
                        "<font color='red'>Error: CSV must contain 'Actual' and 'Predicted' columns.</font>")
                    self.data = None
                    return
                self.info_label.setText(f"Loaded: {file_name.split('/')[-1]}")
                self.plot_data()
            except Exception as e:
                self.info_label.setText(f"<font color='red'>Error loading CSV: {e}</font>")
                self.data = None

    def plot_data(self):
        """Calculates metrics and updates plots based on loaded data."""
        if self.data is None:
            self.clear_plots()
            return

        actual = self.data['Actual'].values
        predicted = self.data['Predicted'].values
        residuals = predicted - actual

        # Compute Metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        # Update Metrics Display
        metrics_str = (
            f"Mean Absolute Error (MAE):\n<b>{mae:.3f}</b>\n\n"
            f"Mean Squared Error (MSE):\n<b>{mse:.3f}</b>\n\n"
            f"R-squared (R²):\n<b>{r2:.3f}</b>"
        )
        self.metrics_text.setHtml(metrics_str)

        # --- Plot Predicted vs Actual ---
        self.pv_a_plot.figure.clear()
        ax1 = self.pv_a_plot.figure.add_subplot(111)
        ax1.scatter(actual, predicted, alpha=0.7, label="Data Points")

        # Add 45-degree line
        lims = [
            min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
            max(ax1.get_xlim()[1], ax1.get_ylim()[1]),
        ]
        ax1.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Ideal Fit")
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title("Predicted vs. Actual Values")
        ax1.legend()
        ax1.grid(True)
        self.pv_a_plot.draw()

        # --- Plot Residuals ---
        self.residuals_plot.figure.clear()
        ax2 = self.residuals_plot.figure.add_subplot(111)
        ax2.scatter(actual, residuals, alpha=0.7, label="Residuals")
        ax2.axhline(0, color='r', linestyle='--', label="Zero Error")
        ax2.set_xlabel("Actual Values")
        ax2.set_ylabel("Residual (Predicted - Actual)")
        ax2.set_title("Residual Plot")
        ax2.legend()
        ax2.grid(True)
        self.residuals_plot.draw()

    def clear_plots(self):
        """Clears plots and shows placeholder text."""
        for canvas, title in [(self.pv_a_plot, "Predicted vs. Actual"), (self.residuals_plot, "Residuals")]:
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Load a CSV to generate plot",
                    ha='center', va='center', fontsize=12, alpha=0.5)
            ax.set_title(title)
            canvas.draw()
        self.metrics_text.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberResidue()
    window.show()
    sys.exit(app.exec())
