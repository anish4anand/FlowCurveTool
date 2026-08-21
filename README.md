
# Flow Curve Tool

A Python GUI application for processing and exporting **flow-curve data from Gleeble tests**.

The tool converts raw force–displacement data into true stress–strain curves, applies robust noise filtering and failure trimming, and supports data exports for DEFORM / QForm simulations.

---

## Features

* **Data Management**: Load and manage multiple raw CSV datasets, with quick "Clear All" functionality.
* **Auto-Matching Geometry**: Import a "Probendaten" Excel/CSV file to automatically map specimen parameters (diameter, height) to loaded datasets.
* **Flow Curve Computation**: Compute true strain and true flow stress with robust preprocessing:
* startup trimming
* failure detection and cutoff
* optional elastic offset removal


* **Auto-Averaging**: Automatically group and average replicate datasets based on filename conventions.
* **Noise Filtering**:
* **none** – unfiltered reference
* **mild** – Savitzky–Golay smoothing
* **strong** – Butterworth low-pass filtering


* **Interactive Plotting**: View stress–strain and temperature curves with dynamic, auto-scaling legends designed to handle large dataset batches.

---

## Export

* **Standard Excel export**
* Processed flow curves for a selected dataset or all datasets.
* *Automatic generation of native Excel scatter charts alongside the data.*


* **DEFORM / QForm export**
* Reduced, solver-friendly flow curves.
* Enforced monotonic hardening and non-negativity.
* *Automatic generation of native Excel charts.*


* **Force–displacement export**
* Processed force–displacement curves (selected dataset only).



---

## Installation

Navigate to the folder directory:

```bash
cd path/to/flow-curve-tool

```

Then, install the required libraries:

```bash
pip install -r requirements.txt

```

---

## Notes

* Raw data is preserved and can be reprocessed at any time.
* Solver-specific exports apply additional constraints for numerical stability.

**Author:** Anish Anand (University of Stuttgart: st192681@stud.uni-stuttgart.de)