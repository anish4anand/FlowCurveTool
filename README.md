# Flow Curve Tool

A Python GUI application for processing, modeling, and exporting **flow-curve data from Gleeble tests**.

The tool converts raw force–displacement data into true stress–strain curves, applies robust noise filtering and failure trimming, fits global constitutive models, and supports advanced data exports for DEFORM and QForm simulations.

---

## Features

* **Data Management**: Load and manage multiple raw CSV datasets, with quick "Clear All" functionality.
* **Auto-Matching Geometry**: Import a "Probendaten" Excel/CSV file to automatically map specimen parameters (diameter, height) to loaded datasets.
* **Flow Curve Computation**: Compute true strain and true flow stress with robust preprocessing:
* startup trimming
* failure detection and cutoff
* optional elastic offset removal


* **Interactive Manual Offset**: Activate "Manual Offset Mode" directly on the plots to custom-trim data, supported by a 1-click "Undo" (Reset) button that restores data from memory backups.
* **Automated Metadata Parsing**: Automatically extracts Temperature and Strain Rate variables directly from `.csv` filenames (automatically handling "RT" as 20°C). Safely prompts the user for manual entry if filename tags are missing.
* **Constitutive Modeling & Extrapolation**:
* Fits the 9-parameter **Hensel-Spittel model** simultaneously across a global multi-condition matrix.
* Supports extrapolating flow curves beyond experimental bounds.


* **Tabbed UI**: Cleanly separates workflows across dedicated tabs: Visualizations (interactive cutting), Hensel-Spittel, and Extrapolations.
* **Noise Filtering**:
* **none** – unfiltered reference
* **mild** – Savitzky–Golay smoothing
* **strong** – Butterworth low-pass filtering



---

## Export

* **Standard Excel Export**
* Processed flow curves for a selected dataset or all datasets.
* *Automatic generation of native Excel scatter charts alongside the data.*


* **DEFORM Material Card (.key)**
* Generates a fully formatted, solver-ready `.key` text file containing the interpolated 3D matrix of stresses across strains, rates, and temperatures.


* **DEFORM / QForm Tables**
* Reduced, solver-friendly flow curves with geometric downsampling.
* Enforced monotonic hardening and non-negativity.
* *Exported as multi-sheet Excel workbooks with automatically generated native charts.*


* **QForm Parameter Clipboard**
* Instantly copies global Hensel-Spittel parameters directly to your OS clipboard (formatted to 4 decimal places with no scientific notation) for direct pasting into QForm.


* **Force–displacement Export**
* Processed force–displacement curves exported into multi-sheet Excel workbooks with auto-generated plots.



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