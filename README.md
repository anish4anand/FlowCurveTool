# Flow Curve Tool

A small Python application for processing and visualizing flow-curve data from compression tests (for example, Gleeble experiments).  
It loads raw CSV files, calculates true stress–strain curves, smooths noisy data, and automatically trims the region after material failure.  
The goal is to have a simple tool that can handle multiple test files quickly and consistently.

---

## Features
- Load one or more CSV files and specify specimen geometry (diameter and height)
- Calculate true strain and flow stress automatically
- Apply different noise filters  
  - **none** – raw data  
  - **light** – Savitzky–Golay smoothing for mildly noisy data  
  - **strong** – Butterworth low-pass filter for heavily noisy data
- Detect and cut off the sharp stress drop after failure
- Optional temperature interpolation from the TC1 channel
- Plot stress–strain and temperature–strain curves
- Export processed data to Excel

---

## How it works
1. Reads jaw and force signals  
2. Applies machine-compliance correction  
3. Computes true strain and flow stress  
4. Interpolates to a uniform strain grid  
5. Applies the selected noise filter  
6. Detects the first large negative slope (failure) and trims the data  
7. Plots and exports the cleaned curves

---

## Interface overview
1. **Add CSV + Parameters** – select a file and enter geometry  
2. **Noise Filter** – choose none / light / strong  
3. **Process Raw Data** – run preprocessing  
4. **Apply Offset / Temperature Compensation** – optional corrections  
5. **Plots** – view results  
6. **Export Results** – save processed curves to Excel

---

## Folder structure
FlowCurveTool/
├── main.py
├── gui_mainwindow.py
├── dialogs.py
│
├── processing/
│ ├── preprocess.py
│ ├── filters.py
│ └── utils.py
│
└── results/

---

## Installation
```bash
pip install -r requirements.txt

