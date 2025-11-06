# 🌲 California Wildfire ML Analysis & Forecasting
# Streamlit Link - 
https://22070521101sejalml-mini-project-gktsnxpansm9m46qgapmvy.streamlit.app/
## Overview

This repository contains a comprehensive machine learning analysis of the California Historic Fire Perimeters dataset. The project explores various supervised, unsupervised, and time series algorithms to model fire characteristics, predict fire outcomes, and forecast long-term trends in burned acreage.

-----

## 💾 Dataset

The primary dataset used is:

  * **File:** `California_Historic_Fire_Perimeters_-6236829869961296710.csv`
  * **Source:** Historical fire data for California.
  * **Key Features Used:** `Year`, `Shape__Area`, `Shape__Length`, `GIS Calculated Acres` (Target for Regression/Forecasting), and `Cause` (Target for Classification).

-----

## ✨ Project Goals

The analysis addresses three main types of machine learning tasks:

### 1\. Supervised Learning

  * **Regression:** Predict the **size of a fire** (`GIS Calculated Acres`) based on its year and geometric properties.
  * **Classification:** Predict the **cause of a fire** (`Cause` code) based on its characteristics.

### 2\. Unsupervised Learning

  * **Clustering (K-Means):** Group fires into similar clusters based on their spatial and temporal features to identify underlying patterns in fire events.

### 3\. Time Series Forecasting

  * **Forecasting (ARIMA):** Aggregate fire size to create an annual time series and forecast **total acres burned** in future years.

-----

## 🛠️ Algorithms Applied

Separate Python scripts (or sections within a single notebook) are used for each algorithm.

| Task | Algorithm | Code Snippet/File |
| :--- | :--- | :--- |
| **Regression** | Linear Regression | `code_regression_linear.py` |
| **Regression** | Random Forest Regressor | `code_regression_rf.py` |
| **Classification** | Logistic Regression | `code_classification_logreg.py` |
| **Classification** | Decision Tree Classifier | `code_classification_dt.py` |
| **Unsupervised** | K-Means Clustering | `code_clustering_kmeans.py` |
| **Time Series** | ARIMA | `code_time_series_arima.py` |

-----

## 📈 Time Series Analysis (ARIMA) Details

The **ARIMA (AutoRegressive Integrated Moving Average)** model was applied to the aggregated annual total burned acreage.

  * **Goal:** Forecast the total annual acres burned.
  * **Model Used:** $\text{ARIMA}(5, 1, 0)$
  * **Result Example (Test Set):**
    | Metric | Value |
    | :--- | :--- |
    | **Root Mean Squared Error (RMSE)** | $\approx 1,432,185$ Acres |

This high RMSE highlights the challenge of forecasting extreme events like massive wildfire years using only historical acreage data, as the system is highly sensitive to external variables (e.g., climate, drought).

-----

## 🚀 Setup and Prerequisites

To run the analysis, you will need Python and the following libraries:

```bash
pip install pandas numpy scikit-learn statsmodels
```

### Required Files

  * `California_Historic_Fire_Perimeters_-6236829869961296710.csv` (The raw data file)

-----

## 📂 Repository Structure

```
.
├── California_Historic_Fire_Perimeters_...csv  # The raw data file
├── code_regression_linear.py                  # Code for Linear Regression
├── code_time_series_arima.py                  # Code for ARIMA Forecasting
├── ... (other ML algorithm files)
└── README.md                                  # This file
```

### Usage

1.  Clone this repository.
2.  Ensure you have the prerequisite libraries installed.
3.  Place the CSV data file in the root directory.
4.  Run any of the Python scripts (e.g., `python code_time_series_arima.py`) to execute the specific model and view the results.
