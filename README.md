# Diabetes Data Engineering & Analysis

## Overview
This project contains the data engineering (ETL) and exploratory data analysis (EDA) pipeline for the Diabetes Prediction Dataset.

## Project Structure
- `data/`: Contains raw and processed datasets.
- `src/`: Source code for ETL and Analysis.
    - `etl.py`: Validation, Cleaning, Encoding, Scaling, Splitting.
    - `analysis.py`: Summary stats, plots, and reports.
- `out/`: Analysis outputs (plots, correlations, risk groups).

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run ETL: `python src/etl.py`
3. Run Analysis: `python src/analysis.py`
