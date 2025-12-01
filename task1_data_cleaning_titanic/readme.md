# Data Science Internship - Matrix Industries

## Task 1: Data Cleaning and Preprocessing

### Overview
This repository contains the Data Cleaning and Preprocessing task for the Matrix Industries Data Science Internship.

### Dataset
- **Name:** Titanic Dataset
- **Size:** 891 rows, 12 columns
- **Source:** Kaggle (loaded from GitHub)

### What Was Done
1. ✓ Loaded and previewed data
2. ✓ Analyzed missing values (Age: 177, Embarked: 2, Cabin: 687)
3. ✓ Handled missing values using median and mode imputation
4. ✓ Removed sparse columns (Cabin with 77% missing)
5. ✓ Checked for and removed duplicates
6. ✓ Fixed data inconsistencies in categorical columns
7. ✓ Detected and documented outliers using IQR method
8. ✓ Encoded categorical variables (Sex: 1/0)

### Final Output
- **Final Dataset Shape:** 891 rows, 11 columns
- **Missing Values:** 0
- **Output File:** titanic_cleaned.csv

### Technologies Used
- Python 3
- pandas (Data manipulation)
- numpy (Numerical operations)
- matplotlib & seaborn (Visualization)
- Google Colab (Development environment)

### How to Run
1. Open the notebook in Google Colab or Jupyter
2. Run all cells in order
3. The cleaned dataset will be saved as `titanic_cleaned.csv`

### Files
- `task1_data_cleaning_titanic.ipynb` - Main notebook with all code and outputs
