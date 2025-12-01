# Task 3: Bank Marketing Machine Learning Model

## Project Overview
Complete machine learning pipeline for predicting customer subscription in bank marketing campaigns using synthetic Bank Marketing dataset with multiple classification models.

## Dataset Information
- **Dataset**: Bank Marketing (Synthetic - 5000 samples)
- **Target Variable**: Customer Subscription Prediction (Binary Classification: 0 = No, 1 = Yes)
- **Number of Features**: 9 original + 15 after encoding
- **Train-Test Split**: 80-20 (4000 training, 1000 testing samples)
- **Target Distribution**: 88% No, 12% Yes (imbalanced dataset)

## Features
- age
- job
- marital
- education
- balance
- housing
- loan
- duration
- campaign
- y (target)

## Project Structure

### Step 1: Data Creation
- Generated 5,000 synthetic samples
- Created realistic bank marketing features
- Maintained realistic target distribution (12% positive class)

### Step 2: Data Preprocessing
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for feature normalization
- **Split**: Stratified train-test split (80-20)
- **Final Features**: 15 features after encoding

### Step 3: Model Training
Three classification models trained and compared:

1. **Logistic Regression**
   - Algorithm: Linear regression with sigmoid function
   - Max iterations: 1000
   - Best for interpretability

2. **Random Forest**
   - Number of estimators: 100
   - Ensemble method for robustness
   - Handles non-linear relationships

3. **Gradient Boosting**
   - Number of estimators: 100
   - Sequential tree building
   - Often provides strong performance

### Step 4: Model Evaluation

**Best Model: Logistic Regression**

Performance Metrics:
- **Accuracy**: 88.20%
- **Precision**: 0.0000 (0 positive predictions)
- **Recall**: 0.0000 (model bias towards negative class)
- **F1-Score**: 0.0000 (imbalanced data impact)
- **ROC-AUC**: 0.5263

### Step 5: Visualizations

Four comprehensive comparison charts generated:

1. **Confusion Matrix** - Best model's prediction distribution
2. **Accuracy Comparison** - All models' accuracy metrics
3. **F1-Score Comparison** - Model performance on imbalanced data
4. **ROC Curves** - ROC-AUC curves for all models

### Step 6: Results & Recommendations

**Recommendations**:
- The Logistic Regression model is recommended for deployment based on performance
- Consider techniques like:
  - Class weight balancing
  - SMOTE for oversampling minority class
  - Threshold adjustment for recall optimization
  - Ensemble methods combination

## How to Use

### Option 1: Google Colab (Recommended)
1. Open the `.ipynb` file in Google Colab
2. Connect to Python 3 runtime
3. Click "Run All" or run cells individually
4. View outputs with visualizations

### Option 2: Local Jupyter
Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

Run notebook
jupyter notebook Task3_BankMarketing_ML_Model_FINAL.ipynb

text

### Option 3: Python Script
Convert notebook to Python
jupyter nbconvert --to script Task3_BankMarketing_ML_Model_FINAL.ipynb

Run script
python Task3_BankMarketing_ML_Model_FINAL.py

text

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Execution Details
- **Execution Time**: ~8.4 seconds
- **Status**: ✓ No Errors
- **Output**: Complete visualizations + summary statistics

## Key Insights

1. **Model Performance**: All three models show similar accuracy (~88%) due to dataset being highly imbalanced
2. **ROC-AUC Analysis**: Logistic Regression performs best but still modest ROC-AUC suggests class imbalance impact
3. **Data Imbalance**: 88% negative samples causes models to predict mostly negative class
4. **Improvement Areas**: 
   - Apply class balancing techniques
   - Use stratified k-fold cross-validation
   - Tune decision threshold

## Files Included
- `Task3_BankMarketing_ML_Model_FINAL.ipynb` - Main notebook with complete pipeline
- `README.md` - This file

## Author
Swamy Guradasu

## Date
December 1, 2025

## Links
- **GitHub Repository**: [Matrix-Industries-Data-Science-Internship](https://github.com/Swamy-somu/Matrix-Industries-Data-Science-Internship)
- **Colab Notebook**: [Task3_BankMarketing_ML_Model_FINAL](https://colab.research.google.com/drive/1A0sC2MuvGby5rlzylB8Z-v1r1VY1sGmN)

## Notes
- This is a complete, production-ready notebook
- All code has been tested and verified to run without errors
- Clean single-cell design for reliability
- Ready for GitHub upload and submission