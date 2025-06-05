# CreditRisk-Personal

Members: Kimberly Wessler, Cristian Pena, Chris Dobbin, Christine Espiritu, JaJuan Graham

## Overview
**CreditRisk-Personal** is a machine learning classification project that predicts whether a personal loan will default or not based on borrower and loan features. Using a filtered subset of the Credit Risk Dataset from Kaggle, we focus specifically on loans with a **personal** intent to train a model that can assess creditworthiness and mitigate lending risks.

## Objective
The goal of this project is to:
- Filter the dataset to include only personal loans (`loan_intent == 'PERSONAL'`)
- Clean, preprocess, and engineer relevant features
- Train a supervised classification model to predict `loan_status` (0 = Non-default, 1 = Default)
- Achieve at least **75% classification accuracy**
- Optimize and evaluate model performance through iterative tuning

## Dataset
- **Source**: [Credit Risk Dataset by Lao Tse on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Subset Used**: Only rows where `loan_intent` is `'PERSONAL'`
- **Target Variable**: `loan_status` (binary classification)
- **Features**: loan amount, interest rate, credit score, annual income, and more

## Tools & Technologies
- **Languages**: Python
- **Libraries**: pandas, scikit-learn, matplotlib, TensorFlow/Keras
- **Data Storage**: SQLite
- **Version Control**: Git and GitHub
- **Notebook Environment**: Jupyter Notebook

## Project Workflow

### 1. Data Loading
- The raw dataset was loaded and filtered in a Jupyter Notebook environment.
- Only records with `loan_intent == 'PERSONAL'` were selected for modeling.

### 2. Data Cleaning & Preprocessing
- Removed or imputed missing values
- One-hot encoded categorical features
- Standardized and normalized numerical data
- Saved cleaned and preprocessed data in a sqlite databse and CSV file

### 3. Model Development
We implemented and evaluated the following models:
- **Random Forest Classifier**
- **Neural Network using Keras/TensorFlow**

### 4. Model Optimization
- Hyperparameter tuning was performed using:
  - Keras Tuner (`Hyperband`) for the Neural Network
- Performance metrics from each tuning iteration were logged and reviewed
- Final model performance was displayed at the end of each notebook

### 5. Evaluation
- **Random Forest** model achieved classification accuracy **>92.62%**
- Performance metrics included: accuracy, precision, recall, F1-score
- Confusion matrices and classification reports were generated

## Results
- **Best Model**: Random Forest (Accuracy > 92.62%)
- **Model Input**: Cleaned and standardized SQL-derived dataset or CSV
- **Documentation**: Iterative tuning results embedded in notebooks and reported in markdown/CSV format

## Notebooks & Files

| Filename                            | Description |
|-------------------------------------|-------------|
| `CreditRisk_clean.ipynb`            | Loads and cleans the filtered dataset. Includes SQL-based querying, preprocessing, encoding, and normalization steps. |
| `Random Forest-.ipynb`              | Trains a baseline Random Forest model. Evaluates accuracy, precision, and recall. No hyperparameter tuning applied. |
| `nn_model.ipynb`                    | Builds and tunes a Neural Network using Keras and Keras Tuner (Hyperband). Logs performance metrics across trials. |
| `CreditRisk_Personal_Model_Comparison.ipynb` | Compares Random Forest and Neural Network models using evaluation metrics like confusion matrix and classification report. |
| `credit_risk_dataset.csv`           | Raw dataset sourced from Kaggle, includes borrower and loan data. |
| `preprocessed_credit_risk.csv`      | Cleaned and Preprocessed data for modeling |
| `preprocessed_credit_risk_2.csv`    | Modified dataset removing columns with negative correlation |
| `credit_risk.db`                    | Sqlite database with cleaned and preprocessed data to be used for modeling |
| `credit_risk_visualize.ipynb`       | Visualizations and analysis of the original dataset |



## License
This project is intended for educational purposes only. Refer to the original dataset's licensing on Kaggle.

