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
- **Libraries**: pandas, scikit-learn, matplotlib, seaborn, xgboost
- **Data Storage**: SQLite or Apache Spark (for querying the filtered dataset)
- **Version Control**: Git and GitHub
- **Notebook Environment**: Jupyter Notebook or .py scripts

## Project Tasks
1. **Data Loading & SQL Integration**  
   Load the raw dataset into a SQL or Spark environment, filter for personal loans, and export for preprocessing.

2. **Data Cleaning & Preprocessing**  
   Handle missing values, encode categorical variables, and normalize numeric features.

3. **Model Development**  
   Train baseline models such as Logistic Regression and Random Forest. Evaluate using accuracy, precision, recall, and F1-score.

4. **Model Optimization**  
   Use hyperparameter tuning with GridSearchCV or RandomizedSearchCV. Document results in CSV format to track improvements.

5. **Results & Evaluation**  
   Select the best-performing model and visualize key metrics (confusion matrix, ROC curve, etc.). Ensure model meets or exceeds 75% accuracy.

6. **Documentation & Presentation**  
   Maintain a clean GitHub repo with a polished README. Prepare and deliver a group presentation summarizing findings and process.

## Deliverables
- Cleaned dataset (filtered for personal loans)
- Python scripts or notebooks for each pipeline stage
- SQL or Spark queries
- Final trained model and evaluation metrics
- Optimization results log (CSV or embedded)
- Complete GitHub repository with organized structure
- Group presentation slides

## License
This project is intended for educational purposes only. Refer to the original dataset's licensing on Kaggle.

