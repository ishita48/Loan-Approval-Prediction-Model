# 🏦 Loan Approval Prediction Machine Learning Project 🏦

## Overview

Welcome to the Loan Approval Prediction Machine Learning Project! This project aims to develop and evaluate machine learning models for predicting loan approval status using a dataset from Analytics Vidhya. The goal is to create accurate models that determine whether a loan will be approved based on various applicant features.

For a detailed exploration of the project, view the Colab notebook [here](https://colab.research.google.com/github/ishita48/Loan-Approval-Prediction-Model/blob/main/loan_approval_prediction_model.ipynb).

## Dataset 📊

The dataset includes essential features such as:

- **Loan_ID**: Unique identifier for the loan
- **Gender**: Gender of the applicant
- **Married**: Marital status of the applicant
- **Dependents**: Number of dependents
- **Education**: Education level of the applicant
- **Self_Employed**: Employment status (self-employed or not)
- **ApplicantIncome**: Income of the applicant
- **CoapplicantIncome**: Income of the coapplicant
- **LoanAmount**: Loan amount applied for
- **Loan_Amount_Term**: Term of the loan in months
- **Credit_History**: Credit history meets guidelines (1: meets, 0: does not meet)
- **Property_Area**: Area type of the property (Urban/Semiurban/Rural)
- **Loan_Status**: Loan approval status (Y/N)

These features are crucial for training models to predict whether a loan will be approved.

## Technologies 🛠️

This project harnesses the power of several technologies:

- **Python** programming language
- **Jupyter Notebook** for interactive development
- **Pandas** and **NumPy** for data manipulation
- **Matplotlib** and **Seaborn** for data visualization
- **Scikit-Learn** for machine learning modeling
- **TensorFlow** and **Keras** for building neural networks
- **Google Colab** for collaborative development
- **GitHub** for version control and collaboration

## Preprocessing 📝

The dataset undergoes preprocessing steps to ensure optimal model performance:

- **Data Loading**: Importing and loading data using Pandas.
- **Handling Missing Values**: Imputing or dropping missing data appropriately.
- **Encoding Categorical Variables**: Converting categorical features to numerical values using LabelEncoder and OneHotEncoder.
- **Normalization**: Scaling numerical features to a standard range using StandardScaler.
- **Train-Test Split**: Dividing data into training and validation sets using `train_test_split`.

## Models 🤖

A diverse set of models are evaluated:

- **Logistic Regression**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**
- **Small Neural Network**
- **Medium Neural Network**
- **Large Neural Network**

Each model is evaluated using metrics like accuracy, precision, recall, and F1-score.

# Conclusion and Analysis Report 📈

## Data Exploration and Preprocessing

The dataset comprising 614 entries and 13 features underwent rigorous preprocessing to ensure data quality. Missing values were handled by imputing `LoanAmount` with the mean and `Credit_History` with the median, followed by dropping rows with any remaining missing data. This meticulous approach resulted in a cleaned dataset of 542 entries, ready for in-depth analysis.

## Exploratory Data Analysis (EDA)

### Univariate Analysis
Exploring numerical features like `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, and `Loan_Amount_Term` revealed skewed distributions, suggesting the presence of outliers in income-related variables. Categorical features such as `Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`, and `Loan_Status` were also scrutinized, with `Loan_Status` indicating an imbalance favoring approved loans (`Y`).

### Bivariate Analysis
The relationship between `Loan_Status` and other variables was extensively examined. Notably, applicants with a `Credit_History` were significantly more likely to have their loans approved, underscoring its pivotal role in lending decisions. Factors like `Married` status and `Education` level also exhibited notable correlations with loan approval rates.

### Multivariate Analysis
A thorough assessment of feature interactions via correlation analysis highlighted moderate relationships among numerical variables without significant multicollinearity concerns, ensuring robustness in subsequent modeling efforts.

## Model Building and Evaluation

Eight distinct classification models were evaluated to predict `Loan_Status`:
- **Logistic Regression**, **Random Forest**, **K-Nearest Neighbors**, **Gradient Boosting**, **Support Vector Machine (SVM)**, and three variants of **Neural Networks** (Small, Medium, Large).

### Model Performance
Following rigorous evaluation, **Logistic Regression** emerged as the top-performing model, achieving an impressive accuracy of approximately 79% on the test set. While other models like **Random Forest** and **Gradient Boosting** exhibited competitive performances, Logistic Regression consistently demonstrated superior predictive capability.

### Insights and Recommendations
The analysis underscored the critical influence of `Credit_History` on loan approval outcomes, aligning with industry standards where creditworthiness plays a pivotal role. The efficacy of Logistic Regression suggests its suitability for this dataset, balancing simplicity with robust predictive accuracy.

## Conclusion

In conclusion, this study provides valuable insights into the factors influencing loan approval decisions. The prominence of `Credit_History` as a determining factor reaffirms its significance in assessing borrower risk. Logistic Regression emerges as an optimal choice for predicting loan status, offering stakeholders a reliable tool for enhancing lending decisions.

Future endeavors could focus on advanced feature engineering techniques or ensemble methods to further refine predictive models. This comprehensive analysis equips stakeholders in the lending industry with actionable insights, emphasizing the importance of data-driven approaches in optimizing loan approval processes.


## Usage 🚀

1. **Data Preparation**: Load and preprocess the loan dataset.
2. **Model Training**: Train models using various algorithms and hyperparameters.
3. **Evaluation**: Assess model performance with validation metrics and visualizations.
4. **Contribution**: Contributions are welcome via pull requests. Fork the repository, create a branch, commit changes, and submit a pull request.

## Visualizations 📊

Visualize model performance with:

- **Confusion Matrices**: Illustrating true positives, false positives, true negatives, and false negatives.
- **Classification Reports**: Detailed metrics for precision, recall, and F1-score.
- **Training Curves**: Loss and accuracy curves for neural networks.

## Installation 🔧

Ensure Python 3.x and required libraries are installed:

```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn
