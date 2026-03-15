# NBA Salary Prediction – End-to-End Machine Learning Analysis

This project explores how well NBA player salaries can be estimated from season-level player performance data using an end-to-end data science workflow.

The analysis includes exploratory data analysis, feature engineering, classical machine learning models, neural network experiments, model evaluation, and interpretability techniques.

The objective is not only to predict salary but also to understand which player characteristics are most associated with salary and how such a model could support decision-making in an NBA front office context.

---

# Business Objective

An NBA finance team wants to better understand player salaries when making decisions about:

- re-signing current players
- evaluating potential acquisitions
- preparing salary negotiation ranges

The goal is to develop a **supervised regression model** that predicts player salary from seasonal performance metrics and contextual variables.

The model should be interpreted as a **decision-support tool**, not a fully automated contract decision system.

---

# Dataset

Source: Kaggle – NBA Salary Prediction Dataset  
https://www.kaggle.com/koki25ando/nba-salary-prediction-using-multiple-regression

Each observation represents a player season containing:

- box score statistics
- contextual variables such as team and position
- salary information

Dataset summary:

- **683 players total**
- **131 rows with missing statistics**
- **552 usable player seasons for modelling**

---

# Project Workflow

## 1. System Definition

The project is framed as a **salary regression problem**.

Input:
- player season statistics
- contextual variables

Output:
- predicted player salary

Evaluation metrics:
- MAE (Mean Absolute Error) in USD
- RMSE in USD
- R² on log-transformed salary

---

## 2. Exploratory Data Analysis

Exploration included:

- missing value analysis
- salary distribution analysis
- log transformation evaluation
- correlation analysis
- scatter plots of salary vs performance variables
- boxplots by player position
- summary statistics by team and position

Key observations:

- Salary distribution is **strongly right-skewed**
- Salary correlates strongly with **usage and availability variables**
  - points
  - minutes played
  - games started
- Player **position contributes additional signal**

Because of skewness, the target variable was transformed using:

```bash
log1p(Salary)
```

---

## 3. Feature Engineering

Additional derived features were created, including:

- points per game
- assists per game
- rebounds per game
- 2-point percentage
- 3-point percentage

Categorical variables such as **position** were encoded through a preprocessing pipeline.

---

## 4. Machine Learning Models

Multiple scikit-learn regressors were evaluated:

- Dummy Regressor (baseline)
- Linear Regression
- Ridge
- Lasso
- ElasticNet
- KNN Regressor
- Support Vector Regression
- Random Forest
- Gradient Boosting
- HistGradientBoosting

Models were compared using both:

- hold-out test evaluation
- cross-validation

---

## 5. Neural Network Experiments

Several multilayer perceptron (MLP) configurations were tested using TensorFlow/Keras.

Experiments varied:

- activation functions (`relu`, `elu`, `tanh`)
- optimizers (`adam`, `rmsprop`, `sgd`)
- learning rates
- dropout regularization
- robust training callbacks

Neural networks did not outperform the best classical model on this dataset.

---

## 6. Model Evaluation

The final model performance on the held-out test set:

- **MAE ≈ 3.53M USD**
- **RMSE ≈ 5.13M USD**
- **R² (log scale) ≈ 0.615**

This indicates that the model provides a useful **reference estimate**, but with significant uncertainty.

---

## 7. Model Interpretability

Several interpretability techniques were used:

- permutation feature importance
- Lasso coefficient analysis
- optional SHAP value analysis for RandomForest

These approaches help identify which player statistics most influence salary predictions.

---

# Final Model Recommendation

The final model selected is a **RandomForest Regressor**, chosen based on strong cross-validated performance and robust behaviour on tabular data.

Reasons for selection:

- strong predictive performance
- ability to model nonlinear relationships
- stable performance across validation folds
- compatibility with feature importance analysis

A **Lasso regression model** was also used as an interpretable baseline to understand directional relationships between variables and salary.

---

# Limitations

The dataset does not include several factors that influence NBA salaries in reality:

- contract rules (rookie scale, max contracts)
- salary cap context
- player age and experience
- injury history
- advanced impact metrics
- negotiation and market dynamics

Therefore the model should be interpreted as a **data-driven anchor**, not a complete salary valuation model.

---

# Future Improvements

Potential extensions include:

- adding multi-season performance history
- including age and years of experience
- incorporating advanced basketball metrics
- modelling positions separately
- estimating prediction intervals more explicitly
- expanding interpretability analysis

---

# Tech Stack

Python

Libraries used:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow / keras
- shap

---

# Repository Structure

```plaintext
.
├── data
|   └── nba_data.csv
├── nba_salary_prediction.ipynb
├── README.md
└── requirements.txt
```

---

# Author

Moritz Philipp Haaf

This project was originally created as part of a machine learning assignment and later refined into a portfolio-ready end-to-end sports analytics project.