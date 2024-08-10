# Football Match Goals Prediction

## Project Overview

This project involves analyzing and predicting football match results using various regression and classification techniques. The dataset used contains historical football match results, including scores, teams, and other relevant features. The goal is to build models that predict:
1. The number of goals scored by the home team (FTHG).
2. The number of goals scored by the away team (FTAG).
3. The match outcome (FTR) classified into win, loss, or draw.

## Project Objective

The primary objectives of this project are:
1. **Predict the Number of Goals Scored by the Home Team (FTHG)**:
   - Implement and evaluate various regression models to predict FTHG.
2. **Predict the Number of Goals Scored by the Away Team (FTAG)**:
   - Implement and evaluate various regression models to predict FTAG.
3. **Classify the Match Outcome (FTR)**:
   - Use a classification model to predict the match outcome.

## Scope

The project covers the following areas:
- Data preprocessing and cleaning.
- Encoding categorical variables.
- Building and evaluating regression models (Linear Regression, Ridge Regression, Lasso Regression) for predicting FTHG and FTAG.
- Building and evaluating a Decision Tree Regressor for predicting FTHG.
- Building and evaluating a Decision Tree Classifier for predicting FTR.
- Comparing the performance of different models using various metrics.

## Tech Stack Used

- **Python**: Programming language used for analysis.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib** & **Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning algorithms and evaluation metrics.
- **Graphviz**: For visualizing decision trees.

## Step-by-Step Code Explanation

### 1. Data Loading and Preprocessing

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'/content/results.csv', encoding='latin-1')
df.head()
```

- **Data Loading**: Load the dataset into a DataFrame `df` using Pandas. The dataset is read from a CSV file and displayed using `df.head()`.

```python
df.isna().sum()
df.dropna(inplace=True)
df.isna().sum()
```

- **Missing Values**: Check for missing values with `df.isna().sum()`, and then drop any rows with missing values to ensure data quality.

```python
df.dtypes
df.info()
df.describe()
df.columns
```

- **Data Inspection**: Examine data types, summary statistics, and column names to understand the dataset’s structure and content.

### 2. Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for column in ['Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']:
    df[column] = le.fit_transform(df[column])
```

- **Label Encoding**: Convert categorical variables into numerical values using `LabelEncoder` to make them compatible with machine learning models.

### 3. Regression Analysis

#### Linear Regression for FTHG

```python
X = df_new.drop('FTHG', axis=1)
y = df_new['FTHG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
```

- **Data Preparation**: Split data into features (`X`) and target (`y`). Then, split the data into training and testing sets using `train_test_split`.

```python
from sklearn.linear_model import LinearRegression

lr_fthg = LinearRegression()
lr_fthg.fit(X_train, y_train)
train_pred = lr_fthg.predict(X_train)
```

- **Model Training**: Train a Linear Regression model to predict FTHG using the training data.

```python
# Visualizing Predictions on Training Data
plt.scatter(y_train, train_pred, color='b')
plt.plot([min(train_pred), max(train_pred)], [min(train_pred), max(train_pred)], color='r')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted - Training Data')
plt.show()
```

- **Visualization**: Plot actual vs. predicted values for the training set to evaluate the model's performance on the training data.

```python
test_pred = lr_fthg.predict(X_test)
# Visualizing Predictions on Test Data
plt.scatter(y_test, test_pred, color='b')
plt.plot([min(test_pred), max(test_pred)], [min(test_pred), max(test_pred)], color='r')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted - Test Data')
plt.show()
```

- **Test Data Visualization**: Plot actual vs. predicted values for the test set to evaluate the model's performance on the test data.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mae_fthg = mean_absolute_error(y_test, test_pred)
mse_fthg = mean_squared_error(y_test, test_pred)
rmse_fthg = np.sqrt(mean_squared_error(y_test, test_pred))
r2_fthg = r2_score(y_test, test_pred)
```

- **Evaluation Metrics**: Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score to evaluate model performance.

#### Linear Regression for FTAG

Follow similar steps as for FTHG, but predicting FTAG:

```python
X = df_new.drop('FTAG', axis=1)
y = df_new['FTAG']
# Training and Evaluation Code Similar to FTHG
```

### 4. Regularization Techniques

#### Ridge Regression

```python
from sklearn.linear_model import Ridge

rr = Ridge(alpha=0.05)
rr.fit(X_train, y_train)
train_pred = rr.predict(X_train)

# Visualizing Predictions on Training Data
plt.scatter(y_train, train_pred, color='b')
plt.plot([min(train_pred), max(train_pred)], [min(train_pred), max(train_pred)], color='r')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Ridge Regression - Training Data')
plt.show()

test_pred = rr.predict(X_test)

# Visualizing Predictions on Test Data
plt.scatter(y_test, test_pred, color='b')
plt.plot([min(test_pred), max(test_pred)], [min(test_pred), max(test_pred)], color='r')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Ridge Regression - Test Data')
plt.show()
```

- **Ridge Regression**: Train a Ridge Regression model with L2 regularization and visualize predictions on both training and test data.

#### Lasso Regression

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.05)
lasso.fit(X_train, y_train)
train_pred = lasso.predict(X_train)

# Visualizing Predictions on Training Data
plt.scatter(y_train, train_pred, color='b')
plt.plot([min(train_pred), max(train_pred)], [min(train_pred), max(train_pred)], color='r')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression - Training Data')
plt.show()

test_pred = lasso.predict(X_test)

# Visualizing Predictions on Test Data
plt.scatter(y_test, test_pred, color='b')
plt.plot([min(test_pred), max(test_pred)], [min(test_pred), max(test_pred)], color='r')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression - Test Data')
plt.show()
```

- **Lasso Regression**: Train a Lasso Regression model with L1 regularization and visualize predictions on both training and test data.

### 5. Decision Tree Regressor for FTHG

```python
from sklearn.tree import DecisionTreeRegressor

dr = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
dr.fit(X_train, y_train)
train_pred = dr.predict(X_train)

from sklearn.tree import export_graphviz

export_graphviz(dr, out_file='reg_tree.dot', feature_names=X.columns)
import graphviz

with open('reg_tree.dot') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)

# Visualizing Predictions on Training Data
plt.scatter(y_train, train_pred, color='b')
plt.plot([min(train_pred), max(train_pred)], [min(train_pred), max(train_pred)], color='r')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Decision Tree Regressor - Training Data')
plt.show()

test_pred = dr.predict(X_test)

# Visualizing Predictions on Test Data
plt.scatter(y_test, test_pred, color='b')
plt.plot([min(test_pred), max(test_pred)], [min(test_pred), max(test_pred)], color='r')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Decision Tree Regressor - Test Data')
plt.show()
```

- **Decision Tree Regressor**: Train

 a Decision Tree Regressor and visualize the decision tree structure using `Graphviz`. Plot predictions on both training and test data.

### 6. Decision Tree Classifier for FTR

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

dt = DecisionTreeClassifier(criterion='entropy', max_depth=5)
dt.fit(X_train, y_train)
train_pred = dt.predict(X_train)

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_train, train_pred)
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', classification_report(y_train, train_pred))

# Visualizing the Decision Tree
export_graphviz(dt, out_file='clf_tree.dot', feature_names=X.columns)
with open('clf_tree.dot') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)

test_pred = dt.predict(X_test)

# Confusion Matrix and Classification Report for Test Data
cm_test = confusion_matrix(y_test, test_pred)
print('Confusion Matrix (Test Data):\n', cm_test)
print('Classification Report (Test Data):\n', classification_report(y_test, test_pred))
```

- **Decision Tree Classifier**: Train a Decision Tree Classifier to predict match outcomes. Visualize the decision tree structure and evaluate model performance using confusion matrix and classification report.

## Model Evaluation Report for FTHG

```python
report = pd.DataFrame({'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Decision Tree Regressor'],
                       'MAE': [mae_fthg, ridge_mae, lasso_mae, dr_mae],
                       'MSE': [mse_fthg, ridge_mse, lasso_mse, dr_mse],
                       'RMSE': [rmse_fthg, ridge_rmse, lasso_rmse, dr_rmse],
                       'R2': [r2_fthg, ridge_r2, lasso_r2, dr_r2]})
report
```

- **Model Comparison**: Create a report comparing the performance of different models for predicting FTHG.

## Future Extensions

1. **Feature Engineering**: Incorporate additional features such as player statistics, weather conditions, and team performance trends.
2. **Hyperparameter Tuning**: Use techniques such as Grid Search or Random Search for tuning hyperparameters of the models.
3. **Advanced Models**: Explore more complex models like Gradient Boosting Machines (GBM) or Neural Networks.
4. **Model Deployment**: Develop a web application or API to deploy the model for real-time predictions.

## References

1. **Scikit-learn Documentation**: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
2. **Graphviz Documentation**: [https://graphviz.gitlab.io/documentation/](https://graphviz.gitlab.io/documentation/)
3. **Pandas Documentation**: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

## Contact

For any queries, feel free to reach out to me at [jameers2003@gmail.com](mailto:jameers2003@gmail.com).
