# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:00:26 2024

@author: lapra
"""

import pandas as pd
import os

path = r"C:\Users\lapra\OneDrive\Desktop\SupMid_NicholasLaprade"
file = "titanic_midterm_comp247.csv"
fullpath = os.path.join(path,file)

df_nicholas = pd.read_csv(fullpath)

# Printing types
print(df_nicholas.dtypes)

# Printing Missing Values
missing_values = df_nicholas.isnull().sum()
for column, count in missing_values.items():
    print(f"{column}: {count} missing values")
    
# Printing Range, Mean, Median
print(df_nicholas.describe().round(2))

import matplotlib.pyplot as plt
import seaborn as sns

# Generating a pairplot for column relationships
sns.pairplot(df_nicholas)
plt.savefig('pairplot.png')
plt.show()

# Separating features
df_nicholas_features = df_nicholas.drop(columns = ["survived"])
df_nicholas_target = df_nicholas["survived"]

from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Training
X_train, X_test, y_train, y_test = train_test_split(df_nicholas_features, df_nicholas_target, test_size = 0.3, random_state = 45)

# Pipeline
num_pipeline_nicholas = Pipeline(
    [
     ('imputer', SimpleImputer(strategy = 'median')),
     ('scaler', StandardScaler())
     ])

cat_pipeline_nicholas = Pipeline(
    [
     ('imputer', SimpleImputer(strategy = 'most_frequent')),
     ('onehot', OneHotEncoder())
     ])

from sklearn.compose import ColumnTransformer

full_pipeline_nicholas = ColumnTransformer(
    [
     ("num", num_pipeline_nicholas, X_train.select_dtypes(include=['int64', 'float64']).columns),
     ("cat", cat_pipeline_nicholas, X_train.select_dtypes(include=['object']).columns)
    ])

X_train_transformed = full_pipeline_nicholas.fit_transform(X_train)

# Logistic Regression + SVM Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

clf_lr_nicholas = LogisticRegression(solver = 'lbfgs', random_state = 45)
clf_svm_nicholas = SVC(gamma = 'auto', random_state = 45)

# Training both Classifiers
clf_lr_nicholas.fit(X_train_transformed, y_train)
clf_svm_nicholas.fit(X_train_transformed, y_train)

lr_scores = cross_val_predict(clf_lr_nicholas, X_train_transformed, y_train, cv = 5)
svm_scores = cross_val_predict(clf_svm_nicholas, X_train_transformed, y_train, cv = 5)

print(f'Logistic Regression Scores: {lr_scores}')
print(f'Mean Logistic Regression Scores: {lr_scores.mean()}')

print(f'SVM Scores: {svm_scores}')
print(f'Mean SVM Scores: {svm_scores.mean()}')

# Fine Tune
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.01, 0.1, 1, 5],
    'gamma': [0.01, 0.02, 0.2, 0.3]
    }

grid_search = GridSearchCV(clf_svm_nicholas, param_grid)
grid_search.fit(X_train_transformed, y_train)

print(f'Best Performance Parameters: {grid_search.best_params_}')

clf_svm_best_fit_nicholas = grid_search.best_estimator_

X_test_transformed = full_pipeline_nicholas.transform(X_test)
data_predict = clf_svm_best_fit_nicholas.predict(X_test_transformed)

from sklearn.metrics import accuracy_score

# Calculating Accuracy Score
accuracy = accuracy_score(y_test, data_predict)
print(f'Accuracy Score: {accuracy}')

from joblib import dump

dump(clf_svm_best_fit_nicholas, 'clf_svm_best_fit_nicholas.joblib')
dump(full_pipeline_nicholas, 'full_pipeline_nicholas.joblib')

















