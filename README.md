# Credit Card Default Prediction

This repository contains a comprehensive machine learning pipeline for predicting credit card defaults using decision trees and random forests. It includes data loading, preprocessing, model training, hyperparameter optimization, and evaluation with techniques to handle class imbalance.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
  - [1. Data Loading and Initial Exploration](#1-data-loading-and-initial-exploration)
  - [2. Data Splitting](#2-data-splitting)
  - [3. Decision Tree Classifier](#3-decision-tree-classifier)
  - [4. Hyperparameter Optimization](#4-hyperparameter-optimization)
  - [5. GridSearchCV for Decision Tree](#5-gridsearchcv-for-decision-tree)
  - [6. Random Forest Classifier](#6-random-forest-classifier)
  - [7. GridSearchCV for Random Forest](#7-gridsearchcv-for-random-forest)
  - [8. Data Resampling](#8-data-resampling)
  - [9. Summary](#9-summary)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Requirements

- Python 3.6 or higher
- pandas
- scikit-learn
- imbalanced-learn
- matplotlib
- openpyxl

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Alex5497/credit-card-default-prediction.git
   cd credit-card-default-prediction
   ```

2. Install the required packages:
   ```bash
   pip install pandas scikit-learn imbalanced-learn matplotlib openpyxl
   ```

## Usage

1. Place the `Credit_card.xlsx` file in the same directory as the script.
2. Run the script:
   ```bash
   python credit_card_default.py
   ```

## Code Explanation

### 1. Data Loading and Initial Exploration

The script starts by loading the dataset and performing initial exploration to understand its structure and class distribution.

```python
df = pd.read_excel("Credit_card.xlsx")
df = df.drop(columns=['ID'])
print("Liczba rekordów, liczba cech:", df.shape)
print("Rozkład kategorii:")
print(df['default payment next month'].value_counts())
df['default payment next month'].value_counts().plot(kind='bar')
```

### 2. Data Splitting

The data is split into training and testing sets.

```python
train, test = train_test_split(df, test_size=0.3, random_state=50, shuffle=True)
x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1:]
x_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1:]
```

### 3. Decision Tree Classifier

A decision tree classifier is trained and evaluated.

```python
dt_classifier = DecisionTreeClassifier(max_depth=6, criterion='entropy')
dt_classifier.fit(x_train, y_train)
print_summary(dt_classifier, x_test, y_test)

# Visualize the decision tree
tree.plot_tree(dt_classifier, feature_names=df.columns[:-1], class_names=["0", "1"], filled=True)
plt.show()
```

### 4. Hyperparameter Optimization

The best hyperparameters for the decision tree are found using a simple loop.

```python
def evaluate_classifier(x_train, y_train, x_test, y_test, max_depth, criterion):
    classifier = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    classifier.fit(x_train, y_train)
    y_val_pred_proba = classifier.predict(x_test)
    return roc_auc_score(y_test, y_val_pred_proba)

# Evaluate various combinations of max_depth and criterion
max_depth_values = [2, 4]
criterion_values = ['gini', 'entropy', 'log_loss']
```

### 5. GridSearchCV for Decision Tree

GridSearchCV is used for a more thorough hyperparameter search.

```python
param_grid_dt = {'max_depth': [2, 4, 6, 8, 10, 12], 'criterion': ['gini', 'entropy', 'log_loss']}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search_dt.fit(x_train, y_train.values.ravel())
print_summary(grid_search_dt, x_test, y_test)
```

### 6. Random Forest Classifier

A random forest classifier is trained and evaluated.

```python
rf_classifier = RandomForestClassifier(n_estimators=10, criterion='log_loss', max_depth=5)
rf_classifier.fit(x_train, y_train.values.ravel())
print_summary(rf_classifier, x_test, y_test)
```

### 7. GridSearchCV for Random Forest

GridSearchCV is used to find the best hyperparameters for the random forest classifier.

```python
param_grid_rf = {'n_estimators': [2, 4, 6, 8, 10, 12, 14, 16], 'max_depth': [2, 4, 6, 8, 10, 12, 14, 16], 'criterion': ['gini', 'entropy', 'log_loss']}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search_rf.fit(x_train, y_train.values.ravel())
print_summary(grid_search_rf, x_test, y_test)
```

### 8. Data Resampling

Techniques like oversampling, undersampling, and SMOTE are used to handle class imbalance.

```python
oversampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = oversampler.fit_resample(x_train, y_train.values.ravel())
# Similar resampling for undersampling and SMOTE
```

### 9. Summary

Models are optimized and evaluated on balanced and unbalanced datasets.

```python
print("\nBez balansowania danych\n")
optimize_model_auto(dt_classifier, x_train, y_train, x_test, y_test)
optimize_model_auto(rf_classifier, x_train, y_train, x_test, y_test)
```

## Results

The script prints out the accuracy, F1 score, and ROC-AUC for each model configuration. Confusion matrices and classification reports are also generated for detailed evaluation.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project utilizes the imbalanced-learn library for handling class imbalance and scikit-learn for model building and evaluation. The dataset is provided in the `Credit_card.xlsx` file.

---

Feel free to modify and adapt the code to better suit your needs. Contributions and suggestions are welcome!
