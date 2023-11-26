import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
import warnings

# Zadanie 1: Ładowanie i wstępna eksploracja
print("==================================================")
df = pd.read_excel("Credit_Card.xlsx")
print(df.shape)
df = df.drop(columns=['ID'])
number_of_columns = df.shape[1] - 1
print("Liczba rekordów, liczba cech:", df.shape)
print("Rozkład kategorii:")
print(df['default payment next month'].value_counts())
print("Czy dane są zbalansowane?", df['default payment next month'].value_counts().plot(kind='bar'))
print("==================================================\n")
# Zadanie 2: Podział zbioru danych
train, test = train_test_split(df, test_size=0.3, random_state=50, shuffle=True)
x_train = train.iloc[:, :number_of_columns]
y_train = train.iloc[:, number_of_columns:]
x_test = test.iloc[:, :number_of_columns]
y_test = test.iloc[:, number_of_columns:]

# Zadanie 3: Tworzenie klasyfikatora Decision Tree i wizualizacja
dt_classifier = DecisionTreeClassifier(max_depth=6, criterion='entropy')
dt_classifier.fit(x_train, y_train)


def print_summary(classifier, x_test, y_test):
    y_pred = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    print(f"Trafność (Accuracy) dla {classifier.__class__.__name__}: {accuracy}")
    print(f"Miara F1 dla {classifier.__class__.__name__}: {f1}")
    print(f"ROC dla {classifier.__class__.__name__}: {roc_auc_score(y_test, y_pred)}")


print_summary(dt_classifier, x_test, y_test)
print("==================================================\n")

fig = plt.figure(figsize=(15, 8))
f_names = list(df.columns.values.tolist())
f_names = f_names[:-1]
t_names = ["0", "1"]
tree.plot_tree(dt_classifier, feature_names=f_names,
               class_names=t_names, filled=True)
plt.show()

# DT - Text form
print("Decision Tree text form")
print(print("=================================================="))
f_names = list(df.columns.values.tolist())
f_names = f_names[:-1]
r = export_text(dt_classifier, feature_names=f_names)
print(r)
print("==================================================\n")


# Zadanie 4: Optymalizacja hyperparametrów DT


def evaluate_classifier(x_train, y_train, x_test, y_test, max_depth, criterion):
    # Tworzenie drzewa decyzyjnego z zadanymi hiperparametrami
    classifier = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

    # Trenowanie klasyfikatora na zbiorze treningowym
    classifier.fit(x_train, y_train)

    # Przewidywanie na zbiorze walidacyjnym
    y_val_pred_proba = classifier.predict(x_test)

    # Ocenianie jakości za pomocą ROC-AUC
    roc_auc = roc_auc_score(y_test, y_val_pred_proba)

    return roc_auc


max_depth_values = [2, 4]
criterion_values = ['gini', 'entropy', 'log_loss']

best_roc_auc = 0
best_params = {'max_depth': None, 'criterion': None}

for max_depth in max_depth_values:
    for criterion in criterion_values:
        roc_auc = evaluate_classifier(x_train, y_train, x_test, y_test, max_depth, criterion)

        # print(f"max_depth={max_depth}, criterion={criterion}, ROC-AUC={roc_auc}")

        # Aktualizacja najlepszych parametrów, jeśli uzyskano lepszy wynik
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_params['max_depth'] = max_depth
            best_params['criterion'] = criterion

dt_classifier = DecisionTreeClassifier(max_depth=best_params['max_depth'], criterion=best_params['criterion'])
dt_classifier.fit(x_train,y_train)
print(f"Najlepsze parametry dla Decision Tree: {best_params}, Najlepszy wynik ROC-AUC: {best_roc_auc}")
print_summary(dt_classifier, x_test, y_test)
print("==================================================\n")

# Zadanie 5: GridSearchCV do wyszukiwania optymalnych hyperparametrów DT
param_grid_dt = {
    'max_depth': [2, 4, 6, 8, 10, 12],
    'criterion': ['gini', 'entropy', 'log_loss']
}

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search_dt.fit(x_train, y_train)

best_dt_depth = grid_search_dt.best_params_['max_depth']
best_dt_criterion = grid_search_dt.best_params_['criterion']
print("Najlepsze parametry dla Decision Tree po GridSearchCV:", grid_search_dt.best_params_)
print_summary(grid_search_dt, x_test, y_test)
print("==================================================\n")

# Zadanie 6: Tworzenie klasyfikatora Random Forest i wizualizacja


warnings.filterwarnings('ignore')
rf_classifier = RandomForestClassifier(n_estimators=10, criterion='log_loss', max_depth=5)
rf_classifier.fit(x_train, y_train)
print_summary(rf_classifier, x_test, y_test)
predictions = rf_classifier.predict(x_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("==================================================\n")

# Zadanie 7: Użyj GridSearchCV do wyszukiwania optymalnych hyperparametrów RF
param_grid_rf = {
    'n_estimators': [2, 4, 6, 8, 10, 12, 14, 16],
    'max_depth': [2, 4, 6, 8, 10, 12, 14, 16],
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, n_jobs=-1, scoring='roc_auc')
grid_search_rf.fit(x_train, y_train)

predictions = grid_search_rf.predict(x_test)

best_rf_number = grid_search_rf.best_params_['n_estimators']
best_rf_depth = grid_search_rf.best_params_['max_depth']
best_rf_criterion = grid_search_rf.best_params_['criterion']

print("Najlepsze parametry dla Random Forest:", grid_search_rf.best_params_)

print_summary(grid_search_rf, x_test, y_test)
print("==================================================\n")

# Zadanie 8: Użyj algorytmów do zbalansowania danych
# OverSampling
oversampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = oversampler.fit_resample(x_train, y_train)

# UnderSampling
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(x_train, y_train)

# SMOT
smot = SMOTE(random_state=42)
X_train_smote, y_train_smote = smot.fit_resample(x_train, y_train)

# Balanced Decision Tree OverSampling
dt_classifier_balanced_over = DecisionTreeClassifier(max_depth=best_dt_depth, criterion=best_dt_criterion)
dt_classifier_balanced_over.fit(X_train_over, y_train_over)

# Balanced Random Forest OverSampling
rf_classifier_balanced_over = RandomForestClassifier(n_estimators=best_rf_number, max_depth=best_rf_depth,
                                                     criterion=best_rf_criterion)
rf_classifier_balanced_over.fit(X_train_over, y_train_over)

# Balanced Decision Tree UnderSampling
dt_classifier_balanced_under = DecisionTreeClassifier(max_depth=best_dt_depth, criterion=best_dt_criterion)
dt_classifier_balanced_under.fit(X_train_under, y_train_under)

# Balanced Random Forest UnderSampling
rf_classifier_balanced_under = RandomForestClassifier(n_estimators=best_rf_number, max_depth=best_rf_depth,
                                                      criterion=best_rf_criterion)
rf_classifier_balanced_under.fit(X_train_under, y_train_under)

# Balanced Decision Tree SMOTE
dt_classifier_balanced_smote = DecisionTreeClassifier(max_depth=best_dt_depth, criterion=best_dt_criterion)
dt_classifier_balanced_smote.fit(X_train_smote, y_train_smote)

# Balanced Random Forest SMOTE
rf_classifier_balanced_smote = RandomForestClassifier(n_estimators=best_rf_number, max_depth=best_rf_depth,
                                                      criterion=best_rf_criterion)
rf_classifier_balanced_smote.fit(X_train_smote, y_train_smote)

# Zadanie 9: Podsumowanie
dt_classifier = DecisionTreeClassifier(max_depth=best_dt_depth, criterion=best_dt_criterion)
dt_classifier.fit(x_train, y_train)
rf_classifier = RandomForestClassifier(n_estimators=best_rf_number, max_depth=best_rf_depth,
                                       criterion=best_rf_criterion)
rf_classifier.fit(x_train, y_train)


def optimize_model_auto(model, x_train, y_train, x_test, y_test):
    model_name = type(model).__name__

    if model_name == 'DecisionTreeClassifier':
        param_grid = {
            'max_depth': [2, 4, 6, 8, 10, 12],
            'criterion': ['gini', 'entropy', 'log_loss']
        }
    elif model_name == 'RandomForestClassifier':
        param_grid = {
            'n_estimators': [2, 4, 6, 8, 10, 12, 14, 16],
            'max_depth': [2, 4, 6, 8, 10, 12, 14, 16],
            'criterion': ['gini', 'entropy', 'log_loss']
        }
    else:
        raise ValueError("Nieobsługiwany typ modelu.")

    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_

    print(f"Najlepsze parametry dla {model_name} po GridSearchCV:", best_params)
    print_summary(grid_search, x_test, y_test)
    print("==================================================\n")


print("\nBez balansowania danych\n")
print("==================================================")
optimize_model_auto(dt_classifier, x_train, y_train, x_test, y_test)
print("==================================================")
optimize_model_auto(rf_classifier, x_train, y_train, x_test, y_test)
print("==================================================\n")
print("OverSampling\n")
print("==================================================")
optimize_model_auto(dt_classifier_balanced_over, X_train_over, y_train_over, x_test, y_test)
print("==================================================")
optimize_model_auto(rf_classifier_balanced_over, X_train_over, y_train_over, x_test, y_test)
print("==================================================\n")
print("UnderSampling\n")
print("==================================================")
optimize_model_auto(dt_classifier_balanced_under, X_train_under, y_train_under, x_test, y_test)
print("==================================================")
optimize_model_auto(rf_classifier_balanced_under, X_train_under, y_train_under, x_test, y_test)
print("==================================================\n")
print("SMOTE\n")
print("==================================================")
optimize_model_auto(dt_classifier_balanced_smote, X_train_smote, y_train_smote, x_test, y_test)
print("==================================================")
optimize_model_auto(rf_classifier_balanced_smote, X_train_smote, y_train_smote, x_test, y_test)
print("==================================================")
