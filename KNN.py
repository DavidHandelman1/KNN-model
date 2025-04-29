import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             classification_report, roc_curve, roc_auc_score)

# load dataset
FILE_PATH = 'cc_institution_details.csv'
data = pd.read_csv(FILE_PATH)

# define features
features = ['chronname', 'state', 'control', 'level', 'fte_value', 
            'exp_award_value', 'hbcu', 'flagship', 'awards_per_value']

# drop rows with missing 'awards_per_value'
df = data[features].dropna(subset=['awards_per_value'])

# convert 'hbcu' and 'flagship' from 'X' flags to binary
df['hbcu'] = (df['hbcu'] == 'X').astype(int)
df['flagship'] = (df['flagship'] == 'X').astype(int)

# create binary target: top 10% in awards_per_value
threshold = df['awards_per_value'].quantile(0.90)
df['is_top10'] = (df['awards_per_value'] >= threshold).astype(int)

# Define features and target variable
X = df.drop(columns=['awards_per_value', 'is_top10'])
y = df['is_top10']

# split data into training/validation (80%) and test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, 
                                                  random_state=42, stratify=y)

# also split training into training (75%) and validation (25%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, 
                                                  random_state=42, stratify=y_temp)

# preprocessing pipeline
category_features = ['chronname', 'state', 'control', 'level', 'hbcu', 'flagship']
numerical_features = ['fte_value', 'exp_award_value']

# preprocessing step
preprocessor = ColumnTransformer([
    ('category', OneHotEncoder(drop='first', handle_unknown='ignore'), category_features),
    ('numerical', StandardScaler(), numerical_features)
])

# build pipeline, preprocessing + KNN
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('knn', KNeighborsClassifier())
])

# hyperparameter tuning
param_grid = {
    'knn__n_neighbors': list(range(3, 30)),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit on training data
random_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(random_search.best_params_)
print("\nBest Cross-Validation Score:", random_search.best_score_)

# evaluate best model on validation and test sets
best_knn_model = random_search.best_estimator_

# predict on validation set
y_val_pred = best_knn_model.predict(X_val)
print("\nValidation Set Results:")
print(classification_report(y_val, y_val_pred))

# predict on test set
y_test_pred = best_knn_model.predict(X_test)
print("\nTest Set Results:")
print(classification_report(y_test, y_test_pred))

# confusion Matrix (test set)
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Top 10%', 'Top 10%'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Test Set")
plt.show()

# ROC Curve (test set)
y_probs = best_knn_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Set')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='control', hue='is_top10')
plt.title('Control Type vs Top 10% Status')
plt.xlabel('Control Type')
plt.ylabel('Count')
plt.legend(title='Top 10%')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()