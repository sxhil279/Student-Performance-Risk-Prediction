"""
CSE274: Applied Machine Learning
Model Comparison — Unit III & V
Run: python compare_models.py
Compares: Logistic Regression, KNN, Decision Tree, SVM, Random Forest, Gradient Boosting
"""
import pandas as pd
import numpy as np
import joblib, warnings, os
warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NUMERICAL_COLS = ['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores',
                  'Tutoring_Sessions','Physical_Activity']
CATEGORICAL_COLS = ['Parental_Involvement','Access_to_Resources','Extracurricular_Activities',
    'Motivation_Level','Family_Income','Teacher_Quality',
    'Parental_Education_Level','Distance_to_School','Gender']

def load_data():
    df = pd.read_excel('cse274_dataset.xlsx')
    df.rename(columns={'Unnamed: 13': 'Distance_to_School'}, inplace=True)
    df.dropna(subset=['Exam_Score'], inplace=True)
    df['At_Risk'] = (df['Exam_Score'] < 65).astype(int)
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
    for col in NUMERICAL_COLS:
        df[col] = df[col].fillna(df[col].median())
    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    return df[feature_cols], df['At_Risk']

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Naive Bayes':         GaussianNB(),
    'KNN':                 KNeighborsClassifier(n_neighbors=7),
    'Decision Tree':       DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
    'SVM':                 SVC(probability=True, class_weight='balanced', random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=10,
                               class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
}

print("=" * 65)
print(f"{'Model':<22} {'Accuracy':>9} {'ROC-AUC':>9} {'F1':>9} {'CV-AUC':>9}")
print("=" * 65)

results = []
for name, clf in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)
    cv  = cross_val_score(pipe, X, y, cv=3, scoring='roc_auc').mean()
    results.append({'Model': name, 'Accuracy': acc, 'ROC_AUC': auc, 'F1': f1, 'CV_AUC': cv})
    print(f"{name:<22} {acc*100:>8.2f}% {auc:>9.4f} {f1:>9.4f} {cv:>9.4f}")

print("=" * 65)

results_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False)

# ── Plot comparison bar chart ────────────────────────────────────────────────
metrics = ['Accuracy', 'ROC_AUC', 'F1']
colors  = ['#818cf8', '#6ee7b7', '#fbbf24']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, metric, color in zip(axes, metrics, colors):
    vals = results_df.set_index('Model')[metric]
    vals.plot(kind='barh', ax=ax, color=color, edgecolor='none', width=0.6)
    ax.set_title(metric.replace('_',' '), fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel('Score')
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

plt.suptitle('Model Comparison — CSE274 Project', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150)
plt.close()

print(f"\nBest model: {results_df.iloc[0]['Model']} (ROC-AUC: {results_df.iloc[0]['ROC_AUC']:.4f})")
print("Comparison chart saved to plots/model_comparison.png")
