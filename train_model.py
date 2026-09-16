"""
CSE274: Applied Machine Learning
Project: Student Performance Risk Prediction
train_model.py
"""
import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

NUMERICAL_COLS = ['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores',
                  'Tutoring_Sessions','Physical_Activity']
CATEGORICAL_COLS = ['Parental_Involvement','Access_to_Resources','Extracurricular_Activities',
    'Motivation_Level','Family_Income','Teacher_Quality',
    'Parental_Education_Level','Distance_to_School','Gender']

def load_and_preprocess(filepath):
    df = pd.read_excel(filepath)
    df.rename(columns={'Unnamed: 13': 'Distance_to_School'}, inplace=True)
    df.dropna(subset=['Exam_Score'], inplace=True)
    df['At_Risk'] = (df['Exam_Score'] < 65).astype(int)

    le_dict = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = df[col].astype(str).fillna('Unknown')
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    for col in NUMERICAL_COLS:
        df[col] = df[col].fillna(df[col].median())

    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    return df[feature_cols], df['At_Risk'], feature_cols, le_dict

def train(filepath='cse274_dataset.xlsx'):
    print("=" * 60)
    print("  Student Performance Risk Prediction — Training")
    print("=" * 60)

    X, y, feature_cols, le_dict = load_and_preprocess(filepath)
    print(f"\nDataset: {len(X)} samples | At-Risk: {y.sum()} ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=10,
            min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1))
    ])

    print("\nTraining Random Forest pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"ROC-AUC       : {auc:.4f}")
    print(f"F1-Score      : {f1_score(y_test, y_pred):.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=['Safe','At Risk']))

    cv = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    print(f"5-Fold CV AUC : {cv.mean():.4f} +/- {cv.std():.4f}")

    os.makedirs('plots', exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe','At Risk'], yticklabels=['Safe','At Risk'])
    plt.title('Confusion Matrix', fontweight='bold')
    plt.tight_layout(); plt.savefig('plots/confusion_matrix.png', dpi=150); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='#4f46e5', lw=2, label=f'AUC={auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve', fontweight='bold')
    plt.legend(); plt.tight_layout(); plt.savefig('plots/roc_curve.png', dpi=150); plt.close()

    # Feature importance
    imp = pipeline.named_steps['clf'].feature_importances_
    fi = pd.Series(imp, index=feature_cols).sort_values()
    plt.figure(figsize=(7,6)); fi.plot(kind='barh', color='#4f46e5')
    plt.title('Feature Importance', fontweight='bold')
    plt.tight_layout(); plt.savefig('plots/feature_importance.png', dpi=150); plt.close()

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6,5)); plt.plot(rec, prec, color='#10b981', lw=2)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.tight_layout(); plt.savefig('plots/precision_recall.png', dpi=150); plt.close()

    # Save
    joblib.dump(pipeline, 'model.pkl')
    joblib.dump(le_dict, 'label_encoders.pkl')
    joblib.dump(feature_cols, 'feature_cols.pkl')
    label_maps = {col: list(le.classes_) for col, le in le_dict.items()}
    joblib.dump(label_maps, 'label_maps.pkl')

    print("\nSaved: model.pkl, label_encoders.pkl, feature_cols.pkl, label_maps.pkl")
    print("Plots saved to plots/")
    print("\nDone! Run: python app.py")

if __name__ == '__main__':
    train('cse274_dataset.xlsx')
