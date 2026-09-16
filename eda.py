"""
CSE274: Applied Machine Learning
EDA — Exploratory Data Analysis
Run: python eda.py
Saves charts to plots/eda_*.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

sns.set_theme(style='darkgrid', palette='muted')
os.makedirs('plots', exist_ok=True)

df = pd.read_excel('cse274_dataset.xlsx')
df.rename(columns={'Unnamed: 13': 'Distance_to_School'}, inplace=True)
df['At_Risk'] = (df['Exam_Score'] < 65).astype(int)

print(f"Dataset shape : {df.shape}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
print(f"\nAt-Risk students: {df['At_Risk'].sum()} / {len(df)} ({df['At_Risk'].mean()*100:.1f}%)")

# ── 1. Class distribution ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['At_Risk'].value_counts().plot(kind='bar', ax=axes[0], color=['#6ee7b7','#f87171'],
    edgecolor='none', width=0.5)
axes[0].set_xticklabels(['Safe (≥65)', 'At Risk (<65)'], rotation=0)
axes[0].set_title('Class Distribution', fontweight='bold')
axes[0].set_ylabel('Count')

df['Exam_Score'].plot(kind='hist', ax=axes[1], bins=30, color='#818cf8', edgecolor='none')
axes[1].axvline(65, color='#f87171', linewidth=2, linestyle='--', label='Risk threshold (65)')
axes[1].set_title('Exam Score Distribution', fontweight='bold')
axes[1].set_xlabel('Score'); axes[1].legend()
plt.tight_layout(); plt.savefig('plots/eda_class_dist.png', dpi=150); plt.close()

# ── 2. Numerical feature distributions by risk ──────────────────────────────
num_cols = ['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores',
            'Tutoring_Sessions','Physical_Activity']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    for risk, color, label in [(0,'#6ee7b7','Safe'),(1,'#f87171','At Risk')]:
        axes[i].hist(df[df['At_Risk']==risk][col], bins=20, alpha=0.6,
                     color=color, label=label, edgecolor='none')
    axes[i].set_title(col.replace('_',' '), fontweight='bold')
    axes[i].legend(fontsize=8)
plt.suptitle('Feature Distributions by Risk Status', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plots/eda_numerical.png', dpi=150); plt.close()

# ── 3. Correlation heatmap ───────────────────────────────────────────────────
corr_df = df[num_cols + ['Exam_Score','At_Risk']].corr()
plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plots/eda_correlation.png', dpi=150); plt.close()

# ── 4. Categorical vs Risk (grouped bar) ────────────────────────────────────
cat_cols = ['Motivation_Level','Parental_Involvement','Access_to_Resources',
            'Family_Income','Teacher_Quality']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = df.groupby(col)['At_Risk'].mean().sort_values() * 100
    ct.plot(kind='bar', ax=axes[i], color='#818cf8', edgecolor='none', width=0.6)
    axes[i].set_title(f'Risk % by {col.replace("_"," ")}', fontweight='bold')
    axes[i].set_ylabel('At-Risk %'); axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=30)
axes[-1].set_visible(False)
plt.suptitle('Risk Rate by Categorical Features', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plots/eda_categorical.png', dpi=150); plt.close()

# ── 5. Hours studied vs Attendance scatter ───────────────────────────────────
plt.figure(figsize=(8,6))
colors = df['At_Risk'].map({0:'#6ee7b7', 1:'#f87171'})
plt.scatter(df['Hours_Studied'], df['Attendance'], c=colors, alpha=0.35, s=18, edgecolors='none')
plt.xlabel('Hours Studied / Week'); plt.ylabel('Attendance (%)')
plt.title('Study Hours vs Attendance by Risk', fontweight='bold')
from matplotlib.lines import Line2D
legend = [Line2D([0],[0],marker='o',color='w',markerfacecolor='#6ee7b7',markersize=9,label='Safe'),
          Line2D([0],[0],marker='o',color='w',markerfacecolor='#f87171',markersize=9,label='At Risk')]
plt.legend(handles=legend)
plt.tight_layout(); plt.savefig('plots/eda_scatter.png', dpi=150); plt.close()

print("\nEDA plots saved to plots/")
print("  eda_class_dist.png")
print("  eda_numerical.png")
print("  eda_correlation.png")
print("  eda_categorical.png")
print("  eda_scatter.png")
