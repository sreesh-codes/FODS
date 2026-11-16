"""
Generate EDA figures for GitHub repository
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)

print("Loading data...")
panel = pd.read_parquet('processed/model_panel.parquet')
train = panel[panel['split'] == 'train'].copy()

print(f"Train set: {len(train):,} events")

# Filter valid SUE data
sue_valid = train[train['sue_quintile'].notna()].copy()
print(f"Events with SUE: {len(sue_valid):,}")

# Figure 1: Events by year
print("\n[1/10] Generating events by year...")
events_by_year = train.groupby('year').size()
fig, ax = plt.subplots(figsize=(8, 4))
events_by_year.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Earnings Announcements by Year (Train Set)')
ax.set_xlabel('Year')
ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'events_by_year.png')
plt.close()

# Figure 2: SUE distribution
print("[2/10] Generating SUE distribution...")
fig, ax = plt.subplots(figsize=(8, 4))
sue_data = train['sue'].dropna()
sue_data_clean = sue_data[np.isfinite(sue_data)]
sue_data_clean.hist(bins=50, ax=ax, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Zero')
ax.set_title('SUE Distribution (Train Set)')
ax.set_xlabel('Standardized Unexpected Earnings')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'sue_distribution.png')
plt.close()

# Figure 3: CAR distributions
print("[3/10] Generating CAR distributions...")
car_cols = ['car_-5_-1', 'car_0_1', 'car_2_20']
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for i, col in enumerate(car_cols):
    data = train[col].dropna()
    data = data[np.isfinite(data)]
    data.hist(bins=40, ax=axes[i], color='seagreen', edgecolor='black', alpha=0.7)
    axes[i].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[i].set_title(col.replace('car_', 'CAR'))
    axes[i].set_xlabel('Return')
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'car_distributions.png')
plt.close()

# Figure 4: Sector composition
print("[4/10] Generating sector composition...")
sector_counts = train['sector'].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
sector_counts.plot(kind='barh', ax=ax, color='teal')
ax.set_title('Events by Sector (Train Set)')
ax.set_xlabel('Count')
ax.set_ylabel('Sector')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'sector_composition.png')
plt.close()

# Figure 5: PEAD by SUE quintile (KEY FIGURE)
print("[5/10] Generating PEAD by SUE quintile...")
sue_drift = sue_valid.groupby('sue_quintile')['car_2_20'].mean()
fig, ax = plt.subplots(figsize=(8, 5))
sue_drift.plot(kind='bar', ax=ax,
               color=['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4'],
               edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_title('Post-Earnings Announcement Drift by SUE Quintile\n(CAR[+2,+20], Train Set 2015-2022)',
             fontsize=12, weight='bold')
ax.set_xlabel('SUE Quintile (1=Worst, 5=Best)', fontsize=11)
ax.set_ylabel('Average CAR[+2,+20]', fontsize=11)
ax.set_xticklabels(['Q1\n(Worst)', 'Q2', 'Q3', 'Q4', 'Q5\n(Best)'], rotation=0)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(sue_drift.values):
    ax.text(i, v + 0.002, f"{v:.3f}", ha='center', va='bottom', fontsize=9, weight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pead_sue_quintiles.png', dpi=300)
plt.savefig(FIGURES_DIR / 'pead_sue_quintiles.svg', format='svg')
plt.close()

# Figure 6: CAR path by SUE quintile
print("[6/10] Generating CAR path...")
fig, ax = plt.subplots(figsize=(10, 5))
for q in [1, 2, 3, 4, 5]:
    q_data = sue_valid[sue_valid['sue_quintile'] == q]
    car_path = q_data[['car_-5_-1', 'car_0_1', 'car_0_5', 'car_2_20']].mean()
    windows = ['Pre\n[-5,-1]', 'Ann\n[0,+1]', 'Short\n[0,+5]', 'Drift\n[+2,+20]']
    ax.plot(windows, car_path.values, marker='o', label=f'Q{q}', linewidth=2)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title('CAR Path by SUE Quintile', fontsize=12, weight='bold')
ax.set_xlabel('Window')
ax.set_ylabel('Average CAR')
ax.legend(title='SUE Quintile', loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'car_path_by_sue.png')
plt.close()

# Figure 7: Announcement return by SUE
print("[7/10] Generating announcement returns...")
sue_announcement = sue_valid.groupby('sue_quintile')['car_0_1'].mean()
fig, ax = plt.subplots(figsize=(8, 4))
sue_announcement.plot(kind='bar', ax=ax, color='darkorange', edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_title('Immediate Announcement Return by SUE Quintile')
ax.set_xlabel('SUE Quintile')
ax.set_ylabel('CAR[0,+1]')
ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], rotation=0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'announcement_return_by_sue.png')
plt.close()

# Figure 8: PEAD by sector
print("[8/10] Generating PEAD by sector...")
sector_sue = sue_valid.groupby(['sector', 'sue_quintile'])['car_2_20'].mean().unstack()
sector_spread = sector_sue[5] - sector_sue[1]
sector_spread = sector_spread.sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
sector_spread.plot(kind='barh', ax=ax, color='slateblue')
ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_title('PEAD Spread (Q5-Q1) by Sector')
ax.set_xlabel('CAR[+2,+20] Spread')
ax.set_ylabel('Sector')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pead_by_sector.png')
plt.close()

# Figure 9: PEAD over time
print("[9/10] Generating PEAD over time...")
yearly_spread = []
for year in range(2015, 2023):
    year_data = sue_valid[sue_valid['year'] == year]
    q5 = year_data[year_data['sue_quintile'] == 5]['car_2_20'].mean()
    q1 = year_data[year_data['sue_quintile'] == 1]['car_2_20'].mean()
    yearly_spread.append({'year': year, 'spread': q5 - q1})
yearly_spread_df = pd.DataFrame(yearly_spread)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(yearly_spread_df['year'], yearly_spread_df['spread'],
       color='mediumvioletred', edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_title('PEAD Spread (Q5-Q1) Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('CAR[+2,+20] Spread')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pead_over_time.png')
plt.close()

# Figure 10: Model comparison (from model registry)
print("[10/10] Generating model comparison...")
try:
    model_reg = pd.read_csv('logs/model_registry.csv')
    model_summary = model_reg.groupby('model')['auc'].agg(['mean', 'std']).sort_values('mean', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(model_summary))
    ax.barh(x, model_summary['mean'], xerr=model_summary['std'],
            color='skyblue', edgecolor='black', capsize=5)
    ax.set_yticks(x)
    ax.set_yticklabels(model_summary.index)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='Random (AUC=0.5)')
    ax.set_xlabel('AUC')
    ax.set_title('Model Performance Comparison (Walk-Forward Validation)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison.png')
    plt.close()
    print("  ✓ Model comparison saved")
except Exception as e:
    print(f"  ⚠ Could not generate model comparison: {e}")

print(f"\n✓ All figures saved to {FIGURES_DIR}/")
print("\nGenerated files:")
for f in sorted(FIGURES_DIR.glob('*.png')):
    print(f"  • {f.name}")
