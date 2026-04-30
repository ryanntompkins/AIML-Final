import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

OUT = "eda_outputs"
os.makedirs(OUT, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("../us_wildfire_data.csv", low_memory=False)
print(f"Shape: {df.shape}\n")

# ── 2. Missing Values ──────────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_df = missing_df[missing_df.missing_count > 0].sort_values('missing_pct', ascending=False)
print("=== Missing Values ===")
print(missing_df.to_string())

fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.4)))
ax.barh(missing_df.index, missing_df.missing_pct, color='tomato')
ax.set_xlabel('% Missing')
ax.set_title('Missing Values by Column')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/missing_values.png", dpi=150)
plt.close()

# ── 3. Spatial ─────────────────────────────────────────────────────────────────
state_counts = df['state'].value_counts()
print("\n=== Top 10 States ===")
print(state_counts.head(10).to_string())

fig, ax = plt.subplots(figsize=(14, 5))
state_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Number of Wildfires by State')
ax.set_xlabel('State')
ax.set_ylabel('Fire Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUT}/fires_by_state.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(14, 7))
sc = ax.scatter(df['longitude'], df['latitude'],
                c=np.log1p(df['fire_size']), cmap='YlOrRd', s=1, alpha=0.4)
plt.colorbar(sc, ax=ax, label='log(fire_size + 1)')
ax.set_xlim(-130, -60)
ax.set_ylim(20, 55)
ax.set_title('Wildfire Locations (coloured by log fire size)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig(f"{OUT}/fire_map.png", dpi=150)
plt.close()

# ── 4. Temporal ────────────────────────────────────────────────────────────────
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_counts = df['discovery_month'].value_counts().reindex(month_order, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 4))
month_counts.plot(kind='bar', ax=ax, color='darkorange', edgecolor='white')
ax.set_title('Wildfire Discoveries by Month')
ax.set_xlabel('Month')
ax.set_ylabel('Fire Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT}/fires_by_month.png", dpi=150)
plt.close()

year_counts = df['disc_pre_year'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(year_counts.index, year_counts.values, marker='o', color='firebrick', linewidth=2)
ax.fill_between(year_counts.index, year_counts.values, alpha=0.15, color='firebrick')
ax.set_title('Wildfire Frequency Over the Years')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Fires')
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(f"{OUT}/fires_by_year.png", dpi=150)
plt.close()

# ── 5. Causes ──────────────────────────────────────────────────────────────────
cause_counts = df['stat_cause_descr'].value_counts()
print("\n=== Fire Causes ===")
print(cause_counts.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
cause_counts.plot(kind='barh', ax=ax, color='mediumpurple', edgecolor='white')
ax.set_title('Wildfire Causes')
ax.set_xlabel('Count')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/fire_causes.png", dpi=150)
plt.close()

# ── 6. Fire Size & Class Imbalance ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(df['fire_size'].dropna(), bins=80, color='darkorange', edgecolor='white', log=True)
axes[0].set_title('Fire Size Distribution (log y-scale)')
axes[0].set_xlabel('Fire Size (acres)')
axes[0].set_ylabel('Count (log)')
axes[1].hist(np.log1p(df['fire_size'].dropna()), bins=60, color='steelblue', edgecolor='white')
axes[1].set_title('log(fire_size + 1) Distribution')
axes[1].set_xlabel('log(fire_size + 1)')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(f"{OUT}/fire_size_dist.png", dpi=150)
plt.close()

class_counts = df['fire_size_class'].value_counts().sort_index()
class_pct = (class_counts / class_counts.sum() * 100).round(2)
print("\n=== Fire Size Class Distribution ===")
for cls, cnt, pct in zip(class_counts.index, class_counts.values, class_pct.values):
    print(f"  Class {cls}: {cnt:>6} fires  ({pct:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(class_counts.index, class_counts.values, color='coral', edgecolor='white')
ax.set_title('Fire Size Class Distribution (A = smallest, G = largest)')
ax.set_xlabel('Fire Size Class')
ax.set_ylabel('Count')
for bar, pct in zip(bars, class_pct.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/class_imbalance.png", dpi=150)
plt.close()

# ── 7. Correlations ────────────────────────────────────────────────────────────
weather_cols = [
    'Temp_pre_30','Temp_pre_15','Temp_pre_7','Temp_cont',
    'Wind_pre_30','Wind_pre_15','Wind_pre_7','Wind_cont',
    'Hum_pre_30','Hum_pre_15','Hum_pre_7','Hum_cont',
    'Prec_pre_30','Prec_pre_15','Prec_pre_7','Prec_cont',
    'fire_size', 'fire_mag', 'remoteness'
]
corr_df = df[weather_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.zeros_like(corr_df, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 7})
ax.set_title('Correlation Matrix - Weather Features & Fire Size')
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=150)
plt.close()

top_corr = corr_df['fire_size'].drop('fire_size').abs().sort_values(ascending=False).head(4).index.tolist()
print(f"\nTop 4 features correlated with fire_size: {top_corr}")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, feat in zip(axes, top_corr):
    sample = df[[feat, 'fire_size']].dropna().sample(min(3000, len(df)), random_state=42)
    ax.scatter(sample[feat], np.log1p(sample['fire_size']), alpha=0.2, s=5, color='steelblue')
    ax.set_xlabel(feat)
    ax.set_ylabel('log(fire_size + 1)')
    ax.set_title(f'{feat} vs fire_size')
plt.suptitle('Top Weather Features vs log(Fire Size)', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/scatter_top_features.png", dpi=150)
plt.close()

# ── 8. Vegetation ──────────────────────────────────────────────────────────────
veg_labels = {
    1:'Trop.Evgr.Broadleaf', 2:'Trop.Decid.Broadleaf', 3:'Temp.Evgr.Broadleaf',
    4:'Temp.Evgr.Needleleaf', 5:'Temp.Decid.Broadleaf', 6:'Bor.Evgr.Needleleaf',
    7:'Bor.Decid.Needleleaf', 8:'Savanna', 9:'C3 Grassland', 10:'C4 Grassland',
    11:'Dense Shrubland', 12:'Open Shrubland', 13:'Tundra', 14:'Desert',
    15:'Polar/Rock/Ice', 16:'Sec.Trop.Evgr', 17:'Sec.Trop.Decid',
    18:'Sec.Temp.Evgr.BL', 19:'Sec.Temp.Evgr.NL', 20:'Sec.Temp.Decid',
    21:'Sec.Bor.Evgr.NL', 22:'Sec.Bor.Decid.NL', 23:'Water',
    24:'C3 Cropland', 25:'C4 Cropland', 26:'C3 Pasture', 27:'C4 Pasture', 28:'Urban'
}
veg_counts = df['Vegetation'].value_counts().sort_index()
veg_counts.index = veg_counts.index.map(lambda x: veg_labels.get(int(x), str(x)) if pd.notna(x) else 'Unknown')

fig, ax = plt.subplots(figsize=(12, 5))
veg_counts.sort_values(ascending=True).plot(kind='barh', ax=ax, color='forestgreen', edgecolor='white')
ax.set_title('Fires by Dominant Vegetation Type')
ax.set_xlabel('Fire Count')
plt.tight_layout()
plt.savefig(f"{OUT}/fires_by_vegetation.png", dpi=150)
plt.close()

# ── 9. Outliers ────────────────────────────────────────────────────────────────
outlier_cols = ['fire_size', 'Temp_pre_7', 'Wind_pre_7', 'Hum_pre_7', 'Prec_pre_7']

fig, axes = plt.subplots(1, len(outlier_cols), figsize=(18, 5))
for ax, col in zip(axes, outlier_cols):
    data = df[col].dropna()
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='navy'),
               medianprops=dict(color='red', linewidth=2),
               flierprops=dict(marker='o', markersize=2, alpha=0.3, color='grey'))
    ax.set_title(col, fontsize=9)
    ax.set_xticks([])
plt.suptitle('Outlier Detection - Box Plots', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/outliers_boxplots.png", dpi=150)
plt.close()

print("\n=== Outlier Summary (IQR method) ===")
for col in outlier_cols:
    data = df[col].dropna()
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    n = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
    print(f"  {col:<20} outliers: {n:>5} ({n/len(data)*100:.1f}%)  max={data.max():.1f}  median={data.median():.1f}")

# ── 10. Summary ────────────────────────────────────────────────────────────────
print("\n========== EDA SUMMARY ==========")
print(f"Total records   : {len(df):,}")
print(f"Total features  : {df.shape[1]}")
print(f"Years covered   : {df['disc_pre_year'].min():.0f} - {df['disc_pre_year'].max():.0f}")
print(f"States          : {df['state'].nunique()}")
print(f"Top state       : {df['state'].value_counts().idxmax()}")
print(f"Peak month      : {df['discovery_month'].value_counts().idxmax()}")
print(f"Most common cause: {df['stat_cause_descr'].value_counts().idxmax()}")
print(f"Fire size -- mean={df['fire_size'].mean():.1f}  median={df['fire_size'].median():.1f}  max={df['fire_size'].max():.0f} acres")
print("\nClass imbalance:")
for cls, pct in (df['fire_size_class'].value_counts(normalize=True)*100).sort_index().items():
    print(f"  {cls}: {pct:.1f}%")

print(f"\nAll plots saved to: {OUT}/")
