import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pathlib
import seaborn as sns
import os

BASE_DIR = pathlib.Path(__file__).parent.resolve()

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

OUT = BASE_DIR / "eda_outputs"
os.makedirs(OUT, exist_ok=True)

# ── Readable column labels ─────────────────────────────────────────────────────────────
COL_LABELS = {
    'fire_size':    'Fire Size (acres)',
    'fire_mag':     'Fire Magnitude',
    'remoteness':   'Remoteness (dist. to city)',
    'Temp_pre_30':  'Temperature 30 Days Prior (°C)',
    'Temp_pre_15':  'Temperature 15 Days Prior (°C)',
    'Temp_pre_7':   'Temperature 7 Days Prior (°C)',
    'Temp_cont':    'Temperature at Containment (°C)',
    'Wind_pre_30':  'Wind Speed 30 Days Prior (m/s)',
    'Wind_pre_15':  'Wind Speed 15 Days Prior (m/s)',
    'Wind_pre_7':   'Wind Speed 7 Days Prior (m/s)',
    'Wind_cont':    'Wind Speed at Containment (m/s)',
    'Hum_pre_30':   'Humidity 30 Days Prior (%)',
    'Hum_pre_15':   'Humidity 15 Days Prior (%)',
    'Hum_pre_7':    'Humidity 7 Days Prior (%)',
    'Hum_cont':     'Humidity at Containment (%)',
    'Prec_pre_30':  'Precipitation 30 Days Prior (mm)',
    'Prec_pre_15':  'Precipitation 15 Days Prior (mm)',
    'Prec_pre_7':   'Precipitation 7 Days Prior (mm)',
    'Prec_cont':    'Precipitation at Containment (mm)',
}

def label(col):
    return COL_LABELS.get(col, col)

# ── Load ─────────────────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(BASE_DIR / "us_wildfire_data.csv", low_memory=False)
print(f"Shape before cleaning: {df.shape}")

# ── Data Cleaning: replace -1 sentinel values with NaN ────────────────────────────
weather_cols = [
    'Temp_pre_30','Temp_pre_15','Temp_pre_7','Temp_cont',
    'Wind_pre_30','Wind_pre_15','Wind_pre_7','Wind_cont',
    'Hum_pre_30','Hum_pre_15','Hum_pre_7','Hum_cont',
    'Prec_pre_30','Prec_pre_15','Prec_pre_7','Prec_cont',
]

print("\n=== Sentinel -1 counts BEFORE cleaning ===")
for col in weather_cols:
    n = (df[col] == -1).sum()
    pct = n / len(df) * 100
    if n > 0:
        print(f"  {label(col):<40} {n:>6} rows ({pct:.1f}%)")

df[weather_cols] = df[weather_cols].replace(-1, np.nan)
print("\nAll -1 sentinel values replaced with NaN.")
print(f"Shape after cleaning: {df.shape}\n")

# ── 1. Missing Values (after cleaning) ────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_df = missing_df[missing_df.missing_count > 0].sort_values('missing_pct', ascending=False)
missing_df.index = [label(c) for c in missing_df.index]

fig, ax = plt.subplots(figsize=(11, max(5, len(missing_df) * 0.42)))
ax.barh(missing_df.index, missing_df.missing_pct, color='tomato')
ax.set_xlabel('% Missing')
ax.set_title('Missing Values by Column (after -1 sentinel removal)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT / "missing_values.png", dpi=150)
plt.close()

# ── 2. Spatial ─────────────────────────────────────────────────────────────────
state_counts = df['state'].value_counts()

fig, ax = plt.subplots(figsize=(14, 5))
state_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Number of Wildfires by State')
ax.set_xlabel('State')
ax.set_ylabel('Number of Fires')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUT / "fires_by_state.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(14, 7))
sc = ax.scatter(df['longitude'], df['latitude'],
                c=np.log1p(df['fire_size']), cmap='YlOrRd', s=1, alpha=0.4)
plt.colorbar(sc, ax=ax, label='log(Fire Size in acres + 1)')
ax.set_xlim(-130, -60)
ax.set_ylim(20, 55)
ax.set_title('Wildfire Locations — Colour = log(Fire Size)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig(OUT / "fire_map.png", dpi=150)
plt.close()

# ── 3. Temporal ────────────────────────────────────────────────────────────────
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_counts = df['discovery_month'].value_counts().reindex(month_order, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 4))
month_counts.plot(kind='bar', ax=ax, color='darkorange', edgecolor='white')
ax.set_title('Wildfire Discoveries by Month of Year')
ax.set_xlabel('Month of Discovery')
ax.set_ylabel('Number of Fires')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "fires_by_month.png", dpi=150)
plt.close()

year_counts = df['disc_pre_year'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(year_counts.index, year_counts.values, marker='o', color='firebrick', linewidth=2)
ax.fill_between(year_counts.index, year_counts.values, alpha=0.15, color='firebrick')
ax.set_title('Wildfire Frequency Over the Years (1991–2015)')
ax.set_xlabel('Year of Discovery')
ax.set_ylabel('Number of Fires')
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(OUT / "fires_by_year.png", dpi=150)
plt.close()

# ── 4. Causes ──────────────────────────────────────────────────────────────────
cause_counts = df['stat_cause_descr'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
cause_counts.plot(kind='barh', ax=ax, color='mediumpurple', edgecolor='white')
ax.set_title('Wildfire Causes')
ax.set_xlabel('Number of Fires')
ax.set_ylabel('Cause of Fire')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT / "fire_causes.png", dpi=150)
plt.close()

# ── 5. Fire Size & Class Imbalance ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(df['fire_size'].dropna(), bins=80, color='darkorange', edgecolor='white', log=True)
axes[0].set_title('Fire Size Distribution (log y-axis)')
axes[0].set_xlabel('Fire Size (acres)')
axes[0].set_ylabel('Number of Fires (log scale)')
axes[1].hist(np.log1p(df['fire_size'].dropna()), bins=60, color='steelblue', edgecolor='white')
axes[1].set_title('log(Fire Size + 1) Distribution')
axes[1].set_xlabel('log(Fire Size in acres + 1)')
axes[1].set_ylabel('Number of Fires')
plt.suptitle('Fire Size — Raw vs Log-Transformed', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "fire_size_dist.png", dpi=150)
plt.close()

class_labels = {
    'A': 'A (< 0.25 ac)', 'B': 'B (0.25–10 ac)', 'C': 'C (10–100 ac)',
    'D': 'D (100–300 ac)', 'E': 'E (300–1k ac)', 'F': 'F (1k–5k ac)', 'G': 'G (> 5k ac)'
}
class_counts = df['fire_size_class'].value_counts().sort_index()
class_pct = (class_counts / class_counts.sum() * 100).round(2)
readable_labels = [class_labels.get(c, c) for c in class_counts.index]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(readable_labels, class_counts.values, color='coral', edgecolor='white')
ax.set_title('Fire Size Class Distribution')
ax.set_xlabel('Fire Size Class')
ax.set_ylabel('Number of Fires')
for bar, pct in zip(bars, class_pct.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(OUT / "class_imbalance.png", dpi=150)
plt.close()

# ── 6. Correlations ────────────────────────────────────────────────────────────
corr_cols = weather_cols + ['fire_size', 'fire_mag', 'remoteness']
corr_df = df[corr_cols].corr()
corr_df_display = corr_df.copy()
corr_df_display.index = [label(c) for c in corr_df_display.index]
corr_df_display.columns = [label(c) for c in corr_df_display.columns]

fig, ax = plt.subplots(figsize=(16, 12))
mask = np.zeros_like(corr_df_display, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_df_display, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 6})
ax.set_title('Correlation Matrix — Weather Features & Fire Size (cleaned data)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=150)
plt.close()

top_corr = corr_df['fire_size'].drop('fire_size').abs().sort_values(ascending=False).head(4).index.tolist()

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, feat in zip(axes, top_corr):
    sample = df[[feat, 'fire_size']].dropna().sample(min(3000, len(df)), random_state=42)
    ax.scatter(sample[feat], np.log1p(sample['fire_size']), alpha=0.2, s=5, color='steelblue')
    ax.set_xlabel(label(feat), fontsize=8)
    ax.set_ylabel('log(Fire Size + 1)')
    ax.set_title(label(feat), fontsize=8)
plt.suptitle('Top Weather Features vs log(Fire Size) — after cleaning', y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "scatter_top_features.png", dpi=150)
plt.close()

# ── 7. Vegetation ──────────────────────────────────────────────────────────────
veg_labels = {
    1:'Tropical Evergreen Broadleaf Forest', 2:'Tropical Deciduous Broadleaf Forest',
    3:'Temperate Evergreen Broadleaf Forest', 4:'Temperate Evergreen Needleleaf Forest',
    5:'Temperate Deciduous Broadleaf Forest', 6:'Boreal Evergreen Needleleaf Forest',
    7:'Boreal Deciduous Needleleaf Forest', 8:'Savanna',
    9:'C3 Grassland / Steppe', 10:'C4 Grassland / Steppe',
    11:'Dense Shrubland', 12:'Open Shrubland', 13:'Tundra', 14:'Desert',
    15:'Polar Desert / Rock / Ice', 16:'Sec. Tropical Evergreen Broadleaf',
    17:'Sec. Tropical Deciduous Broadleaf', 18:'Sec. Temperate Evergreen Broadleaf',
    19:'Sec. Temperate Evergreen Needleleaf', 20:'Sec. Temperate Deciduous Broadleaf',
    21:'Sec. Boreal Evergreen Needleleaf', 22:'Sec. Boreal Deciduous Needleleaf',
    23:'Water / Rivers', 24:'C3 Cropland', 25:'C4 Cropland',
    26:'C3 Pastureland', 27:'C4 Pastureland', 28:'Urban Land'
}
veg_counts = df['Vegetation'].value_counts().sort_index()
veg_counts.index = veg_counts.index.map(lambda x: veg_labels.get(int(x), f'Type {int(x)}') if pd.notna(x) else 'Unknown')

fig, ax = plt.subplots(figsize=(13, 6))
veg_counts.sort_values(ascending=True).plot(kind='barh', ax=ax, color='forestgreen', edgecolor='white')
ax.set_title('Number of Fires by Dominant Vegetation Type')
ax.set_xlabel('Number of Fires')
ax.set_ylabel('Vegetation Type')
plt.tight_layout()
plt.savefig(OUT / "fires_by_vegetation.png", dpi=150)
plt.close()

# ── 8. Outliers ────────────────────────────────────────────────────────────────
outlier_cols = ['fire_size', 'Temp_pre_7', 'Wind_pre_7', 'Hum_pre_7', 'Prec_pre_7']
outlier_labels = [label(c) for c in outlier_cols]

fig, axes = plt.subplots(1, len(outlier_cols), figsize=(18, 5))
for ax, col, lbl in zip(axes, outlier_cols, outlier_labels):
    data = df[col].dropna()
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='navy'),
               medianprops=dict(color='red', linewidth=2),
               flierprops=dict(marker='o', markersize=2, alpha=0.3, color='grey'))
    ax.set_title(lbl, fontsize=8)
    ax.set_xticks([])
plt.suptitle('Outlier Detection — Box Plots (cleaned data)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "outliers_boxplots.png", dpi=150)
plt.close()

print(f"\nAll plots saved to: {OUT}/")
