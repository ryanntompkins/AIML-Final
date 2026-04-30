# EDA Results — U.S. Wildfire Dataset

## Dataset Overview
| | |
|---|---|
| Total records | 55,367 |
| Total features | 43 |
| Years covered | 1991 – 2015 |
| States | 51 |

---

## Key Findings

### Spatial
- **Top state: TX** (6,080 fires), followed by GA (4,811), CA (3,847), MS (3,493), FL (3,115)
- Western states (CA, WA, OR) produce the largest fires by size

### Temporal
- **Peak month: March** (8,271 fires) — spring is the most active season
- Fire frequency is relatively stable across 1991–2015 with some year-to-year variation

### Causes
| Cause | Count |
|---|---|
| Debris Burning | 14,278 |
| Arson | 9,724 |
| Miscellaneous | 8,344 |
| Lightning | 8,218 |
| Missing/Undefined | 5,063 |

### Fire Size
- **Mean: 2,104 acres** | **Median: 4 acres** | **Max: 606,945 acres**
- Heavy right skew — a tiny number of fires account for most burned area
- Use `log(fire_size + 1)` for modelling

### Class Imbalance (fire_size_class)
| Class | Count | % |
|---|---|---|
| B | 36,522 | 66.0% |
| C | 10,811 | 19.5% |
| G | 3,972 | 7.2% |
| F | 1,968 | 3.6% |
| D | 1,394 | 2.5% |
| E | 700 | 1.3% |

> Class B dominates at 66% — Rishi's dummy baseline will exploit this.

### Missing Values
| Column | Missing | % |
|---|---|---|
| cont_date_final | 29,735 | 53.7% |
| fire_name | 29,454 | 53.2% |
| cont_clean_date | 27,890 | 50.4% |
| putout_time | 27,890 | 50.4% |
| disc_date_final | 26,659 | 48.2% |

> Weather columns (Temp, Wind, Hum, Prec) have **no missing values** — good news for Karim.

### Top Features Correlated with fire_size
1. `fire_mag` (scaled version of fire_size — expected)
2. `Temp_cont` — temperature at containment
3. `remoteness` — distance to nearest city
4. `Wind_cont` — wind at containment

### Outliers (IQR method)
| Column | Outliers | % |
|---|---|---|
| fire_size | 9,878 | 17.8% |
| Prec_pre_7 | 10,087 | 18.2% |
| Temp_pre_7 | 1 | ~0% |
| Wind_pre_7 | 15 | ~0% |

> `fire_size` and `Prec_pre_7` have significant outliers — flag for Karim.

---

## Outputs
All plots saved in `eda_outputs/` after running `python run_eda.py`:
- `fires_by_state.png`
- `fire_map.png`
- `fires_by_month.png`
- `fires_by_year.png`
- `fire_causes.png`
- `fire_size_dist.png`
- `class_imbalance.png`
- `correlation_heatmap.png`
- `scatter_top_features.png`
- `fires_by_vegetation.png`
- `outliers_boxplots.png`
- `missing_values.png`
