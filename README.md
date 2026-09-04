# Crime Pattern Detection System

## Project Overview
Ye project "Crime Dataset India" (Kaggle) ka use karke crime patterns detect karta hai —
kaunse crime types zyada hote hain, kaunsi cities high-risk hain, kis time crime zyada
hota hai, aur ek basic ML model se prediction bhi try kiya gaya hai.

## Dataset
- **Source:** Kaggle - Crime Dataset India
- **Records:** 40,160 crime cases
- **Date Range:** 2020 - 2024
- **Columns:** City, Crime Description, Crime Domain, Date/Time of Occurrence,
  Victim Age/Gender, Weapon Used, Case Closed status, etc.

## Project Pipeline (Step-by-Step)

| Step | File | Purpose |
|------|------|---------|
| 1 | `01_data_cleaning.py` | Raw CSV clean karna, dates parse karna, time-features banana |
| 2 | `02_eda_analysis.py` | Basic patterns nikalna (top crimes, cities, time trends) |
| 3 | `03_visualizations.py` | Charts banana (bar, pie, line graphs) |
| 4 | `04_hotspot_detection.py` | City-wise crime hotspots + heatmap + risk ranking |
| 5 | `05_prediction_model.py` | ML model se "Violent Crime" predict karna |

## Key Findings

1. **High-Crime Cities:** Delhi, Mumbai, aur Bangalore sabse zyada crime records
   wali cities hain.
2. **Time Pattern:** Crimes sabse zyada **Night** time me hote hain, uske baad
   Morning, Afternoon, aur sabse kam Evening me.
3. **Crime Domain Split:** Zyadatar crimes "Other Crime" category me aate hain,
   uske baad Violent Crime, Fire Accident, aur Traffic Fatality.
4. **Case Closure Rate:** Lagbhag 50% cases close ho paate hain.
5. **Data Limitation:** Crime-type counts aur day/month/year distribution
   almost perfectly uniform hain — jo batata hai ye dataset synthetically
   generated hai (real-world crime data me itna balance nahi milta). Isliye
   prediction model ki accuracy bhi limited hai.

## How to Run