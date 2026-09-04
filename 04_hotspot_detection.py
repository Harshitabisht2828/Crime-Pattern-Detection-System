"""
Step 4: High-Crime Area (Hotspot) Detection
Crime Pattern Detection System
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_pickle('cleaned_crime_data.pkl')

# 1. Cross-tab: City vs Crime Domain
print("=" * 60)
print("CITY vs CRIME DOMAIN (counts)")
print("=" * 60)
city_domain = pd.crosstab(df['City'], df['Crime Domain'])
print(city_domain)

# 2. Which crime domain is most common in each city
print()
print("=" * 60)
print("TOP CRIME DOMAIN PER CITY")
print("=" * 60)
top_domain_per_city = city_domain.idxmax(axis=1)
print(top_domain_per_city)

# 3. Heatmap: City vs Crime Domain (visual hotspot map)
plt.figure(figsize=(10, 12))
sns.heatmap(city_domain, annot=True, fmt='d', cmap='Reds', linewidths=0.5)
plt.title('Crime Hotspot Heatmap: City vs Crime Domain', fontsize=14, fontweight='bold')
plt.xlabel('Crime Domain')
plt.ylabel('City')
plt.tight_layout()
plt.savefig('charts/hotspot_heatmap.png', dpi=120)
plt.close()
print()
print("Saved: charts/hotspot_heatmap.png")

# 4. Overall risk score per city (total crimes, sorted)
print()
print("=" * 60)
print("CITY RISK RANKING (by total crime count)")
print("=" * 60)
risk_ranking = df['City'].value_counts().reset_index()
risk_ranking.columns = ['City', 'Total_Crimes']
risk_ranking['Risk_Level'] = pd.qcut(
    risk_ranking['Total_Crimes'], q=3, labels=['Low', 'Medium', 'High']
)
print(risk_ranking)