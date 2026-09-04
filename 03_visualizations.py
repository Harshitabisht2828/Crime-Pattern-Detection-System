"""
Step 3: Visualizations
Crime Pattern Detection System
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned data
df = pd.read_pickle('cleaned_crime_data.pkl')

# Create a folder to save all charts
os.makedirs('charts', exist_ok=True)

# 1. Top 10 High-Crime Cities
fig, ax = plt.subplots(figsize=(10, 6))
df['City'].value_counts().head(10).plot(kind='barh', ax=ax, color='crimson')
ax.set_title('Top 10 High-Crime Cities', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Crimes')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('charts/top_cities.png', dpi=120)
plt.close()
print("Saved: charts/top_cities.png")

# 2. Crime Domain Distribution (Pie Chart)
fig, ax = plt.subplots(figsize=(8, 8))
df['Crime Domain'].value_counts().plot(
    kind='pie', autopct='%1.1f%%', ax=ax,
    colors=['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
)
ax.set_title('Crime Domain Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('charts/crime_domain_pie.png', dpi=120)
plt.close()
print("Saved: charts/crime_domain_pie.png")

# 3. Crime by Time of Day
fig, ax = plt.subplots(figsize=(8, 6))
order = ['Morning', 'Afternoon', 'Evening', 'Night']
df['Time_Bucket'].value_counts().reindex(order).plot(kind='bar', ax=ax, color='#8e44ad')
ax.set_title('Crime Frequency by Time of Day', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Crimes')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/time_of_day.png', dpi=120)
plt.close()
print("Saved: charts/time_of_day.png")

# 4. Crime by Hour of Day (Line Chart)
fig, ax = plt.subplots(figsize=(12, 5))
df['Occurrence_Hour'].value_counts().sort_index().plot(
    kind='line', marker='o', ax=ax, color='darkred'
)
ax.set_title('Crime Occurrence by Hour of Day', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour (0-23)')
ax.set_ylabel('Number of Crimes')
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig('charts/hourly_trend.png', dpi=120)
plt.close()
print("Saved: charts/hourly_trend.png")

print()
print("All charts saved in the 'charts' folder!")