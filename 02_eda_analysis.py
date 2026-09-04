"""
Step 2: Exploratory Data Analysis (EDA)
Crime Pattern Detection System
"""
import pandas as pd

# Load the cleaned data we saved in step 1
df = pd.read_pickle('cleaned_crime_data.pkl')

print("=" * 50)
print("TOP 10 CRIME TYPES")
print("=" * 50)
print(df['Crime Description'].value_counts().head(10))

print()
print("=" * 50)
print("CRIME DOMAIN DISTRIBUTION (Violent / Other / etc.)")
print("=" * 50)
print(df['Crime Domain'].value_counts())

print()
print("=" * 50)
print("TOP 10 HIGH-CRIME CITIES")
print("=" * 50)
print(df['City'].value_counts().head(10))

print()
print("=" * 50)
print("CRIME BY TIME OF DAY")
print("=" * 50)
order = ['Morning', 'Afternoon', 'Evening', 'Night']
print(df['Time_Bucket'].value_counts().reindex(order))

print()
print("=" * 50)
print("CASE CLOSURE RATE")
print("=" * 50)
print(df['Case Closed'].value_counts(normalize=True) * 100)