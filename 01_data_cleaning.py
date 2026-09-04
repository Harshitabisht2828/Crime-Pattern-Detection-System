"""
Step 1: Data Cleaning & Preprocessing
Crime Pattern Detection System
"""
import pandas as pd

RAW_PATH = 'crime_dataset_india.csv'
OUT_PATH = 'cleaned_crime_data.pkl'


def smart_parse_datetime(series):
    """Dataset has mixed MM-DD-YYYY and DD-MM-YYYY formats.
    Try MM-DD-YYYY first, fall back to DD-MM-YYYY for rows that fail."""
    dt1 = pd.to_datetime(series, format='%m-%d-%Y %H:%M', errors='coerce')
    dt2 = pd.to_datetime(series, format='%d-%m-%Y %H:%M', errors='coerce')
    return dt1.fillna(dt2)


def load_and_clean():
    df = pd.read_csv(RAW_PATH)

    # Parse all datetime columns
    df['Date of Occurrence'] = smart_parse_datetime(df['Date of Occurrence'])
    df['Date Reported'] = smart_parse_datetime(df['Date Reported'])
    df['Time of Occurrence'] = smart_parse_datetime(df['Time of Occurrence'])
    df['Date Case Closed'] = smart_parse_datetime(df['Date Case Closed'])

    # Feature engineering: time-based features
    df['Occurrence_Hour'] = df['Time of Occurrence'].dt.hour
    df['Occurrence_DayOfWeek'] = df['Date of Occurrence'].dt.day_name()
    df['Occurrence_Month'] = df['Date of Occurrence'].dt.month_name()
    df['Occurrence_Year'] = df['Date of Occurrence'].dt.year
    df['Occurrence_MonthNum'] = df['Date of Occurrence'].dt.month

    # Reporting delay (days between crime occurring and being reported)
    df['Reporting_Delay_Days'] = (df['Date Reported'] - df['Date of Occurrence']).dt.days

    # Clean categoricals
    df['Weapon Used'] = df['Weapon Used'].fillna('Not Specified')
    df['Case Closed'] = df['Case Closed'].str.strip()

    # Time-of-day bucket (useful for pattern grouping)
    def time_bucket(h):
        if pd.isna(h):
            return 'Unknown'
        h = int(h)
        if 5 <= h < 12:
            return 'Morning'
        elif 12 <= h < 17:
            return 'Afternoon'
        elif 17 <= h < 21:
            return 'Evening'
        else:
            return 'Night'

    df['Time_Bucket'] = df['Occurrence_Hour'].apply(time_bucket)

    return df


if __name__ == '__main__':
    df = load_and_clean()
    df.to_pickle(OUT_PATH)
    print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df['Date of Occurrence'].min()} -> {df['Date of Occurrence'].max()}")
    print(f"Saved to {OUT_PATH}")