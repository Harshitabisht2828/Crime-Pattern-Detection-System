"""
Step 6: Interactive Dashboard (with Tabs)
Crime Pattern Detection System
"""
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Crime Pattern Detection System",
    layout="wide",
    page_icon="🚨",
    initial_sidebar_state="collapsed"
)

# ---------- Custom Styling ----------
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #8B0000, #DC143C, #FF6347);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        margin: 0;
    }
    .main-header p {
        color: #f0f0f0;
        margin-top: 5px;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f, #2b2b45);
        border: 1px solid #DC143C;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(220, 20, 60, 0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #FF6347;
        font-size: 1.8rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #14141f;
    }
    h2, h3 {
        border-left: 4px solid #DC143C;
        padding-left: 10px;
    }
        .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0e0e1a;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2b2b45;
        border-radius: 8px;
        padding: 10px 24px;
        color: #ddd;
        font-weight: 600;
        border: 1px solid #3a3a5a;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3a3a5a;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DC143C !important;
        color: white !important;
        border: 1px solid #DC143C !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1>🚨 Crime Pattern Detection System</h1>
        <p>Interactive dashboard for exploring crime patterns across Indian cities</p>
    </div>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
@st.cache_data
def load_data():
    return pd.read_pickle('cleaned_crime_data.pkl')

df = load_data()

# ---------- Sidebar Filters ----------
st.sidebar.header("🔎 Filters")

cities = sorted(df['City'].unique())
selected_cities = st.sidebar.multiselect("Select City", cities, default=cities)

years = sorted(df['Occurrence_Year'].dropna().unique())
selected_years = st.sidebar.multiselect("Select Year", years, default=years)

domains = sorted(df['Crime Domain'].unique())
selected_domains = st.sidebar.multiselect("Select Crime Domain", domains, default=domains)

time_buckets = ['Morning', 'Afternoon', 'Evening', 'Night']
selected_time = st.sidebar.multiselect("Select Time of Day", time_buckets, default=time_buckets)

# Apply filters
filtered = df[
    (df['City'].isin(selected_cities)) &
    (df['Occurrence_Year'].isin(selected_years)) &
    (df['Crime Domain'].isin(selected_domains)) &
    (df['Time_Bucket'].isin(selected_time))
]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered)}")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🗺️ Patterns & Hotspots", "🔮 Live Prediction"])

# ================= TAB 1: OVERVIEW =================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Crimes", len(filtered))
    col2.metric("🏙️ Cities Covered", filtered['City'].nunique())
    closure_rate = (filtered['Case Closed'] == 'Yes').mean() * 100 if len(filtered) > 0 else 0
    col3.metric("✅ Case Closure Rate", f"{closure_rate:.1f}%")
    top_city = filtered['City'].value_counts().idxmax() if len(filtered) > 0 else "N/A"
    col4.metric("🔥 Top Crime City", top_city)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Top 10 High-Crime Cities")
        city_counts = filtered['City'].value_counts().head(10).reset_index()
        city_counts.columns = ['City', 'Count']
        fig = px.bar(city_counts, x='Count', y='City', orientation='h', color='Count',
                     color_continuous_scale='Reds')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Crime Domain Distribution")
        domain_counts = filtered['Crime Domain'].value_counts().reset_index()
        domain_counts.columns = ['Domain', 'Count']
        fig = px.pie(domain_counts, names='Domain', values='Count', hole=0.4,
                     color_discrete_sequence=['#8B0000', '#DC143C', '#FF6347', '#FFB6A3'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2: PATTERNS & HOTSPOTS =================
with tab2:
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Crime by Time of Day")
        time_counts = filtered['Time_Bucket'].value_counts().reindex(time_buckets).reset_index()
        time_counts.columns = ['Time', 'Count']
        fig = px.bar(time_counts, x='Time', y='Count', color='Time',
                     color_discrete_sequence=['#FFB6A3', '#FF6347', '#DC143C', '#8B0000'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Crime by Hour of Day")
        hour_counts = filtered['Occurrence_Hour'].value_counts().sort_index().reset_index()
        hour_counts.columns = ['Hour', 'Count']
        fig = px.line(hour_counts, x='Hour', y='Count', markers=True,
                       color_discrete_sequence=['#FF6347'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ Hotspot Heatmap: City vs Crime Domain")
    if len(filtered) > 0:
        heatmap_data = pd.crosstab(filtered['City'], filtered['Crime Domain'])
        fig = px.imshow(heatmap_data, text_auto=True, color_continuous_scale='Reds', aspect='auto')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=700)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected filters.")

# ================= TAB 3: LIVE PREDICTION =================
with tab3:
    st.header("🔮 Live Crime Prediction")
    st.markdown("Enter details below to predict whether a crime is likely to be **Violent**.")

    @st.cache_resource
    def train_model():
        data = df.copy()
        data['Is_Violent'] = (data['Crime Domain'] == 'Violent Crime').astype(int)
        model_features = ['City', 'Time_Bucket', 'Occurrence_Hour', 'Occurrence_MonthNum',
                    'Victim Gender', 'Victim Age']
        data_model = data[model_features + ['Is_Violent']].dropna()

        encoders = {}
        for col in ['City', 'Time_Bucket', 'Victim Gender']:
            le = LabelEncoder()
            data_model[col] = le.fit_transform(data_model[col])
            encoders[col] = le

        X = data_model[model_features]
        y = data_model['Is_Violent']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        return model, encoders, acc, model_features

    model, encoders, acc, model_features = train_model()

    st.caption(f"Model accuracy on test data: {acc*100:.1f}%")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        input_city = st.selectbox("City", cities)
        input_time = st.selectbox("Time of Day", time_buckets)
    with pc2:
        input_hour = st.slider("Hour of Day", 0, 23, 12)
        input_month = st.slider("Month", 1, 12, 6)
    with pc3:
        input_gender = st.selectbox("Victim Gender", sorted(df['Victim Gender'].dropna().unique()))
        input_age = st.slider("Victim Age", 1, 100, 30)

    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([{
            'City': encoders['City'].transform([input_city])[0],
            'Time_Bucket': encoders['Time_Bucket'].transform([input_time])[0],
            'Occurrence_Hour': input_hour,
            'Occurrence_MonthNum': input_month,
            'Victim Gender': encoders['Victim Gender'].transform([input_gender])[0],
            'Victim Age': input_age,
        }])[model_features]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Predicted: **Violent Crime** (Confidence: {probability*100:.1f}%)")
        else:
            st.success(f"✅ Predicted: **Not Violent** (Confidence: {(1-probability)*100:.1f}%)")