import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import plotly.express as px

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Road Accident Analytics Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# =====================================================================
# LOAD DATA
# =====================================================================
@st.cache_data
def load_data():
    try:
        # put your file here (same folder as this .py)
        df = pd.read_csv("Accident_Information.csv")
    except FileNotFoundError:
        st.error("CSV file not found. Place 'Accident_Information.csv' in the app folder.")
        return None
    return df

df = load_data()
if df is None:
    st.stop()

# =====================================================================
# PREPROCESS
# =====================================================================
@st.cache_data
def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_clean = df_raw.copy()

    # Date / time columns can differ by dataset; adjust if needed
    if "Date" in df_clean.columns:
        df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
        df_clean["Year"] = df_clean["Date"].dt.year
        df_clean["Month"] = df_clean["Date"].dt.month
        df_clean["DayOfWeek"] = df_clean["Date"].dt.day_name()

    if "Time" in df_clean.columns:
        df_clean["Hour"] = pd.to_datetime(df_clean["Time"], errors="coerce").dt.hour

    return df_clean

df_clean = preprocess_data(df)

# =====================================================================
# SIDEBAR FILTERS
# =====================================================================
st.sidebar.title("🎛️ Dashboard Filters")
st.sidebar.markdown("---")

df_filtered = df_clean.copy()
selected_years = []  # make sure variable always exists

# Year filter
if "Year" in df_filtered.columns:
    years = sorted(df_filtered["Year"].dropna().unique())
    selected_years = st.sidebar.multiselect(
        "Select Year(s)",
        options=years,
        default=[years[-1]] if len(years) > 0 else []
    )
    if selected_years:
        df_filtered = df_filtered[df_filtered["Year"].isin(selected_years)]

# Severity filter
if "Accident_Severity" in df_filtered.columns:
    severities = df_filtered["Accident_Severity"].dropna().unique()
    selected_severity = st.sidebar.multiselect(
        "Select Accident Severity",
        options=severities,
        default=list(severities)
    )
    df_filtered = df_filtered[df_filtered["Accident_Severity"].isin(selected_severity)]

# Weather filter
if "Weather_Conditions" in df_filtered.columns:
    weather = df_filtered["Weather_Conditions"].dropna().unique()
    selected_weather = st.sidebar.multiselect(
        "Select Weather Conditions",
        options=weather,
        default=list(weather[:3]) if len(weather) > 3 else list(weather)
    )
    df_filtered = df_filtered[df_filtered["Weather_Conditions"].isin(selected_weather)]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 Filtered records: {len(df_filtered):,}")

# =====================================================================
# MAIN TITLE
# =====================================================================
st.title("🚗 Road Accident Analytics Dashboard")
st.markdown("**Task 4 – Matrix Industries Data Science Internship**")
st.markdown("---")

# =====================================================================
# KPIs
# =====================================================================
st.subheader("📈 Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)

with k1:
    total_acc = len(df_filtered)
    st.metric(
        "Total Accidents",
        f"{total_acc:,}",
        delta="Filtered" if selected_years else "All years"
    )

with k2:
    if "Accident_Severity" in df_filtered.columns and total_acc > 0:
        fatal = (df_filtered["Accident_Severity"] == "Fatal").sum()
        pct = fatal / total_acc * 100
        st.metric("Fatal Accidents", f"{fatal:,}", f"{pct:.1f}%")

with k3:
    if "Number_of_Vehicles" in df_filtered.columns:
        st.metric("Total Vehicles Involved", f"{df_filtered['Number_of_Vehicles'].sum():,}")
    elif "Number_of_Casualties" in df_filtered.columns:
        st.metric("Total Casualties", f"{df_filtered['Number_of_Casualties'].sum():,}")

with k4:
    loc_col = "Location_Easting_OSGR" if "Location_Easting_OSGR" in df_filtered.columns else None
    unique_locations = df_filtered[loc_col].nunique() if loc_col else 0
    st.metric("Unique Locations", f"{unique_locations:,}")

st.markdown("---")

# =====================================================================
# SECTION 1 – SEVERITY & HOUR
# =====================================================================
c1, c2 = st.columns(2)

with c1:
    st.subheader("🚨 Accidents by Severity")
    if "Accident_Severity" in df_filtered.columns:
        severity_counts = df_filtered["Accident_Severity"].value_counts()
        fig_sev = px.bar(
            x=severity_counts.index,
            y=severity_counts.values,
            labels={"x": "Severity", "y": "Number of Accidents"},
            color=severity_counts.index,
        )
        fig_sev.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_sev, use_container_width=True)

        for sev, cnt in severity_counts.items():
            st.write(f"- **{sev}**: {cnt:,} ({cnt/total_acc*100:.1f}%)")

with c2:
    st.subheader("⏰ Accidents by Hour of Day")
    if "Hour" in df_filtered.columns:
        hourly = df_filtered["Hour"].dropna().value_counts().sort_index()
        fig_hr = px.bar(
            x=hourly.index,
            y=hourly.values,
            labels={"x": "Hour of Day", "y": "Number of Accidents"},
        )
        fig_hr.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_hr, use_container_width=True)

        if not hourly.empty:
            peak_hour = int(hourly.idxmax())
            st.write(f"Peak hour: **{peak_hour}:00–{(peak_hour+1)%24}:00**")

st.markdown("---")

# =====================================================================
# SECTION 2 – WEATHER & DAY OF WEEK
# =====================================================================
c3, c4 = st.columns(2)

with c3:
    st.subheader("🌤️ Accidents by Weather Condition")
    if "Weather_Conditions" in df_filtered.columns:
        weather_counts = df_filtered["Weather_Conditions"].value_counts().head(10)
        fig_w = px.bar(
            x=weather_counts.values,
            y=weather_counts.index,
            orientation="h",
            labels={"x": "Number of Accidents", "y": "Weather Condition"},
        )
        fig_w.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_w, use_container_width=True)

with c4:
    st.subheader("📅 Accidents by Day of Week")
    if "DayOfWeek" in df_filtered.columns:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = df_filtered["DayOfWeek"].value_counts().reindex(order)
        fig_d = px.bar(
            x=day_counts.index,
            y=day_counts.values,
            labels={"x": "Day of Week", "y": "Number of Accidents"},
        )
        fig_d.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_d, use_container_width=True)

st.markdown("---")

# =====================================================================
# SECTION 3 – ROAD CONDITIONS
# =====================================================================
st.subheader("🛣️ Road & Lighting Conditions")
r1, r2, r3 = st.columns(3)

with r1:
    if "Road_Surface_Conditions" in df_filtered.columns:
        surf = df_filtered["Road_Surface_Conditions"].value_counts()
        fig_s = px.pie(values=surf.values, names=surf.index, hole=0.4)
        fig_s.update_layout(height=300)
        st.write("Road Surface Conditions")
        st.plotly_chart(fig_s, use_container_width=True)

with r2:
    if "Light_Conditions" in df_filtered.columns:
        light = df_filtered["Light_Conditions"].value_counts()
        fig_l = px.pie(values=light.values, names=light.index, hole=0.4)
        fig_l.update_layout(height=300)
        st.write("Light Conditions")
        st.plotly_chart(fig_l, use_container_width=True)

with r3:
    if "Speed_limit" in df_filtered.columns:
        speed = df_filtered["Speed_limit"].value_counts().head(5)
        fig_sp = px.pie(values=speed.values, names=speed.index, hole=0.4)
        fig_sp.update_layout(height=300)
        st.write("Top Speed Limits")
        st.plotly_chart(fig_sp, use_container_width=True)

st.markdown("---")

# =====================================================================
# SECTION 4 – TRENDS OVER TIME
# =====================================================================
st.subheader("📊 Monthly Accident Trend")

if "Date" in df_filtered.columns:
    # ensure Date is datetime
    dates = pd.to_datetime(df_filtered["Date"], errors="coerce")
    df_temp = df_filtered.copy()
    df_temp["Date"] = dates

    monthly = df_temp.groupby(df_temp["Date"].dt.to_period("M")).size()
    month_labels = monthly.index.astype(str)

    fig_tr = px.line(
        x=month_labels,
        y=monthly.values,
        labels={"x": "Month", "y": "Number of Accidents"},
    )
    fig_tr.update_layout(height=350)
    st.plotly_chart(fig_tr, use_container_width=True)

st.markdown("---")

# =====================================================================
# SECTION 5 – SUMMARY TABLES
# =====================================================================
st.subheader("📋 Dataset Summary")

s1, s2 = st.columns(2)

with s1:
    st.write("**Numeric Columns**")
    num_cols = df_filtered.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        st.dataframe(df_filtered[num_cols].describe(), use_container_width=True)

with s2:
    st.write("**Missing Values (Top 10)**")
    missing = df_filtered.isnull().sum()
    missing_percent = (missing / len(df_filtered)) * 100
    missing_df = (
        pd.DataFrame(
            {"Column": missing.index,
             "Missing Count": missing.values,
             "Percentage": missing_percent.values}
        )
        .sort_values("Missing Count", ascending=False)
        .head(10)
    )
    st.dataframe(missing_df, use_container_width=True)

st.markdown("---")

# =====================================================================
# SECTION 6 – RAW DATA VIEWER
# =====================================================================
st.subheader("🔍 Raw Data Explorer")

if st.checkbox("Show raw data"):
    c1, c2 = st.columns([3, 1])

    with c1:
        num_rows = st.slider("Number of rows to display", 5, 100, 20)

    with c2:
        if st.button("Download as CSV"):
            csv = df_filtered.head(num_rows).to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="accident_data_filtered.csv",
                mime="text/csv",
            )

    st.dataframe(df_filtered.head(num_rows), use_container_width=True)

st.markdown("---")

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("### 📌 About This Dashboard")
st.info(
    """
**Task 4: Road Accident Analytics Dashboard**
- Source: UK Road Safety Dataset from Kaggle
- Purpose: Interactive analysis of traffic accident patterns
- Skills Used: Data cleaning, EDA, Streamlit, Plotly, Pandas
- Matrix Industries Data Science Internship
"""
)

st.caption(
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Total records: {len(df_clean):,}"
)
