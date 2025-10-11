# olympic_data_analytics_final_fixed.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# Data Preprocessing
# -------------------------------
def preprocess(df, region_df):
    """Clean, merge, and prepare Olympic data."""
    if df is None or region_df is None:
        raise ValueError("Input dataframes cannot be None")

    df = df[df['Season'] == 'Summer'].copy()

    cols_to_keep = ['ID','Name','Sex','Age','Height','Weight','Team','NOC','Games','Year','City','Sport','Event','Medal']
    df = df[cols_to_keep]

    region_df = region_df.drop_duplicates(subset=['NOC'])
    df = df.merge(region_df, on='NOC', how='left')
    df.drop_duplicates(inplace=True)

    # Make medal columns safe (even if missing)
    medal_dummies = pd.get_dummies(df['Medal'], dtype='int8')
    for col in ['Gold', 'Silver', 'Bronze']:
        if col not in medal_dummies.columns:
            medal_dummies[col] = 0
    df = pd.concat([df, medal_dummies], axis=1)

    return df


# -------------------------------
# Helper Functions
# -------------------------------
def fetch_medal_tally(df, year, country):
    """Compute medal tally safely."""
    if df is None or df.empty:
        return pd.DataFrame()

    medal_df = df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'])
    flag = 0

    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    elif year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    elif year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    else:
        temp_df = medal_df[(medal_df['Year'] == int(year)) & (medal_df['region'] == country)]

    if temp_df.empty:
        return pd.DataFrame(columns=['region','Gold','Silver','Bronze','total'])

    # Group and ensure missing columns default to 0
    grouped = temp_df.groupby('region' if flag == 0 else 'Year').sum(numeric_only=True)
    for col in ['Gold', 'Silver', 'Bronze']:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped['total'] = grouped['Gold'] + grouped['Silver'] + grouped['Bronze']
    grouped = grouped.reset_index()

    # Sort properly
    if flag == 0:
        grouped = grouped.sort_values('Gold', ascending=False)
    else:
        grouped = grouped.sort_values('Year')

    return grouped


def country_year_list(df):
    """Return dropdown lists for years and countries."""
    if df is None or df.empty:
        return ['Overall'], ['Overall']
    years = sorted(df['Year'].dropna().unique().tolist())
    years.insert(0, 'Overall')
    countries = sorted(df['region'].dropna().unique().tolist())
    countries.insert(0, 'Overall')
    return years, countries


def yearwise_medal_tally(df, country):
    """Year-wise medal trend."""
    temp_df = df.dropna(subset=['Medal']).drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'])
    new_df = temp_df[temp_df['region'] == country]
    if new_df.empty:
        return pd.DataFrame(columns=['Year','Medal'])
    return new_df.groupby('Year').count()['Medal'].reset_index()


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Olympic Data Analytics", layout="wide")
st.sidebar.title("🏅 Olympics Data Analysis")

st.sidebar.image(
    "https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport.png",
    use_container_width=True
)

# File uploaders
athlete_events_file = st.sidebar.file_uploader("📂 Upload athlete_events.csv", type=["csv"])
region_df_file = st.sidebar.file_uploader("📂 Upload noc_regions.csv", type=["csv"])

df, region_df = None, None

# Load CSV files
if athlete_events_file:
    try:
        df = pd.read_csv(athlete_events_file)
        st.sidebar.success("✅ athlete_events.csv loaded.")
    except Exception as e:
        st.sidebar.error(f"❌ Error reading athlete_events.csv: {e}")

if region_df_file:
    try:
        region_df = pd.read_csv(region_df_file)
        st.sidebar.success("✅ noc_regions.csv loaded.")
    except Exception as e:
        st.sidebar.error(f"❌ Error reading noc_regions.csv: {e}")

if df is None or region_df is None:
    st.warning("⚠️ Please upload both CSV files to proceed.")
    st.stop()

# Preprocessing
try:
    with st.spinner("🔄 Processing data..."):
        df = preprocess(df, region_df)
    st.success("✅ Data processed successfully!")
except Exception as e:
    st.error(f"❌ Preprocessing error: {e}")
    st.stop()

# Dashboard
st.title("🏆 Olympic Data Dashboard")

years, countries = country_year_list(df)
col1, col2 = st.columns(2)
with col1:
    selected_year = st.selectbox("Select Year", years)
with col2:
    selected_country = st.selectbox("Select Country", countries)

st.subheader("🥇 Medal Tally")
medal_tally = fetch_medal_tally(df, selected_year, selected_country)

if medal_tally is not None and not medal_tally.empty:
    st.dataframe(medal_tally)
else:
    st.info("No data available for the selected filters.")

# Medal trend
if selected_country != 'Overall':
    st.subheader(f"📈 Medal Trend for {selected_country}")
    trend_df = yearwise_medal_tally(df, selected_country)
    if not trend_df.empty:
        fig = px.line(trend_df, x='Year', y='Medal', markers=True,
                      title=f"Medal Trend Over Years for {selected_country}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No medal trend data available.")
else:
    st.info("Select a specific country to view its medal trend.")
