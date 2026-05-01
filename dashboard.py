import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Accidents Dashboard",
    layout="wide"
)

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    CSV_FILE = 'US_Accidents_March23_sampled_500k.csv'
    if not os.path.exists(CSV_FILE):
        st.error("CSV file not found. Make sure US_Accidents_March23_sampled_500k.csv is in the same folder.")
        st.stop()

    KEEP_COLS = [
        'ID', 'Severity', 'Start_Time',
        'State', 'City',
        'Temperature(F)', 'Visibility(mi)', 'Precipitation(in)',
        'Weather_Condition',
        'Junction', 'Crossing', 'Traffic_Signal',
        'Sunrise_Sunset'
    ]

    df = pd.read_csv(CSV_FILE, usecols=KEEP_COLS, low_memory=False)

    # Clean
    df['Temperature(F)']    = df['Temperature(F)'].fillna(df['Temperature(F)'].median())
    df['Visibility(mi)']    = df['Visibility(mi)'].fillna(df['Visibility(mi)'].median())
    df['Precipitation(in)'] = df['Precipitation(in)'].fillna(0)
    df['Weather_Condition'] = df['Weather_Condition'].fillna('Unknown')
    df['Sunrise_Sunset']    = df['Sunrise_Sunset'].fillna('Unknown')
    df.dropna(subset=['Severity', 'State'], inplace=True)

    # Datetime
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour']       = df['Start_Time'].dt.hour
    df['Month']      = df['Start_Time'].dt.month_name()

    # Booleans
    for col in ['Junction', 'Crossing', 'Traffic_Signal']:
        df[col] = df[col].astype(bool)

    return df

df = load_data()

# ── SIDEBAR FILTERS ────────────────────────────────────────────────────────────
st.sidebar.title("Filters")
st.sidebar.markdown("Use filters to explore the data")

# State filter
all_states = sorted(df['State'].unique().tolist())
selected_states = st.sidebar.multiselect(
    "Select State(s)",
    options=all_states,
    default=all_states[:10],
    help="Filter by US state"
)

# Weather filter
top_weather = df['Weather_Condition'].value_counts().head(10).index.tolist()
selected_weather = st.sidebar.multiselect(
    "Select Weather Condition(s)",
    options=top_weather,
    default=top_weather[:5],
    help="Filter by weather condition"
)

# Hour filter
hour_range = st.sidebar.slider(
    "Select Hour Range",
    min_value=0,
    max_value=23,
    value=(0, 23),
    help="Filter by hour of day (0=midnight, 12=noon)"
)

# ── APPLY FILTERS ──────────────────────────────────────────────────────────────
if not selected_states:
    st.warning("Please select at least one State from the sidebar.")
    st.stop()

if not selected_weather:
    st.warning("Please select at least one Weather Condition from the sidebar.")
    st.stop()

filtered_df = df[
    (df['State'].isin(selected_states)) &
    (df['Weather_Condition'].isin(selected_weather)) &
    (df['Hour'] >= hour_range[0]) &
    (df['Hour'] <= hour_range[1])
]

# ── TITLE ──────────────────────────────────────────────────────────────────────
st.title("Traffic Accidents & Road Safety Dashboard")
st.markdown("**Dataset:** US-Accidents (500k sample) | Moosavi et al., 2019")
st.markdown("---")

# ── KEY METRICS ROW ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Accidents", f"{len(filtered_df):,}")

with col2:
    avg_sev = filtered_df['Severity'].mean()
    st.metric("Avg Severity", f"{avg_sev:.2f} / 4.0")

with col3:
    if len(filtered_df) > 0:
        top_state = filtered_df['State'].value_counts().index[0]
        top_state_count = filtered_df['State'].value_counts().iloc[0]
        st.metric("Top State", f"{top_state}", f"{top_state_count:,} accidents")
    else:
        st.metric("Top State", "N/A")

with col4:
    night_pct = (filtered_df['Sunrise_Sunset'] == 'Night').mean() * 100
    st.metric("Night Accidents", f"{night_pct:.1f}%")

st.markdown("---")

# ── ROW 1: State + Severity ────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Accidents by State")
    state_counts = filtered_df['State'].value_counts().head(15).reset_index()
    state_counts.columns = ['State', 'Count']
    fig = px.bar(
        state_counts, x='State', y='Count',
        color='Count', color_continuous_scale='Blues',
        title="Top 15 States by Accident Count"
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Severity Distribution")
    sev_counts = filtered_df['Severity'].value_counts().sort_index().reset_index()
    sev_counts.columns = ['Severity', 'Count']
    sev_counts['Severity'] = sev_counts['Severity'].astype(str)
    fig = px.pie(
        sev_counts, values='Count', names='Severity',
        color='Severity',
        color_discrete_map={'1':'#2ecc71','2':'#f1c40f','3':'#e67e22','4':'#e74c3c'},
        title="Severity Levels"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')

# ── ROW 2: Hour + Month ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Accidents by Hour of Day")
    hourly = filtered_df['Hour'].value_counts().sort_index().reset_index()
    hourly.columns = ['Hour', 'Count']
    fig = px.line(
        hourly, x='Hour', y='Count',
        markers=True,
        title="Accident Frequency by Hour"
    )
    fig.update_traces(line_color='steelblue', fill='tozeroy', fillcolor='rgba(70,130,180,0.2)')
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Accidents by Weather Condition")
    weather_counts = filtered_df['Weather_Condition'].value_counts().head(10).reset_index()
    weather_counts.columns = ['Weather', 'Count']
    fig = px.bar(
        weather_counts, x='Count', y='Weather',
        orientation='h',
        color='Count', color_continuous_scale='Oranges',
        title="Top Weather Conditions"
    )
    fig.update_layout(showlegend=False, height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, width='stretch')

# ── ROW 3: Road Features + Day/Night ──────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Road Features vs Accident Count")
    road_data = {
        'Feature': ['Junction', 'Crossing', 'Traffic Signal'],
        'Count': [
            filtered_df['Junction'].sum(),
            filtered_df['Crossing'].sum(),
            filtered_df['Traffic_Signal'].sum()
        ]
    }
    road_df = pd.DataFrame(road_data)
    fig = px.bar(
        road_df, x='Feature', y='Count',
        color='Feature',
        color_discrete_sequence=['#e74c3c','#3498db','#2ecc71'],
        title="Accidents at Road Features"
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Day vs Night Accidents")
    day_night = filtered_df['Sunrise_Sunset'].value_counts().reset_index()
    day_night.columns = ['Time', 'Count']
    fig = px.pie(
        day_night, values='Count', names='Time',
        color='Time',
        color_discrete_map={'Day': '#f1c40f', 'Night': '#2c3e50', 'Unknown': '#b0bec5'},
        title="Day vs Night Distribution"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Data Source:** Moosavi, S., et al. US-Accidents Dataset, 2019 | Kaggle")
st.markdown(f"Showing **{len(filtered_df):,}** accidents based on current filters.")