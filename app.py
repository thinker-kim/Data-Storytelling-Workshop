"""
Climate Data Explorer - Interactive Dashboard
Built with Streamlit for a Data Storytelling Workshop
"""

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Climate Data Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DATA LOADING WITH CACHING
# ============================================
@st.cache_data
def load_data():
    """Load and preprocess the climate data"""
    df = pd.read_csv('data/ddbb_surface_temperature_countries.csv')
    df = df.dropna(subset=['Anomaly'])
    return df

@st.cache_data
def get_country_list(df):
    """Get sorted list of countries"""
    return sorted(df['Country'].unique())

@st.cache_data
def get_annual_data(df, country):
    """Calculate annual averages for a country"""
    country_data = df[df['Country'] == country]
    annual = country_data.groupby('Years').agg({
        'Anomaly': 'mean',
        'Temperature': 'mean'
    }).reset_index()
    annual.columns = ['Year', 'Anomaly', 'Temperature']
    return annual

@st.cache_data
def get_decade_data(annual_df):
    """Calculate decade averages"""
    df = annual_df.copy()
    df['Decade'] = (df['Year'] // 10) * 10
    return df.groupby('Decade')['Anomaly'].mean().reset_index()

# ============================================
# LOAD DATA
# ============================================
try:
    df = load_data()
    countries = get_country_list(df)
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'data/ddbb_surface_temperature_countries.csv' exists.")
    st.stop()

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("🌍 Climate Explorer")
st.sidebar.markdown("---")

# Country Selection
default_country = 'South Korea' if 'South Korea' in countries else countries[0]
selected_country = st.sidebar.selectbox(
    "Select Country",
    countries,
    index=countries.index(default_country)
)

# Year Range
min_year = int(df['Years'].min())
max_year = int(df['Years'].max())

year_range = st.sidebar.slider(
    "Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(1900, max_year)
)

# Additional Options
st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")
show_trend = st.sidebar.checkbox("Show Trend Line", value=True)
show_moving_avg = st.sidebar.checkbox("Show Moving Average", value=True)
ma_window = st.sidebar.slider("Moving Average Window", 5, 20, 10) if show_moving_avg else 10

# About Section
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data Source:** [Berkeley Earth](https://berkeleyearth.org/)

**233 countries** | **1743-2020**
""")

# ============================================
# MAIN CONTENT
# ============================================
st.title(f"🌡️ Climate Data: {selected_country}")
st.markdown("Explore temperature anomalies from the Berkeley Earth dataset (39,000+ weather stations)")

# Get processed data
annual_data = get_annual_data(df, selected_country)
annual_filtered = annual_data[
    (annual_data['Year'] >= year_range[0]) & 
    (annual_data['Year'] <= year_range[1])
]

if len(annual_filtered) == 0:
    st.warning(f"No data available for {selected_country} in the selected year range.")
    st.stop()

# ============================================
# KEY METRICS
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    latest_year = annual_filtered['Year'].max()
    latest_anomaly = annual_filtered[annual_filtered['Year'] == latest_year]['Anomaly'].values
    if len(latest_anomaly) > 0:
        delta_color = "normal" if latest_anomaly[0] > 0 else "inverse"
        st.metric(
            label=f"Latest ({int(latest_year)})",
            value=f"{latest_anomaly[0]:.2f}°C",
            delta=f"{latest_anomaly[0]:.2f}°C"
        )

with col2:
    avg_anomaly = annual_filtered['Anomaly'].mean()
    st.metric(
        label="Period Average",
        value=f"{avg_anomaly:.2f}°C"
    )

with col3:
    max_idx = annual_filtered['Anomaly'].idxmax()
    max_row = annual_filtered.loc[max_idx]
    st.metric(
        label=f"Hottest ({int(max_row['Year'])})",
        value=f"{max_row['Anomaly']:.2f}°C"
    )

with col4:
    min_idx = annual_filtered['Anomaly'].idxmin()
    min_row = annual_filtered.loc[min_idx]
    st.metric(
        label=f"Coldest ({int(min_row['Year'])})",
        value=f"{min_row['Anomaly']:.2f}°C"
    )

st.markdown("---")

# ============================================
# TABS FOR DIFFERENT VIEWS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🌡️ Warming Stripes", "📊 Decade Analysis", "🌍 Country Comparison"])

# --------------------------------------------
# TAB 1: TIME SERIES
# --------------------------------------------
with tab1:
    st.subheader("Temperature Anomaly Over Time")
    
    # Calculate moving average
    plot_data = annual_filtered.copy()
    plot_data['MA'] = plot_data['Anomaly'].rolling(window=ma_window, center=True).mean()
    
    # Base chart
    base = alt.Chart(plot_data).encode(
        x=alt.X('Year:Q', title='Year', scale=alt.Scale(domain=[year_range[0], year_range[1]]))
    )
    
    # Points
    points = base.mark_circle(size=40, opacity=0.6, color='steelblue').encode(
        y=alt.Y('Anomaly:Q', title='Temperature Anomaly (°C)'),
        tooltip=[
            alt.Tooltip('Year:Q', title='Year'),
            alt.Tooltip('Anomaly:Q', title='Anomaly', format='.2f')
        ]
    )
    
    # Layers to add
    layers = [points]
    
    # Moving average line
    if show_moving_avg:
        ma_line = base.mark_line(color='#e74c3c', strokeWidth=2.5).encode(
            y='MA:Q'
        )
        layers.append(ma_line)
    
    # Trend line
    if show_trend:
        trend_line = base.transform_regression(
            'Year', 'Anomaly'
        ).mark_line(color='#2ecc71', strokeDash=[5,5], strokeWidth=2).encode(
            y='Anomaly:Q'
        )
        layers.append(trend_line)
    
    # Zero baseline
    baseline_df = pd.DataFrame({'y': [0]})
    baseline = alt.Chart(baseline_df).mark_rule(color='gray', strokeDash=[3,3]).encode(y='y:Q')
    layers.append(baseline)
    
    # Paris 1.5°C target
    paris_df = pd.DataFrame({'y': [1.5]})
    paris_line = alt.Chart(paris_df).mark_rule(color='red', strokeWidth=2).encode(y='y:Q')
    layers.append(paris_line)
    
    # Combine
    chart = alt.layer(*layers).properties(
        width='container',
        height=450
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # Legend explanation
    col_legend1, col_legend2 = st.columns(2)
    with col_legend1:
        st.caption("""
        **Chart Elements:**
        - 🔵 Blue dots: Annual temperature anomaly
        - 🔴 Red line: Moving average (smoothed trend)
        """)
    with col_legend2:
        st.caption("""
        - 🟢 Green dashed: Linear trend line
        - ⚫ Gray dashed: Zero baseline
        - 🔴 Red solid: Paris Agreement 1.5°C target
        """)

# --------------------------------------------
# TAB 2: WARMING STRIPES
# --------------------------------------------
with tab2:
    st.subheader("Warming Stripes Visualization")
    st.markdown("*Each stripe represents one year's temperature anomaly. Created by Ed Hawkins.*")
    
    stripes_data = annual_filtered.copy()
    
    stripes = alt.Chart(stripes_data).mark_rect().encode(
        x=alt.X('Year:O', 
                axis=alt.Axis(
                    labels=True, 
                    labelAngle=-45,
                    values=list(range(year_range[0], year_range[1]+1, 10))
                ),
                title=None),
        color=alt.Color('Anomaly:Q',
                       scale=alt.Scale(scheme='redblue', reverse=True, domain=[-2, 2]),
                       legend=alt.Legend(title='Anomaly (°C)', orient='bottom'))
    ).properties(
        width='container',
        height=150,
        title=f'Warming Stripes: {selected_country} ({year_range[0]}-{year_range[1]})'
    )
    
    st.altair_chart(stripes, use_container_width=True)
    
    st.info("""
    💡 **How to read warming stripes:**
    - **Blue** = Cooler than average (negative anomaly)
    - **Red** = Warmer than average (positive anomaly)  
    - The shift from blue to red shows the warming trend over time
    - Learn more: [Show Your Stripes](https://showyourstripes.info/)
    """)

# --------------------------------------------
# TAB 3: DECADE ANALYSIS
# --------------------------------------------
with tab3:
    st.subheader("Temperature by Decade")
    
    decade_data = get_decade_data(annual_filtered)
    decade_data = decade_data[decade_data['Decade'] >= year_range[0]]
    
    decade_chart = alt.Chart(decade_data).mark_bar().encode(
        x=alt.X('Decade:O', title='Decade'),
        y=alt.Y('Anomaly:Q', title='Average Anomaly (°C)'),
        color=alt.condition(
            alt.datum.Anomaly > 0,
            alt.value('#e74c3c'),
            alt.value('#3498db')
        ),
        tooltip=[
            alt.Tooltip('Decade:O', title='Decade'),
            alt.Tooltip('Anomaly:Q', title='Avg Anomaly', format='.2f')
        ]
    ).properties(
        width='container',
        height=400
    )
    
    st.altair_chart(decade_chart, use_container_width=True)
    
    # Decade statistics table
    st.subheader("Decade Statistics")
    decade_stats = decade_data.copy()
    decade_stats['Anomaly'] = decade_stats['Anomaly'].round(3)
    decade_stats.columns = ['Decade', 'Avg Anomaly (°C)']
    st.dataframe(decade_stats, use_container_width=True, hide_index=True)

# --------------------------------------------
# TAB 4: COUNTRY COMPARISON
# --------------------------------------------
with tab4:
    st.subheader("Compare Multiple Countries")
    
    # Default countries
    default_compare = []
    for c in ['South Korea', 'Japan', 'United Kingdom', 'Germany', 'Australia']:
        if c in countries:
            default_compare.append(c)
    if len(default_compare) == 0:
        default_compare = countries[:3]
    
    # Multi-select for countries
    compare_countries = st.multiselect(
        "Select countries to compare",
        countries,
        default=default_compare[:5]
    )
    
    if len(compare_countries) > 0:
        # Prepare comparison data
        comparison_data = df[df['Country'].isin(compare_countries)]
        comparison_annual = comparison_data.groupby(['Years', 'Country'])['Anomaly'].mean().reset_index()
        comparison_annual = comparison_annual[
            (comparison_annual['Years'] >= year_range[0]) & 
            (comparison_annual['Years'] <= year_range[1])
        ]
        
        # Multi-line chart
        comparison_chart = alt.Chart(comparison_annual).mark_line(strokeWidth=2).encode(
            x=alt.X('Years:Q', title='Year'),
            y=alt.Y('Anomaly:Q', title='Temperature Anomaly (°C)'),
            color=alt.Color('Country:N', legend=alt.Legend(title='Country')),
            tooltip=[
                alt.Tooltip('Years:Q', title='Year'),
                alt.Tooltip('Country:N', title='Country'),
                alt.Tooltip('Anomaly:Q', title='Anomaly', format='.2f')
            ]
        ).properties(
            width='container',
            height=450
        ).interactive()
        
        st.altair_chart(comparison_chart, use_container_width=True)
        
        # Comparison statistics
        st.subheader("Country Statistics")
        stats_list = []
        for country in compare_countries:
            country_annual = get_annual_data(df, country)
            country_filtered = country_annual[
                (country_annual['Year'] >= year_range[0]) & 
                (country_annual['Year'] <= year_range[1])
            ]
            
            if len(country_filtered) > 0:
                stats_list.append({
                    'Country': country,
                    'Avg Anomaly': round(country_filtered['Anomaly'].mean(), 3),
                    'Max Anomaly': round(country_filtered['Anomaly'].max(), 3),
                    'Min Anomaly': round(country_filtered['Anomaly'].min(), 3),
                    'Hottest Year': int(country_filtered.loc[country_filtered['Anomaly'].idxmax(), 'Year']),
                    'Data Points': len(country_filtered)
                })
        
        if stats_list:
            stats_df = pd.DataFrame(stats_list)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Please select at least one country to compare")

# ============================================
# STORY GENERATOR
# ============================================
st.markdown("---")
st.header("📖 Auto-Generated Climate Story")

# Generate story based on data
story_data = annual_filtered.copy()
if len(story_data) > 0:
    latest = story_data[story_data['Year'] == story_data['Year'].max()].iloc[0]
    hottest = story_data.loc[story_data['Anomaly'].idxmax()]
    coldest = story_data.loc[story_data['Anomaly'].idxmin()]
    avg = story_data['Anomaly'].mean()
    
    # Early vs Recent comparison
    mid_year = (year_range[0] + year_range[1]) // 2
    early = story_data[story_data['Year'] < mid_year]['Anomaly'].mean()
    recent = story_data[story_data['Year'] >= mid_year]['Anomaly'].mean()
    change = recent - early
    
    st.markdown(f"""
    ### Climate Summary: {selected_country}
    
    **Key Findings ({year_range[0]}-{year_range[1]}):**
    
    - 📅 In **{int(latest['Year'])}**, the temperature anomaly was **{latest['Anomaly']:.2f}°C**
    - 🔥 The **hottest year** was **{int(hottest['Year'])}** with an anomaly of **{hottest['Anomaly']:.2f}°C**
    - ❄️ The **coldest year** was **{int(coldest['Year'])}** with an anomaly of **{coldest['Anomaly']:.2f}°C**
    - 📊 The **average anomaly** for this period is **{avg:.2f}°C**
    
    **Trend Analysis:**
    
    Comparing the first half ({year_range[0]}-{mid_year-1}) to the second half ({mid_year}-{year_range[1]}):
    - Early period average: **{early:.2f}°C**
    - Recent period average: **{recent:.2f}°C**
    - Change: **{change:+.2f}°C** {'📈 (warming)' if change > 0 else '📉 (cooling)'}
    """)
    
    # Paris Agreement Progress
    st.subheader("Paris Agreement Status")
    paris_target = 1.5
    current_progress = latest['Anomaly'] / paris_target * 100
    
    progress_col1, progress_col2 = st.columns([3, 1])
    with progress_col1:
        st.progress(min(current_progress / 100, 1.0))
    with progress_col2:
        st.write(f"**{current_progress:.1f}%** of 1.5°C target")
    
    if latest['Anomaly'] >= paris_target:
        st.error(f"⚠️ Current anomaly ({latest['Anomaly']:.2f}°C) has exceeded the Paris Agreement 1.5°C target!")
    else:
        remaining = paris_target - latest['Anomaly']
        st.success(f"✅ Remaining budget: {remaining:.2f}°C until 1.5°C threshold")

# ============================================
# DATA DOWNLOAD
# ============================================
st.markdown("---")
st.header("📥 Download Data")

col1, col2 = st.columns(2)

with col1:
    csv = annual_filtered.to_csv(index=False)
    st.download_button(
        label="📄 Download Annual Data (CSV)",
        data=csv,
        file_name=f"{selected_country.replace(' ', '_')}_annual_anomaly.csv",
        mime="text/csv"
    )

with col2:
    decade_csv = decade_data.to_csv(index=False)
    st.download_button(
        label="📊 Download Decade Data (CSV)",
        data=decade_csv,
        file_name=f"{selected_country.replace(' ', '_')}_decade_anomaly.csv",
        mime="text/csv"
    )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Data Source:</strong> <a href="https://berkeleyearth.org/" target="_blank">Berkeley Earth</a></p>
    <p><strong>Citation:</strong> Rohde, R. A. and Hausfather, Z.: The Berkeley Earth Land/Ocean Temperature Record, 
    Earth Syst. Sci. Data, 12, 3469–3479, 2020.</p>
    <p>Built with ❤️ using Streamlit • Climate Data Storytelling Workshop</p>
</div>
""", unsafe_allow_html=True)
