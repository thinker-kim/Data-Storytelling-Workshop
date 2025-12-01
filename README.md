# A-Data-Storytelling-Workshop-Making-Climate-Data-Visible-with-Streamlit

Turn climate data into compelling stories with Python, Altair, and Streamlit.

<br>

## Overview

A hands-on 2-day workshop designed to teach data analysis and interactive visualization through the lens of climate science. Learn to transform raw temperature data into meaningful narratives that communicate the urgency of climate change.

<br>

## Workshop Structure

<table>
<tr>
<td width="50%">

**Day 1 · Data Analysis & Visualization**

3 hours · Beginner to Intermediate

</td>
<td width="50%">

**Day 2 · Interactive Dashboard**

3 hours · Intermediate

</td>
</tr>
</table>

<br>

## What You'll Create

<table>
<tr><td><b>Time Series Analysis</b></td><td>Detect trends and patterns in historical temperature data</td></tr>
<tr><td><b>Warming Stripes</b></td><td>Recreate Ed Hawkins' iconic climate visualization</td></tr>
<tr><td><b>Decade Comparisons</b></td><td>Analyze how warming has accelerated over time</td></tr>
<tr><td><b>Interactive Dashboard</b></td><td>Build a deployable Streamlit application</td></tr>
<tr><td><b>Auto-Generated Stories</b></td><td>Create dynamic narratives from data</td></tr>
</table>

<br>

## Dataset

Data provided by [Berkeley Earth Surface Temperature Study](https://berkeleyearth.org/)

<table>
<tr>
<td><b>Records</b></td>
<td><b>Countries</b></td>
<td><b>Time Span</b></td>
</tr>
<tr>
<td>534,571</td>
<td>233</td>
<td>1743 – 2020</td>
</tr>
</table>

<br>

### Schema

<table>
<tr><td><code>Years</code></td><td>Year of observation</td></tr>
<tr><td><code>Month</code></td><td>Month (1-12)</td></tr>
<tr><td><code>Country</code></td><td>Country name</td></tr>
<tr><td><code>Temperature</code></td><td>Absolute temperature (°C)</td></tr>
<tr><td><code>Monthly_variation</code></td><td>Variation from monthly average</td></tr>
<tr><td><code>Anomaly</code></td><td>Temperature anomaly from baseline (°C)</td></tr>
</table>

<br>

## Quick Start

```bash
git clone https://github.com/yourusername/climate-workshop.git
cd climate-workshop

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

<br>

## Project Structure

```
climate-workshop/
├── DAY1_TUTORIAL.ipynb
├── DAY2_TUTORIAL.ipynb
├── app.py
├── requirements.txt
├── data/
│   └── ddbb_surface_temperature_countries.csv
└── output/
```

<br>

## Curriculum

### Day 1 · Climate Data Storytelling

<table>
<tr><td>1</td><td>Introduction to data storytelling and temperature anomalies</td></tr>
<tr><td>2</td><td>Environment setup with Python and Jupyter</td></tr>
<tr><td>3</td><td>Data exploration with pandas</td></tr>
<tr><td>4</td><td>Time series analysis — trends, moving averages, decades</td></tr>
<tr><td>5</td><td>Finding stories — extremes, comparisons, patterns</td></tr>
<tr><td>6</td><td>Altair fundamentals — declarative visualization</td></tr>
<tr><td>7</td><td>Advanced charts — warming stripes, heatmaps</td></tr>
<tr><td>8</td><td>Narrative structure — Hook, Context, Evidence, Meaning, Action</td></tr>
<tr><td>9</td><td>Export workflows — HTML, CSV</td></tr>
</table>

### Day 2 · Interactive Dashboard

<table>
<tr><td>1</td><td>Streamlit architecture and execution model</td></tr>
<tr><td>2</td><td>Core components — text, data display, status</td></tr>
<tr><td>3</td><td>Input widgets — buttons, sliders, selectors</td></tr>
<tr><td>4</td><td>Layout systems — sidebar, columns, tabs</td></tr>
<tr><td>5</td><td>Chart integration with Altair</td></tr>
<tr><td>6</td><td>Performance optimization with caching</td></tr>
<tr><td>7</td><td>Multi-page applications</td></tr>
<tr><td>8</td><td>Dynamic story generation</td></tr>
<tr><td>9</td><td>Deployment to Streamlit Cloud</td></tr>
</table>

<br>

## Learning Outcomes

<table>
<tr>
<td width="33%">

**Technical**

Manipulate time series data with pandas. Calculate statistical trends and moving averages. Design professional visualizations with Altair. Build and deploy interactive web applications.

</td>
<td width="33%">

**Storytelling**

Identify compelling narratives in datasets. Structure data stories effectively. Select appropriate visualizations for your message.

</td>
<td width="33%">

**Domain**

Understand temperature anomaly methodology. Interpret warming trends and regional patterns. Contextualize data with Paris Agreement targets.

</td>
</tr>
</table>

<br>

## Tech Stack

<table>
<tr>
<td><b>Python</b></td>
<td><b>pandas</b></td>
<td><b>Altair</b></td>
<td><b>Streamlit</b></td>
<td><b>NumPy</b></td>
<td><b>SciPy</b></td>
</tr>
<tr>
<td>Core language</td>
<td>Data manipulation</td>
<td>Visualization</td>
<td>Web apps</td>
<td>Numerical</td>
<td>Statistics</td>
</tr>
</table>

<br>

## Key Concepts

### Temperature Anomaly

```
Anomaly = Observed Temperature − Baseline Average
```

The baseline period is typically 1951–1980. Positive values indicate warmer-than-average conditions; negative values indicate cooler-than-average. Anomalies enable comparison across different regions, highlight change rather than absolute values, and follow scientific convention.

### Paris Agreement Targets

<table>
<tr><td><b>1.5°C</b></td><td>Preferred limit above pre-industrial levels</td></tr>
<tr><td><b>2.0°C</b></td><td>Upper limit to prevent catastrophic impacts</td></tr>
</table>

<br>

## Resources

<table>
<tr>
<td><b>Data</b></td>
<td><a href="https://berkeleyearth.org/data/">Berkeley Earth</a> · <a href="https://data.giss.nasa.gov/gistemp/">NASA GISS</a> · <a href="https://www.climate.gov/maps-data">NOAA Climate</a></td>
</tr>
<tr>
<td><b>Docs</b></td>
<td><a href="https://altair-viz.github.io/">Altair</a> · <a href="https://docs.streamlit.io/">Streamlit</a> · <a href="https://pandas.pydata.org/docs/">pandas</a></td>
</tr>
<tr>
<td><b>Inspiration</b></td>
<td><a href="https://showyourstripes.info/">Show Your Stripes</a> · <a href="https://ourworldindata.org/climate-change">Our World in Data</a></td>
</tr>
</table>

<br>

## Deployment

Push to GitHub → Visit [share.streamlit.io](https://share.streamlit.io) → Connect repository → Select `app.py` → Deploy

<br>

## Citation

```
Berkeley Earth Surface Temperature Study
http://berkeleyearth.org/

Rohde, R. A. and Hausfather, Z.: The Berkeley Earth Land/Ocean Temperature Record,
Earth Syst. Sci. Data, 12, 3469–3479, https://doi.org/10.5194/essd-12-3469-2020, 2020.
```

<br>

---

<p align="center">
<sub>Built for climate awareness</sub>
</p>
