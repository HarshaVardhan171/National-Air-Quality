# National Air Quality Index Analytics – India

### Project Overview
This project analyzes India's city-level air quality using pollutant concentration data and AQI metrics. The goal is to:
- Identify pollution hotspots
- Understand pollutant composition
- Correlate AQI with contributing pollutants
- Recommend actionable strategies

#### 1. Business Problem
Air pollution poses severe health risks across Indian cities. Policymakers and the public need timely, location-specific insights to:
- Track pollution trends
- Understand pollutant composition
- Enable targeted interventions

#### 2. Stakeholders
- Government agencies (e.g., CPCB, state boards)
- Public health authorities
- Urban planners
- Environmental NGOs
- Citizens & vulnerable groups

#### 3. Data Sources
Source	Description	Frequency
- Kaggle (city_hour.csv)	Hourly AQI & pollutant data (2015–2024)	Hourly
- OpenAQ / CPCB APIs	Real-time air quality readings	API - Live
- Weather APIs (OpenWeather)	Temperature, wind, humidity data	Hourly/Daily

#### 4. Tools Used
- Python (pandas, seaborn, matplotlib)
- Excel
- Power BI
- MySQL (optional for storage/querying)

#### 5. Data Preparation
- Import Data in Python:
import pandas as pd
df = pd.read_csv("city_hour.csv", parse_dates=['Datetime'], dayfirst=True)

- Clean & Explore Data:
df.columns = df.columns.str.strip()
df.dropna(subset=['City', 'PM2.5', 'PM10', 'AQI'], inplace=True)
df['Datetime'] = pd.to_datetime(df['Datetime'])

- Summary Statistics:
summary = df[['PM2.5', 'PM10', 'AQI']].describe()

#### 6. Exploratory Data Analysis (EDA)
- Importing necessary libraries for visualization:
import matplotlib.pyplot as plt
import seaborn as sns

- Setting the style for seaborn:
sns.set(style='whitegrid')

- Creating histograms for PM2.5, PM10, and AQI:
plt.figure(figsize=(15, 5))

- PM2.5 Histogram:
plt.subplot(1, 3, 1)
sns.histplot(df['PM2.5'], bins=30, kde=True)
plt.title('Distribution of PM2.5')
plt.xlabel('PM2.5')
plt.ylabel('Frequency')

- PM10 Histogram:
plt.subplot(1, 3, 2)
sns.histplot(df['PM10'], bins=30, kde=True)
plt.title('Distribution of PM10')
plt.xlabel('PM10')
plt.ylabel('Frequency')

- AQI Histogram:
plt.subplot(1, 3, 3)
sns.histplot(df['AQI'], bins=30, kde=True)
plt.title('Distribution of AQI')
plt.xlabel('AQI')
plt.ylabel('Frequency')

- Displaying the summary statistics and the plots:
plt.tight_layout()
plt.show()
summary_statistics

#### 7. Power BI Dashboard Insights
Key Visuals:
- KPI Cards: Average & Median AQI
- Trend Lines: Median AQI by Year and City
- Monthly Line Chart: Seasonal AQI variation
- Heat Map: Average AQI by city (geo view)
- Pollutant Breakdown: Stacked bar % by city
- Filters Used:
Year, AQI_Bucket, City slicers

#### 8. Key Insights
- High AQI Zones: Delhi and Kolkata consistently show higher average AQI.
- Dominant Pollutants: PM10 and PM2.5 contribute nearly 70% of AQI load.
- Health Hazards: AQI levels in many cities remain in “Poor” category over time.
- Seasonality: Winter months show a spike in AQI values.
- Correlation: AQI strongly correlates with PM2.5 and PM10 concentrations.

#### 9. Recommendations
- Policy:	Enforce stricter vehicle and industry emission norms
- Monitoring:	Expand AQI sensors in Tier-2/3 cities
- Urban Planning:	Develop green belts and pollution buffer zones
- Public Health:	Alert systems for at-risk groups during severe AQI days
- Awareness:	Real-time AQI display at traffic intersections and apps

