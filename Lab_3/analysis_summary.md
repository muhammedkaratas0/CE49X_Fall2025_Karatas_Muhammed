# Lab 3: ERA5 Weather Data Analysis Summary

## Data Overview

This analysis examined ERA5 reanalysis wind data for Berlin and Munich covering the period from April 1, 2024 to December 31, 2024. The dataset included 1,100 records for each city with measurements every 6 hours, containing eastward (u10m) and northward (v10m) wind components at 10 meters height.

## Key Findings

### Overall Wind Speed Comparison
- **Berlin average wind speed**: 3.31 m/s
- **Munich average wind speed**: 2.50 m/s
- Berlin consistently experiences stronger winds than Munich throughout the year, approximately 32% higher on average.

### Seasonal Patterns

Both cities showed similar seasonal trends with the strongest winds in winter and weakest in summer:

**Berlin Seasonal Averages:**
- Winter: 3.87 m/s (windiest)
- Spring: 3.25 m/s
- Summer: 2.80 m/s (calmest)
- Fall: 3.68 m/s

**Munich Seasonal Averages:**
- Winter: 2.79 m/s (windiest)
- Spring: 2.75 m/s
- Summer: 2.12 m/s (calmest)
- Fall: 2.62 m/s

### Monthly Trends

- Both cities showed a gradual decrease in wind speed from spring through summer
- A significant increase occurred in September-October (fall transition)
- The windiest month was December for both cities
- The calmest period was July-August (summer) for Munich and July for Berlin

### Extreme Weather Events

**Berlin:**
- Windiest day: December 16, 2024 with maximum wind speed of 8.55 m/s
- Multiple high wind events occurred in October and December
- Top wind direction during extreme events: predominantly from the east (80-85° from north)

**Munich:**
- Windiest day: December 6, 2024 with maximum wind speed of 9.05 m/s
- April also saw significant wind events (April 16-19)
- Extreme winds primarily from the east (60-90° from north)

### Diurnal (Daily) Patterns

**Berlin:**
- Peak wind speeds occur around midday (12:00): 3.77 m/s
- Morning (06:00) shows slightly elevated speeds: 3.33 m/s
- Calmest periods are at midnight (00:00) and evening (18:00): ~3.05-3.10 m/s

**Munich:**
- Similar pattern with peak at midday (12:00): 2.70 m/s
- Less pronounced diurnal variation compared to Berlin
- Calmest in evening (18:00): 2.39 m/s

### Wind Direction Analysis

From the wind rose diagrams:
- Berlin: More variable wind directions with significant contribution from westerly and easterly winds
- Munich: Predominant wind directions from east and west
- Both cities show relatively uniform directional distribution without strong prevailing winds in a single direction

## Implications

1. **Renewable Energy**: Berlin has better wind energy potential due to consistently higher wind speeds, especially in winter months. Both locations show strong diurnal patterns that could be exploited for energy planning.

2. **Urban Planning**: The consistent easterly winds during extreme events should inform building orientation and urban canyon design in both cities.

3. **Climate Patterns**: The summer minimum and winter maximum wind speeds align with typical central European weather patterns, with winter storm systems bringing stronger winds.

4. **Data Quality**: No missing values were found in the dataset, indicating high quality ERA5 reanalysis data suitable for engineering applications.

## Comparison with Historical Events

The extreme wind events identified in the analysis (particularly December 16 in Berlin and December 6 in Munich) would be worth cross-referencing with weather news from 2024 to validate the findings against reported storm systems or severe weather warnings.

## Visualizations Created

1. **Monthly Wind Speeds**: Time series showing seasonal variation and city comparison
2. **Seasonal Comparison**: Bar chart highlighting winter-summer differences
3. **Wind Rose Diagrams**: Directional analysis for both cities
4. **Diurnal Pattern**: Hourly averaging revealing daily wind cycles

## Technical Notes

The wind speed was calculated from orthogonal components using the formula:
```
wind_speed = sqrt(u10m² + v10m²)
```

Wind direction was calculated using meteorological convention (direction FROM which wind blows):
```
wind_direction = atan2(-u10m, -v10m) converted to degrees
```

All analysis was performed using Python with pandas, numpy, matplotlib, and seaborn libraries.
