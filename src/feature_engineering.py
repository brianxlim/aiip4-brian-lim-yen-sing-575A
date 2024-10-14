"""
Insights from EDA:
- Sunshine Duration and Cloud Cover are strongly negatively correlated.
- Rainfall has a slight negative correlation with Sunshine Duration and a positive correlation with Cloud Cover.
- Minimum and Maximum Temperatures have a strong positive correlation, which indicates a stable daily temperature range.
- Wind Speed (Min/Max) shows some variability, with potential outliers.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("feature engineering")
logger.info("Starting feature engineering process...")

def feature_engineering_weather(weather_df):
    """
    Performs feature engineering on the weather dataset.

    Args:
    weather_df (pd.DataFrame): The input weather DataFrame with raw features.

    Returns:
    pd.DataFrame: A new DataFrame with additional engineered features.
    """
    try:
        logger.info("Starting feature engineering process for weather data")

        # Feature 1: Temperature Range (Max Temp - Min Temp)
        weather_df['temp_range'] = weather_df['Maximum Temperature (deg C)'] - weather_df['Min Temperature (deg C)']

        # Feature 2: Cloud to Rain Ratio (Cloud Cover / Daily Rainfall Total)
        weather_df['cloud_rain_ratio'] = np.where(
            weather_df['Daily Rainfall Total (mm)'] > 0,
            weather_df['Cloud Cover (%)'] / weather_df['Daily Rainfall Total (mm)'],
            0
        )
        
        # Feature 3: Wind Speed Range (Max Wind Speed - Min Wind Speed)
        weather_df['wind_range'] = weather_df['Max Wind Speed (km/h)'] - weather_df['Min Wind Speed (km/h)']

        # Feature 4: Squared Maximum Temperature
        weather_df['max_temp_squared'] = weather_df['Maximum Temperature (deg C)'] ** 2

        # Feature 5: Rain Intensity (Binary flag for high rainfall)
        weather_df['high_rain_intensity'] = np.where(weather_df['Daily Rainfall Total (mm)'] >= 50, 1, 0)

        # Feature 6: Daylight Efficiency (Sunshine Duration / max possible sunshine hours, assuming max is 9.15 hrs)
        weather_df['daylight_efficiency'] = weather_df['Sunshine Duration (hrs)'] / 9.15

        logger.info("Feature engineering process for weather data completed successfully.")

        return weather_df
    
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

def feature_engineering_air_quality(air_quality_df):
    """
    Performs feature engineering on the air quality dataset.

    Args:
    air_quality_df (pd.DataFrame): The input air quality DataFrame with raw features.

    Returns:
    pd.DataFrame: A new DataFrame with additional engineered features.
    """
    try:
        logger.info("Starting feature engineering process for air quality data...")

        # Feature 1: Average PM2.5 Levels across all regions
        air_quality_df['avg_pm25'] = air_quality_df[
            ['pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central']
        ].mean(axis=1)

        # Feature 2: Average PSI Levels across all regions
        air_quality_df['avg_psi'] = air_quality_df[
            ['psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central']
        ].mean(axis=1)

        # Feature 3: Maximum PM2.5 Level
        air_quality_df['max_pm25'] = air_quality_df[
            ['pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central']
        ].max(axis=1)

        # Feature 4: Maximum PSI Level
        air_quality_df['max_psi'] = air_quality_df[
            ['psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central']
        ].max(axis=1)

        logger.info("Feature engineering process for air quality data completed successfully.")

        return air_quality_df
    
    except Exception as e:
        logger.error(f"Error during feature engineering (air quality): {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure
