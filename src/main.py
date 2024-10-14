import os
from src.data_ingestion import fetch_data_from_sqlite
from src.initial_preprocessing import handle_infinite_and_missing_values, to_numeric
from src.feature_engineering import feature_engineering_weather, feature_engineering_air_quality
from final_preprocessing import merge_and_preprocess, split_data
from src.model_training import train_and_evaluate_models

def main():

    # Ingest data
    weather_df = fetch_data_from_sqlite('data/weather.db', 'weather')
    air_quality_df = fetch_data_from_sqlite('data/air_quality.db', 'air_quality')

    # Preprocess data
    weather_df = to_numeric(weather_df, [
        'Min Wind Speed (km/h)',
        'Max Wind Speed (km/h)',
        'Min Temperature (deg C)',
        'Maximum Temperature (deg C)',
        'Daily Rainfall Total (mm)',
        'Highest 30 Min Rainfall (mm)',
        'Highest 60 Min Rainfall (mm)',
        'Highest 120 Min Rainfall (mm)'
        ])
    
    air_quality_df = to_numeric(air_quality_df, [
        'pm25_north',
        'pm25_south',
        'pm25_east',
        'pm25_west',
        'pm25_central',
        'psi_north',
        'psi_south',
        'psi_east',
        'psi_west',
        'psi_central'
        ])
    
    weather_df = handle_infinite_and_missing_values(weather_df)
    air_quality_df = handle_infinite_and_missing_values(air_quality_df)
    
    # Feature engineering
    weather_df = feature_engineering_weather(weather_df)
    air_quality_df = feature_engineering_air_quality(air_quality_df)
    
    # Merge data and preprocess
    merged_df = merge_and_preprocess(weather_df, air_quality_df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(merged_df, target_column="Daily Solar Panel Efficiency")
    
    # Train models
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
