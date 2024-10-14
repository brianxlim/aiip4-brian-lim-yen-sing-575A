import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("final preprocessing")
logger.info("Starting final preprocessing...")

def merge_and_preprocess(weather_df, air_quality_df):
    """
    Merge the weather and air quality datasets and perform final preprocessing
    (handling missing values, scaling, and splitting).
    """
    logger.info("Merging weather and air quality datasets...")
    merged_df = pd.merge(weather_df, air_quality_df, on='date', how='inner')
    
    logger.info("Performing final preprocessing...")
    merged_df = final_preprocessing(merged_df)

    # Encode the target column
    try:
        merged_df, class_names = encode_target(merged_df, 'Daily Solar Panel Efficiency')
        logger.info(f"Class names for target encoding: {class_names}")
    except Exception as e:
        logger.error("Error in encoding target")
    
    return merged_df

def final_preprocessing(df):
    """
    Performs final preprocessing steps including handling missing values 
    and scaling numerical features.
    
    Args:
    df (pd.DataFrame): The input DataFrame after feature engineering.
    
    Returns:
    pd.DataFrame: A processed DataFrame ready for model training.
    """
    try:
        logger.info("Starting final preprocessing...")

        # Check for missing values
        if df.isnull().sum().sum() > 0:
            logger.info("Missing values detected, replacing with column medians.")
            df.fillna(df.median(), inplace=True)
        else:
            logger.info("No missing values detected.")

        # Selecting numeric columns for scaling
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        # Normalize numerical features using StandardScaler
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        logger.info("Final preprocessing completed successfully.")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in final preprocessing: {e}")
        return pd.DataFrame()

def split_data(df, target_column, date_columns=None):
    """
    Split the dataset into training and testing sets, remove non-numeric columns, and drop any date-related columns.

    Args:
    df (pd.DataFrame): The DataFrame containing the full dataset.
    target_column (str): The name of the target column.
    date_columns (list): List of date-related columns to be dropped (optional).

    Returns:
    tuple: X_train, X_test, y_train, y_test (split data)
    """
    logger.info("Splitting data into training and test sets...")

    try:
        # Drop target column to create feature set (X) and separate target column (y)
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Drop non-numeric columns
        X = X.select_dtypes(exclude=['object'])  # Exclude non-numeric columns (e.g., strings)
        
        # Drop date-related columns if provided
        if date_columns:
            X = X.drop(date_columns, axis=1)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Data successfully split. Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    except Exception as e:
        logger.error("Error in splitting data", exc_info=True)
        return None, None, None, None

    return X_train, X_test, y_train, y_test

def encode_target(df, target_column):
    """
    Encode the target column into numeric values.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the target column.
    target_column (str): The name of the target column to encode.

    Returns:
    pd.DataFrame: DataFrame with the encoded target column.
    """
    logger.info(f"Encoding target column: {target_column}...")
    
    try:
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        logger.info(f"Successfully encoded {target_column}")
        return df, le.classes_
    except Exception as e:
        logger.error(f"Error in encoding target column: {e}")
        return df, None
