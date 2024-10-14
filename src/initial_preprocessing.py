import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("data preprocessing")
logger.info("Starting initial preprocessing...")

def to_numeric(df, columns):
    """
    Convert specified columns in the dataframe to numeric.
    
    Args:
    df (pd.DataFrame): The DataFrame to convert columns for.
    columns (list): List of columns to be converted to numeric.

    Returns:
    pd.DataFrame: DataFrame with columns converted to numeric.
    """
    logger.info("Converting columns to numeric...")
    try:
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce any non-convertible values to NaN
        logger.info(f"Successfully converted {len(columns)} columns to numeric.")
    except Exception as e:
        logger.error(f"Error in converting columns to numeric: {e}", exc_info=True)
    
    return df

def handle_infinite_and_missing_values(df):
    """
    Handle infinite and missing values in the dataset by replacing infinite values and imputing missing numeric values with the median.
    
    Args:
    df (pd.DataFrame): The DataFrame with missing values.

    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    logger.info("Handling infinite values...")
    try:
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info("Successfully replaced infinite values.")
    except Exception as e:
        logger.error(f"Error in replacing infinite values: {e}", exc_info=True)

    logger.info("Handling missing values...")
    try:
        # Fill missing numeric values with median
        df.fillna(df.median(), inplace=True)
        logger.info("Successfully handled missing values.")
    except Exception as e:
        logger.error("Error in handling missing values", exc_info=True)
    
    return df
