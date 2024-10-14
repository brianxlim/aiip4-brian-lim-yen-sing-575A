import sqlite3
import pandas as pd
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("data ingestion")
logger.info('Starting data ingestion process...')

def fetch_data_from_sqlite(db_path, table_name):
    """
    Fetch data from an SQLite database table and return a pandas DataFrame.
    
    Args:
    db_path (str): Path to the SQLite database.
    table_name (str): Name of the table to fetch data from.

    Returns:
    pd.DataFrame: DataFrame containing the data or empty DataFrame if error is encountered.
    """
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        
        # Query data from the specified table
        query = f"SELECT * FROM {table_name}"
        
        # Load the data into a pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Close the database connection
        conn.close()
        
        logger.info(f"Successfully read {table_name} from {db_path}")
    
    except Exception as e:
        logger.info(f"Failed to get {table_name} from {db_path}")
        logger.info("Using empty DataFrame")
        return pd.DataFrame()

    return df
