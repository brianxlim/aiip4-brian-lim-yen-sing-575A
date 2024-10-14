# Machine Learning Pipeline for Solar Panel Efficiency Prediction

## Full Name and Email Address

Name: BRIAN LIM YEN SING
Email: brianlimyensing@gmail.com

## Overview of Submitted Folder and Folder Structure

The folder structure is organized as follows:

```bash
/src
  ├── data_ingestion.py         # Script to fetch data from SQLite databases
  ├── initial_preprocessing.py  # Script to handle initial preprocessing like missing value handling and numeric conversion
  ├── feature_engineering.py    # Script for feature engineering for weather and air quality data
  ├── final_preprocessing.py    # Script for final preprocessing, merging, and splitting
  ├── model_training.py         # Script for training and evaluating machine learning models
  └── main.py                   # Main script to run the pipeline

/data                           # Contains the SQLite database files
  ├── weather.db                # Weather data
  └── air_quality.db            # Air quality data

run.sh                          # Script to run the pipeline
requirements.txt                # List of Python dependencies
README.md                       # This file
```

## Instructions for Executing the Pipeline and Modifying Parameters

### How to Run the Pipeline
1. Ensure you have Python 3.x and `virtualenv` installed.
2. Create a virtual environment and install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Execute the pipeline using the `run.sh` script:

```bash
./run.sh
```

## Description of Logical Steps/Flow of the Pipeline

**Data Ingestion**: The weather and air quality data are fetched from SQLite databases using the fetch_data_from_sqlite function in data_ingestion.py.

**Initial Preprocessing**: In initial_preprocessing.py, numeric columns are converted, missing values are handled, and infinite values are addressed.

**Feature Engineering**: In feature_engineering.py, new features such as temperature range, cloud-to-rain ratio, average PM2.5 levels, and PSI metrics are engineered based on insights from the EDA.

**Final Preprocessing & Merging**: The datasets are merged, missing values are further handled, and numeric columns are scaled in final_preprocessing.py.

**Model Training and Evaluation**: Various machine learning models are trained and evaluated using metrics like accuracy or mean squared error in model_training.py.

The overall pipeline is orchestrated in the `main.py` script.

## Key Findings from EDA and Feature Engineering

- Sunshine Duration and Cloud Cover have a strong negative correlation, and Daily Rainfall also correlates negatively with sunshine duration.
- Temperature Range (max temp minus min temp) was identified as a feature that may affect solar panel efficiency.
- Air quality features such as PM2.5 Levels and PSI were engineered by taking the average and maximum values across regions.

These insights were crucial in building the feature set for the machine learning models.

## How Features are Processed (Table)
| Feature                    | Processing Step                | Description                                                 |
|----------------------------|---------------------------------|-------------------------------------------------------------|
| `Min Temperature`           | Numeric Conversion, Scaling     | Minimum temperature of the day in Celsius                   |
| `Max Temperature`           | Numeric Conversion, Scaling     | Maximum temperature of the day in Celsius                   |
| `Sunshine Duration`         | Scaling, Feature Engineering    | Duration of sunshine in hours, converted into daylight efficiency |
| `PM2.5 Levels`              | Feature Engineering             | Average and Maximum PM2.5 values across regions             |
| `Cloud Cover`               | Scaling, Feature Engineering    | Cloud coverage, used in combination with rainfall to create cloud-rain ratio |
| `Daily Solar Panel Efficiency` | Encoding (Target Variable)  | Categorical variable converted into numeric form for prediction |

## Explanation of Model Choice

Three models were selected for this pipeline based on their ability to handle tabular data efficiently:

1. Random Forest Classifier: Chosen for its robustness in handling a large number of features and its ability to capture feature importance.
2. Logistic Regression: A simple baseline model that helps establish the linear separability of the dataset.
3. Gradient Boosting Classifier: Selected for its advanced boosting techniques, which generally improve prediction accuracy.

These models are suitable for a classification task where we predict solar panel efficiency as `Low`, `Medium`, or `High`.

## Evaluation of Models

==== TODO ====

## Other Considerations for Deploying the Models

Scalability: The models should be retrained periodically as more weather and air quality data become available.

Feature Importance: Random Forest's feature importance metrics could help guide decisions on feature selection in future iterations of the pipeline.

Computational Efficiency: Gradient boosting, while highly accurate, is more computationally expensive than Logistic Regression. This needs to be balanced for deployment at scale.
