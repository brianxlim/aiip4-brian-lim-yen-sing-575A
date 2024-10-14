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

### Logistic Regression:
- Overall Accuracy: 71.71%

For Class 0:
- Precision: 0.72
- Recall: 0.42
- F1-score: 0.53

Logistic Regression model struggled with correctly identifying instances of class 0, as shown by the relatively low recall (0.42) and F1-score (0.53). This means that many true positives for class 0 were not identified, and the precision is slightly better than random, but not high.

For Class 1:
- Precision: 0.79
- Recall: 0.53
- F1-score: 0.63

The model performed slightly better on class 1 but still has challenges with recall, meaning it's missing many true positives.

For Class 2:
- Precision: 0.70
- Recall: 0.93
- F1-score: 0.80

The model did well on class 2, with high recall (0.93), meaning it captured most true positives. The precision is slightly lower (0.70), indicating that some false positives were predicted, but the high F1-score (0.80) shows solid performance.

Macro Avg F1-Score: 0.65
The macro average (mean of F1-scores for all classes) shows moderate performance across the board.

Summary: Logistic Regression struggled particularly with class 0 and class 1, though it performed relatively well on class 2. The overall accuracy of 71.7% is moderate, and the F1-scores suggest this model might not be ideal for this classification task due to poor recall in certain classes.

### Random Forest:
Overall Accuracy: 89.67%

For Class 0:
- Precision: 0.86
- Recall: 0.82
- F1-score: 0.84

Random Forest shows strong performance in classifying instances of class 0, with high precision and recall, indicating a balanced and effective classification.

For Class 1:
- Precision: 0.91
- Recall: 0.84
- F1-score: 0.87

Class 1 performance is even better, with precision and recall both being strong, indicating the model is both highly accurate and capturing most true positives.

For Class 2:
- Precision: 0.90
- Recall: 0.96
- F1-score: 0.93

Class 2 shows excellent performance with high precision, recall, and F1-score.

Macro Avg F1-Score: 0.88
The macro average F1-score is very high, indicating that Random Forest is performing well across all classes.

Summary: Random Forest clearly outperforms Logistic Regression. It has high precision, recall, and F1-scores across all classes, resulting in an overall accuracy of 89.67%. This model balances false positives and false negatives better, making it an excellent candidate for this classification task.

### Support Vector Machine (SVM):
- Overall Accuracy: 74.88%

For Class 0:
- Precision: 0.74
- Recall: 0.41
- F1-score: 0.53

SVM has similar issues with class 0 as Logistic Regression, with low recall (0.41), meaning many true positives were missed. Precision is moderate at 0.74, but the F1-score (0.53) indicates room for improvement.

For Class 1:
- Precision: 0.80
- Recall: 0.67
- F1-score: 0.73

Performance on class 1 is better than Logistic Regression, with higher precision (0.80) and a decent F1-score (0.73), though recall is still somewhat low.

For Class 2:
- Precision: 0.73
- Recall: 0.93
- F1-score: 0.82

The model performs well on class 2, with strong recall (0.93) and a solid F1-score (0.82). Precision (0.73) is slightly lower, but the model captures most true positives.

Macro Avg F1-Score: 0.69
The macro average F1-score (0.69) shows that the performance across all classes is better than Logistic Regression but worse than Random Forest.

Summary: SVM performs better than Logistic Regression but worse than Random Forest. It does particularly well on class 2, but like Logistic Regression, it struggles with class 0, where recall is low. Its overall accuracy of 74.88% is a noticeable improvement over Logistic Regression, but it still doesn't match the balanced performance of Random Forest.

### Overall Conclusion:

Best Model: Random Forest is the best-performing model with the highest accuracy (89.67%) and strong precision, recall, and F1-scores across all classes.

SVM: SVM is a solid middle-ground option with an accuracy of 74.88%, performing better than Logistic Regression but not as well as Random Forest.

Logistic Regression: Logistic Regression had the weakest performance, with an accuracy of 71.71% and difficulties especially with class 0.

**Recommendation:**
Random Forest should be the primary choice for deployment due to its robust performance across all classes.

## Other Considerations for Deploying the Models

Scalability: The models should be retrained periodically as more weather and air quality data become available.

Feature Importance: Random Forest's feature importance metrics could help guide decisions on feature selection in future iterations of the pipeline.

Computational Efficiency: Gradient boosting, while highly accurate, is more computationally expensive than Logistic Regression. This needs to be balanced for deployment at scale.
