import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("model training")
logger.info("Starting model training...")

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple machine learning models.
    
    Args:
    X_train (pd.DataFrame): Training feature set.
    X_test (pd.DataFrame): Test feature set.
    y_train (pd.Series): Training labels.
    y_test (pd.Series): Test labels.
    
    Returns:
    dict: A dictionary of models and their performance metrics.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }
    
    model_performance = {}
    
    for name, model in models.items():
        try:
            logger.info(f"Training {name} model...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            logger.info(f"Evaluating {name} model...")
            performance = classification_report(y_test, y_pred, output_dict=True)
            model_performance[name] = performance
            logger.info(f"{name} performance: {performance}")
        
        except Exception as e:
            logger.error(f"Error during training/evaluation of {name}: {e}")
    
    return model_performance

def tune_model(model, param_grid, X_train, y_train):
    """
    Perform hyperparameter tuning for a given model using GridSearchCV.
    
    Args:
    model (sklearn estimator): The model to tune.
    param_grid (dict): Hyperparameter grid for tuning.
    X_train (pd.DataFrame): Training feature set.
    y_train (pd.Series): Training labels.
    
    Returns:
    best_model (sklearn estimator): The model with the best parameters.
    """
    logger.info("Starting hyperparameter tuning...")
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    
    return grid_search.best_estimator_
