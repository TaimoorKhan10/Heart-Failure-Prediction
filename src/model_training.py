import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HeartFailureModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.best_model = None
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    def train(self, X_train, y_train):
        """Train the model using GridSearchCV for hyperparameter tuning."""
        try:
            grid_search = GridSearchCV(
                self.model,
                self.param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.best_model = grid_search.best_estimator_
            
            logger.info(f'Best parameters found: {grid_search.best_params_}')
            logger.info(f'Best cross-validation score: {grid_search.best_score_:.3f}')
            
            return self.best_model
        except Exception as e:
            logger.error(f'Error during model training: {str(e)}')
            raise

    def evaluate(self, X_test, y_test):
        """Evaluate the model using multiple metrics."""
        try:
            y_pred = self.best_model.predict(X_test)
            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }

            logger.info('Model Evaluation Metrics:')
            for metric, value in metrics.items():
                logger.info(f'{metric}: {value:.3f}')

            logger.info('\nClassification Report:\n')
            logger.info(classification_report(y_test, y_pred))

            conf_matrix = confusion_matrix(y_test, y_pred)
            logger.info('\nConfusion Matrix:\n')
            logger.info(conf_matrix)

            return metrics
        except Exception as e:
            logger.error(f'Error during model evaluation: {str(e)}')
            raise

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
            logger.info(f'Cross-validation scores: {cv_scores}')
            logger.info(f'Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')
            return cv_scores
        except Exception as e:
            logger.error(f'Error during cross-validation: {str(e)}')
            raise

    def save_model(self, filepath):
        """Save the trained model to disk."""
        try:
            joblib.dump(self.best_model, filepath)
            logger.info(f'Model saved successfully to {filepath}')
        except Exception as e:
            logger.error(f'Error saving model: {str(e)}')
            raise

    def load_model(self, filepath):
        """Load a trained model from disk."""
        try:
            self.best_model = joblib.load(filepath)
            logger.info(f'Model loaded successfully from {filepath}')
            return self.best_model
        except Exception as e:
            logger.error(f'Error loading model: {str(e)}')
            raise

    def get_feature_importance(self, feature_names):
        """Get feature importance scores."""
        if self.best_model is None:
            logger.error('Model has not been trained yet')
            return None

        try:
            importance = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            logger.info('Feature Importance:')
            logger.info(feature_importance)
            
            return feature_importance
        except Exception as e:
            logger.error(f'Error getting feature importance: {str(e)}')
            raise