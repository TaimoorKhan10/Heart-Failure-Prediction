import os
import logging
from data_preprocessing import DataPreprocessor
from model_training import HeartFailureModel
from visualization import DataVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('heart_failure_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HeartFailurePredictionSystem:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = HeartFailureModel()
        self.visualizer = DataVisualizer()
        self.output_dir = 'output'
        self.ensure_output_directory()

    def ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, 'plots'))
            os.makedirs(os.path.join(self.output_dir, 'models'))
            logger.info('Created output directories')

    def train_and_evaluate(self, data_path):
        """Complete training and evaluation pipeline."""
        try:
            # Data preprocessing
            logger.info('Starting data preprocessing...')
            X_train, X_test, y_train, y_test = self.preprocessor.process_data(data_path)

            # Generate and save initial visualizations
            logger.info('Generating data visualizations...')
            data = self.preprocessor.load_data(data_path)
            self.visualizer.plot_feature_distributions(
                data,
                save_path=os.path.join(self.output_dir, 'plots', 'feature_distributions.png')
            )
            self.visualizer.plot_correlation_matrix(
                data,
                save_path=os.path.join(self.output_dir, 'plots', 'correlation_matrix.png')
            )

            # Model training
            logger.info('Training model...')
            self.model.train(X_train, y_train)

            # Model evaluation
            logger.info('Evaluating model...')
            metrics = self.model.evaluate(X_test, y_test)

            # Generate prediction probabilities
            y_pred_proba = self.model.best_model.predict_proba(X_test)[:, 1]

            # Generate and save evaluation visualizations
            self.visualizer.plot_roc_curve(
                y_test,
                y_pred_proba,
                save_path=os.path.join(self.output_dir, 'plots', 'roc_curve.html')
            )

            # Get and plot feature importance
            feature_importance = self.model.get_feature_importance(X_train.columns)
            self.visualizer.plot_feature_importance(
                feature_importance,
                save_path=os.path.join(self.output_dir, 'plots', 'feature_importance.png')
            )

            # Save the trained model
            model_path = os.path.join(self.output_dir, 'models', 'heart_failure_model.joblib')
            self.model.save_model(model_path)

            return metrics

        except Exception as e:
            logger.error(f'Error in training and evaluation pipeline: {str(e)}')
            raise

    def predict(self, patient_data):
        """Make predictions for new patient data."""
        try:
            # Ensure model is loaded
            if self.model.best_model is None:
                model_path = os.path.join(self.output_dir, 'models', 'heart_failure_model.joblib')
                self.model.load_model(model_path)

            # Preprocess the input data
            processed_data = self.preprocessor.prepare_single_prediction(patient_data)

            # Make prediction
            prediction = self.model.best_model.predict(processed_data)
            prediction_proba = self.model.best_model.predict_proba(processed_data)[0, 1]

            result = {
                'prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk',
                'probability': prediction_proba,
                'risk_level': self.get_risk_level(prediction_proba)
            }

            return result

        except Exception as e:
            logger.error(f'Error making prediction: {str(e)}')
            raise

    @staticmethod
    def get_risk_level(probability):
        """Convert probability to risk level."""
        if probability < 0.3:
            return 'Low Risk'
        elif probability < 0.7:
            return 'Moderate Risk'
        else:
            return 'High Risk'

def main():
    # Initialize the system
    system = HeartFailurePredictionSystem()

    # Path to the dataset
    data_path = '../data/heart_failure_clinical_records_dataset.csv'

    try:
        # Train and evaluate the model
        metrics = system.train_and_evaluate(data_path)
        logger.info('Training and evaluation completed successfully')
        logger.info(f'Model performance metrics: {metrics}')

    except Exception as e:
        logger.error(f'Error in main execution: {str(e)}')
        raise

if __name__ == '__main__':
    main()