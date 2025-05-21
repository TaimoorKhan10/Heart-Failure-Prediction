import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
            'ejection_fraction', 'high_blood_pressure', 'platelets',
            'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
        ]
        self.target_column = 'DEATH_EVENT'

    def load_data(self, file_path):
        """Load and validate the dataset."""
        try:
            data = pd.read_csv(file_path)
            logger.info(f'Successfully loaded data from {file_path}')
            return data
        except Exception as e:
            logger.error(f'Error loading data: {str(e)}')
            raise

    def handle_missing_values(self, data):
        """Handle missing values in the dataset."""
        missing_values = data.isnull().sum()
        if missing_values.any():
            logger.warning(f'Found missing values:\n{missing_values[missing_values > 0]}')
            data = data.dropna()
            logger.info('Dropped rows with missing values')
        return data

    def handle_outliers(self, data):
        """Handle outliers using IQR method."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[column] = data[column].clip(lower_bound, upper_bound)
        logger.info('Handled outliers using IQR method')
        return data

    def prepare_features(self, data):
        """Prepare features for model training."""
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        # Scale numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        
        logger.info('Features prepared and scaled')
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f'Data split into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets')
        return X_train, X_test, y_train, y_test

    def process_data(self, file_path):
        """Complete data preprocessing pipeline."""
        data = self.load_data(file_path)
        data = self.handle_missing_values(data)
        data = self.handle_outliers(data)
        X, y = self.prepare_features(data)
        return self.split_data(X, y)