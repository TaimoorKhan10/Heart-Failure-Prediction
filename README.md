# Heart Failure Prediction System

A machine learning-based system for predicting heart failure risk using clinical records. The system implements a complete pipeline from data preprocessing to model evaluation and provides an easy-to-use interface for making predictions.

## Features

- Data preprocessing and feature engineering
- Automated model training and evaluation
- Comprehensive visualization of data and model performance
- Risk level prediction with probability scores
- Logging system for tracking model performance and errors

## Project Structure

```
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── visualization.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TaimoorKhan10/Heart-Failure-Prediction.git
cd Heart-Failure-Prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training and Evaluation

To train the model and evaluate its performance:

```python
from src.main import HeartFailurePredictionSystem

# Initialize the system
system = HeartFailurePredictionSystem()

# Train and evaluate the model
metrics = system.train_and_evaluate('data/heart_failure_clinical_records_dataset.csv')
print(f'Model performance metrics: {metrics}')
```

### Making Predictions

To make predictions for new patient data:

```python
# Example patient data
patient_data = {
    'age': 65,
    'anaemia': 0,
    'creatinine_phosphokinase': 146,
    'diabetes': 0,
    'ejection_fraction': 38,
    'high_blood_pressure': 0,
    'platelets': 262000,
    'serum_creatinine': 1.3,
    'serum_sodium': 140,
    'sex': 1,
    'smoking': 0,
    'time': 129
}

# Get prediction
result = system.predict(patient_data)
print(f'Prediction: {result}')
```

## Model Output

The model provides three levels of risk assessment:
- Low Risk (probability < 0.3)
- Moderate Risk (probability 0.3-0.7)
- High Risk (probability > 0.7)

## Visualizations

The system generates several visualizations to help understand the data and model performance:
- Feature distributions
- Correlation matrix
- ROC curve
- Feature importance plot

All visualizations are saved in the `output/plots` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: UCI Machine Learning Repository
## Contact

Taimoor Khan - [GitHub](https://github.com/TaimoorKhan10)
