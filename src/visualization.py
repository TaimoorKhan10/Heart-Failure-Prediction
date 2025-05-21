import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self):
        self.style_config()

    def style_config(self):
        """Configure default plotting styles."""
        sns.set(style='darkgrid', palette='colorblind')
        plt.style.use('seaborn')

    def plot_feature_distributions(self, data, save_path=None):
        """Plot distributions of all numeric features."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            n_cols = len(numeric_cols)
            n_rows = (n_cols + 2) // 3
            
            fig = plt.figure(figsize=(15, 5*n_rows))
            for i, col in enumerate(numeric_cols, 1):
                plt.subplot(n_rows, 3, i)
                sns.histplot(data=data, x=col, kde=True)
                plt.title(f'Distribution of {col}')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                logger.info(f'Feature distributions plot saved to {save_path}')
            return fig
        except Exception as e:
            logger.error(f'Error plotting feature distributions: {str(e)}')
            raise

    def plot_correlation_matrix(self, data, save_path=None):
        """Plot correlation matrix heatmap."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f'Correlation matrix plot saved to {save_path}')
            return plt.gcf()
        except Exception as e:
            logger.error(f'Error plotting correlation matrix: {str(e)}')
            raise

    def plot_feature_importance(self, importance_df, save_path=None):
        """Plot feature importance scores."""
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Feature Importance Scores')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f'Feature importance plot saved to {save_path}')
            return plt.gcf()
        except Exception as e:
            logger.error(f'Error plotting feature importance: {str(e)}')
            raise

    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """Plot ROC curve using plotly for interactivity."""
        try:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC curve (AUC = {roc_auc:.3f})',
                mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                mode='lines',
                line=dict(dash='dash')
            ))

            fig.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True
            )

            if save_path:
                fig.write_html(save_path)
                logger.info(f'ROC curve plot saved to {save_path}')
            return fig
        except Exception as e:
            logger.error(f'Error plotting ROC curve: {str(e)}')
            raise

    def plot_confusion_matrix(self, conf_matrix, save_path=None):
        """Plot confusion matrix using plotly."""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                hoverongaps=False,
                texttemplate='%{z}'
            ))

            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Label',
                yaxis_title='True Label'
            )

            if save_path:
                fig.write_html(save_path)
                logger.info(f'Confusion matrix plot saved to {save_path}')
            return fig
        except Exception as e:
            logger.error(f'Error plotting confusion matrix: {str(e)}')
            raise

    def plot_prediction_distribution(self, y_pred_proba, save_path=None):
        """Plot distribution of prediction probabilities."""
        try:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=y_pred_proba,
                nbinsx=30,
                name='Prediction Probabilities'
            ))

            fig.update_layout(
                title='Distribution of Prediction Probabilities',
                xaxis_title='Probability of Heart Failure',
                yaxis_title='Count'
            )

            if save_path:
                fig.write_html(save_path)
                logger.info(f'Prediction distribution plot saved to {save_path}')
            return fig
        except Exception as e:
            logger.error(f'Error plotting prediction distribution: {str(e)}')
            raise