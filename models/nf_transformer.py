import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import Informer, Autoformer, FEDformer, PatchTST,VanillaTransformer
from neuralforecast.losses.pytorch import MAE
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import logging
import optuna
import torch


"""
Transformer-based models are re-implemented using Neural Forecast Libary: https://github.com/Nixtla/neuralforecast
"""
class TransformerModels:
    def __init__(self, Y_train_df, Y_test_df, output_length, granularity):
        """
        Initializes the class with training and testing datasets.
        Args:
            Y_train_df (DataFrame): Training dataset.
            Y_test_df (DataFrame): Testing dataset.
            output_length (int): Length of the output horizon for forecasting.
        """
        self.Y_train_df = Y_train_df
        self.Y_test_df = Y_test_df
        self.output_length = output_length
        self.gra = granularity

    def _generate_results(self, nf):
        """
        Generates prediction results for the test set.
        Args:
            nf (NeuralForecast): Trained NeuralForecast model.
        Returns:
            DataFrame: Concatenated results of predictions and actual values.
        """
        Y_test_df = self.Y_test_df.copy()
        start_date = Y_test_df['ds'].min()
        Y_test_df['ds'] = pd.to_datetime(Y_test_df['ds'])
        all_results = []

        for start_date in Y_test_df['ds'].values:
            start_date = pd.to_datetime(start_date).to_pydatetime()
            if (self.gra == 'monthly'):
                end_date = start_date + relativedelta(months=2*self.output_length)
                buffer = relativedelta(months=self.output_length)
            else:
                end_date = start_date + relativedelta(days=2*self.output_length)
                buffer = relativedelta(days=self.output_length)
            
            if end_date > Y_test_df['ds'].max() - buffer:
                break
       
            Y_filtered = Y_test_df[(Y_test_df['ds'] >= start_date) & (Y_test_df['ds'] < end_date)].copy()
            Y_hat_df = nf.predict(Y_filtered, verbose=0)
            Y_actual = Y_test_df[(Y_test_df['ds'] >= end_date) & (Y_test_df['ds'] < end_date + buffer)].copy()

            result_df = Y_hat_df.copy()
            result_df['actual'] = Y_actual['y'].tolist()
            all_results.append(result_df)

        final_result = pd.concat(all_results, ignore_index=True)
        return final_result

    def _compute_metrics(self, final_result):
        """
        Computes evaluation metrics for the predictions.
        Args:
            final_result (DataFrame): Results containing predictions and actual values.
            filename (str): Path to save the metrics as a CSV.
        Returns:
            DataFrame: Metrics computed for the predictions.
        """
        actuals = final_result['actual'].values.tolist()
        metrics = {'model': [], 'MAE': [], 'RMSE': [], 'NMAE': [], 'NRMSE': []}
        
        for col in final_result.columns[2:-1]:
            predictions = final_result[col].values
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            nrmse = rmse / (np.max(actuals) - np.min(actuals))
            nmae = mae / (np.max(actuals) - np.min(actuals))

            metrics['model'].append(col)
            metrics['MAE'].append(mae)
            metrics['RMSE'].append(rmse)
            metrics['NMAE'].append(nmae)
            metrics['NRMSE'].append(nrmse)

        metrics_df = pd.DataFrame(metrics)     
        return metrics_df

    def runAdvTransformers(self):
        """
        Runs the Transformer Models and computes metrics.
        Returns:
            DataFrame: Metrics for the Transformer model.
        """      
        models = [  Informer(h=self.output_length,                # Forecasting horizon
                        input_size=2 * self.output_length,         # Input size
                        max_steps=100,               # Number of training iterations
                        val_check_steps=50,          # Compute validation loss every 100 steps
                        early_stop_patience_steps=2,
                        enable_progress_bar=False,
                         logger = False), # Stop training if validation loss does not improve
                    Autoformer(h=self.output_length,
                        input_size=2*self.output_length,
                        hidden_size=64,
                        conv_hidden_size=32,
                        n_head = 4,
                        loss=MAE(),
                        scaler_type='robust',
                        learning_rate=1e-3,
                        max_steps=100,
                        val_check_steps=50,
                        early_stop_patience_steps=2,
                        enable_progress_bar=False,
                        logger = False),                                           
                    VanillaTransformer(h=self.output_length,
                        input_size=2*self.output_length,
                        hidden_size=64,
                        conv_hidden_size=32,
                        n_head=4,
                        loss=MAE(),
                        scaler_type='robust',
                        learning_rate=1e-3,
                        max_steps=100,
                        val_check_steps=50,
                        early_stop_patience_steps=2,
                        enable_progress_bar=False,
                        logger = False)

                ]
        

        if (self.gra == 'monthly'):
            nf = NeuralForecast(models=models, freq='ME')
        else:
            nf = NeuralForecast(models=models, freq='D')

        nf.fit(df=self.Y_train_df, val_size=self.Y_test_df.shape[0])
        result_df = self._generate_results(nf)
        metrics = self._compute_metrics(result_df)
        return metrics, result_df


"""

"""