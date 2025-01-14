import pandas as pd
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

class TimeSeriesSTL:
    def __init__(self, time_series, granularity = 'monthly',  feature = 'fatalities',                  
                 result_path = './results/images/'):
        """
        Initialize the TimeSeriesSTL class.

        Args:
            time_series (pd.Series): The time series data with a DateTime index.
            seasonal_period (int): The length of the seasonal component.
        """
        self.time_series = time_series   
        self.feature = feature
        self.granularity = granularity
   
        self.result_path = result_path


    def decompose(self, seasonal_period):
        try:                  
            stl = STL(self.time_series[self.feature], seasonal = seasonal_period)
            result = stl.fit()
            print("STL decomposition successful.")
            
        except ValueError as e:
            print(f"Error: {e}")
            return None

        else:
            return result
        
    def perform_stl(self):
        result_path = self.result_path + self.feature
        color_code = {'fatalities': '#ff0000', 'event':'#0000ff'}
        if (self.granularity == 'monthly'):  
            self.time_series = self.time_series.asfreq('MS')  # Month: MS, Day: D
            self.time_series = self.time_series.fillna(0)             
            seasonal_period = 13

        elif(self.granularity == 'daily'):           
            self.time_series = self.time_series.asfreq('D')  # Month: MS, Day: D
            self.time_series = self.time_series.fillna(0)      
            seasonal_period = 31
        
        else:
            print(f"Unknown granularity: '{self.granularity}'. Please choose either 'daily' or 'monthly'.")
            return  
        
        try:                     
            result = self.decompose(seasonal_period)
            self.plot_components(result, color_code[self.feature])
            adf_result = self.adf_test()
            return adf_result

        except Exception as e:
            print(f"Error : {e}")
            return  # Stop execution if there is an error

    def plot_components(self, result, color_code):
        """
        Plot the decomposed components: Original, Trend, Seasonal, and Residual.
        """
        if result is None:
            raise ValueError("STL decomposition has not been performed. Call decompose() first.")

        last_data = self.time_series.index.max()

        plt.figure(figsize=(16, 8))
        plt.subplot(4, 1, 1)
        plt.plot(self.time_series[self.feature], label=self.feature, color = color_code)
        plt.xlim([self.time_series.index.min(), self.time_series.index.max()])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))  
        locs, labels = plt.xticks()
        plt.xticks(locs[::50], labels[::50])
        plt.legend(loc='best')

        plt.subplot(4, 1, 2)
        plt.plot(result.trend, label='Trend', color = color_code)
        plt.xlim([self.time_series.index.min(), self.time_series.index.max()])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))  
        locs, labels = plt.xticks()
        plt.xticks(locs[::50], labels[::50])
        plt.legend(loc='best')        

        plt.subplot(4, 1, 3)
        plt.plot(result.seasonal, label='Seasonal', color = color_code)
        plt.xlim([self.time_series.index.min(), self.time_series.index.max()])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))  
        locs, labels = plt.xticks()
        plt.xticks(locs[::50], labels[::50])
        plt.legend(loc='best')

        plt.subplot(4, 1, 4)
        plt.plot(result.resid, label='Residual', color = color_code)
        plt.xlim([self.time_series.index.min(), self.time_series.index.max()])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))  
        locs, labels = plt.xticks()
        plt.xticks(locs[::50], labels[::50])
        plt.legend(loc='best')

        plt.tight_layout()    
        plt.savefig(f'{self.result_path}{self.feature}_{self.granularity}_stl.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_residuals(self):
        """
        Analyze the residuals using Ljung-Box test and identify anomalies.

        Returns:
            dict: A dictionary containing residual diagnostics and anomalies.
        """
        if not self.result:
            raise ValueError("STL decomposition has not been performed. Call decompose() first.")

        residuals = self.result.resid.dropna()
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)

        anomalies = residuals[abs(residuals) > 2 * residuals.std()]
        diagnostics = {
            "Ljung-Box Test": lb_test,
            "Anomalies": anomalies
        }
        return diagnostics
    
    def adf_test(self):        
        adf_test = adfuller(self.time_series[self.feature])

        return adf_test
    
    def get_seasonal_summary(self):
        """
        Summarize the seasonal component by aggregating values by month.

        Returns:
            pd.Series: A summary of the seasonal component.
        """
        if not self.result:
            raise ValueError("STL decomposition has not been performed. Call decompose() first.")

        seasonal = self.result.seasonal
        summary = seasonal.groupby(seasonal.index.month).mean()
        return summary
