import pandas as pd
import argparse

class TimeSeriesPVTW:
    def __init__(self, data_source):
        self.data_source = data_source
        self.df = pd.read_csv(data_source)
        
        print(f"Number of Records:{self.df.shape[0]}")
        print(f"Number of Attibutes:{self.df.shape[1]}")

    def transform(self):
        
        df_time = self.df[['event_date', 'country', 'fatalities']].copy()
        df_time['event_date'] = pd.to_datetime(df_time['event_date'])  
        df_time['country'] = df_time['country'].astype('category')
        df_time['year'] = df_time['event_date'].dt.year
        df_time['month'] = df_time['event_date'].dt.strftime('%b')

        # perform data cleaning
        df_time['country'] = df_time['country'].str.strip()  # Remove leading/trailing spaces
        df_time['country'] = df_time['country'].str.lower()  # Standardize case
        df_time['event_date'] = pd.to_datetime(df_time['event_date'])     

        daily_result = df_time.groupby('event_date', observed=False).agg(
            fatalities=('fatalities', 'sum'), event=('fatalities', 'count')).reset_index()

        df_time['year_month'] = df_time['event_date'].dt.to_period('M')
        monthly_result = df_time.groupby('year_month', observed=False).agg(
            fatalities=('fatalities', 'sum'), event=('fatalities', 'count')).reset_index()        
        monthly_result['year_month'] = monthly_result['year_month'].dt.strftime('%Y-%m')

        return daily_result, monthly_result, df_time

    def save(self, output_filepath ):

        daily_output_filepath = f"{output_filepath}_daily.csv"
        monthly_output_filepath = f"{output_filepath}_monthly.csv"
        total_output_filepath = f"{output_filepath}_all_data.csv"
        daily_result, monthly_result, df_time = self.transform()
        daily_result.to_csv(daily_output_filepath, index=False)
        df_time.to_csv(total_output_filepath, index=False)

        monthly_result.to_csv(monthly_output_filepath, index=False)

        print(f"Number of Daily Records:{daily_result.shape[0]}")
        print(f"Number of Monthly Records:{monthly_result.shape[0]}")
        
        for col in daily_result.columns[1:]:
            daily_output_filepath = f"{output_filepath}_daily_{col}.csv"
            monthly_output_filepath = f"{output_filepath}_monthly_{col}.csv"
            daily_result[['event_date', col]].to_csv(daily_output_filepath, index=False)
            monthly_result[['year_month', col]].to_csv(monthly_output_filepath, index=False)


