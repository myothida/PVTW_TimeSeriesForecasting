import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas as pd


def plot_pvtw(time_series, granularity, result_path):
    """
    Plots the number of political violence targeting women events and fatalities over time.
    Args:
    - time_series (DataFrame): The time series data with 'event' and 'fatalities'.
    - granularity (str): The granularity of the data ('daily' or 'monthly').
    - result_path (str): The path to save the plot image.

    Returns:
    - None: Displays the plot and saves it as a PNG file.
    """

    plt.figure(figsize=(16, 4))
    
    plt.plot(time_series.index, time_series['event'], marker='.', alpha=0.5,  label='Number of PVTW Events')
    plt.plot(time_series.index, time_series['fatalities'], marker='.', color='r', label='Number of Fatalities')
   
    plt.xlim([time_series.index.min(), time_series.index.max()])

    # Set the x-axis to display full date (day-month-year)
    if granularity == 'monthly':
        date_format = '%Y-%m'
    else:
        date_format = '%Y-%m-%d'

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format)) 
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  
    locs, labels = plt.xticks()
    plt.xticks(locs[::50], labels[::50])
    plt.title(f'Number of Conflict Events and Fatalities on {granularity} Basis')
    plt.legend()  
    plt.savefig(f'{result_path}{granularity}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pvtw_subplots(time_series, granularity, result_path):
    """
    Plots the number of political violence targeting women events and fatalities over time.
    Args:
    - time_series (DataFrame): The time series data with 'event' and 'fatalities'.
    - granularity (str): The granularity of the data ('daily' or 'monthly').
    - result_path (str): The path to save the plot image.

    Returns:
    - None: Displays the plot and saves it as a PNG file.
    """
    # Set the x-axis to display full date (day-month-year)
    if granularity == 'monthly':
        date_format = '%Y-%m'
    else:
        date_format = '%Y-%m-%d'
    colors = {'event': 'blue', 'fatalities': 'red'}
    fig, axs = plt.subplots(2, 1, figsize=(16,8))    
    for k, feature in enumerate(time_series.columns):

        axs[k].plot(time_series.index, time_series[feature], color = colors.get(feature, 'black'),
                    marker='.', alpha=0.5,  label=f'Number of PVTW {feature}')      
        axs[k].set_xlim([time_series.index.min(), time_series.index.max()])
        axs[k].legend()
        axs[k].set_title(f'Number of Conflict {feature} on {granularity} Basis')   
        axs[k].xaxis.set_major_formatter(mdates.DateFormatter(date_format)) 
        axs[k].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  
        locs, labels = axs[k].get_xticks(), axs[k].get_xticklabels()
        axs[k].set_xticks(locs[::50])  # Update ticks with step
        axs[k].set_xticklabels([label.get_text() for label in labels[::50]])  # Set the labels accordingly


     
    plt.savefig(f'{result_path}{granularity}_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_ts_sequence(df_filtered, input_seq_length, output_seq_length, result_path):

    time_col, values_col = df_filtered.columns
    fig, axs = plt.subplots(3, 1, figsize=(16,8))
    for k in range(3):

        st_ip = 10*k
        end_ip = st_ip + input_seq_length-1
        st_op = end_ip+1
        end_op = st_op + output_seq_length-1       
        
        input_sequence = df_filtered.loc[st_ip:end_ip, values_col]
        output_sequence = df_filtered.loc[st_op:end_op, values_col]


        axs[k].plot(range(df_filtered.shape[0]), df_filtered.loc[:, values_col], label='data', 
                    color = 'black', marker='.') #plot the data points        
        axs[k].plot(range(st_ip, end_ip+1), input_sequence, color = 'blue', label='Input', marker='o')
        axs[k].plot(range(st_op, end_op+1), output_sequence, color = 'green', label='Output', marker='o', linestyle='-.',)
            

        axs[k].axvline(x=st_ip, color='black', linestyle='--')
        axs[k].axvline(x=end_ip, color='black', linestyle='--')
        axs[k].axvline(x=end_op, color='black', linestyle='--')
        
        axs[k].legend()

    df_filtered[time_col] = df_filtered[time_col].dt.to_period('D').astype(str)  
    dates = df_filtered[time_col].values
    data_length = len(dates)
    time_duration = f"{dates[0]}_to_{dates[-1]}"
        
    plt.xticks(ticks=np.arange(0, data_length, step = 10), labels=dates[::10], rotation=0)
    plt.savefig(f'{result_path}{time_duration}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_sequence_predictions(input_sequence, actual_sequence, predicted_sequence, timestamps, target, result_path):
    
    plt.figure(figsize=(16, 5))

    plt.plot(range(len(input_sequence)), input_sequence, label='Input Sequence', marker='o')  
    plt.plot(range(len(input_sequence), len(input_sequence) + len(actual_sequence)), 
                actual_sequence,  label='Actual Sequence', marker='o', linestyle='-.',)
    plt.plot(range(len(input_sequence), len(input_sequence) + len(predicted_sequence)), 
                predicted_sequence, label='Predicted Sequence', marker='o')

    # Connect the end of the input sequence to the start of the actual sequence
    plt.plot([len(input_sequence) - 1, len(input_sequence)], [input_sequence[-1], actual_sequence[0]], 'k--')
    plt.plot([len(input_sequence) - 1, len(input_sequence)], [input_sequence[-1], predicted_sequence[0]], 'k--')

    data_length = len(timestamps)
    time_duration = f"{timestamps[0]}_to_{timestamps[-1]}"

    plt.ylabel(f'Daily Count of Political Violence {target} ')
    plt.xticks(ticks=np.arange(0, data_length, 10), labels=timestamps[::10], rotation=0)
    plt.title('Input Sequence, Actual Future Sequence, and Predicted Future Sequence')
    plt.legend(loc='upper right')
    plt.savefig(f'{result_path}/{time_duration}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_compare_predictions(input_sequence, actual_sequence, predict_data, timestamps, target, result_path):
    
    plt.figure(figsize=(16, 5))
    plt.plot(range(len(input_sequence)), input_sequence, label='Input Sequence', marker='o')  
    plt.plot(range(len(input_sequence), len(input_sequence) + len(actual_sequence)), 
                actual_sequence,  label='Actual Sequence', marker='o', linestyle='-.',)
    plt.plot([len(input_sequence) - 1, len(input_sequence)], [input_sequence[-1], actual_sequence[0]], 'k--')

    for method in predict_data.columns:
        predicted_sequence = predict_data[method].values
        plt.plot(range(len(input_sequence), len(input_sequence) + len(predicted_sequence)), 
                predicted_sequence, label=method, marker='o')
        plt.plot([len(input_sequence) - 1, len(input_sequence)], [input_sequence[-1], predicted_sequence[0]], 'k--')

    data_length = len(timestamps)
    time_duration = f"{timestamps[0]}_to_{timestamps[-1]}"

    plt.ylabel(f'Daily Count of Political Violence {target} ')
    plt.xticks(ticks=np.arange(0, data_length, 10), labels=timestamps[::10], rotation=0)
    plt.title(f'Model Performance for Predicting Polticial Violence {target}, Showing Both input and future prediction')
    plt.legend(loc='upper right')
    plt.savefig(f'{result_path}/{time_duration}.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(16, 5))    
    plt.plot(range(len(actual_sequence)), 
                actual_sequence,  label='Actual Sequence', marker='o', linestyle='-.',)    

    for method in predict_data.columns:
        predicted_sequence = predict_data[method].values
        plt.plot(range(len(predicted_sequence)), 
                predicted_sequence, label=method, marker='o')
        
    data_length = len(actual_sequence)
    short_time_stamps = timestamps[len(input_sequence):]   

    plt.ylabel(f'Daily Count of Political Violence {target} ')
    plt.xticks(ticks=np.arange(0, data_length,5), labels=short_time_stamps[::5], rotation=0)
    plt.title(f'Comparison of Model Performance for Predicting Polticial Violence {target}: Zoom in on the Predictions')
    plt.legend(loc='upper right')

    plt.savefig(f'{result_path}/{time_duration}_ZoomIn.png', dpi=300, bbox_inches='tight')
    plt.close()