import pandas as pd

def resample_and_save_data(file_path):
    data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    data_5min = data.resample('5min').mean()
    data_10min = data.resample('10min').mean()
    data_30min = data.resample('30min').mean()
    path_5min = file_path.replace('.csv', '_5min.csv')
    path_10min = file_path.replace('.csv', '_10min.csv')
    path_30min = file_path.replace('.csv', '_30min.csv')
    data_5min.to_csv(path_5min)
    data_10min.to_csv(path_10min)
    data_30min.to_csv(path_30min)
    print("5-minute resampled data saved to:", path_5min)
    print("10-minute resampled data saved to:", path_10min)
    print("30-minute resampled data saved to:", path_30min)

input_file_path = 'cleaned_room_1_data1.csv'
resample_and_save_data(input_file_path)