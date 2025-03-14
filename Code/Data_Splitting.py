import pandas as pd

def split_and_filter_dataset(file_path, date_ranges, output_suffix):
    data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    office_hours = (data.index.time >= pd.to_datetime('09:00').time()) & \
                   (data.index.time <= pd.to_datetime('18:00').time())
    data = data[office_hours]
    for start_date, end_date in date_ranges:
        range_data = data[(data.index.date >= pd.to_datetime(start_date).date()) &
                          (data.index.date <= pd.to_datetime(end_date).date())]
        range_data = range_data.loc[:, (range_data != 0).any(axis=0)]
        output_file_path = file_path.replace('.csv', f'_{start_date}_to_{end_date}_{output_suffix}.csv')
        range_data.to_csv(output_file_path)
        print(f"Data for {start_date} to {end_date} saved to: {output_file_path}")

date_ranges = [
    ('2021-08-23', '2021-08-28'),
    ('2021-08-30', '2021-09-04'),
    ('2021-09-06', '2021-09-11')
]
file_paths = [
    'cleaned_room_1_data1_5min_normalized.csv',
    'cleaned_room_1_data1_10min_normalized.csv',
    'cleaned_room_1_data1_30min_normalized.csv'
]
suffixes = ['5min', '10min', '30min']
for file_path, suffix in zip(file_paths, suffixes):
    split_and_filter_dataset(file_path, date_ranges, suffix)