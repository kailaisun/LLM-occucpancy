import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def check_and_clean_data(data):
    if np.any(np.isinf(data)) or data.isnull().any().any():
        print("Data contains infinite or NaN values. Handling...")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(data.mean(), inplace=True)
    return data

def convert_normalize_and_save(file_path):
    data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    data = check_and_clean_data(data)
    data['occupant_num'] = (data['occupant_num'] > 0).astype(int)
    features = data.drop('occupant_num', axis=1)
    target = data['occupant_num']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns, index=data.index)
    features_scaled_df['occupant_num'] = target
    output_file_path = file_path.replace('.csv', '_normalized.csv')
    features_scaled_df.to_csv(output_file_path)
    print(f"Normalized data saved to: {output_file_path}")

resampled_paths = [
    'cleaned_room_1_data1_5min.csv',
    'cleaned_room_1_data1_10min.csv',
    'cleaned_room_1_data1_30min.csv'
]
for path in resampled_paths:
    convert_normalize_and_save(path)