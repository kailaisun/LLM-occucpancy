import pandas as pd

def balance_dataset(file_path):
    data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    class_0 = data[data['occupant_num'] == 0]
    class_1 = data[data['occupant_num'] == 1]
    class_0_undersampled = class_0.sample(n=len(class_1), random_state=42)
    balanced_data = pd.concat([class_0_undersampled, class_1], axis=0).sample(frac=1, random_state=42)
    output_file_path = file_path.replace('.csv', '_undersampled_balanced.csv')
    balanced_data.to_csv(output_file_path)
    print(f"Balanced dataset saved to: {output_file_path}")
    print(f"Class distribution:\n{balanced_data['occupant_num'].value_counts()}")

input_file_path = '/mnt/data/R1W1_5min.csv'
balance_dataset(input_file_path)