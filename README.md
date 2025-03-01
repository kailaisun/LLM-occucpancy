# LLM-Occupancy

This project predicts room occupancy using machine learning models, including LLMs (llama3.2:latest, Gemini-1.5-pro, DeepSeek_R1), alongside baseline models. Data is preprocessed, normalized, balanced, and split for training and evaluation.

## Overview

- **Data Processing**: Resamples time-series data into 5, 10, and 30-minute intervals.
- **Normalization**: Standardizes features using `StandardScaler`.
- **Data Splitting**: Splits data into weekly ranges and filters for office hours (9 AM–6 PM).
- **Data Balancing**: Undersamples the majority class to balance occupancy data.
- **Models**: Includes LLMs (llama3.2:latest, Gemini-1.5-pro, DeepSeek_R1) and baseline models (Logistic Regression, Random Forest, Decision Tree, XGBoost).

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/LLM-occupancy.git
   cd LLM-occupancy

2. ## Create a Conda Environment:

 ```bash
conda create --name ml_env python=3.8
conda activate ml_env
conda install pandas scikit-learn numpy requests
 ```

3. ## Prepare Your Data:
Place your CSV files in a folder named Dataset within the project directory.
Ensure the CSV files have a datetime column in "YYYY-MM-DD HH:MM:SS" format and a binary occupant_num column.

# Usage
To run the project, follow these steps:

1. ## Run Data Processing and Normalization:
- Update input_file_path and resampled_paths in the scripts to match your file names.
- Execute:
 ```bash
python data_processing.py
python normalization.py
 ```
2. ## Balance the Dataset:
- Modify input_file_path in the balancing script.
- Run:
 ```bash
python data_balancing.py
 ```
3. ## Run LLM Models:
- Ensure the LLM API is hosted locally (e.g., Ollama for llama3.2:latest).
- Adjust the model parameter in generate_llm_response() for other LLMs.
- Execute:
 ```bash
python llm_models.py
 ```
4. ## Run Baseline Models:
- Ensure processed CSV files are in the Dataset folder.
- Run:
 ```bash
python baseline_models.py
 ```
# Code
## Data Processing

```bash
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
```
# Normalization
```bash
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
```

# Data Splitting
```bash
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
```
# Data Balancing
```bash
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
```
# LLM Models

```bash
import os
import pandas as pd
import requests
import json
import time
from sklearn.metrics import classification_report

RESULT_DIR = "result_OCC"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def generate_llm_response(prompt, model="llama3.2:latest", retries=3, delay=2):
    url = "http://localhost:11434/api/chat"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            messages = []
            for line in response.text.strip().splitlines():
                try:
                    message = json.loads(line)
                    if "content" in message.get("message", {}):
                        messages.append(message["message"]["content"])
                    if message.get("done"):
                        break
                except json.JSONDecodeError:
                    print(f"Error parsing response line: {line}")
            return " ".join(messages).strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    return "Error"

def construct_prompt(training_data, test_row):
    default_temperature = 75
    temp_mapping = {
        "2021-08-23": 78, "2021-08-24": 77, "2021-08-25": 79, "2021-08-26": 78, "2021-08-27": 77,
        "2021-08-28": 78, "2021-08-29": 77, "2021-08-30": 76, "2021-08-31": 75, "2021-09-01": 74,
        "2021-09-02": 73, "2021-09-03": 72, "2021-09-04": 72, "2021-09-05": 71, "2021-09-06": 70,
        "2021-09-07": 69, "2021-09-08": 68, "2021-09-09": 67, "2021-09-10": 66, "2021-09-11": 65,
        "2021-09-12": 64
    }
    training_data_sample = training_data.sample(n=min(20, len(training_data)), random_state=42)
    training_examples = "\n".join([
        f"- {', '.join([f'{col}: {row[col]}' for col in training_data_sample.columns if col != 'occupant_num'])}, "
        f"Temperature: {temp_mapping.get(str(row['datetime'].date()), default_temperature)}°F, "
        f"Occupancy: {'Occupied' if row['occupant_num'] > 0 else 'Not Occupied'}"
        for _, row in training_data_sample.iterrows()
    ])
    test_date_str = str(test_row['datetime'].date())
    test_temperature = temp_mapping.get(test_date_str, default_temperature)
    test_example = ", ".join([f"{col}: {test_row[col]}" for col in training_data.columns if col != 'occupant_num']) + f", Temperature: {test_temperature}°F"
    prompt = (
        "You are a model trained to detect room occupancy status (Occupied or Not Occupied) in an office building "
        "in Baoding City, Hebei Province, using data from August 23 to September 12, 2023, with provided temperatures.\n"
        f"Training Data:\n{training_examples}\n"
        "Predict the occupancy status for:\n"
        f"- {test_example}"
    )
    return prompt

def process_data(file_name):
    data = pd.read_csv(file_name)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data[(data['datetime'].dt.hour >= 9) & (data['datetime'].dt.hour < 18)]
    data['day_of_week'] = data['datetime'].dt.dayofweek
    training_data = data[data['day_of_week'] < 4]
    test_data = data[data['day_of_week'] == 4]
    predictions = []
    for _, test_row in test_data.iterrows():
        prompt = construct_prompt(training_data, test_row)
        prediction = generate_llm_response(prompt)
        predictions.append({
            "timestamp": test_row['datetime'],
            "true_label": 'Occupied' if test_row['occupant_num'] > 0 else 'Not Occupied',
            "pred_label": prediction
        })
    output_file = os.path.join(RESULT_DIR, f"{os.path.splitext(os.path.basename(file_name))[0]}_pred.csv")
    pd.DataFrame(predictions).to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")
    print("Accuracy Report:")
    predictions_df = pd.DataFrame(predictions)
    print(classification_report(predictions_df["true_label"], predictions_df["pred_label"]))

def main():
    directory = os.getcwd()
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    for file_name in csv_files:
        file_path = os.path.join(directory, file_name)
        print(f"Processing file: {file_name}")
        process_data(file_path)

if __name__ == "__main__":
    main()
```
# Baseline Models

```bash
import pandas as pd
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(filepath):
    data = pd.read_csv(filepath)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

def prepare_data(data):
    train_data = data[data.index.weekday < 4]
    test_data = data[data.index.weekday == 4]
    X_train = train_data.drop(columns='occupant_num')
    y_train = train_data['occupant_num']
    X_test = test_data.drop(columns='occupant_num')
    y_test = test_data['occupant_num']
    return X_train, y_train, X_test, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'XGBoost': GradientBoostingClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = accuracy_score(y_test, predictions)
    return results

def main():
    data_directory = './Dataset'
    result_directory = 'Result_ML'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_directory, filename)
            data = load_data(filepath)
            X_train, y_train, X_test, y_test = prepare_data(data)
            results = train_and_evaluate(X_train, y_train, X_test, y_test)
            results_path = os.path.join(result_directory, f'{filename}_results.json')
            with open(results_path, 'w') as file:
                json.dump(results, file)
            print(f"Results for {filename}:")
            for model_name, accuracy in results.items():
                print(f"{model_name}: {accuracy:.2%}")

if __name__ == "__main__":
    main()
```
# Model Accuracy Results

| Interval        | Logistic Regression | Random Forest | Decision Tree | XGBoost | LLM (llama3.2:latest) |
|-----------------|---------------------|---------------|---------------|---------|-----------------------|
| Week 1 - 5 min  | 91.67%              | 94.44%        | 88.89%        | 93.06%  | 91.67%                |
| Week 1 - 10 min | 86.36%              | 88.64%        | 68.18%        | 88.64%  | 86.36%                |
| Week 1 - 30 min | 83.33%              | 83.33%        | 72.22%        | 83.33%  | 88.89%                |
| Week 2 - 5 min  | 86.73%              | 89.80%        | 83.67%        | 88.78%  | 86.73%                |
| Week 2 - 10 min | 87.50%              | 91.67%        | 77.08%        | 89.58%  | 89.58%                |
| Week 2 - 30 min | 91.67%              | 100.00%       | 66.67%        | 100.00% | 91.67%                |
| Week 3 - 5 min  | 76.71%              | 65.75%        | 42.47%        | 63.01%  | 82.19%                |
| Week 3 - 10 min | 65.12%              | 60.47%        | 46.51%        | 55.81%  | 86.05%                |
| Week 3 - 30 min | 63.16%              | 63.16%        | 63.16%        | 63.16%  | 94.74%                |

**Note**: Results shown are for llama3.2:latest. To generate results for Gemini-1.5-pro or DeepSeek_R1, adjust the model parameter in the LLM script.

# Conclusion
The LLM (llama3.2:latest) consistently outperforms Decision Tree and Logistic Regression, particularly at longer intervals (10 and 30 minutes), and demonstrates strong generalization in Week 3 with up to 94.74% accuracy. It remains competitive with Random Forest and XGBoost, especially in Week 2. LLMs, including Gemini-1.5-pro and DeepSeek_R1, provide robust occupancy prediction across various intervals and weeks.

# Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:
```bash
git checkout -b feature/new-feature
```
3. Commit your changes:
```bash
git commit -m 'Add new feature'
```
4. Push to the branch:
```bash
git push origin feature/new-feature
```
5. Open a pull request.
Please ensure your code adheres to the project’s style and includes tests where applicable.

