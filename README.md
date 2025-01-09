# LLM-occucpancy
## Data Processing
The purpose of this code is to process and simplify time-series data by resampling it into fixed intervals (5, 10, and 30 minutes). By averaging data points within these intervals, we reduce data granularity, making it easier to analyze broader trends and patterns over time. The function loads the dataset, resamples it, and saves the results into separate files, ensuring the original dataset remains intact. This step is particularly useful for applications where high-frequency data is unnecessary or overwhelming, allowing for more efficient visualization, analysis, and modeling. The resampled data files are saved with appropriate filenames, ensuring clear identification of the time interval used.

## Code

```python
import pandas as pd

def resample_and_save_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)

    # Resample the dataset into different time windows using 'min' instead of 'T'
    data_5min = data.resample('5min').mean()  # 5 minutes
    data_10min = data.resample('10min').mean()  # 10 minutes
    data_30min = data.resample('30min').mean()  # 30 minutes

    # Define file paths for the resampled data
    path_5min = file_path.replace('.csv', '_5min.csv')
    path_10min = file_path.replace('.csv', '_10min.csv')
    path_30min = file_path.replace('.csv', '_30min.csv')

    # Save the resampled data to new CSV files
    data_5min.to_csv(path_5min)
    data_10min.to_csv(path_10min)
    data_30min.to_csv(path_30min)

    # Print the paths of saved files as a confirmation
    print("5-minute resampled data saved to:", path_5min)
    print("10-minute resampled data saved to:", path_10min)
    print("30-minute resampled data saved to:", path_30min)

# Specify the path to your input file
input_file_path = 'cleaned_room_1_data1.csv'
'''

# Call the function to process and save the data
resample_and_save_data(input_file_path)
```

## Normalization for each resampling
The normalization is performed using the StandardScaler from the sklearn.preprocessing library. The purpose of normalization is to standardize the features (predictor variables) so that they have a mean of 0 and a standard deviation of 1. This is achieved by applying the StandardScaler to the feature columns of the dataset. The fit_transform() method calculates the mean and standard deviation for each feature and then scales the values accordingly. This ensures that all features are on a comparable scale, which is important for many machine learning algorithms to work effectively. After scaling, the normalized features are converted back into a DataFrame with the original column names and indices, and the binary occupant_num target column is re-added to the dataset. The final result is a dataset where the features are normalized, making them suitable for further analysis or modeling.

## Code

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def check_and_clean_data(data):
    # Check for any infinite or NaN values in the data
    if np.any(np.isinf(data)) or data.isnull().any().any():
        print("Data contains infinite or NaN values. Handling...")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
        data.fillna(data.mean(), inplace=True)  # Replace NaNs with the mean of each column
    return data


def convert_normalize_and_save(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)

    # Clean data if necessary
    data = check_and_clean_data(data)

    # Convert 'occupant_num' to binary: 0 if no occupants, 1 if there are one or more occupants
    data['occupant_num'] = (data['occupant_num'] > 0).astype(int)

    # Separate features and target
    features = data.drop('occupant_num', axis=1)
    target = data['occupant_num']

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Convert the scaled features back to a DataFrame
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns, index=data.index)

    # Include the binary 'occupant_num' back into the DataFrame
    features_scaled_df['occupant_num'] = target

    # Define the output file path
    output_file_path = file_path.replace('.csv', '_normalized.csv')

    # Save the normalized and binary converted data to a new CSV file
    features_scaled_df.to_csv(output_file_path)

    print(f"Binary converted and normalized data saved to: {output_file_path}")


# Paths to the resampled datasets
resampled_paths = [
    'cleaned_room_1_data1_5min.csv',
    'cleaned_room_1_data1_10min.csv',
    'cleaned_room_1_data1_30min.csv'
]

# Convert 'occupant_num' to binary, normalize, and save each dataset
for path in resampled_paths:
    convert_normalize_and_save(path)
```
## Split dataset
The normalization datasets split into weekly date ranges, filter for office hours (9 AM to 6 PM), and remove columns containing only zeros. By isolating office hours, the analysis focuses on relevant periods for occupancy and energy usage patterns. Splitting the data into weekly blocks makes it easier to study trends over time while removing zero-only columns enhances clarity and reduces unnecessary complexity. The processed data is saved with clear filenames indicating the date range and time resolution, ensuring efficient organization and accessibility for subsequent analysis. This structured approach supports meaningful exploration of time-series data.

## Code

```python
import pandas as pd


def split_and_filter_dataset(file_path, date_ranges, output_suffix):
    """
    Splits the dataset into specified date ranges, filters for office hours (9 AM to 6 PM),
    removes columns with only zero values, and saves the results as new CSV files.

    Parameters:
    - file_path: str, path to the input dataset.
    - date_ranges: list of tuples, date ranges to filter (start_date, end_date).
    - output_suffix: str, suffix to add to the output file names.
    """
    # Load the dataset
    data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)

    # Filter for office hours: 9 AM to 6 PM
    office_hours = (data.index.time >= pd.to_datetime('09:00').time()) & \
                   (data.index.time <= pd.to_datetime('18:00').time())
    data = data[office_hours]

    # Split the data into specified date ranges
    for start_date, end_date in date_ranges:
        # Filter data for the date range
        range_data = data[(data.index.date >= pd.to_datetime(start_date).date()) &
                          (data.index.date <= pd.to_datetime(end_date).date())]

        # Drop columns that only contain zeros
        range_data = range_data.loc[:, (range_data != 0).any(axis=0)]

        # Save the filtered data to a new file
        output_file_path = file_path.replace('.csv', f'_{start_date}_to_{end_date}_{output_suffix}.csv')
        range_data.to_csv(output_file_path)

        print(f"Data for {start_date} to {end_date} saved to: {output_file_path}")


# Define date ranges for splitting
date_ranges = [
    ('2021-08-23', '2021-08-28'),
    ('2021-08-30', '2021-09-04'),
    ('2021-09-06', '2021-09-11')
]

# Paths to the resampled datasets
file_paths = [
    'cleaned_room_1_data1_5min_normalized.csv',
    'cleaned_room_1_data1_10min_normalized.csv',
    'cleaned_room_1_data1_30min_normalized.csv'
]

# Suffixes for output files to indicate time window
suffixes = ['5min', '10min', '30min']

# Process each dataset with the defined date ranges
for file_path, suffix in zip(file_paths, suffixes):
    split_and_filter_dataset(file_path, date_ranges, suffix)
```
## What is Data Balancing?

Data balancing refers to addressing the issue of unequal class distribution in a dataset. This is common in binary classification problems, like your task of detecting occupancy (`0` for unoccupied, `1` for occupied), where one class (e.g., `0`) may have significantly more samples than the other (e.g., `1`).

## Why is Balancing Important?

1. **Model Bias:** Most machine learning algorithms tend to favor the majority class, leading to poor performance on the minority class.
2. **Performance Metrics:** Metrics like accuracy can be misleading in imbalanced datasets (e.g., if 90% of data is of one class, a model predicting that class 100% of the time would appear "accurate").
3. **Fair Representation:** Balancing ensures that both classes are represented adequately during training, improving model robustness.

## Methods for Data Balancing

1. **Resampling Techniques:**
    - **Oversampling:** Increase the size of the minority class by duplicating samples or generating synthetic samples.
        - **SMOTE (Synthetic Minority Oversampling Technique):** Generates synthetic samples in feature space.
        - **Random Oversampling:** Simply duplicates minority class samples.
    - **Undersampling:** Reduce the size of the majority class by randomly removing samples.
        - **Random Undersampling:** Removes majority class samples.
        - **Cluster Centroids:** Uses clustering to find centroids of the majority class and downsamples to those points.
2. **Class Weights:**
    - Adjust the class weights during model training so that the minority class has higher weight, penalizing misclassification more heavily.
3. **Data Augmentation:**
    - For cases like images or text, create new samples for the minority class by transformations (e.g., rotation, flipping for images).
4. **Generate Synthetic Data:**
    - Use techniques like GANs (Generative Adversarial Networks) for creating synthetic samples of the minority class.
5. **Stratified Sampling:**
    - When splitting the dataset into training, validation, and testing, ensure that the class distribution is maintained across splits.

## When to Apply Balancing?

- **Training Data Only:** Apply balancing techniques to the training data, not the validation or test sets. This ensures the evaluation reflects real-world class distributions.
- **Before Model Training:** Balancing is typically done after splitting the data into training and testing sets.

## Dataset balancing to our Datasets

Here's a script that balances the dataset by **undersampling the majority class (`0: unoccupied`)** to match the minority class (`1: occupied`). The `datetime` column will remain intact throughout the process:
## Code

```python
python
Copy code
import pandas as pd

def balance_dataset(file_path):
    """
    Balances the dataset by undersampling the majority class to match the minority class.

    Parameters:
    - file_path: str, path to the input dataset.
    """
    # Load the dataset
    data = pd.read_csv(file_path, index_col='datetime', parse_dates=True)

    # Separate the two classes
    class_0 = data[data['occupant_num'] == 0]  # Majority class
    class_1 = data[data['occupant_num'] == 1]  # Minority class

    # Undersample the majority class to match the size of the minority class
    class_0_undersampled = class_0.sample(n=len(class_1), random_state=42)

    # Combine the undersampled majority class with the minority class
    balanced_data = pd.concat([class_0_undersampled, class_1], axis=0)

    # Shuffle the combined data
    balanced_data = balanced_data.sample(frac=1, random_state=42)

    # Save the balanced dataset to a new file
    output_file_path = file_path.replace('.csv', '_undersampled_balanced.csv')
    balanced_data.to_csv(output_file_path)

    print(f"Balanced dataset saved to: {output_file_path}")
    print(f"Balanced class distribution:\n{balanced_data['occupant_num'].value_counts()}")

# Path to the input file
input_file_path = '/mnt/data/R1W1_5min.csv'

# Call the function to balance the dataset
balance_dataset(input_file_path)
```
## What is LLM (Large Language Model)?

A Large Language Model (LLM) is a type of artificial intelligence designed to understand and generate human language. These models are trained on massive datasets that include vast amounts of text from various sources, allowing them to learn patterns in language, context, and meaning. LLMs, such as GPT (Generative Pre-trained Transformer) or LLaMA (Large Language Model Meta AI), are capable of processing and generating coherent text, interpreting structured data, answering questions, and making predictions based on the input provided. Their ability to process complex language structures makes them a powerful tool in a variety of applications, including natural language understanding, text generation, and data interpretation.

In this resarch, we will apply the llama3.2:latest model, a version of the LLaMA family of LLMs, to predict room occupancy based on historical environmental and control system data. The model uses natural language processing to analyze the structured data provided in the form of a prompt, making it suitable for tasks like occupancy prediction. The model generates predictions based on a prompt that is dynamically created with context and historical data. This prompt incorporates a selection of training data, which consists of records detailing environmental and control system factors, such as date, time, temperature, and occupancy. These training examples, which are randomly sampled from the historical data, inform the model by providing examples of when rooms were occupied and the associated temperatures for those dates. If a date from the test data is missing from the temperature mapping, a default temperature is used instead.

For each row of test data, the prompt is constructed by excluding the occupancy column (the true occupancy value) and including other relevant features. The resulting test data is then appended to the prompt, asking the model to predict whether the room is occupied or not, based on the input details. This prompt is sent to a locally hosted version of the "llama3.2:latest" model via the LLM API, which processes the prompt and returns a prediction of the occupancy status. The LLM uses the historical context within the prompt to evaluate the likelihood of the room being occupied, leveraging its natural language processing capabilities. By interpreting the structured data, the LLM transforms it into a prediction that provides valuable insights for managing building occupancy, making the information accessible and actionable.

## Code

```python
import os
import pandas as pd
import requests
import json
import time
from sklearn.metrics import classification_report

# Define result directory
RESULT_DIR = "result_occ"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


# Function to interact with LLM API with retry logic
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


# Construct prompt dynamically using all columns except `occupant_num`
def construct_prompt(training_data, test_row):
    # Define a default temperature in case the date is not found in the mapping
    default_temperature = 75  # This should be a reasonable estimate or the average temperature

    # Temperature mapping from the user-provided data
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
    if test_date_str not in temp_mapping:
        print(
            f"Warning: Temperature for date {test_date_str} not found. Using default temperature {default_temperature}°F.")
    test_example = ", ".join([f"{col}: {test_row[col]}" for col in training_data.columns if col != 'occupant_num']) + f", Temperature: {test_temperature}°F"

    prompt = (
        "You are a model trained to detect room occupancy status (occupied or not occupied) in an office building "
        "located in Baoding City, Hebei Province. This analysis considers environmental and control system data, "
        "covering the period from August 23 to September 12, 2023, with the specific temperature fluctuations provided. "
        "This setting includes typical office hours and building dynamics.\n\n"
        f"Training Data:\n{training_examples}\n"
        "Predict the occupancy status (Occupied or Not Occupied) for the following data:\n\n"
        f"- {test_example}"
    )
    return prompt


# Main processing function
def process_data(file_name):
    try:
        data = pd.read_csv(file_name)
        data['datetime'] = pd.to_datetime(data['datetime'])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if data.isnull().any().any():
        print("Dataset contains missing values. Please clean the data before running.")
        return

    data = data[(data['datetime'].dt.hour >= 9) & (data['datetime'].dt.hour < 18)]
    data['date'] = data['datetime'].dt.date
    data['day_of_week'] = data['datetime'].dt.dayofweek

    training_data = data[data['day_of_week'] < 4]  # Monday to Thursday
    test_data = data[data['day_of_week'] == 4]  # Friday

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
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_file, index=False)

    print(f"Predictions saved to: {output_file}")
    print("Accuracy Report:")
    print(classification_report(predictions_df["true_label"], predictions_df["pred_label"]))


# Entry point
def main():
    directory = os.getcwd()  # Assumes the CSV files are in the same directory as the script
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    for file_name in csv_files:
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            print(f"Dataset file '{file_name}' not found. Please check the file path.")
            continue
        print(f"Processing file: {file_name}")
        process_data(file_path)


if __name__ == "__main__":
    main()
```

