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