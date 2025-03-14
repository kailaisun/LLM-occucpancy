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