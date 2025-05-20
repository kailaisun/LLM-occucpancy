import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np


def load_data(filepath):
    # Load the dataset and set datetime as index
    data = pd.read_csv(filepath)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data


def prepare_data(data, strategy):
    # Different data splitting strategies
    if strategy == 1:
        train_data = data[data.index.weekday < 4]
        test_data = data[data.index.weekday == 4]
    elif strategy == 2:
        train_data = data[data.index.weekday < 3]
        test_data = data[data.index.weekday >= 3]
    elif strategy == 3:
        train_data = data[data.index.weekday < 2]
        test_data = data[data.index.weekday >= 2]
    elif strategy == 4:
        train_data = data[data.index.weekday == 0]
        test_data = data[data.index.weekday > 0]

    X_train = train_data.drop(columns='occupant_num')
    y_train = train_data['occupant_num']
    X_test = test_data.drop(columns='occupant_num')
    y_test = test_data['occupant_num']

    return X_train, y_train, X_test, y_test


def evaluate_metrics(y_true, y_pred):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape == (1, 1):  # If only one class exists in y_true
        return 0.0, 0.0, 0.0, 0.0, np.array([[0, 0], [0, 0]])
    tn = cm[0, 0] if cm.shape[0] > 1 else 0
    tp = cm[1, 1] if cm.shape[0] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 else 0
    fp = cm[0, 1] if cm.shape[0] > 1 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return round(accuracy * 100, 2), round(recall * 100, 2), round(precision * 100, 2), round(f1 * 100, 2), cm


def train_and_evaluate(X_train, y_train, X_test, y_test, filename, strategy):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'XGBoost': GradientBoostingClassifier()
    }

    results = {}

    print(f"\n--- Evaluation Results for {filename} (Strategy {strategy}) ---")

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy, recall, precision, f1, cm = evaluate_metrics(y_test, predictions)

        print(f'[{name}]')
        print(f'  Accuracy  : {accuracy:.2f}%')
        print(f'  Recall    : {recall:.2f}%')
        print(f'  Precision : {precision:.2f}%')
        print(f'  F1-Score  : {f1:.2f}%')
        print(f'  Confusion Matrix:\n{cm}\n')
        print("---------------------------------------------------")

        results[name] = predictions

    return results


def main():
    data_directory = './Dataset'
    result_directory = './ML_detection_result'  # Folder for saving prediction results
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    strategies = {1: "Mon-Thur train, Fri test", 2: "Mon-Wed train, Thu-Fri test", 3: "Mon-Tue train, Wed-Fri test",
                  4: "Mon train, Tue-Fri test"}

    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_directory, filename)
            data = load_data(filepath)
            for strategy, description in strategies.items():
                X_train, y_train, X_test, y_test = prepare_data(data, strategy)
                results = train_and_evaluate(X_train, y_train, X_test, y_test, filename, strategy)
                combined_results = pd.DataFrame(index=X_test.index)
                combined_results['True Values'] = y_test
                for model_name, predictions in results.items():
                    combined_results[f'{model_name} Predictions'] = predictions

                csv_filename = f"{result_directory}/{filename.replace('.csv', '')}_strategy_{strategy}_all_predictions.csv"
                combined_results.to_csv(csv_filename)
                print(f"All model predictions for strategy {strategy} ({description}) saved to {csv_filename}")


if __name__ == "__main__":
    main()
