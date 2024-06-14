import os
import numpy as np
import pandas as pd
import gc
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import glob

def report_results(metrics_result, num_branches, dataset_name, model_used):
    with open("new_results.txt", 'a') as file:
        file.write("\n")
        file.write("MODEL USED: {0}".format(model_used) + "\n")
        file.write("NUMBER OF CONDITION BRANCHES: {0}".format(num_branches) + "\n")
        file.write("DATASET: {0}".format(dataset_name) + "\n")
        file.write("ACCURACY: {0}".format(metrics_result[1]) + "\n")
        file.write("ROC-AUC SCORE: {0}".format(metrics_result[0]) + "\n")
        file.write("F1 SCORE: {0}".format(metrics_result[2]) + "\n")

def read_and_decompress(file_path):
    data = pd.read_csv(file_path)
    return data

def plot_probability_distribution(probabilities, y_test, dataset_name):
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities, bins=50, alpha=0.75, color='blue', label='Predicted Probabilities')
    plt.hist(y_test, bins=50, alpha=0.5, color='red', label='Actual Labels')
    plt.xlabel('Probability / Label')
    plt.ylabel('Frequency')
    plt.title(f'Probability Distribution for {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'probability_distribution_{dataset_name}.png')
    plt.close()

def test_model(test_files, category):
    model = xgb.Booster()
    model.load_model('models/xgb_model_{}.json'.format(category))

    accuracies = []
    total_errors = 0  # 初始化錯誤預測次數

    for file_path in test_files:
        data = read_and_decompress(file_path)
        print(f"read testing data from {file_path} done.")

        # 去除每列的第一個數值
        repeat_counts = data.iloc[:, 0].astype(int)
        X_test = np.array(data.iloc[:, 1:-1], dtype=np.float32)
        y_test = np.array(data.iloc[:, -1], dtype=np.float32)
        data = None

        dtest = xgb.DMatrix(X_test)
        y_pred_proba = model.predict(dtest)

        plot_probability_distribution(y_pred_proba, y_test, os.path.basename(file_path))

        threshold = 0.5
        y_pred = (y_pred_proba > threshold).astype(int)

        # 計算錯誤預測次數
        errors = (y_test != y_pred).astype(int)
        total_errors += np.sum(errors * repeat_counts)

        # 根據重複次數更新評估指標
        expanded_y_test = np.repeat(y_test, repeat_counts)
        expanded_y_pred = np.repeat(y_pred, repeat_counts)
        expanded_y_pred_proba = np.repeat(y_pred_proba, repeat_counts)

        # 檢查 y_true 中的類別數量
        if len(np.unique(expanded_y_test)) == 1:
            print(f"Only one class present in y_true for file {file_path}. Skipping ROC AUC calculation.")
            continue

        metrics = roc_auc_score(expanded_y_test, expanded_y_pred_proba), accuracy_score(expanded_y_test, expanded_y_pred), f1_score(expanded_y_test, expanded_y_pred)
        report_results(metrics, len(expanded_y_test), os.path.basename(file_path), 'XGBoost test')

        accuracies.append(metrics[1])  # 收集準確率

        del X_test, y_test, y_pred_proba, y_pred, expanded_y_test, expanded_y_pred, expanded_y_pred_proba
        gc.collect()

    # 計算並輸出準確率的平均值
    if accuracies:
        avg_accuracy = np.mean(accuracies)
        print(f"Average accuracy for category {category}: {avg_accuracy}")
        with open("results.txt", 'a') as file:
            file.write(f"\nAverage accuracy for category {category}: {avg_accuracy}\n")

    # 輸出錯誤預測次數
    print(f"Total errors for category {category}: {total_errors}")
    with open("results.txt", 'a') as file:
        file.write(f"Total errors for category {category}: {total_errors}\n")

    # 保存進度
    with open("last_category.txt", 'w') as file:
        file.write(category)

if __name__ == "__main__":
    categories = ["SHORT_SERVER", "LONG_MOBILE","LONG_SERVER", "SHORT_MOBILE"]

    # 檢查是否有保存的進度
    if os.path.exists("last_category.txt"):
        with open("last_category.txt", 'r') as file:
            last_category = file.read().strip()
        start_index = categories.index(last_category) + 1
    else:
        start_index = 0

    for category in categories[start_index:]:
        pattern = f"../../data/generated/{category}.test/{category}-*.bt9.trace"
        files = glob.glob(pattern)
        files.sort(key=lambda x: int(os.path.basename(x).split('-')[1].split('.')[0]))

        print(f"Testing category: {category}")
        test_model(files, category)
