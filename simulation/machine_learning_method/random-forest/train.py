import os
import numpy as np
import pandas as pd
import joblib
import time
import gc
import xgboost as xgb

NUM_SAMPLES = 10
BATCH_SIZE = 1  # 每次處理 1 個數據集
EARLY_STOPPING_ROUNDS = 10  # 早停輪數
GRID_SEARCH_PARAMS = {
    'max_depth': [10, 15, 20],
    'eta': [0.01, 0.1, 0.2],
    'min_child_weight': [1, 5, 10],
    'num_boost_round': [3, 5, 7]  # 加入樹的數量範圍
}

def read_and_decompress(file_path):
    data = pd.read_csv(file_path)
    return data

def save_checkpoint(model, checkpoint_file, current_batch):
    model.save_model(checkpoint_file)
    with open('last_batch.txt', 'w') as f:
        f.write(str(current_batch))
    print(f"Checkpoint saved at batch {current_batch}")

def load_checkpoint(checkpoint_file):
    model = xgb.Booster()
    model.load_model(checkpoint_file)
    with open('last_batch.txt', 'r') as f:
        last_batch = int(f.read().strip())
    print(f"Checkpoint loaded from batch {last_batch}")
    return model, last_batch

def perform_grid_search(X_train, y_train, params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    best_params = None
    best_score = 0

    for max_depth in GRID_SEARCH_PARAMS['max_depth']:
        for eta in GRID_SEARCH_PARAMS['eta']:
            for min_child_weight in GRID_SEARCH_PARAMS['min_child_weight']:
                for num_boost_round in GRID_SEARCH_PARAMS['num_boost_round']:
                    grid_params = {
                        'max_depth': max_depth,
                        'eta': eta,
                        'min_child_weight': min_child_weight
                    }

                    cv_results = xgb.cv(
                        grid_params,
                        dtrain,
                        num_boost_round=num_boost_round,
                        nfold=3,
                        metrics='auc',
                        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        seed=42
                    )

                    mean_auc = cv_results['test-auc-mean'].max()  # 將 AUC 取最大值
                    if mean_auc > best_score:
                        best_score = mean_auc
                        best_params = (max_depth, eta, min_child_weight, num_boost_round)

                    print(f"Params: max_depth={max_depth}, eta={eta}, min_child_weight={min_child_weight}, num_boost_round={num_boost_round}, AUC: {mean_auc}")

    print(f"Best params found: max_depth={best_params[0]}, eta={best_params[1]}, min_child_weight={best_params[2]}, num_boost_round={best_params[3]}, AUC: {best_score}")
    return {
        'max_depth': best_params[0],
        'eta': best_params[1],
        'min_child_weight': best_params[2],
        'num_boost_round': best_params[3]
    }

# 訓練部分
def train_model(train_list,type):
    checkpoint_file = 'xgb_model_checkpoint.json'
    if (os.path.exists(checkpoint_file) and os.path.exists('last_batch.txt')):
        model, start_batch = load_checkpoint(checkpoint_file)
        print(f"Resuming from batch {start_batch + 1}")
    else:
        model = None
        start_batch = 0

    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # 使用 hist 樹方法
        'device': 'cuda',  # 使用 GPU
        'random_state': 42
    }

    for i in range(start_batch, len(train_list), BATCH_SIZE):
        X_train_list = []
        y_train_list = []

        batch = train_list[i:i + BATCH_SIZE]

        for j in batch:
            csv_name = os.path.join(f"../../data/generated/{type}.train/",j)
            data = read_and_decompress(csv_name)
            print(f"read data {j} done.")

            X_train = np.array(data.iloc[:, :-1], dtype=np.float32)
            y_train = np.array(data.iloc[:, -1], dtype=np.float32)
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            data = None

        # 合併批次數據
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)

        dtrain = xgb.DMatrix(X_train, label=y_train)

        start_time = time.time()

        if model is None:
            # 初始化模型並進行網格搜索
            print("model is none... Initializing new model and performing grid search.")
            best_params = perform_grid_search(X_train, y_train, params)
            params.update({
                'max_depth': best_params['max_depth'],
                'eta': best_params['eta'],
                'min_child_weight': best_params['min_child_weight']
            })
            # 初始化模型
            model = xgb.train(params, dtrain, num_boost_round=best_params['num_boost_round'], early_stopping_rounds=EARLY_STOPPING_ROUNDS, evals=[(dtrain, 'train')])
        else:
            # 增量訓練
            print("model retraining... Continuing from loaded model.")
            model = xgb.train(params, dtrain, num_boost_round=best_params['num_boost_round'], xgb_model=model, early_stopping_rounds=EARLY_STOPPING_ROUNDS, evals=[(dtrain, 'train')])

        end_time = time.time()

        print(f"Training time for batch {i // BATCH_SIZE + 1}: {end_time - start_time} seconds")

        # 儲存當前模型和批次信息
        save_checkpoint(model, checkpoint_file, i + 1)  # 儲存為下一次的起始批次

        # 清除不必要的變數以節省記憶體
        del X_train, y_train, X_train_list, y_train_list
        gc.collect()

    # 最後儲存完整模型
    model.save_model('xgb_model_{}.json'.format(type))
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if os.path.exists('last_batch.txt'):
        os.remove('last_batch.txt')

# 主程式
if __name__ == "__main__":

    train_type = ["LONG_SERVER","SHORT_MOBILE","SHORT_SERVER","LONG_MOBILE","ALL"]
    for type in train_type:
        data_dir = "../../data/generated/{}.train".format(type)
        train_list = [f for f in os.listdir(data_dir) if f.startswith(type)]
        train_model(train_list,type)
