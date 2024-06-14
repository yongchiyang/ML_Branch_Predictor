import os
import numpy as np
import pandas as pd
import joblib
import time
import gc
import random
import glob
from snapml import DecisionTreeClassifier

BATCH_SIZE = 2  # Process 1 dataset at a time

def save_checkpoint(model, checkpoint_file, current_batch):
    joblib.dump(model, checkpoint_file)
    with open('last_batch.txt', 'w') as f:
        f.write(str(current_batch))
    print(f"Checkpoint saved at batch {current_batch}")

def load_checkpoint(checkpoint_file):
    model = joblib.load(checkpoint_file)
    with open('last_batch.txt', 'r') as f:
        last_batch = int(f.read().strip())
    print(f"Checkpoint loaded from batch {last_batch}")
    return model, last_batch

def train_model(train_files, data):
    checkpoint_file = 'rf_model_checkpoint.pkl'
    if os.path.exists(checkpoint_file) and os.path.exists('last_batch.txt'):
        model, start_batch = load_checkpoint(checkpoint_file)
        print(f"Resuming from batch {start_batch + 1}")
    else:
        model = None
        start_batch = 0

    for i in range(start_batch, len(train_files), BATCH_SIZE):
        X_train_list = []
        y_train_list = []

        batch_files = train_files[i:i + BATCH_SIZE]
        for file_path in batch_files:
            data = pd.read_csv(file_path)
            print(f"read data from {file_path} done.")

            X_train = np.array(data.iloc[:, :-1], dtype=np.float32)
            y_train = np.array(data.iloc[:, -1], dtype=np.float32)
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            data = None

        # Merge batch data
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)

        start_time = time.time()

        if model is None:
            # Initialize model
            print("model is none...Initializing new model.")
            
            model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=7,
                # use_gpu=False, #uncomment this if cannot use gpu
                use_gpu=True,
                random_state=42,
                gpu_id=0
            )
            model.fit(X_train, y_train)
        else:
            # Incremental training
            print("model retraining... Continuing from loaded model.")
            model.fit(X_train, y_train)

        end_time = time.time()

        print(f"Training time for batch {i // BATCH_SIZE + 1}: {end_time - start_time} seconds")

        # Save current model and batch information
        save_checkpoint(model, checkpoint_file, i + 1)  # Save as the starting batch for next time

        # Clear unnecessary variables to save memory
        del X_train, y_train, X_train_list, y_train_list
        gc.collect()

    # Finally save the complete model
    joblib.dump(model, f'rf_model_{data}.pkl')
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if os.path.exists('last_batch.txt'):
        os.remove('last_batch.txt')

# Main program
if __name__ == "__main__":
    # Four type of training dataset
    DATA_TYPE = ['LONG_SERVER','LONG_MOBILE','SHORT_SERVER','SHORT_MOBILE','ALL']

    for data in DATA_TYPE:
        # Find all matching files
        train_files = glob.glob(f"../../data/data_info/{data}.train/{data}-fin-*")
        random.shuffle(train_files)  # Shuffle the list of files
        train_model(train_files, data)
