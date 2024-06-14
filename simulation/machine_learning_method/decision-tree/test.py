from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import joblib
import glob
import os
import gzip
from io import StringIO
import csv

def test_model(data, trace_paths, model):

    with open(f'{data}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['name', 'mispred', 'accuracy'])
        for trace_path in trace_paths:

            with open(trace_path, mode='rt', newline='') as file:
                test_content = file.read()

            test_data = StringIO(test_content)
            test_df = pd.read_csv(test_data, delimiter=',')
            test_df.columns = range(len(test_df.columns))
            repeat = test_df[0]
            test_df = test_df.drop(0, axis=1)
            x_test = test_df.drop(41, axis=1).to_numpy()
            y_test = test_df[41].to_numpy()
            
            y_pred = model.predict(x_test)

            total_instruction = np.sum(repeat)
            mispre_cnt = np.sum(np.array(repeat) * np.abs(y_pred - y_test))
            accuracy = 1 - np.sum(np.array(repeat) * np.abs(y_pred - y_test)) / total_instruction

            final_path = trace_path.split('/')[-1].replace('.bt9.trace', '')
            logger.info(f'accuracy_{final_path} : {accuracy}, {mispre_cnt}')
            writer.writerow([final_path, mispre_cnt, accuracy])

if __name__ == '__main__':
    # Four type of training dataset
    DATA_TYPE = ['LONG_SERVER', 'LONG_MOBILE', 'SHORT_SERVER', 'SHORT_MOBILE']

    for data in DATA_TYPE:
        filename = f"models/rf_model_{data}.pkl"
        model = joblib.load(filename)

        trace_paths = sorted(glob.glob(f'../../data/generated/{data}.test/{data}-*.bt9.trace'))
        test_model(data, trace_paths, model)

