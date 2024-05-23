import os
import numpy as np
import pandas as pd


import tensorflow as tf

from sklearn.metrics import roc_auc_score,accuracy_score,f1_score

def report_results(metrics_result,num_branches,dataset_name,model_used):
    with open("results.txt", 'a') as file:
        file.write("\n")
        file.write("MODEL USED: {0}".format(model_used)+"\n")
        file.write("NUMBER OF CONDITION BRANCHES: {0}".format(num_branches)+"\n")
        file.write("DATASET: {0}".format(dataset_name)+"\n")
        file.write("ACCURACY: {0}".format(metrics_result[1])+"\n")
        file.write("ROC-AUC SCORE: {0}".format(metrics_result[0])+"\n")
        file.write("F1 SCORE: {0}".format(metrics_result[2])+"\n")


model = tf.keras.models.load_model('my_model.keras')

# testing
test_list = [15,30,39,43]
for i in test_list:
    csv_name = os.path.join("../../sim_final_project/data/test_1/","SHORT_MOBILE-{id}.bt9.trace.gz.csv".format(id=i))
    data = pd.read_csv(csv_name)
    print("read testing data {i} done.".format(i=i))
    X_test = np.array(data.iloc[:,:-1]/256.0)
    y_test = np.array(data.iloc[:,-1])
    y_test = y_test[:,np.newaxis]
    data = None
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    X_len = X_test.shape[0]
    X_test = None
    metrics = roc_auc_score(y_test,y_pred),accuracy_score(y_test,y_pred),f1_score(y_test,y_pred)
    report_results(metrics,X_len,"SHORT_MOBILE-{id}.bt9.trace.gz.csv".format(id=i),'MLP')

