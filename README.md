## Branch Prediction based on Machine Learning
* Computer Architecture Final Project
* Team members : 312551164 盧孟璇, 109654020 楊永琪, 312512082 胡弘沅
## CBP-16-Simulation evaluation
### evaluation on branch prediction machine learning model
* Make sure to install all the requirment in order to run simpython in CBP-16-Simulation Repo
* run `pip install -r requirements.txt`
* `cd CBP-16-Simulation/cbp16sim`
* run `make`
#### Decision Tree
#### Random Forest
#### Multi-layer Perceptron
* `PYTHONPATH=src/simpython/ ./simpython ../../../CBP-16-Simulation-master/cbp2016.eval/evaluationTraces/<trace-name> mlp_predictor.py`
    * Change the model name in `mlp_predictor.py` to evaluate models trained on different dataset.

## CBP-16 Training and Testing
Since running the CBP-16-Simulation repo takes a lot of time, we decided to collect the training and testing data ourselves for analysis.
### training and testing data generation
* Run `sim_final/dataset_gen.sh` to generated specific trace types.
* We also save the generated files on [google cloud](https://drive.google.com/drive/folders/19cVRMmxUc_1lO16mOBd7fITPS_rOU2rr?usp=sharing).
* After data generated, you may need to rename the generated files to ensure the program runs succesfully.
* During generation, OOM-killed problem may occur when using the `shuf` command due to the size of the data, we find an alternative program that can help us to shuffle the dataset : [terashuf](https://github.com/alexandres/terashuf)
* Ensure the directory structure is as follows:
    ```
    .
    ├── sim_final
    ├── machine_learning_method
    │   ├── decision_tree
    │   ├── random_forest
    │   └── mlp
    └── data
        ├── evaluationTraces
        ├── traces
        └── generated
           ├── ALL.train
           ├── LONG_MOBILE.test
           ├── LONG_MOBILE.train
           ├── LONG_SERVER.train
           ├── LONG_SERVER.test
           ├── SHORT_MOBILE.train
           ├── SHORT_MOBILE.test
           ├── SHORT_SERVER.train
           └── SHORT_SERVER.test
    ```
### training and evaluation
* `pip install -r requirements.txt`
#### Decision Tree
* To train decision tree branch predictor:
    * `cd machine_learning_method/decision-tree/`
    * run `python train.py`
    * It will generated four models according to each trace type, and a model that is trained on mixed data.
* To evaluate the decision tree result, run `python test.py`
* The evaluation metric of our trained decision models are stored in `machine_learning_method/decision-tree/models/analyze_data`
#### Random Forest
* To train random forest branch predictor:
    * `cd machine_learning_method/random-forest`
    * run `python train.py`
* To evalutate:
    * run `python test.py`
#### Multi-layer Perceptron
* To train mlp branch predictor:
    * `cd machine_learning_method/mlp`
    * run `python train.py` 
* To evaluate mlp branch predictor, run `python test.py`
* The evaluation metric of our trained mlp models are stored in `machine_learning_method/mlp/models/analyze_data`
* If need to train and evaluate on mixed dataset, use the combined data in `data/ALL.train` directory to train the model and edit the model name in `test.py
