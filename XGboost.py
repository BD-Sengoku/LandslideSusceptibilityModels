import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix
import xgboost as xgb

# Read Data
data_valid = pd.read_csv(r'C:\Users\1\Desktop\sampleSize\554ValidSet.csv', encoding='gbk')
data_train = pd.read_csv(r'C:\Users\1\Desktop\sampleSize\1290TrainSet_0.1.csv', encoding='gbk')

filename = r'C:\Users\1\Desktop\sampleSize\XG\Result_0.1.txt'
seed = 4
max_evals = 10
# Remove the useless columns in the CSV dataset
columns = ["target", "OBJECTID", "X", "Y"]

# Select the training and validation sets from the data
train_y = data_train.target
train_X = data_train.drop(columns=columns, axis=1)
valid_y = data_valid.target
valid_X = data_valid.drop(columns=columns, axis=1)

# Define xgboost's training and validation sets
dtrain = xgb.DMatrix(train_X, label=train_y)
dvalid = xgb.DMatrix(valid_X, label=valid_y)
evallist = [(dtrain, 'train'), (dvalid, 'eval')]

import hyperopt

def cross_validation(model_params, data):
    """
    Function to perform cross-validation using XGBoost model with given parameters.
    
    Returns:
    - float: Inverted ROC AUC score (1 - ROC AUC) as the objective to minimize.
    """
    
    dtrain, dvalid = data
    evallist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(model_params, dtrain, 5000, evals=evallist, early_stopping_rounds=50)
    y_pred = bst.predict(dvalid)
    auc = roc_auc_score(valid_y, y_pred)
    return 1 - auc

def hyperopt_objective(params):
	"""
    Objective function for Hyperopt optimization.

    Parameters:
    - params (dict): Dictionary containing hyperparameters to optimize.

    Returns:
    - float: Value of the objective function to minimize.
    """
    cur_param = {
        'max_depth': params["max_depth"],
        'eta': params["eta"],
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': 1,
        'min_child_weight': params["min_child_weight"],
        'subsample': params["subsample"],
        'colsample_bytree': params["colsample_bytree"],
        'gamma': params["gamma"],
        'reg_alpha': params["reg_alpha"],
        'verbose': 100
    }
    data = (dtrain, dvalid)
    res = cross_validation(cur_param, data)
    return res  

params_space = {
    "eta": hyperopt.hp.uniform("eta", 0.001, 0.5),
    "max_depth": hyperopt.hp.choice("max_depth", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
    "min_child_weight": hyperopt.hp.choice("min_child_weight", [1, 2, 3, 4, 5, 6, 7, 8]),
    "subsample": hyperopt.hp.uniform("subsample", 0, 0.9),
    'colsample_bytree': hyperopt.hp.uniform("colsample_bytree", 0, 0.9),
    "gamma": hyperopt.hp.uniform("gamma", 0.1, 0.5),
    "reg_alpha": hyperopt.hp.choice("reg_alpha", [0, 0.01, 0.05, 0.1, 0.5]),
}

trials = hyperopt.Trials()
import warnings
warnings.filterwarnings("ignore")

best = hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=max_evals,
    trials=trials)
print("BestParameter")
print(best)

# The optimal parameters after training are obtained
best_params = hyperopt.space_eval(params_space, best)
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'auc'
best_params['silent'] = '1'
best_params['verbose'] = '100'

#Substitute the optimal parameters into the model
xgb_model = xgb.train(best_params, dtrain, 2000, evals=evallist, early_stopping_rounds=50)
y_pred1 = xgb_model.predict(dvalid)
auc1 = roc_auc_score(valid_y, y_pred1)
y_pred1 = (y_pred1 >= 0.5) * 1
a = confusion_matrix(valid_y, y_pred1)
a = a.tolist()
a0 = str(a[0])
a1 = str(a[1])

# Calculate the commonly used evaluation indicators of machine learning algorithms
Precision = 'Precision: %.4f' % metrics.precision_score(valid_y, y_pred1)
Recall = 'Recall: %.4f' % metrics.recall_score(valid_y, y_pred1)
F1_score = 'F1-score: %.4f' % metrics.f1_score(valid_y, y_pred1)
Accuracy = 'Accuracy: %.4f' % metrics.accuracy_score(valid_y, y_pred1)
AUC = 'AUC: %.4f' % auc1
AP = 'AP: %.4f' % metrics.average_precision_score(valid_y, y_pred1)
Log_loss = 'Log_loss: %.4f' % metrics.log_loss(valid_y, y_pred1, eps=1e-15, normalize=True, sample_weight=None, labels=None)
Kappa_score = 'Kappa_score: %.4f' % metrics.cohen_kappa_score(valid_y, y_pred1)

# Calculate the sum of selected metrics
total_metric_sum = (
    metrics.precision_score(valid_y, y_pred1) +
    metrics.recall_score(valid_y, y_pred1) +
    metrics.f1_score(valid_y, y_pred1) +
    metrics.accuracy_score(valid_y, y_pred1) +
    auc1 + 
    metrics.average_precision_score(valid_y, y_pred1)
)
total_model_performance_sum = 'total_model_performance(sum): %.4f' % total_metric_sum
total_model_performance_mean = 'total_model_performance(mean): %.4f' % (total_metric_sum / 6)

# Splitting multiple strings into multiple lines to improve readability
confusion_matrix = f'{a0}\n{a1}\n'
metrics = f'{AUC}\n{Precision}\n{Recall}\n{F1_score}\n{Accuracy}\n{AP}\n{Log_loss}\n{Kappa_score}\n'

# Feature Importance Handling Area
my_dict = xgb_model.get_score(importance_type='weight')
# New dictionary sorted from largest to smallest by value
sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
# Calculate the sum of values
total = sum(sorted_dict.values())
# Calculate the percentage of each value and save it to a new dictionary
dict1 = {key: (value / total) * 100 for key, value in sorted_dict.items()}

# Open file, create automatically if file does not exist
with open(filename, 'w') as f:
    # Write Confusion Matrix
    f.write('---------Confusion Matrix---------\n')
    f.write(confusion_matrix)
    # Write Evaluation Metrics
    f.write('--------Evaluation Metrics-----------\n')
    f.write(metrics)
    f.write('--------Total Model Performance-----------\n')
    f.write(total_model_performance_sum)
    f.write('\n')
    f.write(total_model_performance_mean)
    # Write Importance
    f.write('\n')
    f.write('-------Importance------------\n')
    for key, value in dict1.items():
        f.write(f'{key}: {value:.2f}\n')
    # Write Best Parameters for future reference
    f.write('----------Best Parameters-----------\n')
    f.write(str(best_params))
    f.write('\n')
    seed_str = f'seed = {seed}'
    f.write('----------Seed-----------\n')
    f.write(seed_str)
