import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import hyperopt
from sklearn import metrics



# Read Data
data_valid = pd.read_csv(r'C:\Users\1\Desktop\sampleSize\554ValidSet.csv', encoding='gbk')
data_train = pd.read_csv(r'C:\Users\1\Desktop\sampleSize\1290TrainSet_0.1.csv', encoding='gbk')

filename = r'C:\Users\1\Desktop\sampleSize\RF\Result_0.1.txt'
seed = 4
max_evals = 1000
# Remove the useless columns in the CSV dataset
columns = ["target", "OBJECTID", "X", "Y"]


# Select the training and validation sets from the data
train_y = data_train.target
train_X = data_train.drop(columns=columns, axis=1)
valid_y = data_valid.target
valid_X = data_valid.drop(columns=columns, axis=1)



def objective(params):
    """
    Objective function for hyperparameter optimization.

    Parameters:
    - params (dict): Dictionary containing hyperparameters to optimize.

    Returns:
    - float: Value of the objective function to minimize.
    """
    # Extract hyperparameters from the params dictionary
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    min_samples_split = int(params['min_samples_split'])
    min_samples_leaf = int(params['min_samples_leaf'])

    # Create a Random Forest classifier with the specified hyperparameters
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)

    # Train the classifier on the training data
    rf.fit(train_X, train_y)
    # Compute the ROC AUC score on the validation data
    auc = roc_auc_score(valid_y, rf.predict_proba(valid_X)[:, 1])
    # Return 1 - AUC because Hyperopt aims to minimize the objective function
    return 1 - auc


# Define the search space for hyperparameters
params_space = {
    'n_estimators': hp.choice('n_estimators', range(100,1000,50)),
    'max_depth': hp.choice('max_depth', range(1,20)),
    'min_samples_split': hp.choice('min_samples_split', range(2,10)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1,10))
}

import warnings
warnings.filterwarnings("ignore")
# 定义超参数优化器
trials = Trials()
best = fmin(fn=objective,
            space=params_space,
            algo=tpe.suggest,
            max_evals = max_evals,
            trials=trials)
print("BestParameter: ", best)


# The optimal parameters after training are obtained
best_params = hyperopt.space_eval(params_space, best)
best_params['random_state'] = 42

#Substitute the optimal parameters into the model
rf = RandomForestClassifier(**best_params)
rf.fit(train_X, train_y)
y_pred1 = rf.predict_proba(valid_X)[:, 1]
auc1 = roc_auc_score(valid_y, y_pred1)
y_pred1 = (y_pred1 >= 0.5) * 1
a =  confusion_matrix(valid_y, y_pred1)
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
my_dict = dict(zip(train_X.columns, rf.feature_importances_ ))
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
