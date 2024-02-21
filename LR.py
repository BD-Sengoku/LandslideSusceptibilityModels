import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn import metrics
import hyperopt


# Read Data
data_valid = pd.read_csv(r'C:\Users\1\Desktop\sampleSize\554ValidSet.csv', encoding='gbk')
data_train = pd.read_csv(r'C:\Users\1\Desktop\sampleSize\1290TrainSet_0.1.csv', encoding='gbk')

filename = r'C:\Users\Administrator\Desktop\LR\Result_0.1.txt'
seed = 4
max_evals = 1000
# Remove the useless columns in the CSV dataset
columns = ["target", "OBJECTID", "X", "Y"]


# Select the training and validation sets from the data
train_y = data_train.target
train_X = data_train.drop(columns=columns, axis=1)
valid_y = data_valid.target
valid_X = data_valid.drop(columns=columns, axis=1)



params_space = {
    'C': hp.loguniform('C', -5, 2), 
    'solver': hp.choice('solver', ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'])  
}


def objective(params):
    model = LogisticRegression(C = params['C'], solver = params['solver'], random_state = 42)
    model.fit(train_X, train_y)
    y_pred =  model.predict_proba(valid_X)[:, 1]
    auc = roc_auc_score(valid_y, y_pred)
    return 1 - auc  


trials = Trials()
import warnings
warnings.filterwarnings("ignore")


best = fmin(
    fn = objective, 
    space = params_space, 
    algo = tpe.suggest,
    max_evals = max_evals, 
    trials = trials,
    verbose = 0)

print("BestParameter: ", best)

# The optimal parameters after training are obtained
best_params = hyperopt.space_eval(params_space, best)
model = LogisticRegression(**best_params, random_state=42)
model.fit(train_X, train_y)
y_pred1 =  model.predict_proba(valid_X)[:, 1]   
auc1 = roc_auc_score(valid_y, y_pred1)
y_pred1 = (y_pred1 >= 0.5) * 1
a =  confusion_matrix(valid_y, y_pred1)
a = a.tolist()
a0 = str(a[0])
a1 = str(a[1])

#Substitute the optimal parameters into the model
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

# Importance of looking at features (absolute value of coefficients)
feature_importance  = np.abs(model.coef_[0])
train_X_header  = train_X.columns
my_dict = dict(zip(train_X_header, feature_importance ))
sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
# Calculate the sum of values
total = sum(sorted_dict.values())
# Calculate the percentage of each value and save it to a new dictionary
dict1 = {key: (value / total) * 100 for key, value in sorted_dict.items()}


# View the coefficient and intercept area
y_pred =  model.predict_proba(valid_X)[:, 1]
best_score = roc_auc_score(valid_y, y_pred)
print("*******best_score*******")
print(best_score)


non_scientific_array = [format(x, '.5f') for x in model.coef_[0]]
train_X_header  = train_X.columns
my_dict_lrpara = dict(zip(train_X_header, non_scientific_array))
my_dict_lrpara['截距'] = format(model.intercept_[0], '.5f')


# Open file, create automatically if file does not exist
with open(filename, 'w') as f:
    # Write Confusion Matrix
    f.write('---------Confusion Matrix---------\n')
    f.write(confusion_matrix)
    # Write Evaluation Metrics
    f.write('--------Evaluation Metrics-----------\n')
    f.write(metrics)
    f.write('--------total_model_performance-----------\n')
    f.write(total_model_performance_sum)
    f.write('\n')
    f.write(total_model_performance_mean)
    f.write('\n')
    # Write Importance
    f.write('-------importance------------\n')

    for key, value in dict1.items():
        f.write(f'{key}: {value:.2f}\n')
    # Write Best Parameters for future reference
    f.write('----------best_parm-----------\n')
    f.write(str(best_params))
    f.write('\n')
    seed_str = f'seed = {seed}'
    f.write('----------seed-----------\n')
    f.write(seed_str)
    f.write('-------coefficient and intercept area------------\n')
    for key, value in my_dict_lrpara.items():
        f.write(f'{key}: {value:}\n')
