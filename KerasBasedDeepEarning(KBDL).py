import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tensorflow import keras
from hyperopt import fmin, tpe, hp, Trials
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import hyperopt
from sklearn import metrics

# Read Data
data_valid = pd.read_csv(r'C:\Users\1\Desktop\sampleSize\554ValidSet.csv', encoding='gbk')
data_train = pd.read_csv(r'C:\Users\1\Desktop\sampleSize\1290TrainSet_0.1.csv', encoding='gbk')

# File paths for saving results and models
filename = r'C:\Users\1\Desktop\sampleSize\XG\Result_0.1.txt'
filename_SaveModel = r'C:\Users\1\Desktop\Keras\new0.5.h5' #Keras can't reproduce the model by saving parameters, so the model with the largest AUC is saved here for reproduction

seed = 4
max_evals = 100
# Remove the useless columns in the CSV dataset
columns = ["target", "OBJECTID", "X", "Y"]
global_auc = 0  #Since the model cannot be saved, an if statement is added to the function, if AUC is greater than the previous one, the model is saved (overwritten) and the AUC is changed to the greater

# Select the training and validation sets from the data
train_y = data_train.target
train_X = data_train.drop(columns=columns, axis=1)
valid_y = data_valid.target
valid_X = data_valid.drop(columns=columns, axis=1)

valid_y_array = valid_y.values 

# Parameter Space
space = {
    'units': hp.choice('units', [16, 32, 64, 128]),
    'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh']),
    'dropout_rate': hp.uniform('dropout_rate', 0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'batch_size': hp.choice('batch_size', [16, 32, 64]),
    'epochs': hp.choice('epochs', [50, 100, 200])
}


# Objective function for hyperparameter optimization
def objective(params):
    model = keras.Sequential([
        keras.layers.Dense(params['units'],
                           activation=params['activation'],
                           input_shape=(train_X.shape[1],)
                           ),
        keras.layers.Dropout(params['dropout_rate']),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_X, train_y,
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              verbose=0,
              validation_data=(valid_X, valid_y))
    y_pred = model.predict(valid_X)[:, 0]
    global global_auc
    if 1 - roc_auc_score(valid_y_array, y_pred) < 1 - global_auc:
        global_auc = roc_auc_score(valid_y_array, y_pred)
        model.save(filename_savlModel)
    return 1 - roc_auc_score(valid_y_array, y_pred)



trials = Trials()  
# Hyperparameter optimization
best = fmin(
    fn=objective, 
    space=space,  
    algo=tpe.suggest,
    max_evals=max_evals,  
    trials=trials, 
    rstate=np.random.RandomState(42) 
)


print("BestParameter: ", best)
# The optimal parameters after training are obtained
best_params = hyperopt.space_eval(space, best)


# Load the best model and evaluate its performance
from tensorflow import keras
model = keras.models.load_model(filename_savlModel)
y_pred1 = model.predict(valid_X)[:, 0]
auc1 = roc_auc_score(valid_y_array, y_pred1)
y_pred1 = (y_pred1 >= 0.5) * 1
a =  confusion_matrix(valid_y_array, y_pred1)
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




import plotly.express as px

feature_df = pd.DataFrame(columns=['feature', 'layer', 'neuron', 'weight', 'abs_weight'])

for i, layer in enumerate(model.layers[:-1]):
    w = layer.get_weights()
    if len(w) > 0:  
        w = np.array(w[0])
        n = 0
        for neuron in w.T:
            for f, name in zip(neuron, train_X.columns):
                feature_df.loc[len(feature_df)] = [name, i, n, f, abs(f)]
            n += 1

feature_df = feature_df.sort_values(by=['abs_weight'])
feature_df.reset_index(inplace=True)
feature_df = feature_df.drop(['index'], axis=1)


feature_df = feature_df.sort_values(by=['abs_weight'], ascending=False)
feature_df['abs_weight_decimal'] = feature_df['abs_weight'] / feature_df['abs_weight'].sum()

# Feature Importance Handling Area
my_dict = dict(zip(feature_df['feature'], feature_df['abs_weight_decimal']))
sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
# Calculate the sum of values
total = sum(sorted_dict.values())
# Calculate the percentage of each value and save it to a new dictionary
dict1 = {key: (value / total) * 100 for key, value in sorted_dict.items()}



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
