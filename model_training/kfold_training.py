""" 
# K fold cross validation script

Written by Ebrahim Feghhi, UCLA Neural Computation and Engineering Lab, efeghhi@gmail.com
"""

# packages
import matplotlib
matplotlib.use('Agg')
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.activations import relu
from tensorflow.keras import Input
from tensorflow.keras import Model
from numpy import save
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import detrend
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from numpy.random import default_rng

# modules
from data_prep_model import * 
from results_storage import * 
from metrics import * 
from model_defined import *

      
def detrend_func(X):
    
    X_raw_detrend = detrend(X[:, :, 0])
    X[:, :, 0] = X_raw_detrend 
    return X


# MODIFY BASE PATH TO POINT TO DIRECTORY WHERE DATA IS STORED
base_path = '/home3/ebrahim/ripple_code/yasa_labels/weighted_labels/' 

# --------------------------- load data ----------------------------------
model_mode = 'cnn'

# select detector and epoch 
detector_type = 'C'
identifier = '489'
epoch = 'q1'

# Select folder name 
fname = 'global'
if fname == 'global':
    run_number = detector_type + '_' + fname + '_3_' + identifier + '_kfold'
    X = np.load(base_path + fname + '/X_train_all_3_' + identifier + '_' + detector_type + '.npy')
    y = np.load(base_path + fname + '/y_train_all_3_' + identifier + '_' + detector_type + '.npy')
else:
    run_number = detector_type + '_' + epoch + '_' + fname[0:3] + '_500_kfold_pt_lr_5e-6'
    X = np.load(base_path + fname + '/X_' + epoch + '_' + detector_type + '_500.npy')
    y = np.load(base_path + fname + '/y_' + epoch + '_' + detector_type + '_500.npy')

print("X shape: ", X.shape)
print("y shape: ", y.shape)
# ------------------------------------------------------------------------------------

# -----------Hyperparameters------------
learning_rate = 5e-6
batch_size = X.shape[0]
epochs = 30000
drop = .15
l2_out = 0
patience = 1000
random_state = 42
np.random.seed(random_state)
class_weights = None
detrend_bool = True
dataset = 'HFO + distractor'
num_classes = 2
kfold = True
testing = False
modeltype = 0 # 0 for CNN, 1 for groundtruth dense, 2 for domain adaptation
model_desc = 'paper_model'
num_epochs_ran=epochs
load_model = True
load_model_path = '/home3/ebrahim/results/test_detector/cnn/final_models/A_q1_509_3_1/best_model.h5'
dense_num = 10
optim = 'adam'
eps = 1e-7
flip_grad_coeff = 1.0
    
# note, models in the paper were ran with this setting, meaning there were 2 output units but 
# sigmoid and bce was applied. This was a minor mistake, and
# future models should either use 1 output unit with sigmoid and bce or 2 output units and softmax and
# cce 
if num_classes >= 2:
    nonlin_out = 'softmax'
    loss_func = 'categorical_crossentropy'
else:
    nonlin_out = 'sigmoid'
    loss_func = 'binary_crossentropy'
    
# -----------Define Model-----------        
size  = X.shape[1]
nonlin = 'selu'
# -----------Results Storage-----------
results_storage_path = results_storage(run_number)

if detrend_bool:
    X = detrend_func(X)

# define optimizer 
adam = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=eps, amsgrad=False)

splits=5

kf = StratifiedKFold(n_splits=splits, random_state=random_state, shuffle=True)

val_loss = np.zeros(splits)
num_epochs_list = np.zeros(splits)
accuracy_list = np.zeros(splits)

for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        
    X_train_k, X_val_k = X[train_index], X[val_index]
    y_train_k_int, y_val_k_int = y[train_index], y[val_index]


    y_train_k = to_categorical(y_train_k_int)
    y_val_k = to_categorical(y_val_k_int)
       
   
    if modeltype==0: 
        kernel_size = [20, 10]
        filters = [48,96]
        dims = X.shape[2]
        if (load_model):
            print("LOADING MODEL")
            model_k = tf.keras.models.load_model(load_model_path)
        else:
            print("STARTING FRESH")
            model_k = create_model_classify_2_dense_BN_avg(size, dims, filters, kernel_size, 
                          nonlin, nonlin_out, num_classes, drop, dense_num)
        loss = loss_func
        monitor_val = 'val_loss'
        save_val = 'val_acc'
        es = EarlyStopping(monitor=monitor_val, mode='min', verbose=1, patience=patience)
        mc = ModelCheckpoint(results_storage_path + '/best_model.h5', monitor=monitor_val, mode='min', verbose=1, save_best_only=True) 
        model_k.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        print("Model Parameters: ", model_k.count_params())
        history = model_k.fit(X_train_k, y_train_k, batch_size=batch_size, epochs=epochs, validation_data=(X_val_k, y_val_k), shuffle=False, callbacks=[es, mc], class_weight=class_weights)
        val_loss[i] = np.min(history.history[monitor_val])
        accuracy_list[i] = np.max(history.history[save_val])

    elif modeltype==2:
        num_classes = 1
        domain_classes=1
        max_acc = 1-(y_val_k_int[y_val_k_int==-1].shape[0]/y_val_k_int.shape[0])
        thresh_acc = .9*max_acc
        print("THRESH ACC: ", thresh_acc)
        y_train_d_k_int, y_val_d_k_int = y_domain[train_index], y_domain[val_index]
        kernel_size = [20, 10]
        filters = [48,96]
        dims = X.shape[2]
        nonlin_domain='sigmoid'
        val_plot_1 = 'val_domain_binary_acc'
        val_plot_2 = 'val_label_binary_acc'
        loss = binary_crossentropy_mask
        model_k = domain_adaptation_model(size, dims, filters, kernel_size, nonlin, nonlin_out, nonlin_domain, num_classes, domain_classes, drop, flip_grad_coeff, dense_num)
        model_k.compile(optimizer=adam, loss=loss, metrics=[binary_acc])
        history = model_k.fit(X_train_k, [y_train_d_k_int, y_train_k_int], batch_size=batch_size, epochs=epochs, validation_data=(X_val_k, [y_val_d_k_int, y_val_k_int]), shuffle=False, callbacks=[Custom_ES(patience, thresh_acc)], class_weight=class_weights)
        val_loss[i] = np.min(history.history['val_loss'])
        accuracy_list[i] = np.max(history.history['val_label_binary_acc'])/max_acc
        np.save(results_storage_path + "/" + 'num_epochs', num_epochs_list)   
        plt.plot(history.history[val_plot_1])
        plt.plot(history.history[val_plot_2])
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend([val_plot_1, val_plot_2], loc='upper left')
        plt.savefig(results_storage_path + "/" + "Train--Val--Acc_" + str(i) + ".png")
        plt.close()
        
    else:
        model_k = ground_truth_model(size, nonlin, nonlin_out, num_classes, dense_num)
        
 

    # if maximum epochs are reached, store normally, otherwise subtract patient value
    if len(history.history['loss']) == epochs:
        num_epochs_list[i] = epochs
    else:
        num_epochs_list[i] = len(history.history['loss']) - patience
    
    
np.save(results_storage_path + "/" + 'val_loss', val_loss)
np.save(results_storage_path + "/" + 'num_epochs', num_epochs_list)
np.save(results_storage_path + "/" + 'accuracy_list', accuracy_list)
        
        
# -----------store hyperparam information-----------
hyperparams_file = (results_storage_path + "/hyper_params.txt")
with open(hyperparams_file, 'w') as f:
    
    f.write("epochs: " + str(epochs))
    f.write("\n")
    f.write("patience: " + str(patience))
    f.write("\n")
    f.write("dropout: " + str(drop))
    f.write("\n")
    f.write("learning rate: " + str(learning_rate))
    f.write("\n")
    f.write("batch size: " + str(batch_size))
    f.write("\n")
    f.write("optimizer: " + optim)
    f.write("\n")
    f.write("dataset: " + dataset)
    f.write("\n")  
    f.write("load_model: " + str(load_model))
    f.write("\n")  
    f.write("detrend: " + str(detrend_bool))
    f.write("\n") 
    f.write("normalize: " + str(normalize))
    f.write("\n") 
    f.write("kfold: " + str(kfold))
    f.write("\n") 
    f.write("class_weights: " + str(class_weights))
    f.write("\n") 
    f.write("testing: " + str(testing))
    f.write("\n") 
    f.write("num_epochs_ran: " + str(num_epochs_ran))
    f.write("\n") 
    f.write("num classes: " + str(num_classes))
    f.write("\n")
    f.write("patient name: " + fname)
    f.write("\n")
    f.write("dense_num: " + str(dense_num))
    f.write("\n")
    f.write("eps: " + str(eps))
    f.write("\n")

    f.write("modeltype: " + str(modeltype))
    f.write("\n") 
    f.write("model_desc: " + model_desc)
    f.write("\n") 
    f.write("nonlin: " + nonlin)
    f.write("\n")
    f.write("nonlin_out: " + nonlin_out)
    f.write("\n")
    f.write("filters: " + str(filters))
    f.write("\n")
    f.write("kernel_size: " + str(kernel_size))
    f.write("\n")
    f.write("size: " + str(size))
    f.write("\n")
