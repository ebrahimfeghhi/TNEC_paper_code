""" 
Script options:

Model architecutres: 

# ConvLSTM
# LSTM
# CNN

Data Type: f

# Raw and filtered (80-110 Hz)
# Filtered (80-110 Hz)
# Raw 

Classification mode: 

# Events (ripples and transients) and non events
# Ripples and transients
# Ripples, transients, and non events 

Mode:

# K fold cross validation
# Train and test 

Written by Ebrahim Feghhi, UCLA Neural Computation and Engineering Lab, efeghhi@gmail.com
"""

# packages
import matplotlib
matplotlib.use('Agg')
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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


def run_model(num): 

    base_path = '/home3/ebrahim/ripple_code/yasa_labels/weighted_labels/' 
    
    # --------------------------- load data ----------------------------------
    model_mode = 'cnn'

   # select detector and epoch 
    detector_type = 'C'
    identifier = '489'

    # Select folder name 
    fname = 'global'
    
    if fname == 'global':
        run_number = detector_type + '_' + fname + '_3_' + identifier + '_' + str(num)
        X = np.load(base_path + fname + '/X_train_all_3_' + identifier + '_' + detector_type + '.npy')
        y = np.load(base_path + fname + '/y_train_all_3_' + identifier + '_' + detector_type + '.npy')
        
    else:
        run_number = detector_type + '_' + epoch + '_' + fname[0:3] + '_kfold'
        X = np.load(base_path + fname + '/X_' + epoch + '_' + detector_type + '.npy')
        y = np.load(base_path + fname + '/y_' + epoch + '_' + detector_type + '.npy')

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    # ------------------------------------------------------------------------------------

    # -----------Hyperparameters------------
    learning_rate = 1e-4
    batch_size = int(X.shape[0]/2)
    epochs = 2000
    drop = .1
    patience = 600
    random_state = 42
    np.random.seed(random_state)
    class_weights = None
    detrend_bool = True
    dataset = 'HFO + distractor'
    num_classes = 2
    kfold = False
    testing = True
    modeltype = 0
    num_epochs_ran=epochs
    load_model = False
    load_model_path = '/home3/ebrahim/results/test_detector/cnn/A_q1_487_val/best_model.h5'
    dense_num = 10
    optim = 'adam'
    eps = 1e-7
    domain_adaptation=False
    flip_grad_coeff = 1.0

    if num_classes > 2:
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
        
    # -----------Train model-----------
    
    adam = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=eps, amsgrad=False)
    
    
    if (load_model):
        model = tf.keras.models.load_model(load_model_path)
        
    elif modeltype==0:
        filters = [48, 96]
        dims = X.shape[-1]
        kernel_size = [20,10]
        loss = loss_func
        model = create_model_classify_2_dense_BN_avg(size, dims, filters, kernel_size, nonlin, nonlin_out, num_classes, drop, dense_num)
        model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        y_cat = to_categorical(y)
        history = model.fit(X, y_cat, batch_size=batch_size, epochs=epochs, shuffle=True, class_weight=class_weights)
        plot_val = 'acc'   
        
    elif modeltype==2:
        kernel_size = [20, 10]
        filters = [48,96]
        dims = X.shape[2]
        domain_classes=1
        nonlin_domain='sigmoid'
        model = domain_adaptation_model(size, dims, filters, kernel_size, nonlin, nonlin_out, nonlin_domain, num_classes, domain_classes, drop, flip_grad_coeff, dense_num)
        loss = binary_crossentropy_mask
        model.compile(optimizer=adam, loss=loss, metrics=[binary_acc])
        history = model.fit(X, [y_domain,y], batch_size=batch_size, epochs=epochs, shuffle=True, class_weight=class_weights)
        plot_val = 'label_binary_acc'
        plot_val2 = 'domain_binary_acc'
    else:
        model = ground_truth_model(size, nonlin, nonlin_out, num_classes, dense_num)
        
    
    plot_title = 'model acc'
    plot_yaxis = 'acc'
    
    model.save(results_storage_path + '/best_model.h5')

    plt.plot(history.history[plot_val], label=plot_val)
    plt.title(plot_title)
    plt.ylabel(plot_yaxis)
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(results_storage_path + "/" + "Val--Acc.png")
    plt.close()

    # -----------store hyperparm information-----------
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
        f.write("dense_num: " + str(dense_num))
        f.write("\n")
        f.write("eps: " + str(eps))
        f.write("\n")
        f.write("size: " + str(size))
        f.write("\n")
        f.write("nonlin: " + nonlin)
        f.write("\n")
        f.write("nonlin_out: " + nonlin_out)
        f.write("\n")
        f.write("identifier: " + identifier)
        f.write("\n")
        
        if modeltype==0:
            f.write("filters: " + str(filters))
            f.write("\n")
            f.write("kernel_size: " + str(kernel_size))
            f.write("\n")
            
        
        
        
for i in range(1,11):
    run_model(i)
    
