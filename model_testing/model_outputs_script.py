import matplotlib
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#session = tf.Session(config=config)
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
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import detrend
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import confusion_matrix
import pandas as pd


# True for global patient experiments, else False 
global_bool = True


def detrend_func(X):
    
    X_raw = X[:, :, 0]
    X_raw_detrend = detrend(X_raw)
    X[:, :, 0] = X_raw_detrend
    return X


def load_models(model_name, model_path, n):
    
    model_arr=[]
    
    # load models and store models in model_arr
    for i in range(1,n+1):
        print(i)
        load_model_path = model_path + model_name + str(i)
        model_best = tf.keras.models.load_model(load_model_path + "/best_model.h5")
        model_arr.append(model_best)
        
    return model_arr


def load_models_total(patients, train_q, model_name, model_path, base, ltype, quarters, save_folder, n):
    
    '''
    patients: patient data used for training
    base: directory where patient data is stored
    save_folder: folder in which to save the sensitivity values 
    '''
    
    print("Model name: ", model_name)
    print("Loading N models...")
    model_arr = load_models(model_name, model_path, n)
    
    # Note: the terms quarters and epochs are used interchangeably
    # init. arrays to store results
    sens_mat_A = np.zeros((4, n))
    sens_mat_B = np.zeros((4, n))
    sens_mat_C = np.zeros((4, n))
    
    p = patients[0]
    
    # loop through each epoch x ltype
    for i, q in enumerate(quarters):
        # skip epochs overlapping with training epoch
        if q == train_q:
            continue
        for m, l in enumerate(ltype):
                
            print("Testing epoch")
            print(q, l)

            # load data 
            X = detrend_func(np.load(base + p + 'X_' + q + '_' + l + '.npy'))
            y = np.load(base + p + 'y_' + q + '_' + l + '.npy')

            # loop through models and compute fraction of HFOs correctly classified 
            for j in range(0,n):

                y_pred = model_arr[j].predict(X)
                y_pred_int = np.argmax(y_pred, axis=-1) 
                y_pred_int_HFOs = y_pred_int[y==1] # select HFOs
                frac_consistent_total = np.argwhere(y_pred_int_HFOs==1).shape[0]/y_pred_int_HFOs.shape[0] 

                if l == 'A':
                    sens_mat_A[i,j] = frac_consistent_total

                if l == 'B':
                    sens_mat_B[i,j] = frac_consistent_total

                if l == 'C':
                    sens_mat_C[i,j] = frac_consistent_total
                    
    
    np.save(save_folder + 'model_' + model_name + 'sens_A', sens_mat_A)
    np.save(save_folder + 'model_' + model_name + 'sens_B', sens_mat_B)
    np.save(save_folder + 'model_' + model_name + 'sens_C', sens_mat_C)
    tf.keras.backend.clear_session()
    
    
    
def load_models_unique_overlap(train_q, train_l, model_name, model_path, pname, base, ltype, quarters, save_folder, n, save_pred_mode):
    
    print("Loading N models...")
    print(model_name)
    model_arr = load_models(model_name, model_path, n)
    unique_pred_arr = {}
    overlap_pred_arr = {}
    unique_arr = np.zeros((len(ltype), len(quarters), n))
    overlap_arr = np.zeros((len(ltype), len(quarters), n))
    
    
    for i, q in enumerate(quarters):
        
        if q == train_q:
            continue
            
        for m, l in enumerate(ltype):
            
            if l == train_l:
                continue

            print(q,l)
            
            unique_exists = False
            
            # hfos which are unique (inconsistent) to the hfos the classifer was trained on
            unique = np.load(base + pname + 'hfo_' + q + '_' + l + '_' + train_l + '_unique.npy')
            
            if unique.shape[0] != 0:
                unique = np.squeeze(detrend_func(unique))
                unique_exists = True 
           
            # hfos which are overlapping (consistent) with the hfos the classifier was trained on 
            try:               
                overlap = detrend_func(np.squeeze(np.load(base + pname + 'hfo_' + q + '_' + train_l + '_' + l + '_overlap.npy')))
            except:
                overlap = detrend_func(np.squeeze(np.load(base + pname + 'hfo_' + q + '_' + l + '_' + train_l + '_overlap.npy')))
                
            
            
            for j in range(0,n):
                
                
                if unique_exists:
                    unique_pred = model_arr[j].predict(unique)
                    if j == 0:
                        unique_pred_arr[q + '_' + l] = unique_pred
                    else:
                        unique_pred_arr[q + '_' + l] += unique_pred
                        
                    unique_int = np.argmax(unique_pred, axis=-1)
                    unique_correct = np.count_nonzero(unique_int) / unique_int.shape[0]
                    
                else:
                    unique_correct = np.nan
                    unqiue_pred = np.nan
      
                overlap_pred = model_arr[j].predict(overlap)
        
                if j == 0:
                    overlap_pred_arr[q + '_' + l] = overlap_pred
                else:
                    overlap_pred_arr[q + '_' + l] += overlap_pred
                        
                overlap_int = np.argmax(overlap_pred, axis=-1)
                overlap_correct = np.count_nonzero(overlap_int) / overlap_int.shape[0]
                
                unique_arr[m, i, j] = unique_correct
                overlap_arr[m, i, j] = overlap_correct
                
            unique_pred_arr[q + '_' + l] /= n
            overlap_pred_arr[q + '_' + l] /= n
                
    if save_pred_mode:
        #np.save(save_folder + 'model_' + model_name + 'unique_preds', unique_pred_arr)
        np.save(save_folder + 'model_' + model_name + 'overlap_preds', overlap_pred_arr)
        
    else:
        #np.save(save_folder + 'model_' + model_name + 'unique', unique_arr)
        np.save(save_folder + 'model_' + model_name + 'overlap', overlap_arr)
    tf.keras.backend.clear_session()
    
                
                
if global_bool == False:
    ltype = ['A', 'B', 'C']
    patients = ['509_d/']
    quarters=['q1', 'q2','q3', 'q4']
    model_path = '/home3/ebrahim/results/test_detector/cnn/final_models/'
    base = '/home3/ebrahim/ripple_code/yasa_labels/weighted_labels/'
    save_folder = '/home3/ebrahim/results/updated_results/unique_overlap_preds/'
    pname = patients[0][0:3]
    n=10
    save_pred_mode = True
    for l in ltype:
        for q in quarters:
            model_name = l + '_' + q + '_' + pname + '_3_'
            #load_models_total(patients, q, model_name, model_path, base, ltype, quarters, save_folder, n=10)
            load_models_unique_overlap(q, l, model_name, model_path, patients[0], base, ltype, quarters, save_folder, n, save_pred_mode)
        
        
def load_models_unique_overlap_global(train_l, model_name, model_path, pname, base, ltype, save_folder, n):
    
    #print("Loading N models...")
    #print(model_name)
    model_arr = load_models(model_name, model_path, n)
    unique_pred_arr = {}
    overlap_pred_arr = {}
    unique_arr = np.zeros((len(ltype), n))
    overlap_arr = np.zeros((len(ltype), n))
    
    for m, l in enumerate(ltype):
        

        if l == train_l:
            continue
        
        print("L TEST: ", l)

        unique_exists = False

        # hfos which are unique (inconsistent) to the hfos the classifer was trained on
        unique = np.load(base + 'hfo_' + pname + '_' + l + '_' + train_l + '_unique.npy')
        

        if unique.shape[0] != 0:
            unique = np.squeeze(detrend_func(unique))
            unique_exists = True 

        # hfos which are overlapping (consistent) with the hfos the classifier was trained on 
        try:               
            overlap = detrend_func(np.squeeze(np.load(base + 'hfo_' + pname + '_' + l + '_' + train_l + '_overlap.npy')))
        except:
            overlap = detrend_func(np.squeeze(np.load(base + 'hfo_' + pname + '_' + train_l + '_' + l + '_overlap.npy')))
        
        print("# of unique HFOS: ", unique.shape[0])
        print("# of overlapping HFOs: ", overlap.shape[0])
        print("# of total HFOs: ", unique.shape[0] + overlap.shape[0])

        for j in range(0,n):
            if unique_exists:
                
                unique_pred = model_arr[j].predict(unique)
                
                if j == 0:
                    unique_pred_arr[l] = unique_pred
                else:
                    unique_pred_arr[l] += unique_pred
                    
                unique_int = np.argmax(unique_pred, axis=-1)
                unique_correct = np.count_nonzero(unique_int) / unique_int.shape[0]

            else:
                unique_correct = np.nan
                unqiue_pred = np.nan
                
            overlap_pred = model_arr[j].predict(overlap)
            
            if j == 0:
                overlap_pred_arr[l] = overlap_pred
            else:
                overlap_pred_arr[l] += overlap_pred

            overlap_int = np.argmax(overlap_pred, axis=-1)
            overlap_correct = np.count_nonzero(overlap_int) / overlap_int.shape[0]

            unique_arr[m, j] = unique_correct
            overlap_arr[m, j] = overlap_correct

        unique_pred_arr[l] /= n
        overlap_pred_arr[l] /= n
        
                
    np.save(save_folder + 'model_' + model_name + 'unique_preds', unique_pred_arr)
    np.save(save_folder + 'model_' + model_name + 'unique', unique_arr)
    np.save(save_folder + 'model_' + model_name + 'overlap', overlap_arr)
    np.save(save_folder + 'model_' + model_name + 'overlap_preds', overlap_pred_arr)
    
    tf.keras.backend.clear_session()

if global_bool:
    
    model_path = '/home3/ebrahim/results/test_detector/cnn/final_models/'
    base = '/home3/ebrahim/ripple_code/yasa_labels/weighted_labels/global/'
    save_folder = '/home3/ebrahim/results/updated_results/unique_overlap_global/'
    pname = ['509', '493', '489', '487']
    n = 10
    ltype=['A', 'B', 'C']

    for p in pname: 
        
        for train_l in ltype:
        
            model_name = train_l + '_global_3_' + p + '_'
        
            load_models_unique_overlap_global(train_l, model_name, model_path, p, base, ltype, save_folder, n)
        
    
    

