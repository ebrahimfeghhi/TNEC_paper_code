# Defines model architectures 
# Custom layers and loss functions are defined at the end 

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, BatchNormalization, UpSampling1D, Reshape, LSTM
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

def create_model():
    data = Input(shape=(size,dims,1))

    conv_1 = Conv2D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(data)
    conv_2 = Conv2D(filters=filters[1], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(conv_1)
    average_pool_1 = MaxPooling2D(pool_size=(2,1), strides=(2,2), padding='valid')(conv_2)
    drop_1 = Dropout(drop)(average_pool_1)

    conv_3 = Conv2D(filters=filters[2], kernel_size=kernel_size[1], padding='same', activation=cnn_nonlin)(drop_1)
    conv_4 = Conv2D(filters=filters[2], kernel_size=kernel_size[1], padding='same', activation=cnn_nonlin)(conv_3)
    average_pool_2 = MaxPooling2D(pool_size=(2,1), strides=(2,2), padding='valid')(conv_4)
    drop_2 = Dropout(drop)(average_pool_2)

    conv_rel = Conv2D(filters=1, kernel_size=kernel_size[3], padding='same', activation=cnn_nonlin, name='conv_rel')(drop_2)
    conv_corr = Conv2D(filters=1, kernel_size=kernel_size[3], padding='same', activation=cnn_nonlin, name='conv_corr')(drop_2)
    conv_rms = Conv2D(filters=1, kernel_size=kernel_size[3], padding='same', activation=cnn_nonlin, name='conv_rms')(drop_2)

    drop_rel = Dropout(drop)(conv_rel)
    drop_corr = Dropout(drop)(conv_corr)
    drop_rms = Dropout(drop)(conv_rms)

    flatten_rel = Flatten()(drop_rel)
    flatten_corr = Flatten()(drop_corr)
    flatten_rms = Flatten()(drop_rms)

    rel_dense_1 = Dense(10, activation='relu')(flatten_rel)
    corr_dense_1 = Dense(10, activation='relu')(flatten_corr)
    rms_dense_1 = Dense(10, activation='relu')(flatten_rms)

    out_rel = Dense(1, activation='linear', name='rel_output')(rel_dense_1)
    out_corr = Dense(1, activation='linear', name='corr_output')(corr_dense_1)
    out_rms = Dense(1, activation='linear', name='rms_output')(rms_dense_1)

    model = Model(inputs=data, outputs=[out_rel, out_corr, out_rms])

    return model

def create_model_classify(size, dims, filters, kernel_size, cnn_nonlin, nonlin, num_classes, drop):
    data = Input(shape=(size,dims))

    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(data)
    max_pool_1 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_1)
    drop_1 = Dropout(drop)(max_pool_1)

    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_1)
    max_pool_2 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_2)
    drop_2 = Dropout(drop)(max_pool_2)
    
    flatten_output = Flatten()(drop_2)
    
    dense_1 = Dense(num_classes, activation=nonlin)(flatten_output)
    
    model = Model(inputs=data, outputs=dense_1)

    return model

def create_model_classify_3_conv(size, dims, filters, kernel_size, cnn_nonlin, nonlin, num_classes, drop):
    data = Input(shape=(size,dims))

    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(data)
    drop_1 = Dropout(drop)(conv_1)

    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_1)
    max_pool_1 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_2)
    drop_2 = Dropout(drop)(max_pool_1)
    
    conv_3 = Conv1D(filters=filters[2], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_2)
    max_pool_2 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_3)
    drop_3 = Dropout(drop)(max_pool_2)
    
    flatten_output = Flatten()(drop_3)
    
    dense_1 = Dense(num_classes, activation=nonlin)(flatten_output)
    
    model = Model(inputs=data, outputs=dense_1)

    return model 

def create_model_classify_2_dense(size, dims, filters, kernel_size, cnn_nonlin, nonlin, num_classes, drop, dense_num=10):
    data = Input(shape=(size,dims))

    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(data)
    max_pool_1 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_1)
    drop_1 = Dropout(drop)(max_pool_1)
    
    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_1)
    max_pool_2 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_2)
    drop_2 = Dropout(drop)(max_pool_2)
    
    flatten_output = Flatten()(drop_2)
    
    dense_1 = Dense(dense_num, activation=cnn_nonlin)(flatten_output)
    dense_2 = Dense(num_classes, activation=nonlin)(dense_1)
    
    model = Model(inputs=data, outputs=dense_2)

    return model 

def create_model_classify_2_dense_BN(size, dims, filters, kernel_size, cnn_nonlin, nonlin, num_classes, drop, dense_num=10):
    data = Input(shape=(size,dims))

    
    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(data)
    batch_norm_1 = BatchNormalization()(conv_1)
    max_pool_1 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_1)
    drop_1 = Dropout(drop)(max_pool_1)
    
    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    max_pool_2 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_2)
    drop_2 = Dropout(drop)(max_pool_2)
    
    flatten_output = Flatten()(drop_2)
    
    dense_1 = Dense(dense_num, activation=cnn_nonlin)(flatten_output)
    batch_norm_3 = BatchNormalization()(dense_1)
    dense_2 = Dense(num_classes, activation=nonlin)(batch_norm_3)
    
    model = Model(inputs=data, outputs=dense_2)

    return model 


def create_model_dense(size, dims, nonlin, nonlin_out, num_classes, drop):
    
    data = Input(shape=(size,dims))
    
    x = Flatten()(data)
    
    x = Dense(100, activation=nonlin)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    
    x = Dense(50, activation=nonlin)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    
    x = Dense(25, activation=nonlin)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    
    x = Dense(10, activation=nonlin)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    
    
    out = Dense(num_classes, activation=nonlin_out)(x)
    
    model = Model(inputs=data, outputs=out)

    return model 


def create_model_classify_2_dense_3_conv(size, dims, filters, kernel_size, cnn_nonlin, nonlin, num_classes, drop, dense_num=10):
    data = Input(shape=(size,dims))

    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(data)
    drop_1 = Dropout(drop)(conv_1)

    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_1)
    max_pool_1 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_2)
    drop_2 = Dropout(drop)(max_pool_1)
    
    conv_3 = Conv1D(filters=filters[2], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_2)
    max_pool_2 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(conv_3)
    drop_3 = Dropout(drop)(max_pool_2)
    
    flatten_output = Flatten()(drop_3)
    
    dense_1 = Dense(dense_num, activation=cnn_nonlin)(flatten_output)
    dense_2 = Dense(num_classes, activation=nonlin)(dense_1)
    
    model = Model(inputs=data, outputs=dense_2)

    return model 

def create_model_autoencoder(size, dims, filters, kernel_size, cnn_nonlin, drop, dense_num, up):
    data = Input(shape=(size,dims))

    # encoder 
    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(data)
    batch_norm_1 = BatchNormalization()(conv_1)
    max_pool_1 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_1) # downsamples by a factor of 2
    drop_1 = Dropout(drop)(max_pool_1)
    
    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    max_pool_2 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_2) # downsamples by a factor of 2
    drop_2 = Dropout(drop)(max_pool_2)
    
    flatten_output = Flatten()(drop_2)
    
    dense_1 = Dense(dense_num, activation=cnn_nonlin)(flatten_output) # vector of size (num_timesteps/4)*features*filters
    
    # decoder 
    reshape = tf.keras.layers.Reshape((dense_num,1))(dense_1)
    up_1 = UpSampling1D(size=up[0])(reshape)
    conv_3 = Conv1D(filters=filters[2], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(up_1)
    batch_norm_3 = BatchNormalization()(conv_3)
    
    
    up_2 = UpSampling1D(size=up[1])(batch_norm_3)
    conv_4 = Conv1D(filters=filters[3], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(up_2)
    out = Conv1D(filters=filters[4], kernel_size=kernel_size[0], padding='same', activation='sigmoid')(conv_4)
    
    model = Model(inputs=data, outputs=out)

    return model


def create_model_autoencoder_denseonly(size, nonlin, nonlin_out, dense_num, drop):
    data = Input(shape=(size,1))
    
    data_flat = Flatten()(data)

    # encoder 
    dense_0 = Dense(50, activation=nonlin)(data_flat)
    batch_norm_0 = BatchNormalization()(dense_0)  
    drop_0 = Dropout(drop)(batch_norm_0)
    
    dense_1 = Dense(25, activation=nonlin)(drop_0)
    batch_norm_1 = BatchNormalization()(dense_1)  
    drop_1 = Dropout(drop)(batch_norm_1)
    
    dense_2 = Dense(dense_num, activation=nonlin)(drop_1)
    batch_norm_2 = BatchNormalization()(dense_2)  
    
    # decoder
    dense_3 = Dense(50, activation=nonlin)(batch_norm_2)
    batch_norm_3 = BatchNormalization()(dense_3)  
    drop_3 = Dropout(drop)(batch_norm_3)
    
    dense_4 = Dense(100, activation=nonlin_out)(drop_3)
    out = Reshape((100,1))(dense_4)

    model = Model(inputs=data, outputs=out)

    return model

def create_model_autoencoder_denseonly_filt(size, nonlin, nonlin_out, dense_num, drop):
    data = Input(shape=(size,2))
    
    data_flat = Flatten()(data)

    # encoder 
    dense_0 = Dense(100, activation=nonlin)(data_flat)
    batch_norm_0 = BatchNormalization()(dense_0)  
    drop_0 = Dropout(drop)(batch_norm_0)
    
    dense_0 = Dense(50, activation=nonlin)(drop_0)
    batch_norm_0 = BatchNormalization()(dense_0)  
    drop_0 = Dropout(drop)(batch_norm_0)
    
    dense_1 = Dense(25, activation=nonlin)(drop_0)
    batch_norm_1 = BatchNormalization()(dense_1)  
    drop_1 = Dropout(drop)(batch_norm_1)
    
    dense_2 = Dense(dense_num, activation=nonlin)(drop_1)
    batch_norm_2 = BatchNormalization()(dense_2)  
    
    # decoder
    dense_3 = Dense(25, activation=nonlin)(batch_norm_2)
    batch_norm_3 = BatchNormalization()(dense_3)  
    drop_3 = Dropout(drop)(batch_norm_3)
    
    dense_4 = Dense(50, activation=nonlin)(drop_3)
    batch_norm_4 = BatchNormalization()(dense_4)  
    drop_4 = Dropout(drop)(batch_norm_4)
    
    dense_4 = Dense(100, activation=nonlin)(drop_3)
    batch_norm_4 = BatchNormalization()(dense_4)  
    drop_4 = Dropout(drop)(batch_norm_4)
    
    dense_5 = Dense(200, activation=nonlin_out)(drop_4)
    out = Reshape((100,2))(dense_5)

    model = Model(inputs=data, outputs=out)

    return model


def create_model_classify_2_dense_BN_avg(size, dims, filters, kernel_size, cnn_nonlin, nonlin, num_classes, drop, dense_num=10):
    data = Input(shape=(size,dims))

    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin, )(data)
    batch_norm_1 = BatchNormalization()(conv_1)
    max_pool_1 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_1)
    drop_1 = Dropout(drop)(max_pool_1)
    
    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[1], padding='same', activation=cnn_nonlin)(drop_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    max_pool_2 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_2)
    drop_2 = Dropout(drop)(max_pool_2)
    
    flatten_output = Flatten()(drop_2)
    
    dense_1 = Dense(dense_num, activation=cnn_nonlin)(flatten_output)
    batch_norm_3 = BatchNormalization()(dense_1)
    dense_2 = Dense(num_classes, activation=nonlin)(batch_norm_3)
    
    model = Model(inputs=data, outputs=dense_2)

    return model 


def create_model_classify_2_dense_3_cnn(size, dims, filters, kernel_size, cnn_nonlin, nonlin, num_classes, drop, dense_num=10):
    
    data = Input(shape=(size,dims))

    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(data)
    batch_norm_1 = BatchNormalization()(conv_1)
    max_pool_1 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_1)
    drop_1 = Dropout(drop)(max_pool_1)
    
    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    max_pool_2 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_2)
    drop_2 = Dropout(drop)(max_pool_2)
    
    conv_3 = Conv1D(filters=filters[2], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_2)
    batch_norm_3 = BatchNormalization()(conv_3)
    max_pool_3 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_3)
    drop_3 = Dropout(drop)(max_pool_3)
    
    flatten_output = Flatten()(drop_3)
    
    dense_1 = Dense(20, activation=cnn_nonlin)(flatten_output)
    batch_norm_4 = BatchNormalization()(dense_1)
    dense_2 = Dense(dense_num, activation=cnn_nonlin)(batch_norm_4)
    dense_3 = Dense(num_classes, activation=nonlin)(dense_2)
    
    model = Model(inputs=data, outputs=dense_3)

    return model 

def ground_truth_model(size, nonlin, nonlin_out, num_classes, dense_num=100):
    
    
    data = Input(shape=(size,))
    
    dense = Dense(dense_num, activation=nonlin)(data)
    dense_2 = Dense(10, activation=nonlin)(dense)
    dense_out = Dense(num_classes, activation=nonlin_out)(dense_2)
    
    model = Model(inputs=data, outputs=dense_out)

    return model 

def autoencoder_domain_adaptation(size, dims, filters, kernel_size, cnn_nonlin, nonlin, num_classes, drop, l2_out, dense_num=10):
    
    # feature extractor 
    data = Input(shape=(size,dims))

    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(data)
    batch_norm_1 = BatchNormalization()(conv_1)
    max_pool_1 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_1)
    drop_1 = Dropout(drop)(max_pool_1)
    
    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(drop_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    max_pool_2 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_2)
    drop_2 = Dropout(drop)(max_pool_2)
    
    flatten_output = Flatten()(drop_2)
    
    # decoder
    dense_raw_1 = Dense(size*2, activation=cnn_nonlin)(flatten_output)
    batch_norm_raw = BatchNormalization()(dense_raw_1)
    dense_raw_2 = Dense(size, activation=None, name='raw')(batch_norm_raw)
    
    # label classifier 
    dense_label_1 = Dense(dense_num, activation=cnn_nonlin, 
                        kernel_regularizer=tf.keras.regularizers.l2(l2_out))(flatten_output)
    batch_norm_label = BatchNormalization()(dense_label_1)
    dense_label_2 = Dense(num_classes, activation=nonlin, name='label')(batch_norm_label)
    
    model = Model(inputs=data, outputs=[dense_raw_2, dense_label_2])

    return model 


def domain_adaptation_model(size, dims, filters, kernel_size, cnn_nonlin, nonlin, nonlin_domain, num_classes, domain_classes, drop, flip_grad_coeff, dense_num=10):
    
    # feature extractor 
    data = Input(shape=(size,dims))

    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], padding='same', activation=cnn_nonlin)(data)
    batch_norm_1 = BatchNormalization()(conv_1)
    max_pool_1 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_1)
    drop_1 = Dropout(drop)(max_pool_1)
    
    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[1], padding='same', activation=cnn_nonlin)(drop_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    max_pool_2 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_2)
    drop_2 = Dropout(drop)(max_pool_2)
    
    flatten_output = Flatten()(drop_2)
    
    # domain classifier 
    flip_gradient_out = FlipGradient(flip_grad_coeff)(flatten_output)
    dense_domain_1 = Dense(dense_num, activation=cnn_nonlin)(flip_gradient_out)
    batch_norm_domain = BatchNormalization()(dense_domain_1)
    dense_domain_2 = Dense(domain_classes, activation=nonlin_domain, name='domain')(batch_norm_domain)
    
    # label classifier 
    dense_label_1 = Dense(dense_num, activation=cnn_nonlin, 
                        kernel_regularizer=tf.keras.regularizers.l2(0))(flatten_output)
    batch_norm_label = BatchNormalization()(dense_label_1)
    dense_label_2 = Dense(num_classes, activation=nonlin, name='label')(batch_norm_label)
    
    model = Model(inputs=data, outputs=[dense_domain_2, dense_label_2])

    return model 



# ------------------------------ Custom layers -------------------------------------------

# negative of gradient is passed during backpropagation when using this layer 
# x*(l+1) - (x)*l = x
# no gradient on first product
def flip_gradient(x, l=1.0):
	positive_path = tf.stop_gradient(x * tf.cast(1 + l, tf.float32)) 
	negative_path = -x * tf.cast(l, tf.float32)
	return positive_path + negative_path

class FlipGradient(tf.keras.layers.Layer):
    def __init__(self, l=1.0):
        super(FlipGradient, self).__init__()
        self.l = l

    def call(self, x):
        return flip_gradient(x, self.l)
    
    def get_config(self):
        return {"l":self.l}
    
# ------------------------------ Custom loss functions ---------------------------------------
# zeros out predictions corresponding to missing (-1) labels so that they are not used in G.D updates
def binary_crossentropy_mask(y_true, y_pred):
    return tf.reduce_mean(K.binary_crossentropy(tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), 
    tf.float32)), tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))),axis=-1) 

def mean_absolute_error_mask(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))-
    tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32))),axis=-1)

def binary_acc(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)
    
# ------------------------------ Custom callbacks ------------------------------------------

class Custom_ES(tf.keras.callbacks.Callback):
    
    def __init__(self, patience, thresh_acc):
        
        super(Custom_ES, self).__init__()
        self.patience = patience
        self.thresh_acc = thresh_acc
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_label_acc = 0
        self.saved_label_acc = 0
        self.exceeded_thresh = False
        self.max_domain_loss = 0 
        self.saved_domain_acc = 0
            
    def on_epoch_end(self, epoch, logs=None): 
        
        label_acc = logs.get ('val_label_binary_acc')
            
        # once threshold label accuracy has been exceeded
        if label_acc > self.thresh_acc:
            self.exceeded_thresh = True
            domain_loss = logs.get('val_domain_loss')
            if domain_loss > self.max_domain_loss:
                self.max_domain_loss = domain_loss
                self.saved_domain_acc = logs.get('val_domain_binary_acc')
                self.saved_label_acc = label_acc
                self.wait = 0
                self.best_weights = self.model.get_weights()

            else:
                self.wait += 1

        # a best model has been saved, but accuracy is not above threshold now
        elif self.exceeded_thresh:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self.best_weights)
            
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
            print("Saved label acc: ", self.saved_label_acc)
            
