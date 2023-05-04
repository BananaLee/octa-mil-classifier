''' Basic CNN model with hardcoded parameters
'''
import numpy as np
import tensorflow as tf
import os
import pathlib
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

import octa_utilities as util
from octa_utilities import process_path

def preprocess(datapath, channels=None, mode="eval", val_prop=None):
    
    channels = 1 if channels is None else channels
    val_prop = 0.2 if val_prop is None else val_prop
    full_datapath = pathlib.Path(datapath)

    AUTOTUNE = tf.data.AUTOTUNE
    ## NEED TO TRANSFER CLASS_NAMES AND EVTL IMAGE_SIZE TO UTILS SOMEHOW
    CLASS_NAMES = ['diabetic', 'healthy']#np.array([item.name for item in full_datapath.glob('*')])

    list_ds = tf.data.Dataset.list_files(str(full_datapath/'*/*'))
    
    if mode == "train":
        #split into train/val, return two dses
        ## CONFIRM THAT MIL SIMPLY INVOLVES CHOIPPING IMAGE INTO 10x10?? YES.
        ## CHECK THAT IMAGES ARE STANDARDISED
        pass
    else: 
        labelled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
        return labelled_ds, None


def train_model():
    model = model_architecture()

def make_predictions():
    # if class is just initiated, it will load a model, otheriwse
    pass

def model_architecture():
    input_shape = (image_size, image_size, 1) # need way of feeding image size
    num_labels = 3 # need way of number of labels
    batch_size = 128
    kernel_size = 3
    filters = 64
    dropout = 0.3

    inputs = Input(shape=input_shape)
    
    y = Conv2D(filters=filters,
     kernel_size=kernel_size,
     activation='relu')(inputs)
    y = MaxPooling2D()(y)
    
    y = Conv2D(filters=filters,
     kernel_size=kernel_size,
     activation='relu')(y)
    y = MaxPooling2D()(y)
    
    y = Conv2D(filters=filters,
     kernel_size=kernel_size,
     activation='relu')(y)
    
    # convert image to vector 
    y = Flatten()(y)
    
    # dropout regularization
    y = Dropout(dropout)(y)
    
    outputs = Dense(num_labels, activation='softmax')(y)
     # model building by supplying inputs/outputs
    
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss = BinaryCrossentropy(from_logits=True), 
        optimizer = Adam(learning_rate=1e-4),
        metrics = ['accuracy'])

    model.summary()
    
    return model
