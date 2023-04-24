''' Basic CNN model with hardcoded parameters
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from ..utils.utils import _image_to_numpy

def preprocess():
    print('I love Marieke')

    _image_to_numpy()

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
