''' Basic CNN model with hardcoded parameters
'''
import numpy as np
import tensorflow as tf
import os
import pathlib

import octa_utilities as util
from octa_utilities import process_path, configure_for_performance

def preprocess(params):
    """
    Takes params from the config file to look into a folder with image data
    and returns a batched up Tensorflow Dataset
    """
    
    if params['mode'] == 'train':
        datapath = params['train_path']
    elif params['mode'] == 'eval':
        datapath = params['test_path']

    full_datapath = pathlib.Path(datapath)

    channels = 1 if params.get("channels") is None else params.get("channels")
    val_prop = 0.2 if params.get("val_prop") is None \
      else params.get("val_prop")
    CLASS_NAMES = ['diabetic', 'healthy'] if params.get("class_names")\
      is None else params.get("class_names")
    width = 2048 if params.get("width") is None else params.get("width")
    height = 2044 if params.get("height") is None else params.get("height")
    batch_size = 8 if params.get("batch_size") is None \
      else params.get("batch_size")

    AUTOTUNE = tf.data.AUTOTUNE
    
    image_count = len(list(full_datapath.glob('*/*.png')))
    val_size = int(image_count*val_prop)
    list_ds = tf.data.Dataset.list_files(str(full_datapath/'*/*'))
    
    if params['mode'] == "train":
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)
        
        train_ds = train_ds.map(lambda x: process_path(x, CLASS_NAMES, 
            channels, width, height), num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(lambda x: process_path(x, CLASS_NAMES, 
            channels, width, height), num_parallel_calls=AUTOTUNE)

        print("Total Training Images: "+
            str(tf.data.experimental.cardinality(train_ds).numpy()))
        print("Total Validation Images: "+
            str(tf.data.experimental.cardinality(val_ds).numpy()))

        train_ds = configure_for_performance(train_ds, batch_size)
        val_ds = configure_for_performance(val_ds, batch_size)

        return train_ds, val_ds
    else: 
        labelled_ds = list_ds.map(lambda x: process_path(x, CLASS_NAMES, 
            channels, width, height), num_parallel_calls=AUTOTUNE)

        print("Total Eval Images: "+
            str(tf.data.experimental.cardinality(labelled_ds).numpy()))

        labelled_ds = configure_for_performance(labelled_ds, batch_size)

        return labelled_ds, None

def train_model(params, train_ds, val_ds):
    
    channels = 1 if params.get("channels") is None else params.get("channels")
    width = 2048 if params.get("width") is None else params.get("width")
    height = 2044 if params.get("height") is None else params.get("height")
    batch_size = 8 if params.get("batch_size") is None \
      else params.get("batch_size")

    model = model_architecture(batch_size, width, height, channels)

    print(f"Before {tf.config.experimental.get_memory_info('GPU:0')}")

    model.fit(train_ds, validation_data=val_ds, epochs=3)

    print(f"After {tf.config.experimental.get_memory_info('GPU:0')}")

    return model

def make_predictions():
    # if class is just initiated, it will load a model, otheriwse
    pass

def model_architecture(batch_size, width, height, channels):
    '''input_shape = (2048, 2044, 1) # need way of feeding image size
    num_labels = 1 # need way of number of labels
    batch_size = 32
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
    
    model = Model(inputs=inputs, outputs=outputs)'''
    input_shape = (batch_size, width, height, channels)
    num_classes = 2

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        optimizer = 'adam',
        metrics = ['accuracy'])

    model.build(input_shape)
    model.summary()
    
    return model
