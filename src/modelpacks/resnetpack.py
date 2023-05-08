''' Basic CNN model with hardcoded parameters
'''
import numpy as np
import tensorflow as tf
import os
import pathlib

import octa_utilities as util
from octa_utilities import process_path, configure_for_performance

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img

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

    input_shape = (width, height, channels)
    num_classes = 2

    resnet = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)

    for layer in resnet.layers:
        layer.trainable = False

    x = Flatten()(resnet.output)

    prediction = Dense(num_classes, activation='softmax')(x)

    # create a model object
    model = Model(inputs=resnet.input, outputs=prediction)

    model.compile(loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy'])

    model.summary()
    
    return model
