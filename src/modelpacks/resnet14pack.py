''' 
ResNet14 Scratch Adaptation based on ResNet18 Implementation from
https://github.com/jimmyyhwu/resnet18-tf2/blob/master/resnet.py
and proper architecture from Awan et al
https://doi.org/10.3390/diagnostics11010105


'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.models
from tensorflow.keras import layers 
import os
import pathlib
import math

import octa_utilities as util
from octa_utilities import process_path, augment_and_performance

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
    height = 2048 if params.get("height") is None else params.get("height")
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

        train_ds = augment_and_performance(train_ds, batch_size, shuffle=True, augment=True)
        val_ds = augment_and_performance(val_ds, batch_size)

        return train_ds, val_ds
    else: 
        labelled_ds = list_ds.map(lambda x: process_path(x, CLASS_NAMES, 
            channels, width, height), num_parallel_calls=AUTOTUNE)

        print("Total Eval Images: "+
            str(tf.data.experimental.cardinality(labelled_ds).numpy()))

        labelled_ds = augment_and_performance(labelled_ds, batch_size)

        return labelled_ds, None

def train_model(params, train_ds, val_ds):
    
    model = model_architecture(params)

    experiment_path = os.path.join(os.getcwd(), 'experiments', params['name'], 
        params['mode'])

    #print(f"Before {tf.config.experimental.get_memory_info('GPU:0')}")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', 
        mode='max', restore_best_weights=True, patience=3)
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(experiment_path,'training.log'))


    model.fit(train_ds, validation_data=val_ds, epochs=100, 
        callbacks=[early_stopping, csv_logger])

    #print(f"After {tf.config.experimental.get_memory_info('GPU:0')}")

    return model

def make_predictions(model, test_ds):
    return np.rint(model.predict(test_ds).ravel())

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_classes=1000):
    x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = layers.Conv2D(filters=32, kernel_size=7, strides=2, use_bias=False, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=1, name='maxpool')(x)

    x = make_layer(x, 32, blocks_per_layer[0], stride=1, name='layer1')
    x = make_layer(x, 64, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 128, blocks_per_layer[2], stride=2, name='layer3')
    #x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    x = layers.Dense(units=num_classes, activation="sigmoid", name='fc')(x)

    return x


def model_architecture(params):

    channels = 1 if params.get("channels") is None else params.get("channels")
    width = 2048 if params.get("width") is None else params.get("width")
    height = 2044 if params.get("height") is None else params.get("height")
    batch_size = 8 if params.get("batch_size") is None \
      else params.get("batch_size")

    input_shape = (width, height, channels)

    inputs = keras.Input(shape=input_shape)
    outputs = resnet(inputs, [2, 2, 2], num_classes = 1)
    model = keras.Model(inputs, outputs)

    model.compile(loss = tf.keras.losses.BinaryCrossentropy(), 
        optimizer = 'adam',
        metrics = ['accuracy', 
                    keras.metrics.AUC(), 
                    keras.metrics.F1Score(average='micro', threshold=0.5),
                    keras.metrics.Precision(), 
                    keras.metrics.Recall()])
    model.summary()
    
    return model