import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import os
from keras import backend as K

def get_label(file_path, class_names):
	parts = tf.strings.split(file_path, os.path.sep)
	one_hot = parts[-2] == class_names
	return tf.argmax(one_hot)

def decode_img(img, channels=1, width=2048, height=2048, crop_percentage=0.8):
	img = tf.image.decode_png(img, channels=channels)
	img = tf.image.central_crop(img, crop_percentage)# centre crop
	img = tf.image.per_image_standardization(img)# standardise

	return tf.image.resize(img, [width, height]) # blows image back up

def process_path(file_path, class_names, channels, width, height):
	label = get_label(file_path, class_names)
	img = tf.io.read_file(file_path)
	img = decode_img(img, channels, width, height)
	return img, label

def configure_for_performance(ds, batch_size, shuffle=False, augment=False):
	ds = ds.cache()
	
	if shuffle: # only shuffle the dataset if it is training
		ds = ds.shuffle(buffer_size=1000)

	ds = ds.batch(batch_size)

	if augment: # only augment on the training set

		data_augmentation = tf.keras.Sequential([
		  layers.RandomFlip("horizontal_and_vertical"),
		  layers.RandomRotation(0.25), # random rotation up to 90 degrees
		])

		ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=tf.data.AUTOTUNE)

	ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
	return ds

def f1_metric(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val