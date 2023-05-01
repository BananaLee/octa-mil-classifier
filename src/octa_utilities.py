import numpy as np
import pandas as pd
import tensorflow as tf
import os

'''def images_to_buffer(folder, buffer_path, label, channels=0):
	"""
    Takes a folder list and converts it into a list of tensors. If labels are 
    provided, it will also return a convenient label list

    """
	image_list = []
    label_list = []

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        raw = tf.io.read_file(file_path)
        image = tf.image.decode_png(raw, channels=channels)

        image_list.append(image)
        label_list.append(label)

    # use label to setup folder structure to match buffer. is there even a need to decode png? 

    if label = None:
    	return image_list
    else:
    	return image_list, label_list'''

def get_label(file_path):
	parts = tf.strings.split(file_path, os.path.sep)
	one_hot = parts[-2] == ['diabetic', 'healthy']#CLASS_NAMES
	return tf.argmax(one_hot)

def decode_img(img, channels=1, width=2048, height=2044):
	img = tf.image.decode_png(img, channels=channels)
	return tf.image.resize(img, [width, height]) 

def process_path(file_path):
	label = get_label(file_path)
	img = tf.io.read_file(file_path)
	img = decode_img(img)
	return img, label