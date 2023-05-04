import numpy as np
import pandas as pd
import tensorflow as tf
import os

def get_label(file_path):
	parts = tf.strings.split(file_path, os.path.sep)
	one_hot = parts[-2] == ['diabetic', 'healthy']#CLASS_NAMES
	return tf.argmax(one_hot)

def decode_img(img, channels=1, width=2048, height=2044):
	img = tf.image.decode_png(img, channels=channels)
	# add crop ability 2040x2040 width to height (in config?)
	return tf.image.resize(img, [width, height]) 

#STANDARDISE CHECK!!!!

def process_path(file_path):
	label = get_label(file_path)
	img = tf.io.read_file(file_path)
	img = decode_img(img)
	return img, label