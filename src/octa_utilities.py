import numpy as np
import pandas as pd
import tensorflow as tf
import os

def get_label(file_path, class_names):
	parts = tf.strings.split(file_path, os.path.sep)
	one_hot = parts[-2] == class_names
	return tf.argmax(one_hot)

def decode_img(img, channels=1, width=2048, height=2044):
	img = tf.image.decode_png(img, channels=channels)
	# add crop ability 2040x2040 width to height (in config?)

	# standardise
	return tf.image.resize(img, [width, height]) 

def process_path(file_path, class_names, channels, width, height):
	label = get_label(file_path, class_names)
	img = tf.io.read_file(file_path)
	img = decode_img(img, channels, width, height)
	return img, label

def configure_for_performance(ds, batch_size):
	ds = ds.cache()
	ds = ds.shuffle(buffer_size=1000)
	ds = ds.batch(batch_size)
	ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
	return ds