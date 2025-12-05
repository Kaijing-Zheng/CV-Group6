import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
directory = "/Users/dli3/Desktop/Duke Materials/DAML/CV-Group6/asl_alphabet_train"
train_data = keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

val_data = keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    seed=42,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


tf.data.experimental.save(train_data, "/Users/dli3/Desktop/Duke Materials/DAML/CV-Group6/train_dataset")

# Later, load it back
train = tf.data.Dataset.load("/Users/dli3/Desktop/Duke Materials/DAML/CV-Group6/train_dataset")
tf.data.experimental.save(val_data, "/Users/dli3/Desktop/Duke Materials/DAML/CV-Group6/val_dataset")

# Later, load it back
validation = tf.data.Dataset.load("/Users/dli3/Desktop/Duke Materials/DAML/CV-Group6/val_dataset")