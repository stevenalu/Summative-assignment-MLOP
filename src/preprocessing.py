import os
import numpy as np
import tensorflow as tf

img_size = (160, 160)
batch = 32

train_dir = os.path.join("data", "train")
test_dir = os.path.join("data", "test")
model_path = os.path.join("models", "maize_model.keras")

default_classes = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]


def get_class_names(directory=train_dir):
    if os.path.isdir(directory):
        names = [d for d in sorted(os.listdir(directory))
                 if os.path.isdir(os.path.join(directory, d))]
        if names:
            return names
    return default_classes


def load_image(file, target_size=img_size):
    img = tf.keras.utils.load_img(file, target_size=target_size)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)


def build_dataset(directory, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch,
        shuffle=shuffle,
    )