import tensorflow as tf
from tensorflow.keras import layers

def build_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal",fill_mode='nearest'),
            layers.RandomRotation(0.05,fill_mode='nearest'),      
            layers.RandomTranslation(0.05, 0.05,fill_mode='nearest'),
            layers.RandomZoom(0.05,fill_mode='nearest'),
        ],
        name="data_augmentation",
    )