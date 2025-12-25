import tensorflow as tf
from tensorflow.keras import layers

def build_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),      
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )