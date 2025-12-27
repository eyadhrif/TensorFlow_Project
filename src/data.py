import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

def load_data(val_split=0.1, random_state=42):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_split,
        random_state=random_state,
        stratify=y_train
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
