import tensorflow as tf
from tensorflow.keras import layers, models

# Import augmentation - handle both direct and module import
try:
    from augment import build_augmentation
except ImportError:
    from .augment import build_augmentation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {len(gpus)}")
        print(f"GPU devices: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a CNN model for Fashion MNIST classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Flatten and dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout1'),
        layers.Dense(64, activation='relu', name='dense2'),
        layers.Dropout(0.3, name='dropout2'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_cnn_model_with_augmentation(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a CNN model with data augmentation for Fashion MNIST classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Keras model with data augmentation
    """
    model = models.Sequential([
        # Data augmentation layer
        build_augmentation(),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Flatten and dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout1'),
        layers.Dense(64, activation='relu', name='dense2'),
        layers.Dropout(0.3, name='dropout2'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def get_callbacks(patience=10, model_path='best_model.h5', reduce_lr_patience=5):
    """
    Create training callbacks for standard model.
    
    Args:
        patience: Number of epochs with no improvement for early stopping
        model_path: Path to save the best model
        reduce_lr_patience: Patience for learning rate reduction
    
    Returns:
        List of callbacks
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def get_callbacks_augmented(patience=15, model_path='best_model_aug.h5', reduce_lr_patience=7):
    """
    Create training callbacks for augmented model.
    Augmented models need MORE patience because:
    - Training is harder due to augmentation
    - Convergence is slower but more stable
    - Better generalization takes more epochs
    
    Args:
        patience: Number of epochs with no improvement for early stopping (default: 15)
        model_path: Path to save the best model
        reduce_lr_patience: Patience for learning rate reduction (default: 7)
    
    Returns:
        List of callbacks
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


if __name__ == "__main__":
    from data import load_data
    
    # Load data
    print("Loading Fashion MNIST data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    
    print(f"Train samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Image shape: {x_train.shape[1:]}")
    
    # Train model without augmentation
    print("\n" + "="*50)
    print("Training Model WITHOUT Augmentation")
    print("="*50)
    model = create_cnn_model()
    model = compile_model(model, learning_rate=0.001)
    model.summary()
    
    print("\nTraining...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=128,
        callbacks=get_callbacks(patience=10, model_path='best_model.h5'),
        verbose=1
    )
    
    model.save('model_no_augmentation.h5')
    print("\nModel without augmentation saved as 'model_no_augmentation.h5'")
    
    # Train model with augmentation
    print("\n" + "="*50)
    print("Training Model WITH Augmentation")
    print("="*50)
    model_aug = create_cnn_model_with_augmentation()
    model_aug = compile_model(model_aug, learning_rate=0.001)
    model_aug.summary()
    
    print("\nTraining...")
    history_aug = model_aug.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=128,
        callbacks=get_callbacks_augmented(patience=15, model_path='best_model_aug.h5'),
        verbose=1
    )
    
    model_aug.save('model_with_augmentation.h5')
    print("\nModel with augmentation saved as 'model_with_augmentation.h5'")