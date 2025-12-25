# src/augment.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CUSTOM LAYERS (Points 3 & 4) - GRAPH COMPATIBLE
# ==========================================

class RandomGaussianBlur(tf.keras.layers.Layer):
    """
    Point 3: Kernel Filters (Blurring).
    Fixed to use tf.cond for Graph compatibility.
    """
    def __init__(self, probability=0.2, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability

    def call(self, images, training=True):
        if not training:
            return images
        
        # Using tf.cond instead of Python 'if' allows the graph to compile
        return tf.cond(
            tf.random.uniform([]) < self.probability,
            lambda: self._apply_blur(images),
            lambda: images
        )

    def _apply_blur(self, images):
        # Blur trick: Downsample then Upsample using bilinear interpolation
        shape = tf.shape(images)
        h, w = shape[1], shape[2]
        img_small = tf.image.resize(images, [h // 2, w // 2], method='bilinear')
        return tf.image.resize(img_small, [h, w], method='bilinear')

class RandomErasing(tf.keras.layers.Layer):
    """
    Point 4: Random Erasing (Cutout).
    Uses SpatialDropout2D which is natively graph-compatible.
    """
    def __init__(self, probability=0.1, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.drop = tf.keras.layers.SpatialDropout2D(rate=self.probability)

    def call(self, images, training=True):
        # SpatialDropout randomly sets entire channels/blocks to zero
        return self.drop(images, training=training)

# ==========================================
# PIPELINE CREATION (Points 1, 2, 3, 4)
# ==========================================

def create_pipeline():
    """
    Creates the Sequential model for single-image augmentations.
    Addresses Points 1, 2, 3, and 4.
    """
    return tf.keras.Sequential([
        # Point 1: Geometric (removed RandomHeight/RandomWidth to preserve 28x28 dimensions)
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomFlip("horizontal"),

        # Point 2: Color Space (Contrast/Brightness)
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),

        # Point 3: Kernel Filters
        RandomGaussianBlur(probability=0.2),

        # Point 4: Random Erasing
        RandomErasing(probability=0.1) 
    ])

# ==========================================
# MIXUP LOGIC (Point 5)
# ==========================================

def mixup(images, labels, alpha=0.2):
    """
    Point 5: Mixing Images (MixUp augmentation).
    """
    batch_size = tf.shape(images)[0]
    
    # Ensure labels are float32 for mixing
    labels = tf.cast(labels, tf.float32)
    
    # Generate mixing weights (Beta distribution approximation)
    weight = tf.random.uniform([batch_size], minval=0.0, maxval=alpha)
    
    # Shuffle indices to pick images to mix with
    indices = tf.range(batch_size)
    shuffled_indices = tf.random.shuffle(indices)
    
    # Gather pairs
    images_two = tf.gather(images, shuffled_indices)
    labels_two = tf.gather(labels, shuffled_indices)
    
    # Reshape weight for broadcasting with images
    weight_img = tf.reshape(weight, [batch_size, 1, 1, 1])
    
    # Linear interpolation of images and labels
    images_mixed = (1.0 - weight_img) * images + weight_img * images_two
    labels_mixed = (1.0 - weight) * labels + weight * labels_two
    
    return images_mixed, labels_mixed

# ==========================================
# DATASET FACTORY
# ==========================================

def get_augmented_dataset(x_train, y_train, batch_size=32, use_mixup=True):
    """
    Main entry point to get the tf.data.Dataset.
    """
    y_train = tf.cast(y_train, tf.float32)

    augmentation_pipeline = create_pipeline()
    
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    
    # Define a proper function instead of a lambda to satisfy AutoGraph
    @tf.function # This helps performance and clarifies the graph structure
    def apply_pipeline(images, labels):
        return augmentation_pipeline(images, training=True), labels

    # Apply Points 1-4
    dataset = dataset.map(apply_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply Point 5 (MixUp)
    if use_mixup:
        dataset = dataset.map(mixup, num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ==========================================
# VISUALIZATION UTILITY
# ==========================================

def visualize_augmentation(x_train, y_train):
    """
    Helper to visualize the results inside the notebook.
    """
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    
    # Create dataset with mixup for visualization
    ds = get_augmented_dataset(x_train, y_train, batch_size=5, use_mixup=True)
    images, labels = next(iter(ds))
    
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        
        # Display mixed label value (after MixUp, labels are floats)
        lbl_val = labels[i].numpy()
        # Get the closest class for display
        closest_class = int(np.round(lbl_val))
        if 0 <= closest_class < len(class_names):
            plt.title(f"{class_names[closest_class]}\n(mixed: {lbl_val:.2f})")
        else:
            plt.title(f"Label: {lbl_val:.2f}")
             
        plt.axis("off")
    plt.show()