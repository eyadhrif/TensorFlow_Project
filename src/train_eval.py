import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_data


def evaluate_model(model_path, x_test, y_test, model_name):
    """
    Load and evaluate a trained model.
    
    Args:
        model_path: Path to saved model (.h5)
        x_test: Test images
        y_test: Test labels
        model_name: Name used for display
    """
    print(f"\nEvaluating {model_name}")
    print("=" * 50)

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return test_acc


if __name__ == "__main__":

    # Load data
    print("Loading test data...")
    (_, _), (_, _), (x_test, y_test) = load_data()

    acc_no_aug = evaluate_model(
        "../models/best_model.h5",       # path to model without augmentation
        x_test,
        y_test,
        "CNN Without Augmentation"
    )

    acc_aug = evaluate_model(
        "../models/best_model_augmented.h5",  # path to model with augmentation
        x_test,
        y_test,
        "CNN With Augmentation"
    )

    # Final comparison
    print("\nFinal Comparison")
    print("=" * 50)
    print(f"Accuracy without augmentation: {acc_no_aug:.4f}")
    print(f"Accuracy with augmentation:    {acc_aug:.4f}")

    if acc_aug > acc_no_aug:
        print("Data augmentation improves generalization.")
    else:
        print("No significant improvement from augmentation.")
