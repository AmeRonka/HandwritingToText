"""
Training script for Scikit-learn models (SVM and Random Forest).
Trains on EMNIST uppercase letters dataset (A-Z).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

from models.sklearn_model import SklearnLetterClassifier, print_model_info
from prepare_dataset import load_prepared_data, prepare_emnist_uppercase


def prepare_data_for_sklearn(data_dict, use_subset=False, subset_size=50000):
    """
    Prepare data for sklearn models.

    Args:
        data_dict: Dictionary from load_prepared_data()
                  Contains X_train, y_train, X_test, y_test
        use_subset: If True, use only subset of training data (for faster SVM training)
        subset_size: Size of training subset

    Returns:
        Tuple (X_train, y_train, X_test, y_test) - normalized to [0, 1]
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    # Normalize to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Optional: Use subset for faster training
    if use_subset and len(X_train) > subset_size:
        print(f"\nUsing subset of {subset_size} training samples (out of {len(X_train)})")
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    print(f"\nData prepared:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")
    print(f"  Image shape:   {X_train.shape[1:]}")
    print(f"  Value range:   [{X_train.min():.2f}, {X_train.max():.2f}]")

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, model_type='svm'):
    """
    Train sklearn model.

    Args:
        X_train: Training images (N, 28, 28)
        y_train: Training labels (N,)
        model_type: 'svm' or 'random_forest'

    Returns:
        Trained SklearnLetterClassifier
    """
    print("\n" + "="*70)
    print(f"TRAINING {model_type.upper()} MODEL")
    print("="*70)

    # Create model
    model = SklearnLetterClassifier(model_type=model_type)
    print_model_info(model)

    # Train
    model.fit(X_train, y_train)

    print("="*70)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.

    Args:
        model: Trained SklearnLetterClassifier
        X_test: Test images (N, 28, 28)
        y_test: Test labels (N,)

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)

    # Predict
    print("Making predictions on test set...")
    predictions, probabilities = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*70)

    # Classification report
    class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test, predictions,
        target_names=class_names,
        digits=3
    ))

    # Per-class accuracy
    print("\nPER-CLASS ACCURACY:")
    for i, letter in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (predictions[mask] == y_test[mask]).mean()
            print(f"  {letter}: {class_acc:.3f} ({class_acc*100:.1f}%)")

    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'probabilities': probabilities,
        'confusion_matrix': confusion_matrix(y_test, predictions)
    }


def plot_confusion_matrix(cm, save_path):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix from sklearn
        save_path: Path to save figure
    """
    print(f"\nGenerating confusion matrix...")

    plt.figure(figsize=(12, 10))

    # Class names
    class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    # Plot
    sns.heatmap(
        cm,
        annot=False,  # Too many classes for annotations
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )

    plt.title('Confusion Matrix - Letter Recognition', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Letter', fontsize=12)
    plt.ylabel('True Letter', fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def save_model(model, model_type, save_dir='./models/saved'):
    """
    Save trained model.

    Args:
        model: Trained SklearnLetterClassifier
        model_type: 'svm' or 'random_forest'
        save_dir: Directory to save model
    """
    os.makedirs(save_dir, exist_ok=True)

    if model_type == 'svm':
        save_path = os.path.join(save_dir, 'sklearn_svm.pkl')
    else:
        save_path = os.path.join(save_dir, 'sklearn_rf.pkl')

    print(f"\nSaving model to {save_path}...")
    model.save(save_path)
    print("Model saved successfully!")

    return save_path


def main():
    """Main training function."""
    print("="*70)
    print("SCIKIT-LEARN MODEL TRAINING - LETTER RECOGNITION A-Z")
    print("="*70)

    # Set random seed
    np.random.seed(42)

    # 1. Load data
    try:
        print("\nLoading prepared EMNIST data...")
        data = load_prepared_data('./data/processed')
    except FileNotFoundError:
        print("Prepared data not found. Downloading and preparing EMNIST...")
        data = prepare_emnist_uppercase(
            data_dir='./data',
            save_dir='./data/processed'
        )

    # 2. Choose model type
    print("\n" + "="*70)
    print("Choose model type:")
    print("  1. SVM (higher accuracy, slower training ~20-30 min)")
    print("  2. Random Forest (faster training ~5-10 min)")
    print("="*70)

    choice = input("Enter choice (1 or 2) [default: 1]: ").strip()

    if choice == '2':
        model_type = 'random_forest'
        use_subset = False  # RF can handle full dataset
    else:
        model_type = 'svm'
        # Ask about subset for SVM
        subset_choice = input("Use subset (50k samples) for faster training? (y/n) [default: n]: ").strip().lower()
        use_subset = (subset_choice == 'y')

    # 3. Prepare data
    X_train, y_train, X_test, y_test = prepare_data_for_sklearn(
        data,
        use_subset=use_subset,
        subset_size=50000
    )

    # 4. Train model
    model = train_model(X_train, y_train, model_type=model_type)

    # 5. Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # 6. Plot confusion matrix
    cm_path = f'./models/saved/sklearn_{model_type}_confusion_matrix.png'
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)

    # 7. Save model
    save_path = save_model(model, model_type)

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModel type: {model_type.upper()}")
    print(f"Test accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"\nModel saved to:")
    print(f"  {save_path}")
    print(f"\nConfusion matrix saved to:")
    print(f"  {cm_path}")
    print("\nYou can now use this model in the GUI by selecting:")
    print(f"  'Scikit-learn {model_type.upper()}' from the OCR options")
    print("="*70)


if __name__ == "__main__":
    main()
