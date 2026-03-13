"""
Scikit-learn models for letter recognition (A-Z).
Supports SVM and Random Forest classifiers.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib


class SklearnLetterClassifier:
    """
    Wrapper for sklearn models (SVM or RandomForest) for letter recognition.
    Handles feature extraction from 28x28 images and prediction.
    """

    def __init__(self, model_type='svm'):
        """
        Args:
            model_type: 'svm' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.num_classes = 26  # A-Z

    def build_model(self):
        """Creates and configures the sklearn model."""
        if self.model_type == 'svm':
            # SVM with RBF kernel
            # C=10: regularization parameter
            # gamma='scale': kernel coefficient
            # probability=True: enables predict_proba for confidence
            self.model = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42,
                verbose=True
            )
        elif self.model_type == 'random_forest':
            # Random Forest classifier
            # n_estimators=100: number of trees
            # n_jobs=-1: use all CPU cores
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return self.model

    def extract_features(self, X):
        """
        Extract features from 28x28 images.

        Args:
            X: Numpy array of shape (N, 28, 28) or (28, 28)

        Returns:
            Features array of shape (N, 784) or (784,)
        """
        # Simple flatten: 28x28 -> 784 features
        # This works well for both SVM and Random Forest
        original_shape = X.shape

        if len(original_shape) == 2:
            # Single image (28, 28) -> (784,)
            return X.flatten()
        elif len(original_shape) == 3:
            # Batch of images (N, 28, 28) -> (N, 784)
            return X.reshape(X.shape[0], -1)
        else:
            raise ValueError(f"Unexpected shape: {original_shape}")

    def fit(self, X_train, y_train):
        """
        Train the model.

        Args:
            X_train: Images of shape (N, 28, 28), normalized to [0, 1]
            y_train: Labels of shape (N,), values 0-25 for A-Z
        """
        if self.model is None:
            self.build_model()

        print(f"Extracting features from {len(X_train)} training samples...")
        X_features = self.extract_features(X_train)

        print(f"Training {self.model_type} model...")
        self.model.fit(X_features, y_train)
        print("Training complete!")

    def predict(self, X):
        """
        Predict letters with confidence scores.

        Args:
            X: Images of shape (N, 28, 28) or (28, 28)

        Returns:
            Tuple (predictions, probabilities):
                - predictions: Predicted class indices (N,) or scalar
                - probabilities: Probability distributions (N, 26) or (26,)
        """
        X_features = self.extract_features(X)

        # Handle single image vs batch
        if len(X_features.shape) == 1:
            X_features = X_features.reshape(1, -1)
            predictions = self.model.predict(X_features)
            probabilities = self.model.predict_proba(X_features)
            return predictions[0], probabilities[0]
        else:
            predictions = self.model.predict(X_features)
            probabilities = self.model.predict_proba(X_features)
            return predictions, probabilities

    def save(self, filepath):
        """
        Save model to file using joblib.

        Args:
            filepath: Path to save .pkl file
        """
        joblib.dump(self, filepath, compress=3)
        print(f"Model saved to: {filepath}")

    @staticmethod
    def load(filepath):
        """
        Load model from file.

        Args:
            filepath: Path to .pkl file

        Returns:
            SklearnLetterClassifier instance
        """
        return joblib.load(filepath)


def print_model_info(model):
    """
    Display model information.

    Args:
        model: SklearnLetterClassifier instance
    """
    print("\n" + "="*70)
    print(f"SKLEARN MODEL INFO: {model.model_type.upper()}")
    print("="*70)
    print(f"Model type: {model.model_type}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Feature dimension: 784 (28x28 flattened)")

    if model.model is not None:
        if model.model_type == 'svm':
            print(f"Kernel: {model.model.kernel}")
            print(f"C: {model.model.C}")
            print(f"Gamma: {model.model.gamma}")
        elif model.model_type == 'random_forest':
            print(f"Number of estimators: {model.model.n_estimators}")
            print(f"Max depth: {model.model.max_depth}")

    print("="*70 + "\n")


if __name__ == "__main__":
    # Test model creation
    print("Creating SVM model...")
    svm_model = SklearnLetterClassifier(model_type='svm')
    svm_model.build_model()
    print_model_info(svm_model)

    print("Creating Random Forest model...")
    rf_model = SklearnLetterClassifier(model_type='random_forest')
    rf_model.build_model()
    print_model_info(rf_model)
