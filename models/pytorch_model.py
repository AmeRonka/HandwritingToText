"""
Model CNN w PyTorch dla rozpoznawania wielkich liter (A-Z).
Architektura zoptymalizowana dla obrazów 28x28 (format EMNIST).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LetterCNN(nn.Module):
    """
    Model CNN w PyTorch do rozpoznawania liter A-Z.

    Architektura:
        Input (1x28x28)
        -> Conv2D(32) + ReLU + MaxPool2D
        -> Conv2D(64) + ReLU + MaxPool2D
        -> Conv2D(64) + ReLU
        -> Flatten
        -> Dense(128) + ReLU + Dropout
        -> Dense(num_classes)
    """

    def __init__(self, num_classes=26, dropout_rate=0.3):
        """
        Args:
            num_classes: Liczba klas (26 dla A-Z)
            dropout_rate: Współczynnik dropout (0.0-1.0)
        """
        super(LetterCNN, self).__init__()

        # warstwy konwolucyjne
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # Po 2x MaxPool2D (28->14->7), z conv3 (64 channels) mamy: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass przez sieć.

        Args:
            x: Tensor wejściowy (batch_size, 1, 28, 28)

        Returns:
            Tensor wyjściowy (batch_size, num_classes) - logity (przed softmax)
        """
        # Block 1: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)

        # Block 2: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)

        # Block 3: Conv -> ReLU (bez pool)
        x = F.relu(self.conv3(x))             # (batch, 64, 7, 7)

        # Flatten
        x = x.view(-1, 64 * 7 * 7)            # (batch, 3136)

        # Fully connected
        x = F.relu(self.fc1(x))               # (batch, 128)
        x = self.dropout(x)
        x = self.fc2(x)                       # (batch, num_classes)

        return x


class LetterCNNAdvanced(nn.Module):
    """
    Bardziej zaawansowany model z Batch Normalization.
    Użyj tego jeśli podstawowy model nie osiąga zadowalających wyników.
    """

    def __init__(self, num_classes=26):
        super(LetterCNNAdvanced, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Classifier
        # Po 2x pool: 28 -> 14 -> 7, z 128 channels: 128 * 7 * 7 = 6272
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(-1, 128 * 7 * 7)

        # Classifier
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def print_model_summary(model, input_size=(1, 1, 28, 28)):
    """
    Wyświetla podsumowanie modelu PyTorch.

    Args:
        model: Model PyTorch
        input_size: Rozmiar wejścia (batch_size, channels, height, width)
    """
    print("\n" + "="*70)
    print(f"PODSUMOWANIE MODELU: {model.__class__.__name__}")
    print("="*70)

    # Policz parametry
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Całkowita liczba parametrów: {total_params:,}")
    print(f"Treningowe parametry: {trainable_params:,}")
    print("="*70)

    # Wyświetl strukturę
    print("\nStruktura modelu:")
    print(model)
    print("="*70)

    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test budowy modelu
    print("Tworzę model PyTorch...")
    model = LetterCNN(num_classes=26)
    print_model_summary(model)

    print("\nTworzę zaawansowany model PyTorch...")
    model_adv = LetterCNNAdvanced(num_classes=26)
    print_model_summary(model_adv)
