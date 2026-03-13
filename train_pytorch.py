"""
Skrypt treningowy dla modelu PyTorch.
Trenuje model CNN do rozpoznawania wielkich liter (A-Z) na datasecie EMNIST.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from models.pytorch_model import LetterCNN, print_model_summary
from prepare_dataset import load_prepared_data, prepare_emnist_uppercase


def prepare_data_for_pytorch(data_dict, validation_split=0.1, batch_size=128):
    """
    Przygotowuje dane dla PyTorch (normalizacja, DataLoader).

    Args:
        data_dict: Dictionary z danymi (X_train, y_train, X_test, y_test)
        validation_split: Część danych treningowych do walidacji
        batch_size: Rozmiar batcha

    Returns:
        Tuple (train_loader, val_loader, test_loader)
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    # Normalizuj do [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Reshape do (N, 1, 28, 28) - dodaj wymiar kanału
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    # Konwertuj na PyTorch tensory
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test).long()

    # Utwórz dataset
    full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Podziel na train i validation
    val_size = int(len(full_train_dataset) * validation_split)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Utwórz DataLoadery
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Przygotowano dane dla PyTorch:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    print(f"  Batch size:    {batch_size}")

    return train_loader, val_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trenuje model przez jedną epokę.

    Returns:
        Tuple (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statystyki
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Waliduje model.

    Returns:
        Tuple (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_model(train_loader, val_loader, epochs=20, learning_rate=0.001):
    """
    Trenuje model PyTorch.

    Returns:
        Tuple (model, history)
    """
    print("\n" + "="*70)
    print("BUDOWANIE MODELU")
    print("="*70)

    # wybranie karty graficznej, w przeciwnym wypadku cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używam urządzenia: {device}")

    # Model LetterCNN to sieć splotowa która jest tworzona z 26 wyjściami oraz dropout ratem na 30%
    model = LetterCNN(num_classes=26, dropout_rate=0.3)
    model.to(device)

    print_model_summary(model)

    print("\n" + "="*70)
    print("TRENING MODELU")
    print("="*70)
    print(f"Epoki: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print("="*70 + "\n")

    # Loss i optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler (reduce lr on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Historia
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    # Katalog na modele
    os.makedirs('./models/saved', exist_ok=True)

    # Pętla treningowa
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)

        # Trening
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Walidacja
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Zapisz historię
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Wyświetl wyniki
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%)")

        # Scheduler step
        scheduler.step(val_loss)

        # Zapisz najlepszy model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), './models/saved/pytorch_best.pth')
            print("✓ Zapisano najlepszy model")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered (patience={patience})")
            break

    print("\n✓ Trening zakończony!")

    # Załaduj najlepszy model
    model.load_state_dict(torch.load('./models/saved/pytorch_best.pth'))

    return model, history


def evaluate_model(model, test_loader, device):
    """
    Ewaluuje model na danych testowych.

    Returns:
        Dictionary z metrykami
    """
    print("\n" + "="*70)
    print("EWALUACJA NA DANYCH TESTOWYCH")
    print("="*70)

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Metryki
    accuracy = (all_predictions == all_labels).mean()

    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*70)

    # Classification report
    from sklearn.metrics import classification_report

    class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    print("\nRAPORT KLASYFIKACJI:")
    print(classification_report(
        all_labels, all_predictions,
        target_names=class_names,
        digits=3
    ))

    return {
        'test_accuracy': accuracy,
        'predictions': all_predictions
    }


def plot_training_history(history, save_path='./models/saved/pytorch_training_history.png'):
    """
    Generuje wykresy historii treningu.
    """
    print(f"\n📊 Generowanie wykresów treningu...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Wykresy zapisane w: {save_path}")
    plt.close()


def save_final_model(model, save_path='./models/saved/pytorch_final.pth'):
    """Zapisuje finalny model."""
    print(f"\n💾 Zapisywanie finalnego modelu...")
    torch.save(model.state_dict(), save_path)
    print(f"✓ Model zapisany w: {save_path}")


def main():
    """Główna funkcja treningowa."""
    print("="*70)
    print("TRENING MODELU PYTORCH - ROZPOZNAWANIE LITER A-Z")
    print("="*70)

    # Ustaw seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Załaduj lub przygotuj dane
    try:
        print("\n📂 Próba załadowania przygotowanych danych...")
        data = load_prepared_data('./data/processed')
    except FileNotFoundError:
        print("⚠ Nie znaleziono przygotowanych danych. Pobieram EMNIST...")
        data = prepare_emnist_uppercase(
            data_dir='./data',
            save_dir='./data/processed'
        )

    # 2. Przygotuj DataLoadery
    train_loader, val_loader, test_loader = prepare_data_for_pytorch(
        data,
        validation_split=0.1,
        batch_size=128
    )

    # 3. Trenuj model
    model, history = train_model(
        train_loader, val_loader,
        epochs=20,
        learning_rate=0.001
    )

    # 4. Ewaluuj model
    metrics = evaluate_model(model, test_loader, device)

    # 5. Generuj wykresy
    plot_training_history(history)

    # 6. Zapisz finalny model
    save_final_model(model, './models/saved/pytorch_final.pth')

    print("\n" + "="*70)
    print("✓ TRENING ZAKOŃCZONY POMYŚLNIE!")
    print("="*70)
    print(f"\nFinalny wynik:")
    print(f"  Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
    print(f"\nModel zapisany w:")
    print(f"  - Najlepszy model: ./models/saved/pytorch_best.pth")
    print(f"  - Finalny model:   ./models/saved/pytorch_final.pth")
    print(f"\nWykresy zapisane w:")
    print(f"  - ./models/saved/pytorch_training_history.png")
    print("="*70)


if __name__ == "__main__":
    main()
