+-"""
Skrypt treningowy dla Vision Transformer (ViT).
Trenuje model Transformer do rozpoznawania wielkich liter (A-Z) na datasecie EMNIST.

Vision Transformer wymaga:
- Mniejszego learning rate (transformery są wrażliwe)
- Więcej epok (wolniej się uczą ale lepiej generalizują)
- Warmup learning rate (stopniowe zwiększanie LR na początku)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from models.vit_model import VisionTransformer, count_parameters
from prepare_dataset import load_prepared_data


def prepare_data_for_pytorch(data_dict, validation_split=0.1, batch_size=64):
    """
    Przygotowuje dane dla PyTorch (normalizacja, DataLoader).
    Mniejszy batch_size dla ViT (więcej pamięci GPU).
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

    print(f"Przygotowano dane:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    print(f"  Batch size:    {batch_size}")

    return train_loader, val_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    """Trenuje model przez jedną epokę."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping - ważne dla Transformerów!
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    # Step scheduler
    if scheduler is not None:
        scheduler.step()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """Ewaluuje model na zbiorze danych."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_model(epochs=30, learning_rate=1e-3, batch_size=64, embed_dim=64, num_heads=4, num_layers=4):
    """
    Główna funkcja treningowa dla Vision Transformer.

    Domyślne parametry ViT:
    - embed_dim=64: Wymiar embeddingu (mały dla szybkiego treningu)
    - num_heads=4: Liczba głów attention
    - num_layers=4: Liczba bloków Transformer
    - learning_rate=1e-3: Learning rate (z warmup)
    """
    print("=" * 60)
    print("TRENING: Vision Transformer (ViT)")
    print("=" * 60)

    # Sprawdź GPU jeśli nie ma to puść na cpu + wypisanie nazwy karty raficznej jeśli znajdzie i użyje
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Wczytaj dane
    print("\nWczytywanie danych EMNIST...")
    data_path = './data/processed'

    if not os.path.exists(data_path):
        print(f"Brak danych w {data_path}!")
        print("Uruchom najpierw: python prepare_dataset.py")
        return None

    #wczytuje dane i tworzy Dataloaders które są odpowiedzialne za dzielenie danych na batche i ich mieszanie
    data = load_prepared_data(data_path)
    train_loader, val_loader, test_loader = prepare_data_for_pytorch(
        data, batch_size=batch_size
    )

    # Inicjalizacja modelu vit. Obraz zostaje pocięty na 16 kawałków(patchy) o rozmiarze 7x7
    # i z 26 wyjściami(tyle ile liter w alfabecie)
    print(f"\nTworzenie modelu Vision Transformer...")
    model = VisionTransformer(
        img_size=28,
        #Rozmiar pojedynczego "kafelka", na który dzielony jest obraz. ViT nie analizuje obrazu piksel po pikselu,
        # lecz jako sekwencję małych fragmentów. Przy obrazie 28x28 i patchu 7x7, otrzymujemy siatkę 4x4,
        # czyli sekwencję 16 patchy.
        patch_size=7,
        #Liczba kanałów koloru. 1 oznacza obraz w skali szarości
        in_channels=1,
        #Liczba neuronów na ostatniej warstwie.
        num_classes=26,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        #Prawdopodobieństwo (10%) losowego "wyłączania" neuronów podczas treningu. Zapobiega to przeuczeniu,
        # zmuszając sieć do niepolegania na pojedynczych połączeniach.
        dropout=0.1
    )
    # Przenosi wszystkie wagi modelu do pamięci urządzenia
    model = model.to(device)

    print(f"\nArchitektura ViT:")
    print(f"  - Obraz: 28x28 px")
    print(f"  - Patch size: 7x7 px")
    print(f"  - Liczba patchy: {model.patch_embedding.num_patches}")
    print(f"  - Embedding dim: {embed_dim}")
    print(f"  - Liczba glów attention: {num_heads}")
    print(f"  - Liczba bloków Transformer: {num_layers}")
    print(f"  - Parametry: {count_parameters(model):,}")

    # Definiuje funkcję kosztu, która mierzy różnicę między rozkładem prawdopodobieństwa z modelu a faktyczną etykietą.
    criterion = nn.CrossEntropyLoss()

    # AdamW - lepszy dla Transformerów oraz weight decay który pozwoli chronić przed overfittingiem danych
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05,  # Regularyzacja
        betas=(0.9, 0.999)
    )

    # Ustawia harmonogram zmiany Learning Rate. LR będzie się zmieniać według krzywej cosinusowej,
    # co pozwala na szybki start i precyzyjne dotarcie do minimum pod koniec treningu.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )

    # Przygotowuje słownik do gromadzenia statystyk
    # oraz zmienne do przechowywania najlepszej wersji modelu (checkpointing).
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_model_state = None

    print(f"\nRozpoczynam trening ({epochs} epok)...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # Trening
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )

        # Walidacja
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Zapisz historię
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Wyświetl wyniki
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Zapisz najlepszy model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"         ^^^ Nowy najlepszy model! Val Acc: {val_acc:.2f}%")

    print("-" * 60)
    print(f"\nTrening zakonczony!")
    print(f"Najlepsza walidacja: {best_val_acc:.2f}%")

    # Załaduj najlepszy model do testu
    model.load_state_dict(best_model_state)

    # Test końcowy
    print(f"\nTest koncowy na zbiorze testowym...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"TEST ACCURACY: {test_acc:.2f}%")

    # Zapisuje paczkę danych (wagi, konfigurację i wyniki) do pliku binarnego .pth.
    save_dir = './models/saved'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'vit_best.pth')

    torch.save({
        'model_state_dict': best_model_state,
        'config': {
            'img_size': 28,
            'patch_size': 7,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'num_classes': 26
        },
        'test_accuracy': test_acc,
        'val_accuracy': best_val_acc
    }, save_path)

    print(f"\nModel zapisany do: {save_path}")

    # Wykres treningu
    plot_training_history(history, epochs)

    return model, history


def plot_training_history(history, epochs):
    """Rysuje wykresy treningu."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(range(1, epochs+1), history['train_loss'], label='Train Loss')
    axes[0].plot(range(1, epochs+1), history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Vision Transformer - Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(range(1, epochs+1), history['train_acc'], label='Train Acc')
    axes[1].plot(range(1, epochs+1), history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Vision Transformer - Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('./models/saved/vit_training_history.png', dpi=150)
    print("Wykres zapisany do: ./models/saved/vit_training_history.png")
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VISION TRANSFORMER - TRENING")
    print("=" * 60)

    print("\nParametry modelu:")
    print("  1. Maly (szybki)   - embed_dim=64,  num_layers=4  (~50k parametrow)")
    print("  2. Sredni          - embed_dim=128, num_layers=6  (~200k parametrow)")
    print("  3. Duzy (wolny)    - embed_dim=256, num_layers=8  (~800k parametrow)")

    choice = input("\nWybierz rozmiar modelu (1/2/3) [domyslnie 1]: ").strip()

    if choice == '2':
        embed_dim, num_heads, num_layers, epochs = 128, 8, 6, 40
        batch_size = 32
    elif choice == '3':
        embed_dim, num_heads, num_layers, epochs = 256, 8, 8, 50
        batch_size = 16
    else:
        embed_dim, num_heads, num_layers, epochs = 64, 4, 4, 30
        batch_size = 64

    print(f"\nWybrano: embed_dim={embed_dim}, num_layers={num_layers}, epochs={epochs}")

    model, history = train_model(
        epochs=epochs,
        learning_rate=1e-3,
        batch_size=batch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
