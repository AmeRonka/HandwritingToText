"""
Skrypt do przygotowania datasetu EMNIST (tylko wielkie litery A-Z).
Pobiera dataset, filtruje tylko wielkie litery i przygotowuje do treningu.
"""

import os
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm


# Mapowanie klas EMNIST ByClass do liter
# EMNIST ByClass ma 62 klasy:
#   0-9: cyfry (0-9)
#   10-35: wielkie litery (A-Z)
#   36-61: małe litery (a-z)
#
# Chcemy tylko wielkie litery (klasy 10-35)

UPPERCASE_CLASS_OFFSET = 10  # Wielkie litery zaczynają się od klasy 10
NUM_UPPERCASE_CLASSES = 26   # A-Z


def download_emnist(data_dir='./data'):
    """
    Pobiera EMNIST ByClass dataset.

    Args:
        data_dir: Katalog do zapisania danych

    Returns:
        Tuple (train_dataset, test_dataset)
    """
    print("📥 Pobieranie EMNIST ByClass dataset...")
    print("   (To może chwilę potrwać przy pierwszym uruchomieniu)")

    # Pobierz dane treningowe
    train_dataset = datasets.EMNIST(
        root=data_dir,
        split='byclass',
        train=True,
        download=True
    )

    # Pobierz dane testowe
    test_dataset = datasets.EMNIST(
        root=data_dir,
        split='byclass',
        train=False,
        download=True
    )

    print(f"✓ Pobrano dataset")
    print(f"  Dane treningowe: {len(train_dataset)} obrazków")
    print(f"  Dane testowe: {len(test_dataset)} obrazków")

    return train_dataset, test_dataset


def filter_uppercase_letters(dataset, verbose=True):
    """
    Filtruje tylko wielkie litery (A-Z) z EMNIST ByClass.

    Args:
        dataset: Dataset EMNIST ByClass
        verbose: Czy wyświetlać progress bar

    Returns:
        Tuple (images, labels):
            - images: Numpy array (N, 28, 28)
            - labels: Numpy array (N,) z wartościami 0-25 (dla A-Z)
    """
    images = []
    labels = []

    iterator = tqdm(dataset, desc="Filtrowanie liter") if verbose else dataset

    for img, label in iterator:
        # Sprawdź czy to wielka litera (klasa 10-35)
        if UPPERCASE_CLASS_OFFSET <= label < UPPERCASE_CLASS_OFFSET + NUM_UPPERCASE_CLASSES:
            # Konwertuj PIL Image na numpy
            img_array = np.array(img)

            # EMNIST ma obrazy obrócone i odbite - popraw to
            img_array = np.rot90(img_array, k=-1)  # Obróć o 90° w prawo
            img_array = np.fliplr(img_array)        # Odbij poziomo

            images.append(img_array)

            # Przekonwertuj label: 10->0 (A), 11->1 (B), ..., 35->25 (Z)
            new_label = label - UPPERCASE_CLASS_OFFSET
            labels.append(new_label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def prepare_emnist_uppercase(data_dir='./data', save_dir='./data/processed'):
    """
    Przygotowuje cały dataset EMNIST z tylko wielkimi literami.

    Args:
        data_dir: Katalog źródłowy EMNIST
        save_dir: Katalog docelowy dla przetworzonych danych

    Returns:
        Dictionary z danymi treningowymi i testowymi
    """
    # Utwórz katalog jeśli nie istnieje
    os.makedirs(save_dir, exist_ok=True)

    # Pobierz EMNIST
    train_dataset, test_dataset = download_emnist(data_dir)

    # Filtruj wielkie litery
    print("\n🔍 Filtrowanie wielkich liter z danych treningowych...")
    X_train, y_train = filter_uppercase_letters(train_dataset, verbose=True)

    print("\n🔍 Filtrowanie wielkich liter z danych testowych...")
    X_test, y_test = filter_uppercase_letters(test_dataset, verbose=True)

    # Statystyki
    print("\n" + "="*70)
    print("STATYSTYKI DATASETU (TYLKO WIELKIE LITERY A-Z)")
    print("="*70)
    print(f"Dane treningowe: {len(X_train)} obrazków")
    print(f"Dane testowe:    {len(X_test)} obrazków")
    print(f"Kształt obrazka: {X_train.shape[1:]}")
    print(f"Liczba klas:     {NUM_UPPERCASE_CLASSES} (A-Z)")
    print(f"Zakres pikseli:  [{X_train.min()}, {X_train.max()}]")
    print("="*70)

    # Rozkład klas
    print("\nRozkład klas w danych treningowych:")
    unique, counts = np.unique(y_train, return_counts=True)
    for class_id, count in zip(unique, counts):
        letter = chr(ord('A') + class_id)
        print(f"  {letter} (klasa {class_id:2d}): {count:5d} obrazków")

    # Zapisz przetworzone dane
    save_path_train_X = os.path.join(save_dir, 'X_train_uppercase.npy')
    save_path_train_y = os.path.join(save_dir, 'y_train_uppercase.npy')
    save_path_test_X = os.path.join(save_dir, 'X_test_uppercase.npy')
    save_path_test_y = os.path.join(save_dir, 'y_test_uppercase.npy')

    print(f"\n💾 Zapisywanie przetworzonych danych do {save_dir}...")
    np.save(save_path_train_X, X_train)
    np.save(save_path_train_y, y_train)
    np.save(save_path_test_X, X_test)
    np.save(save_path_test_y, y_test)

    print("✓ Dane zapisane pomyślnie")

    # Zwróć dane
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def visualize_samples(data_dict, samples_per_class=5, save_path='./data/samples_visualization.png'):
    """
    Wizualizuje przykładowe litery z datasetu.

    Args:
        data_dict: Dictionary z danymi (z prepare_emnist_uppercase)
        samples_per_class: Ile przykładów pokazać dla każdej klasy
        save_path: Ścieżka do zapisania wizualizacji
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']

    print(f"\n🎨 Tworzenie wizualizacji ({samples_per_class} próbek na klasę)...")

    fig, axes = plt.subplots(NUM_UPPERCASE_CLASSES, samples_per_class,
                             figsize=(samples_per_class * 2, NUM_UPPERCASE_CLASSES * 2))

    for class_id in range(NUM_UPPERCASE_CLASSES):
        # Znajdź wszystkie obrazki tej klasy
        class_indices = np.where(y_train == class_id)[0]

        # Wybierz losowe próbki
        selected_indices = np.random.choice(class_indices,
                                           size=min(samples_per_class, len(class_indices)),
                                           replace=False)

        # Wyświetl
        for i, idx in enumerate(selected_indices):
            ax = axes[class_id, i] if NUM_UPPERCASE_CLASSES > 1 else axes[i]
            ax.imshow(X_train[idx], cmap='gray')
            ax.axis('off')

            # Dodaj literę jako tytuł (tylko dla pierwszej kolumny)
            if i == 0:
                letter = chr(ord('A') + class_id)
                ax.set_title(f'{letter}', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Wizualizacja zapisana w: {save_path}")
    plt.close()


def load_prepared_data(data_dir='./data/processed'):
    """
    Ładuje wcześniej przygotowane dane.

    Args:
        data_dir: Katalog z przetworzonymi danymi

    Returns:
        Dictionary z danymi treningowymi i testowymi
    """
    print(f"📂 Ładowanie przetworzonych danych z {data_dir}...")

    X_train = np.load(os.path.join(data_dir, 'X_train_uppercase.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_uppercase.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_uppercase.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_uppercase.npy'))

    print(f"✓ Załadowano {len(X_train)} próbek treningowych i {len(X_test)} testowych")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


if __name__ == "__main__":
    print("="*70)
    print("PRZYGOTOWANIE DATASETU EMNIST (WIELKIE LITERY A-Z)")
    print("="*70)

    # Przygotuj dataset
    data = prepare_emnist_uppercase(
        data_dir='./data',
        save_dir='./data/processed'
    )

    # Wizualizuj przykłady
    visualize_samples(
        data,
        samples_per_class=5,
        save_path='./data/samples_visualization.png'
    )

    print("\n" + "="*70)
    print("✓ GOTOWE!")
    print("="*70)
    print("\nMożesz teraz trenować modele używając:")
    print("  - python train_tensorflow.py")
    print("  - python train_pytorch.py")
    print("="*70)
