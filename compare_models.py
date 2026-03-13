"""
Skrypt do porównania wydajności trzech modeli OCR.
Testuje wszystkie modele na tym samym zestawie liter i porównuje wyniki.
"""

import numpy as np
from prepare_dataset import load_prepared_data
import recognition


def compare_ocr_models(num_samples=100):
    """
    Porównuje wszystkie 3 modele OCR na próbce danych testowych.

    Args:
        num_samples: Liczba losowych próbek do przetestowania
    """
    print("="*70)
    print("PORÓWNANIE MODELI OCR - ROZPOZNAWANIE WIELKICH LITER A-Z")
    print("="*70)

    # 1. Załaduj dane testowe
    print("\n📂 Ładowanie danych testowych...")
    try:
        data = load_prepared_data('./data/processed')
        X_test = data['X_test']
        y_test = data['y_test']
    except FileNotFoundError:
        print("❌ Brak danych testowych!")
        print("Uruchom najpierw: python prepare_dataset.py")
        return

    # Wybierz losowe próbki
    indices = np.random.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)
    X_samples = X_test[indices]
    y_samples = y_test[indices]

    print(f"✓ Załadowano {len(X_samples)} próbek testowych")

    # 2. Załaduj modele
    print("\n🤖 Ładowanie modeli OCR...")

    models = []

    # Tesseract
    tesseract = recognition.TesseractRecognizer()
    tesseract.load()
    if tesseract.is_loaded:
        models.append(tesseract)

    # TensorFlow
    tensorflow_path = './models/saved/tensorflow_best.keras'
    tensorflow = recognition.TensorFlowRecognizer(tensorflow_path)
    tensorflow.load()
    if tensorflow.is_loaded:
        models.append(tensorflow)

    # PyTorch
    pytorch_path = './models/saved/pytorch_best.pth'
    pytorch = recognition.PyTorchRecognizer(pytorch_path)
    pytorch.load()
    if pytorch.is_loaded:
        models.append(pytorch)

    if len(models) == 0:
        print("\n❌ Brak dostępnych modeli OCR!")
        print("Wytrenuj modele używając train_tensorflow.py i train_pytorch.py")
        return

    print(f"✓ Załadowano {len(models)} modeli")

    # 3. Porównaj modele
    print("\n" + "="*70)
    print("TESTOWANIE MODELI")
    print("="*70)

    results = {}

    for model in models:
        print(f"\n🔍 Testuję: {model.name}")

        correct = 0
        total = 0
        confidences = []
        times = []

        for i, (img, true_label) in enumerate(zip(X_samples, y_samples)):
            # Predykcja z pomiarem czasu
            pred_letter, confidence, elapsed = model.predict_with_time(img)

            # Konwertuj true_label na literę
            true_letter = chr(ord('A') + true_label)

            # Sprawdź czy poprawne
            if pred_letter == true_letter:
                correct += 1

            total += 1
            confidences.append(confidence)
            times.append(elapsed)

            # Progress
            if (i + 1) % 20 == 0:
                print(f"  Przetestowano {i+1}/{len(X_samples)} próbek...")

        # Metryki
        accuracy = correct / total
        avg_confidence = np.mean(confidences)
        avg_time = np.mean(times)
        total_time = np.sum(times)

        results[model.name] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_confidence': avg_confidence,
            'avg_time_ms': avg_time,
            'total_time_ms': total_time
        }

        print(f"  ✓ Dokładność: {accuracy*100:.2f}% ({correct}/{total})")
        print(f"  ✓ Średnia pewność: {avg_confidence:.3f}")
        print(f"  ✓ Średni czas: {avg_time:.2f}ms")

    # 4. Podsumowanie
    print("\n" + "="*70)
    print("PODSUMOWANIE WYNIKÓW")
    print("="*70)
    print(f"Liczba próbek testowych: {len(X_samples)}")
    print("\n{:<25} {:>12} {:>12} {:>15} {:>15}".format(
        "Model", "Dokładność", "Pewność", "Śr. czas [ms]", "Całk. czas [ms]"
    ))
    print("-"*70)

    for model_name, data in results.items():
        print("{:<25} {:>11.2f}% {:>12.3f} {:>15.2f} {:>15.2f}".format(
            model_name,
            data['accuracy'] * 100,
            data['avg_confidence'],
            data['avg_time_ms'],
            data['total_time_ms']
        ))

    print("="*70)

    # 5. Zwycięzca
    best_accuracy = max(results.values(), key=lambda x: x['accuracy'])
    best_model = [name for name, data in results.items() if data['accuracy'] == best_accuracy['accuracy']][0]

    fastest_model = min(results.values(), key=lambda x: x['avg_time_ms'])
    fastest_name = [name for name, data in results.items() if data['avg_time_ms'] == fastest_model['avg_time_ms']][0]

    print("\n🏆 ZWYCIĘZCY:")
    print(f"  Najdokładniejszy: {best_model} ({best_accuracy['accuracy']*100:.2f}%)")
    print(f"  Najszybszy:       {fastest_name} ({fastest_model['avg_time_ms']:.2f}ms)")
    print("="*70)


if __name__ == "__main__":
    # Ustaw seed
    np.random.seed(42)

    # Porównaj modele na 100 losowych próbkach
    compare_ocr_models(num_samples=100)
