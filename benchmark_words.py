"""
Benchmark system dla porównania modeli OCR na prawdziwych słowach.
Testuje na zdjęciach odręcznie napisanych słów.
"""

import os
import glob
import time
import numpy as np
from PIL import Image, ImageOps
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend bez GUI

import binarization
import segmentation
import recognition

# Groundtruth - słowa do testowania
WORDS = ['waltz', 'nymph', 'vexed', 'quick', 'fjord', 'big']


def extract_groundtruth_from_filename(filename):
    """
    Wyciąga groundtruth z nazwy pliku.

    Args:
        filename: np. 'waltz_1.jpg' lub 'path/to/nymph_3.png'

    Returns:
        Groundtruth (uppercase) np. 'WALTZ' lub None jeśli nie rozpoznano
    """
    basename = os.path.basename(filename).lower()

    for word in WORDS:
        if word in basename:
            return word.upper()

    return None


def preprocess_image(image_path, binarization_method='sauvola'):
    """
    Wczytuje i przetwarza obraz (jak w main.py).

    Args:
        image_path: Ścieżka do obrazu
        binarization_method: 'sauvola', 'adaptive' lub 'global'

    Returns:
        Zbinaryzowany obraz PIL (200x200)
    """
    # Wczytaj obraz
    image = Image.open(image_path)

    # Konwersja do grayscale
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image.copy()

    # Zwiększ kontrast
    gray = ImageOps.autocontrast(gray)

    # Binaryzacja W PEŁNEJ ROZDZIELCZOŚCI
    if binarization_method == 'global':
        binary = binarization.global_threshold(gray, threshold=127)
    elif binarization_method == 'sauvola':
        binary = binarization.sauvola_threshold(gray, window_size=25, k=0.2)
    else:  # adaptive
        binary = binarization.adaptive_threshold(gray)

    # Usuń szum
    orig_w, orig_h = binary.size
    scale_factor = max(orig_w, orig_h) / 200
    min_noise_size = int(20 * scale_factor * scale_factor)
    binary = binarization.remove_noise(binary, min_size=max(20, min_noise_size))

    # Operacje morfologiczne
    binary = binarization.morphological_clean(binary)

    # Skaluj do 200x200 (jak w GUI)
    target_w, target_h = 200, 200
    margin = 15

    scale = min((target_w - 2*margin) / orig_w, (target_h - 2*margin) / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = binary.resize((new_w, new_h), Image.Resampling.NEAREST)

    # Wklej na białe tło
    result = Image.new('L', (target_w, target_h), 255)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    result.paste(resized, (paste_x, paste_y))

    return result


def segment_and_recognize(image_pil, model, segmentation_method='profile'):
    """
    Segmentuje obraz i rozpoznaje litery.

    Args:
        image_pil: Zbinaryzowany obraz PIL
        model: Model rozpoznawania (RecognitionModel)
        segmentation_method: 'profile' lub 'cca'

    Returns:
        Tuple (recognized_text, time_taken)
    """
    start_time = time.time()

    # Konwertuj do numpy (inwertuj: czarne pismo -> białe wartości)
    img_array = 255 - np.array(image_pil)

    # Segmentacja
    if segmentation_method == 'profile':
        profile, boundaries = segmentation.segment_by_profile(img_array)
        if len(boundaries) < 2:
            return "?", time.time() - start_time
        letters = segmentation.extract_letters_profile(img_array, boundaries, profile)
    else:  # cca
        labels, num = segmentation.segment_by_cca(img_array)
        if num == 0:
            return "?", time.time() - start_time
        letters = segmentation.extract_letters_cca(img_array, labels, num)

    # Rozpoznaj każdą literę
    recognized = []
    for letter_img, _ in letters:
        # Normalizuj do 28x28
        normalized = binarization.normalize_letter(letter_img, target_size=(28, 28))
        # Rozpoznaj
        letter, confidence = model.predict(normalized)
        recognized.append(letter)

    recognized_text = "".join(recognized)
    elapsed = time.time() - start_time

    return recognized_text, elapsed


def test_tesseract_whole_text(image_pil):
    """
    Testuje Tesseract w trybie whole text (bez segmentacji).

    Args:
        image_pil: Zbinaryzowany obraz PIL

    Returns:
        Tuple (recognized_text, time_taken)
    """
    start_time = time.time()

    # Użyj TesseractWholeTextRecognizer
    tesseract_whole = recognition.TesseractWholeTextRecognizer()
    tesseract_whole.load()

    if not tesseract_whole.is_loaded:
        return "?", 0.0

    recognized_text, confidence = tesseract_whole.recognize_full_image(image_pil)
    elapsed = time.time() - start_time

    return recognized_text, elapsed


def calculate_accuracy(recognized, groundtruth):
    """
    Oblicza accuracy (procent poprawnych liter).

    Args:
        recognized: Rozpoznany tekst np. "WALTS"
        groundtruth: Prawdziwy tekst np. "WALTZ"

    Returns:
        Accuracy (0.0 - 1.0)
    """
    if not groundtruth:
        return 0.0

    # Jeśli różna długość, uzupełnij krótszy
    max_len = max(len(recognized), len(groundtruth))
    recognized = recognized.ljust(max_len, '?')
    groundtruth = groundtruth.ljust(max_len, ' ')

    correct = sum(r == g for r, g in zip(recognized, groundtruth))
    return correct / len(groundtruth)


def run_benchmark(data_dir='./benchmark_data'):
    """
    Główna funkcja benchmarku.

    Args:
        data_dir: Katalog z obrazami testowymi

    Returns:
        Dictionary z wynikami
    """
    print("="*70)
    print("BENCHMARK MODELI OCR NA SŁOWACH")
    print("="*70)

    # 1. Znajdź wszystkie obrazy
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))

    if len(image_files) == 0:
        print(f"\n❌ Błąd: Brak obrazów w katalogu {data_dir}")
        print("   Sprawdź INSTRUKCJA.md jak przygotować dane!")
        return None

    print(f"\n✓ Znaleziono {len(image_files)} obrazów")

    # 2. Załaduj modele
    print("\n" + "="*70)
    print("ŁADOWANIE MODELI")
    print("="*70)

    # PyTorch
    pytorch_path = './models/saved/pytorch_best.pth'
    pytorch_model = recognition.PyTorchRecognizer(pytorch_path)
    pytorch_model.load()

    # SVM
    sklearn_path = './models/saved/sklearn_svm.pkl'
    sklearn_model = recognition.SklearnRecognizer(sklearn_path, model_type='svm')
    sklearn_model.load()

    # Tesseract single
    tesseract_single = recognition.TesseractRecognizer()
    tesseract_single.load()

    models = {
        'PyTorch CNN': pytorch_model,
        'SVM': sklearn_model,
        'Tesseract Single': tesseract_single
    }

    # 3. Definicje konfiguracji do testowania
    configs = [
        ('PyTorch CNN + Profile', 'PyTorch CNN', 'profile'),
        ('PyTorch CNN + CCA', 'PyTorch CNN', 'cca'),
        ('SVM + Profile', 'SVM', 'profile'),
        ('SVM + CCA', 'SVM', 'cca'),
        ('Tesseract Single + Profile', 'Tesseract Single', 'profile'),
        ('Tesseract Single + CCA', 'Tesseract Single', 'cca'),
        ('Tesseract Whole Text', None, None),  # Specjalny przypadek
    ]

    # 4. Uruchom testy
    print("\n" + "="*70)
    print("URUCHAMIANIE TESTÓW")
    print("="*70)

    results = defaultdict(lambda: {'correct': 0, 'total': 0, 'time': 0.0, 'details': []})

    for img_file in image_files:
        groundtruth = extract_groundtruth_from_filename(img_file)

        if groundtruth is None:
            print(f"\n⚠️  Pominięto: {os.path.basename(img_file)} (nie rozpoznano słowa)")
            continue

        print(f"\n📄 Testowanie: {os.path.basename(img_file)} (groundtruth: {groundtruth})")

        # Przetwórz obraz
        try:
            processed_img = preprocess_image(img_file, binarization_method='sauvola')
        except Exception as e:
            print(f"   ❌ Błąd przetwarzania: {e}")
            continue

        # Testuj każdą konfigurację
        for config_name, model_name, seg_method in configs:
            try:
                if config_name == 'Tesseract Whole Text':
                    # Specjalny przypadek - bez segmentacji
                    recognized, elapsed = test_tesseract_whole_text(processed_img)
                else:
                    # Normalna segmentacja + rozpoznawanie
                    model = models[model_name]
                    if not model.is_loaded:
                        continue
                    recognized, elapsed = segment_and_recognize(processed_img, model, seg_method)

                # Oblicz accuracy
                accuracy = calculate_accuracy(recognized, groundtruth)

                # Zapisz wyniki
                results[config_name]['total'] += 1
                results[config_name]['correct'] += accuracy * len(groundtruth)
                results[config_name]['time'] += elapsed
                results[config_name]['details'].append({
                    'file': os.path.basename(img_file),
                    'groundtruth': groundtruth,
                    'recognized': recognized,
                    'accuracy': accuracy
                })

                # Wyświetl wynik
                status = "✓" if recognized == groundtruth else "✗"
                print(f"   {status} {config_name:35s}: {recognized:10s} ({accuracy*100:.0f}%)")

            except Exception as e:
                print(f"   ❌ {config_name}: Błąd - {e}")
                continue

    # 5. Oblicz finalne statystyki
    print("\n" + "="*70)
    print("WYNIKI KOŃCOWE")
    print("="*70)

    final_results = {}
    for config_name, data in results.items():
        if data['total'] == 0:
            continue

        # Oblicz całkowitą liczbę liter
        total_letters = sum(len(d['groundtruth']) for d in data['details'])

        accuracy = (data['correct'] / total_letters) * 100 if total_letters > 0 else 0
        avg_time = (data['time'] / data['total']) * 1000  # ms

        final_results[config_name] = {
            'accuracy': accuracy,
            'avg_time_ms': avg_time,
            'total_words': data['total'],
            'total_letters': total_letters,
            'correct_letters': int(data['correct']),
            'details': data['details']
        }

        print(f"\n{config_name}:")
        print(f"  Dokładność:     {accuracy:.1f}%")
        print(f"  Średni czas:    {avg_time:.0f} ms/słowo")
        print(f"  Testowanych:    {data['total']} słów ({total_letters} liter)")

    return final_results


def generate_html_report(results, output_file='benchmark_report.html'):
    """
    Generuje raport HTML z wykresami.

    Args:
        results: Dictionary z wynikami z run_benchmark()
        output_file: Ścieżka do pliku HTML
    """
    if not results:
        print("\n❌ Brak wyników do wygenerowania raportu!")
        return

    print("\n" + "="*70)
    print("GENEROWANIE RAPORTU")
    print("="*70)

    # 1. Wykres słupkowy - Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    config_names = list(results.keys())
    accuracies = [results[c]['accuracy'] for c in config_names]
    times = [results[c]['avg_time_ms'] for c in config_names]

    # Wykres accuracy
    colors = ['#2ecc71' if acc >= 90 else '#f39c12' if acc >= 80 else '#e74c3c' for acc in accuracies]
    bars1 = ax1.barh(config_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Dokładność (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Porównanie dokładności modeli OCR', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', alpha=0.3)

    # Dodaj wartości na słupkach
    for bar, acc in zip(bars1, accuracies):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', ha='left', va='center', fontweight='bold')

    # Wykres czasu
    bars2 = ax2.barh(config_names, times, color='#3498db', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Średni czas (ms/słowo)', fontsize=12, fontweight='bold')
    ax2.set_title('Porównanie szybkości modeli OCR', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Dodaj wartości
    for bar, time_ms in zip(bars2, times):
        width = bar.get_width()
        ax2.text(width + max(times)*0.02, bar.get_y() + bar.get_height()/2,
                f'{time_ms:.0f} ms', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('benchmark_charts.png', dpi=150, bbox_inches='tight')
    print("✓ Wykresy zapisane do: benchmark_charts.png")
    plt.close()

    # 2. Generuj HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Benchmark Modeli OCR</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .best {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
        .good {{
            background-color: #fff3cd !important;
        }}
        .poor {{
            background-color: #f8d7da !important;
        }}
        .chart {{
            text-align: center;
            margin: 30px 0;
        }}
        .chart img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .details {{
            font-size: 0.9em;
            color: #666;
        }}
        .correct {{
            color: #27ae60;
            font-weight: bold;
        }}
        .incorrect {{
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>📊 Benchmark Modeli OCR - Raport</h1>

    <div class="summary">
        <h2>Podsumowanie</h2>
        <p><strong>Data testu:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Liczba słów:</strong> {results[config_names[0]]['total_words']}</p>
        <p><strong>Testowane konfiguracje:</strong> {len(results)}</p>
    </div>

    <div class="chart">
        <img src="benchmark_charts.png" alt="Wykresy porównawcze">
    </div>

    <h2>Wyniki szczegółowe</h2>
    <table>
        <tr>
            <th>Konfiguracja</th>
            <th>Dokładność</th>
            <th>Czas (ms/słowo)</th>
            <th>Poprawnych liter</th>
        </tr>
"""

    # Sortuj po dokładności
    sorted_configs = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    for i, (config_name, data) in enumerate(sorted_configs):
        acc = data['accuracy']
        css_class = 'best' if i == 0 else 'good' if acc >= 85 else 'poor' if acc < 75 else ''

        html += f"""
        <tr class="{css_class}">
            <td>{config_name}</td>
            <td>{acc:.1f}%</td>
            <td>{data['avg_time_ms']:.0f} ms</td>
            <td>{data['correct_letters']} / {data['total_letters']}</td>
        </tr>
"""

    html += """
    </table>

    <h2>Szczegóły per słowo</h2>
"""

    # Tabela per konfiguracja
    for config_name in config_names:
        data = results[config_name]
        html += f"""
    <h3>{config_name}</h3>
    <table>
        <tr>
            <th>Plik</th>
            <th>Groundtruth</th>
            <th>Rozpoznano</th>
            <th>Dokładność</th>
        </tr>
"""
        for detail in data['details']:
            correct_class = 'correct' if detail['accuracy'] == 1.0 else 'incorrect'
            html += f"""
        <tr>
            <td>{detail['file']}</td>
            <td><strong>{detail['groundtruth']}</strong></td>
            <td class="{correct_class}">{detail['recognized']}</td>
            <td>{detail['accuracy']*100:.0f}%</td>
        </tr>
"""
        html += """
    </table>
"""

    html += """
</body>
</html>
"""

    # Zapisz HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Raport HTML zapisany do: {output_file}")
    print(f"\n🌐 Otwórz w przeglądarce: file:///{os.path.abspath(output_file)}")


def main():
    """Główna funkcja."""
    print("\n" + "🚀 "*25)
    print("   BENCHMARK SYSTEM - Porównanie modeli OCR")
    print("🚀 "*25 + "\n")

    # Sprawdź czy katalog istnieje
    if not os.path.exists('./benchmark_data'):
        print("❌ Katalog benchmark_data/ nie istnieje!")
        print("   Tworzę katalog...")
        os.makedirs('./benchmark_data')
        print("   ✓ Katalog utworzony")
        print("\n📖 Przeczytaj instrukcję: benchmark_data/INSTRUKCJA.md")
        return

    # Uruchom benchmark
    results = run_benchmark('./benchmark_data')

    if results:
        # Generuj raport
        generate_html_report(results)

        print("\n" + "="*70)
        print("✓ BENCHMARK ZAKOŃCZONY POMYŚLNIE!")
        print("="*70)
        print("\n📊 Wyniki dostępne w:")
        print("   - benchmark_report.html (raport HTML)")
        print("   - benchmark_charts.png (wykresy)")
    else:
        print("\n❌ Benchmark nie został ukończony")


if __name__ == "__main__":
    main()
