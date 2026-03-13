"""
Moduł rozpoznawania liter (OCR).
Zawiera implementacje trzech różnych metod:
1. Tesseract OCR (gotowe rozwiązanie)
2. TensorFlow/Keras CNN (własny model)
3. PyTorch CNN (własny model)
"""

import time
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict


# ==================== KLASA BAZOWA ====================

class RecognitionModel(ABC):
    """
    Abstrakcyjna klasa bazowa dla modeli rozpoznawania liter.
    Każdy model musi implementować metodę predict().
    """

    def __init__(self, name: str):
        self.name = name
        self.is_loaded = False

    @abstractmethod
    def load(self):
        """Ładuje model (wagi, konfigurację, itp.)"""
        pass

    @abstractmethod
    def predict(self, letter_img: np.ndarray) -> Tuple[str, float]:
        """
        Rozpoznaje pojedynczą literę.

        Args:
            letter_img: Numpy array 28x28 (znormalizowany obraz litery)

        Returns:
            Tuple (predicted_letter, confidence):
                - predicted_letter: Rozpoznana litera (A-Z)
                - confidence: Pewność predykcji (0.0 - 1.0)
        """
        pass

    def predict_batch(self, letters: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Rozpoznaje wiele liter naraz.

        Args:
            letters: Lista obrazów liter (każdy 28x28)

        Returns:
            Lista krotek (predicted_letter, confidence)
        """
        results = []
        for letter in letters:
            results.append(self.predict(letter))
        return results

    def predict_with_time(self, letter_img: np.ndarray) -> Tuple[str, float, float]:
        """
        Rozpoznaje literę i mierzy czas.

        Returns:
            Tuple (predicted_letter, confidence, time_ms)
        """
        start = time.perf_counter()
        letter, confidence = self.predict(letter_img)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        return letter, confidence, elapsed


# ==================== TESSERACT OCR ====================

class TesseractRecognizer(RecognitionModel):
    """
    Rozpoznawanie liter przy użyciu Tesseract OCR.
    """

    def __init__(self):
        super().__init__("Tesseract OCR")
        self.tesseract_available = False

    def load(self):
        """Sprawdza dostępność Tesseract."""
        try:
            import pytesseract
            from PIL import Image

            # Ustaw ścieżkę do Tesseract (Windows)
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            # Test czy Tesseract jest zainstalowany
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            self.is_loaded = True
            print(f"[OK] {self.name} załadowany pomyślnie")
        except Exception as e:
            print(f"[X] {self.name} niedostępny: {e}")
            print("  Zainstaluj Tesseract OCR:")
            print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("  Linux: sudo apt-get install tesseract-ocr")
            print("  Mac: brew install tesseract")
            self.is_loaded = False

    def predict(self, letter_img: np.ndarray) -> Tuple[str, float]:
        """
        Rozpoznaje literę przy użyciu Tesseract.

        Args:
            letter_img: Numpy array 28x28

        Returns:
            Tuple (letter, confidence)
        """
        if not self.is_loaded:
            return "?", 0.0

        import pytesseract
        from PIL import Image

        # Konwertuj na PIL Image
        pil_img = Image.fromarray(letter_img.astype(np.uint8))

        # Powiększ obraz (Tesseract działa lepiej na większych obrazach)
        scale = 4
        pil_img = pil_img.resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)

        # Konfiguracja Tesseract:
        # --oem 3: OCR Engine Mode 3 (Default, based on what is available)
        # --psm 10: Page Segmentation Mode 10 (Treat the image as a single character)
        # -c tessedit_char_whitelist: Ogranicz do wielkich liter A-Z
        custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        try:
            # Rozpoznaj literę
            result = pytesseract.image_to_string(pil_img, config=custom_config)
            letter = result.strip().upper()

            # Jeśli puste lub nieprawidłowe - zwróć ?
            if not letter or len(letter) == 0:
                return "?", 0.0

            # Weź pierwszą literę (czasami Tesseract zwraca więcej)
            letter = letter[0]

            # Tesseract nie zwraca confidence dla pojedynczych znaków w prosty sposób
            # Dlatego zwracamy stałą wartość 0.8 jeśli rozpoznał, 0.0 jeśli nie
            confidence = 0.8 if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" else 0.0

            return letter, confidence

        except Exception as e:
            print(f"Tesseract error: {e}")
            return "?", 0.0

    def predict_text(self, img) -> str:
        """
        Rozpoznaje caly tekst z obrazu (bez segmentacji na pojedyncze litery).

        Args:
            img: PIL Image lub numpy array z tekstem

        Returns:
            Rozpoznany tekst (wielkie litery)
        """
        if not self.is_loaded:
            return "?"

        import pytesseract
        from PIL import Image

        # Konwertuj na PIL Image jesli trzeba
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img.astype(np.uint8))
        else:
            pil_img = img

        # Konfiguracja Tesseract dla calego tekstu:
        # --oem 3: Default OCR Engine
        # --psm 7: Treat the image as a single text line
        # -c tessedit_char_whitelist: Ogranicz do wielkich liter A-Z i spacji
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        try:
            result = pytesseract.image_to_string(pil_img, config=custom_config)
            text = result.strip().upper()
            # Usun znaki ktore nie sa literami
            text = ''.join(c for c in text if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            return text if text else "?"
        except Exception as e:
            print(f"Tesseract error: {e}")
            return "?"


# ==================== PYTORCH CNN ====================

class PyTorchRecognizer(RecognitionModel):
    """
    Rozpoznawanie liter przy użyciu własnego modelu CNN w PyTorch.
    """

    def __init__(self, model_path: str = None):
        super().__init__("PyTorch CNN")
        self.model_path = model_path
        self.model = None
        self.device = None
        self.class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z

    def load(self):
        """Ładuje wytrenowany model z pliku."""
        try:
            import torch
            from models.pytorch_model import LetterCNN  # Będziemy tworzyć

            if self.model_path is None:
                print(f"[X] {self.name}: Brak ścieżki do modelu")
                print("  Najpierw wytrenuj model używając train_pytorch.py")
                self.is_loaded = False
                return

            # Ustaw device (GPU jeśli dostępne)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Załaduj model
            self.model = LetterCNN(num_classes=26)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()  # Tryb ewaluacji

            self.is_loaded = True
            print(f"[OK] {self.name} załadowany z: {self.model_path} (device: {self.device})")

        except Exception as e:
            print(f"[X] {self.name} nie mógł zostać załadowany: {e}")
            self.is_loaded = False

    def predict(self, letter_img: np.ndarray) -> Tuple[str, float]:
        """
        Rozpoznaje literę przy użyciu modelu PyTorch.

        Args:
            letter_img: Numpy array 28x28

        Returns:
            Tuple (letter, confidence)
        """
        if not self.is_loaded:
            return "?", 0.0

        try:
            import torch

            # Normalizuj obraz do [0, 1]
            img_normalized = letter_img.astype(np.float32) / 255.0

            # Konwertuj na PyTorch tensor: (1, 1, 28, 28)
            img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            # Predykcja (bez gradientów)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

            # Konwertuj na Python types
            predicted_class = predicted_class.item()
            confidence = confidence.item()

            # Konwertuj class index na literę
            letter = self.class_names[predicted_class]

            return letter, confidence

        except Exception as e:
            print(f"PyTorch prediction error: {e}")
            return "?", 0.0


# ==================== VISION TRANSFORMER (ViT) ====================

class ViTRecognizer(RecognitionModel):
    """
    Rozpoznawanie liter przy użyciu Vision Transformer (ViT).

    Vision Transformer to nowoczesna architektura używająca mechanizmu
    Self-Attention zamiast konwolucji. Każdy patch obrazu może "patrzeć"
    na wszystkie inne patche jednocześnie.
    """

    def __init__(self, model_path: str = None):
        super().__init__("Vision Transformer (ViT)")
        self.model_path = model_path
        self.model = None
        self.device = None
        self.class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z

    def load(self):
        """Ładuje wytrenowany model ViT z pliku."""
        try:
            import torch
            from models.vit_model import VisionTransformer

            if self.model_path is None:
                print(f"[X] {self.name}: Brak sciezki do modelu")
                print("  Najpierw wytrenuj model uzywajac train_vit.py")
                self.is_loaded = False
                return

            # Ustaw device (GPU jeśli dostępne)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Załaduj checkpoint z konfiguracją
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Odczytaj konfigurację modelu
            config = checkpoint.get('config', {})
            embed_dim = config.get('embed_dim', 64)
            num_heads = config.get('num_heads', 4)
            num_layers = config.get('num_layers', 4)

            # Stwórz model z odpowiednią konfiguracją
            self.model = VisionTransformer(
                img_size=28,
                patch_size=7,
                in_channels=1,
                num_classes=26,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers
            )

            # Załaduj wagi
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()  # Tryb ewaluacji

            self.is_loaded = True
            print(f"[OK] {self.name} zaladowany z: {self.model_path} (device: {self.device})")

        except Exception as e:
            print(f"[X] {self.name} nie mogl zostac zaladowany: {e}")
            self.is_loaded = False

    def predict(self, letter_img: np.ndarray) -> Tuple[str, float]:
        """
        Rozpoznaje literę przy użyciu Vision Transformer.

        Args:
            letter_img: Numpy array 28x28

        Returns:
            Tuple (letter, confidence)
        """
        if not self.is_loaded:
            return "?", 0.0

        try:
            import torch

            # Normalizuj obraz do [0, 1]
            img_normalized = letter_img.astype(np.float32) / 255.0

            # Konwertuj na PyTorch tensor: (1, 1, 28, 28)
            img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            # Predykcja (bez gradientów)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

            # Konwertuj na Python types
            predicted_class = predicted_class.item()
            confidence = confidence.item()

            # Konwertuj class index na literę
            letter = self.class_names[predicted_class]

            return letter, confidence

        except Exception as e:
            print(f"ViT prediction error: {e}")
            return "?", 0.0


# ==================== SCIKIT-LEARN SVM/RANDOM FOREST ====================

class SklearnRecognizer(RecognitionModel):
    """
    Rozpoznawanie liter przy użyciu Scikit-learn (SVM lub Random Forest).
    """

    def __init__(self, model_path: str = None, model_type: str = 'svm'):
        """
        Args:
            model_path: Path to saved .pkl model
            model_type: 'svm' or 'random_forest' (for display name)
        """
        name = f"Scikit-learn {model_type.upper()}"
        super().__init__(name)
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z

    def load(self):
        """Ładuje wytrenowany model z pliku .pkl"""
        try:
            from models.sklearn_model import SklearnLetterClassifier

            if self.model_path is None:
                print(f"[X] {self.name}: Brak ścieżki do modelu")
                print("  Najpierw wytrenuj model używając train_sklearn.py")
                self.is_loaded = False
                return

            # Load model using joblib (through SklearnLetterClassifier.load)
            self.model = SklearnLetterClassifier.load(self.model_path)
            self.is_loaded = True
            print(f"[OK] {self.name} załadowany z: {self.model_path}")

        except Exception as e:
            print(f"[X] {self.name} nie mógł zostać załadowany: {e}")
            print(f"  Sprawdź czy plik istnieje: {self.model_path}")
            print(f"  Wytrenuj model używając: python train_sklearn.py")
            self.is_loaded = False

    def predict(self, letter_img: np.ndarray) -> Tuple[str, float]:
        """
        Rozpoznaje literę przy użyciu modelu sklearn.

        Args:
            letter_img: Numpy array 28x28

        Returns:
            Tuple (letter, confidence)
        """
        if not self.is_loaded:
            return "?", 0.0

        try:
            # Normalize to [0, 1]
            img_normalized = letter_img.astype(np.float32) / 255.0

            # Predict (model handles single image)
            predicted_class, probabilities = self.model.predict(img_normalized)

            # Get confidence for predicted class
            confidence = float(probabilities[predicted_class])

            # Convert class index to letter
            letter = self.class_names[predicted_class]

            return letter, confidence

        except Exception as e:
            print(f"Sklearn prediction error: {e}")
            return "?", 0.0


# ==================== TESSERACT WHOLE TEXT ====================

class TesseractWholeTextRecognizer(RecognitionModel):
    """
    Rozpoznawanie całego tekstu przy użyciu Tesseract OCR.
    Przetwarza cały obraz jako tekst bez segmentacji na pojedyncze litery.
    """

    def __init__(self):
        super().__init__("Tesseract Whole Text")
        self.tesseract_available = False

    def load(self):
        """Sprawdza dostępność Tesseract."""
        try:
            import pytesseract
            from PIL import Image

            # Ustaw ścieżkę do Tesseract (Windows)
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            # Test czy Tesseract jest zainstalowany
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            self.is_loaded = True
            print(f"[OK] {self.name} załadowany pomyślnie")
        except Exception as e:
            print(f"[X] {self.name} niedostępny: {e}")
            print("  Zainstaluj Tesseract OCR:")
            print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            self.is_loaded = False

    def recognize_full_image(self, image_pil) -> Tuple[str, float]:
        """
        Rozpoznaje cały tekst z obrazu (bez segmentacji).

        Args:
            image_pil: PIL Image (grayscale, binarized)
                      Format: czarny tekst (0) na białym tle (255)

        Returns:
            Tuple (recognized_text, average_confidence):
                - recognized_text: Rozpoznany tekst (string)
                - average_confidence: Średnia pewność (0.0 - 1.0)
        """
        if not self.is_loaded:
            return "?", 0.0

        import pytesseract
        from PIL import Image

        try:
            # Powiększ obraz (Tesseract działa lepiej na większych obrazach)
            # Minimalna wysokość: 100px dla dobrej jakości OCR
            scale = max(3, int(100 / image_pil.height) + 1)
            width, height = image_pil.size
            upscaled = image_pil.resize(
                (width * scale, height * scale),
                Image.Resampling.NEAREST  # NEAREST dla binarnych obrazów
            )

            # Konfiguracja Tesseract dla całego tekstu:
            # --oem 3: OCR Engine Mode 3 (Default)
            # --psm 7: Page Segmentation Mode 7 (single text line)
            #          Use PSM 6 for multiple lines, PSM 13 for raw line
            # -c tessedit_char_whitelist: Ogranicz do wielkich liter A-Z
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'

            # Pobierz szczegółowe dane z confidence
            data = pytesseract.image_to_data(
                upscaled,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )

            # Zbierz tekst i confidence
            text_parts = []
            confidences = []

            for i, conf in enumerate(data['conf']):
                # Confidence > 0 oznacza prawidłowe rozpoznanie
                if int(conf) > 0:
                    text = data['text'][i].strip()
                    if text:  # Pomijamy puste stringi
                        text_parts.append(text)
                        confidences.append(int(conf) / 100.0)  # Konwertuj 0-100 na 0.0-1.0

            # Złącz tekst
            if text_parts:
                recognized_text = "".join(text_parts).upper()  # Bez spacji między literami
                avg_confidence = np.mean(confidences)
            else:
                # Fallback: użyj prostego image_to_string
                recognized_text = pytesseract.image_to_string(
                    upscaled,
                    config=custom_config
                ).strip().upper()
                avg_confidence = 0.5  # Nie mamy confidence, założmy średnią

            # Usuń whitespace
            recognized_text = recognized_text.replace(" ", "").replace("\n", "")

            return recognized_text, avg_confidence

        except Exception as e:
            print(f"Tesseract whole text error: {e}")
            import traceback
            traceback.print_exc()
            return "?", 0.0

    def predict(self, letter_img: np.ndarray) -> Tuple[str, float]:
        """
        NOT USED for whole text mode.

        This method is required by RecognitionModel interface,
        but whole text mode uses recognize_full_image() instead.
        """
        return "?", 0.0


# ==================== PORÓWNYWANIE MODELI ====================

def compare_models(models: List[RecognitionModel], letters: List[np.ndarray]) -> Dict:
    """
    Porównuje wydajność kilku modeli na tym samym zestawie liter.

    Args:
        models: Lista modeli do porównania
        letters: Lista obrazów liter (każdy 28x28)

    Returns:
        Słownik z wynikami porównania
    """
    results = {
        'models': {},
        'letters_count': len(letters)
    }

    for model in models:
        if not model.is_loaded:
            print(f"⚠ Pomijam {model.name} - model nie został załadowany")
            continue

        print(f"\n🔍 Testuję {model.name}...")

        predictions = []
        confidences = []
        times = []

        for i, letter in enumerate(letters):
            pred_letter, confidence, elapsed = model.predict_with_time(letter)
            predictions.append(pred_letter)
            confidences.append(confidence)
            times.append(elapsed)

        # Agreguj wyniki
        results['models'][model.name] = {
            'predictions': predictions,
            'avg_confidence': float(np.mean(confidences)),
            'avg_time_ms': float(np.mean(times)),
            'total_time_ms': float(np.sum(times))
        }

        print(f"  Średnia pewność: {np.mean(confidences):.3f}")
        print(f"  Średni czas: {np.mean(times):.2f}ms")
        print(f"  Całkowity czas: {np.sum(times):.2f}ms")

    return results


def print_comparison_table(results: Dict):
    """
    Wyświetla tabelę porównawczą wyników.

    Args:
        results: Słownik wyników z compare_models()
    """
    print("\n" + "="*70)
    print("PORÓWNANIE MODELI OCR")
    print("="*70)
    print(f"Liczba liter: {results['letters_count']}")
    print("\n{:<25} {:>15} {:>15} {:>15}".format(
        "Model", "Śr. pewność", "Śr. czas [ms]", "Całk. czas [ms]"
    ))
    print("-"*70)

    for model_name, data in results['models'].items():
        print("{:<25} {:>15.3f} {:>15.2f} {:>15.2f}".format(
            model_name,
            data['avg_confidence'],
            data['avg_time_ms'],
            data['total_time_ms']
        ))

    print("="*70)


# ==================== HELPER: BATCH RECOGNITION ====================

def recognize_letters(letters: List[np.ndarray], model: RecognitionModel) -> List[str]:
    """
    Rozpoznaje listę liter i zwraca tekst.

    Args:
        letters: Lista znormalizowanych obrazów liter (28x28)
        model: Model do rozpoznawania

    Returns:
        Lista rozpoznanych liter (lub "?" jeśli nie rozpoznano)
    """
    if not model.is_loaded:
        print(f"⚠ Model {model.name} nie jest załadowany")
        return ["?" for _ in letters]

    results = model.predict_batch(letters)
    return [letter for letter, confidence in results]


def recognize_to_text(letters: List[np.ndarray], model: RecognitionModel) -> str:
    """
    Rozpoznaje listę liter i zwraca jako string.

    Args:
        letters: Lista znormalizowanych obrazów liter (28x28)
        model: Model do rozpoznawania

    Returns:
        String z rozpoznanym tekstem
    """
    recognized = recognize_letters(letters, model)
    return "".join(recognized)
