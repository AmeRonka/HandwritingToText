"""
Moduł binaryzacji obrazów.
Zawiera funkcje do konwersji obrazów grayscale na binarne (czarno-białe).
"""

import numpy as np
from PIL import Image
from scipy import ndimage


def adaptive_threshold(image, block_size=15, c=10):
    """
    Binaryzacja adaptacyjna (lokalna).
    Dla każdego piksela próg jest obliczany na podstawie średniej z sąsiedztwa.

    Args:
        image: Obraz PIL w trybie 'L' (grayscale)
        block_size: Rozmiar okna do obliczania lokalnej średniej
        c: Stała odejmowana od lokalnej średniej

    Returns:
        Obraz PIL zbinaryzowany (0 = czarny, 255 = biały)
    """
    img_array = np.array(image, dtype=np.float32)

    # Rozmycie jako lokalna średnia
    # mode='reflect' - odbicie lustrzane, nie tworzy artefaktów na krawędziach
    blurred = ndimage.uniform_filter(img_array, size=block_size, mode='reflect')

    # Binaryzacja: piksel jest czarny jeśli jest ciemniejszy niż lokalna średnia - c
    binary = img_array < (blurred - c)

    # Konwertuj z powrotem: True (ciemny) -> 0 (czarny), False (jasny) -> 255 (biały)
    result_array = np.where(binary, 0, 255).astype(np.uint8)

    return Image.fromarray(result_array, mode='L')


def global_threshold(image, threshold=127):
    """
    Najprostsza globalna binaryzacja - stały próg dla całego obrazu.

    Args:
        image: Obraz PIL w trybie 'L' (grayscale)
        threshold: Próg binaryzacji (0-255)

    Returns:
        Obraz PIL zbinaryzowany (0 = czarny, 255 = biały)
    """
    img_array = np.array(image, dtype=np.uint8)
    binary = img_array < threshold
    result_array = np.where(binary, 0, 255).astype(np.uint8)
    return Image.fromarray(result_array, mode='L')


def sauvola_threshold(image, window_size=25, k=0.2, R=128):
    """
    Binaryzacja metodą Sauvola - standard dla dokumentów.
    Lepszy niż zwykła adaptacyjna dla tekstów o nierównej grubości i jakości.

    Algorytm:
        T(x,y) = m(x,y) * [1 + k * (s(x,y)/R - 1)]

    gdzie:
        - m(x,y) = lokalna średnia
        - s(x,y) = lokalne odchylenie standardowe
        - R = maksymalne odchylenie standardowe (128 dla obrazów 8-bit)
        - k = parametr kontrolujący wpływ odchylenia std (domyślnie 0.2)

    Piksel jest czarny jeśli jego wartość < T(x,y)

    Args:
        image: Obraz PIL w trybie 'L' (grayscale)
        window_size: Rozmiar okna do obliczania lokalnych statystyk (nieparzysty)
        k: Parametr Sauvola (0.2-0.5). Większy = więcej pikseli uznanych za tło
        R: Dynamiczny zakres odchylenia standardowego (domyślnie 128)

    Returns:
        Obraz PIL zbinaryzowany (0 = czarny, 255 = biały)
    """
    img_array = np.array(image, dtype=np.float64)

    # Oblicz lokalną średnią
    mean = ndimage.uniform_filter(img_array, size=window_size, mode='reflect')

    # Oblicz lokalne odchylenie standardowe
    # std = sqrt(E[X^2] - E[X]^2)
    mean_sq = ndimage.uniform_filter(img_array**2, size=window_size, mode='reflect')
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))  # maximum dla stabilności numerycznej

    # Oblicz próg Sauvola dla każdego piksela
    threshold = mean * (1.0 + k * (std / R - 1.0))

    # Binaryzacja: piksel < próg → czarny (tekst)
    binary = img_array < threshold

    # Konwertuj: True (tekst) -> 0 (czarny), False (tło) -> 255 (biały)
    result_array = np.where(binary, 0, 255).astype(np.uint8)

    return Image.fromarray(result_array, mode='L')


def remove_noise(image, min_size=15):
    """
    Usuwa małe komponenty (szum) - skupiska pikseli mniejsze niż min_size.

    Args:
        image: Obraz PIL zbinaryzowany
        min_size: Minimalna liczba pikseli w komponencie (mniejsze są usuwane)

    Returns:
        Obraz PIL z usuniętym szumem
    """
    img_array = np.array(image)

    # Inwertuj (tekst jako biały na czarnym tle dla label)
    binary = (img_array < 128).astype(np.uint8)

    # Znajdź połączone komponenty
    labeled, num_features = ndimage.label(binary)

    if num_features == 0:
        return image

    # Policz piksele w każdym komponencie
    component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))

    # Usuń małe komponenty
    for i, size in enumerate(component_sizes):
        if size < min_size:
            binary[labeled == (i + 1)] = 0

    # Konwertuj z powrotem: 1 (tekst) -> 0 (czarny), 0 (tło) -> 255 (biały)
    result_array = np.where(binary, 0, 255).astype(np.uint8)

    return Image.fromarray(result_array, mode='L')


def morphological_clean(image):
    """
    Operacje morfologiczne do wygładzenia tekstu.
    Opening (erozja + dylatacja) usuwa drobny szum.
    Closing (dylatacja + erozja) łączy drobne przerwy.

    Args:
        image: Obraz PIL zbinaryzowany

    Returns:
        Obraz PIL po operacjach morfologicznych
    """
    img_array = np.array(image)

    # Inwertuj (tekst jako biały)
    binary = (img_array < 128).astype(np.uint8)

    # Element strukturalny (mały krzyżyk)
    struct = ndimage.generate_binary_structure(2, 1)

    # Opening - usuwa małe wypustki
    opened = ndimage.binary_opening(binary, structure=struct, iterations=1)

    # Closing - łączy małe przerwy
    closed = ndimage.binary_closing(opened, structure=struct, iterations=1)

    # Konwertuj z powrotem
    result_array = np.where(closed, 0, 255).astype(np.uint8)

    return Image.fromarray(result_array, mode='L')


def normalize_letter(letter_img, target_size=(28, 28), padding_ratio=0.15):
    """
    Normalizuje literę do standardowego rozmiaru zachowując aspect ratio.
    Przygotowuje obraz do rozpoznawania przez modele ML (EMNIST format).

    Args:
        letter_img: Numpy array litery (inwertowany: tekst = białe wartości)
                   lub obraz PIL w trybie 'L'
        target_size: Docelowy rozmiar (wysokość, szerokość) - domyślnie 28x28
        padding_ratio: Padding wokół litery (0.15 = 15% z każdej strony)

    Returns:
        Numpy array znormalizowanego obrazu w formacie EMNIST:
        - Rozmiar: 28x28
        - Tekst: białe piksele (255) na czarnym tle (0)
        - Centrowany z paddingiem
    """
    # Konwertuj na PIL jeśli numpy array
    if isinstance(letter_img, np.ndarray):
        # Upewnij się że tekst to białe wartości
        if np.max(letter_img) <= 1:  # normalized [0, 1]
            pil_img = Image.fromarray((letter_img * 255).astype(np.uint8))
        else:
            pil_img = Image.fromarray(letter_img.astype(np.uint8))
    else:
        pil_img = letter_img.copy()

    # Usuń puste brzegi (crop do contentu)
    pil_img = _crop_to_content(pil_img)

    # Oblicz aspect ratio
    w, h = pil_img.size
    if w == 0 or h == 0:
        # Pusty obraz - zwróć czarne tło
        return np.zeros(target_size, dtype=np.uint8)

    aspect = w / h

    # Oblicz nowy rozmiar z paddingiem
    pad_pixels = int(target_size[0] * padding_ratio)
    max_size = target_size[0] - 2 * pad_pixels

    if max_size <= 0:
        max_size = target_size[0] - 4  # minimum 2px padding

    # Skaluj zachowując aspect ratio
    if aspect > 1:  # szeroka litera (W, M)
        new_w = max_size
        new_h = int(max_size / aspect)
    else:  # wysoka litera (I, J) lub kwadratowa
        new_h = max_size
        new_w = int(max_size * aspect)

    # Zapewnij minimum 1px
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # Resize z antyaliasowaniem dla lepszej jakości
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Wklej na czarne tło z centrowaniem (EMNIST format)
    result = Image.new('L', target_size, 0)  # czarne tło
    paste_x = (target_size[1] - new_w) // 2
    paste_y = (target_size[0] - new_h) // 2
    result.paste(resized, (paste_x, paste_y))

    return np.array(result)


def _crop_to_content(image):
    """
    Przycina obraz do obszaru zawierającego content (nie-czarne piksele).

    Args:
        image: Obraz PIL w trybie 'L'

    Returns:
        Obraz PIL przycięty do contentu
    """
    img_array = np.array(image)

    # Znajdź współrzędne nie-czarnych pikseli
    coords = np.column_stack(np.where(img_array > 10))

    if len(coords) == 0:
        # Brak contentu - zwróć oryginalny
        return image

    # Znajdź bbox
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop (dodaj 1 do max bo crop jest exclusive)
    return image.crop((x_min, y_min, x_max + 1, y_max + 1))
