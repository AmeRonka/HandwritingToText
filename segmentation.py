"""
Moduł segmentacji tekstu.
Zawiera algorytmy do dzielenia obrazu na poszczególne litery.
"""

import numpy as np
from scipy import ndimage
import cv2


# ==================== METODA 1: PROJEKCJA PROFILU ====================

def segment_by_profile(img_array):
    """
    Segmentacja metodą projekcji profilu.
    Sumuje piksele w kolumnach i znajduje granice liter tam gdzie suma spada.

    Args:
        img_array: Numpy array obrazu (inwertowany: tekst = białe wartości)

    Returns:
        tuple: (profile, boundaries)
            - profile: Tablica sum pikseli w kolumnach
            - boundaries: Lista indeksów granic liter
    """
    # Oblicz projekcję profilu (suma pikseli w każdej kolumnie)
    profile = np.sum(img_array, axis=0)

    # Znajdź granice liter
    boundaries = find_letter_boundaries_profile(profile)

    return profile, boundaries


def find_letter_boundaries_profile(profile, threshold_ratio=0.05):
    """
    Znajduje granice między literami na podstawie profilu.

    Args:
        profile: Tablica sum pikseli w kolumnach
        threshold_ratio: Stosunek do maksymalnej wartości (poniżej = przerwa)

    Returns:
        Lista indeksów granic (naprzemiennie: początek, koniec litery)
    """
    max_val = np.max(profile)
    threshold = max_val * threshold_ratio

    above_threshold = profile > threshold

    boundaries = []
    in_letter = False
    for i, val in enumerate(above_threshold):
        if val and not in_letter:
            boundaries.append(i)
            in_letter = True
        elif not val and in_letter:
            boundaries.append(i)
            in_letter = False

    if in_letter:
        boundaries.append(len(profile))

    return boundaries


def extract_letters_profile(img_array, boundaries, profile):
    """
    Wycina poszczególne litery na podstawie granic z profilu.

    Args:
        img_array: Numpy array obrazu
        boundaries: Lista granic liter
        profile: Profil projekcji

    Returns:
        Lista krotek (letter_img, letter_profile) dla każdej litery
    """
    # Znajdź wiersze z treścią
    rows_with_content = np.any(img_array > 10, axis=1)
    if not np.any(rows_with_content):
        return []

    row_start = np.argmax(rows_with_content)
    row_end = len(rows_with_content) - np.argmax(rows_with_content[::-1])

    letters = []
    for i in range(0, len(boundaries) - 1, 2):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(profile)

        letter_img = img_array[row_start:row_end, start:end]
        letter_profile = profile[start:end]

        if letter_img.size > 0:
            letters.append((letter_img, letter_profile))

    return letters


# ==================== METODA 2: CCA (Connected Component Analysis) ====================

def segment_by_cca(img_array, threshold=30, dilate_iterations=0):
    """
    Segmentacja metoda Connected Component Analysis.
    Znajduje polaczone grupy pikseli (komponenty) i traktuje je jako litery.

    Args:
        img_array: Numpy array obrazu (inwertowany: tekst = biale wartosci)
        threshold: Prog binaryzacji dla wykrywania komponentow
        dilate_iterations: Liczba iteracji dylatacji (pogrubienia).
                          0 = bez dylatacji (domyslnie)
                          1-2 = lekkie pogrubienie (laczy male przerwy)
                          3-5 = srednie pogrubienie (laczy wieksze przerwy)
                          Wartosci niecalkowite (np. 1.5) sa zaokraglane w gore.

    Returns:
        tuple: (labeled_array, num_components, merged_components)
            - labeled_array: Tablica z etykietami komponentow
            - num_components: Liczba znalezionych komponentow
            - merged_components: Lista scalonych komponentow (slowniki z bbox)
    """
    binary = (img_array > threshold).astype(np.uint8)

    # OPCJONALNIE: Dylatacja (pogrubienie) - laczy przerwy w literach
    if dilate_iterations > 0:
        # Konwersja na int (zaokraglenie w gore dla wartosci niecalkowitych)
        iterations = int(np.ceil(dilate_iterations))
        # Element strukturalny (maly krzyzyk 3x3)
        struct = ndimage.generate_binary_structure(2, 1)
        # Dylatacja - "pogrubia" biale piksele (tekst)
        binary = ndimage.binary_dilation(binary, structure=struct, iterations=iterations).astype(np.uint8)

    # Znajdź połączone komponenty
    labeled_array, num_components = ndimage.label(binary)

    if num_components == 0:
        return labeled_array, 0, []

    # Pobierz bounding boxy komponentów
    components = get_component_bboxes(labeled_array, num_components)

    # Scal bliskie komponenty (dla liter jak i, j, ł, ó)
    merged_components = merge_close_components(components)

    # Posortuj od lewej do prawej
    merged_components.sort(key=lambda c: c['x_min'])

    return labeled_array, num_components, merged_components


def get_component_bboxes(labeled_array, num_components):
    """
    Pobiera bounding boxy dla każdego komponentu.

    Args:
        labeled_array: Tablica z etykietami komponentów
        num_components: Liczba komponentów

    Returns:
        Lista słowników z informacjami o komponentach
    """
    components = []

    for i in range(1, num_components + 1):
        # Znajdź piksele należące do tego komponentu
        coords = np.where(labeled_array == i)

        if len(coords[0]) == 0:
            continue

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        components.append({
            'id': i,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'pixel_count': len(coords[0])
        })

    return components


def merge_close_components(components, x_threshold=15, size_ratio_threshold=0.3):
    """
    Scala komponenty tylko gdy jeden jest znacznie mniejszy (kropka w i, j, ł, ó).
    Nie scala dwóch podobnej wielkości komponentów (pochylone litery).

    Args:
        components: Lista komponentów
        x_threshold: Maksymalna odległość w X do scalenia
        size_ratio_threshold: Maksymalny stosunek wielkości (mniejszy/większy)

    Returns:
        Lista scalonych komponentów
    """
    if len(components) <= 1:
        return components

    # Sortuj po x_min
    sorted_comps = sorted(components, key=lambda c: c['x_min'])

    merged = []
    current = sorted_comps[0].copy()

    for next_comp in sorted_comps[1:]:
        # Sprawdź czy komponenty nachodzą na siebie w poziomie lub są blisko
        x_overlap = (current['x_min'] <= next_comp['x_max'] and
                    next_comp['x_min'] <= current['x_max'])
        x_close = (next_comp['x_min'] - current['x_max']) < x_threshold

        # Sprawdź stosunek wielkości - scalaj tylko gdy jeden jest znacznie mniejszy
        smaller_pixels = min(current['pixel_count'], next_comp['pixel_count'])
        larger_pixels = max(current['pixel_count'], next_comp['pixel_count'])
        size_ratio = smaller_pixels / larger_pixels if larger_pixels > 0 else 1

        # Scalaj tylko gdy:
        # 1. Są blisko w X (nachodzą lub < threshold)
        # 2. Jeden komponent jest znacznie mniejszy (< 30% drugiego) - to kropka/akcent
        is_small_component = size_ratio < size_ratio_threshold

        if (x_overlap or x_close) and is_small_component:
            # Scal komponenty (mały z dużym)
            current['x_min'] = min(current['x_min'], next_comp['x_min'])
            current['x_max'] = max(current['x_max'], next_comp['x_max'])
            current['y_min'] = min(current['y_min'], next_comp['y_min'])
            current['y_max'] = max(current['y_max'], next_comp['y_max'])
            current['pixel_count'] += next_comp['pixel_count']
        else:
            # Zapisz obecny i zacznij nowy
            merged.append(current)
            current = next_comp.copy()

    merged.append(current)
    return merged


def extract_letters_cca(img_array, components, padding=2):
    """
    Wycina poszczególne litery na podstawie komponentów CCA.

    Args:
        img_array: Numpy array obrazu
        components: Lista scalonych komponentów
        padding: Margines wokół litery

    Returns:
        Lista krotek (letter_img, letter_profile) dla każdej litery
    """
    letters = []

    for comp in components:
        # Wytnij literę z paddingiem
        y_min = max(0, comp['y_min'] - padding)
        y_max = min(img_array.shape[0], comp['y_max'] + padding)
        x_min = max(0, comp['x_min'] - padding)
        x_max = min(img_array.shape[1], comp['x_max'] + padding)

        letter_img = img_array[y_min:y_max, x_min:x_max]

        if letter_img.size > 0:
            # Oblicz profil dla tej litery
            letter_profile = np.sum(letter_img, axis=0)
            letters.append((letter_img, letter_profile))

    return letters


# ==================== METODA 3: GRID/KRATKI (dla formularzy) ====================

def merge_close_lines(lines, threshold=10):
    """
    Scala linie które są blisko siebie (duplikaty z Hough Transform).

    Args:
        lines: Lista pozycji linii (współrzędne X lub Y)
        threshold: Maksymalna odległość do scalenia

    Returns:
        Lista unikalnych linii (uśrednione pozycje)
    """
    if len(lines) == 0:
        return []

    lines = sorted(lines)
    merged = [lines[0]]

    for line in lines[1:]:
        if line - merged[-1] < threshold:
            # Scal - zastąp średnią
            merged[-1] = (merged[-1] + line) // 2
        else:
            # Nowa linia
            merged.append(line)

    return merged


def segment_by_grid(img_array, min_line_length=20, line_gap=5, merge_threshold=10):
    """
    Segmentacja dla formularzy z kratkami używając Hough Line Transform.
    Działa dla STYKAJĄCYCH SIĘ kratek (wspólne krawędzie).

    Algorytm:
    1. Wykryj krawędzie (Canny)
    2. Znajdź linie pionowe i poziome (Hough Transform)
    3. Scal bliskie linie (duplikaty)
    4. Zrekonstruuj siatkę kratek z przecięć linii

    Args:
        img_array: Numpy array obrazu (inwertowany: tekst = białe wartości)
        min_line_length: Minimalna długość linii do wykrycia (px)
        line_gap: Maksymalna przerwa w linii (px)
        merge_threshold: Odległość do scalenia duplikatów linii (px)

    Returns:
        tuple: (grid_boxes, num_boxes)
            - grid_boxes: Lista słowników z informacjami o kratkach
            - num_boxes: Liczba znalezionych kratek
    """
    # Konwertuj do uint8
    binary = (img_array > 30).astype(np.uint8) * 255

    # Wykryj krawędzie (Canny edge detection)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Wykryj linie (Hough Line Transform - probabilistic)
    # Obniżony threshold dla małych obrazów
    lines = cv2.HoughLinesP(
        edges,
        rho=1,                    # Rozdzielczość w pikselach
        theta=np.pi/180,          # Rozdzielczość kąta (1 stopień)
        threshold=20,             # Minimalna liczba przecięć (obniżone z 50)
        minLineLength=min_line_length,
        maxLineGap=line_gap
    )

    if lines is None or len(lines) == 0:
        return [], 0

    # Rozdziel linie na pionowe i poziome
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Oblicz kąt linii
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            angle = 90
        else:
            angle = abs(np.arctan(dy / dx) * 180 / np.pi)

        # Klasyfikuj jako pionowa lub pozioma
        if angle > 80:  # Pionowa (~90°)
            x_avg = (x1 + x2) // 2
            vertical_lines.append(x_avg)
        elif angle < 10:  # Pozioma (~0°)
            y_avg = (y1 + y2) // 2
            horizontal_lines.append(y_avg)

    # Scal bliskie linie (usuń duplikaty)
    vertical_lines = merge_close_lines(vertical_lines, merge_threshold)
    horizontal_lines = merge_close_lines(horizontal_lines, merge_threshold)

    # Muszą być przynajmniej 2 linie w każdym kierunku (min. 1 kratka)
    if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
        return [], 0

    # Posortuj linie
    vertical_lines.sort()
    horizontal_lines.sort()

    # Zrekonstruuj kratki z siatki przecięć
    grid_boxes = []

    for i in range(len(vertical_lines) - 1):
        for j in range(len(horizontal_lines) - 1):
            x = vertical_lines[i]
            y = horizontal_lines[j]
            w = vertical_lines[i + 1] - x
            h = horizontal_lines[j + 1] - y

            # Filtruj zbyt małe kratki (szum)
            if w < 10 or h < 10:
                continue

            grid_boxes.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': w * h
            })

    # Sortuj od lewej do prawej, potem od góry do dołu
    grid_boxes.sort(key=lambda box: (box['y'], box['x']))

    return grid_boxes, len(grid_boxes)


def extract_letters_grid(img_array, grid_boxes, crop_margin=0.15):
    """
    Wycina litery z kratek formularza.

    Args:
        img_array: Numpy array obrazu
        grid_boxes: Lista kratek wykrytych przez segment_by_grid()
        crop_margin: Ile % brzegów kratki usunąć (unika linii kratek)

    Returns:
        Lista krotek (letter_img, letter_profile) dla każdej kratki
    """
    letters = []

    for box in grid_boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']

        # Oblicz margines (usuwa brzegi kratki - linie)
        margin_x = int(w * crop_margin)
        margin_y = int(h * crop_margin)

        # Wytnij środek kratki (unikamy linii)
        x_start = max(0, x + margin_x)
        x_end = min(img_array.shape[1], x + w - margin_x)
        y_start = max(0, y + margin_y)
        y_end = min(img_array.shape[0], y + h - margin_y)

        # Sprawdź czy po przycięciu coś zostało
        if x_end <= x_start or y_end <= y_start:
            continue

        letter_img = img_array[y_start:y_end, x_start:x_end]

        if letter_img.size > 0:
            # Oblicz profil
            letter_profile = np.sum(letter_img, axis=0)
            letters.append((letter_img, letter_profile))

    return letters
