"""
Handwriting to Text - Porównanie metod segmentacji:
1. Projekcja profilu
2. Connected Component Analysis (CCA)

GUI aplikacji do testowania różnych metod binaryzacji i segmentacji tekstu.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import os

# Import modułów projektu
import binarization
import segmentation
import recognition


class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentacja Liter - Porównanie Metod")
        self.root.resizable(False, False)

        # Rozmiar canvasa do rysowania
        self.canvas_width = 200
        self.canvas_height = 200

        # Grubość linii
        self.brush_size = 12

        # Ostatnia pozycja myszy
        self.last_x = None
        self.last_y = None

        # Wybrana metoda segmentacji
        self.segmentation_method = tk.StringVar(value="profile")

        # Źródło obrazu (rysowanie lub plik)
        self.image_source = tk.StringVar(value="draw")
        self.loaded_image = None
        self.photo_image = None

        # Metoda binaryzacji
        self.binarization_method = tk.StringVar(value="sauvola")

        # OCR - modele rozpoznawania
        self.ocr_model_name = tk.StringVar(value="tesseract")
        self.ocr_models = {}  # Dictionary modeli OCR
        self.current_recognized_text = ""  # Rozpoznany tekst

        # CCA - siła pogrubienia (dylatacji)
        self.cca_dilate = tk.IntVar(value=2)  # 0-3, domyślnie 2 (najlepsze wyniki)

        self._setup_ui()
        self._setup_drawing_image()
        self._load_ocr_models()  # Załaduj modele OCR

    def _setup_ui(self):
        """Tworzy interfejs użytkownika."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Instrukcja
        ttk.Label(
            main_frame,
            text="Narysuj tekst lub wczytaj obraz:",
            font=("Arial", 12)
        ).grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # Canvas do rysowania
        self.canvas = tk.Canvas(
            main_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            cursor="crosshair",
            relief="solid",
            borderwidth=1
        )
        self.canvas.grid(row=1, column=0, columnspan=3, pady=5)

        # Bindowanie myszy
        self.canvas.bind("<Button-1>", self._start_drawing)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drawing)

        # Ramka na wybór metody segmentacji
        method_frame = ttk.LabelFrame(main_frame, text="Metoda segmentacji", padding=5)
        method_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky="ew")

        ttk.Radiobutton(
            method_frame, text="Projekcja profilu",
            variable=self.segmentation_method, value="profile"
        ).grid(row=0, column=0, padx=10)

        ttk.Radiobutton(
            method_frame, text="Connected Component Analysis (CCA)",
            variable=self.segmentation_method, value="cca"
        ).grid(row=0, column=1, padx=10)

        ttk.Radiobutton(
            method_frame, text="Grid/Kratki (formularze)",
            variable=self.segmentation_method, value="grid"
        ).grid(row=0, column=2, padx=10)

        # Slider dla pogrubienia CCA
        ttk.Label(method_frame, text="Pogrubienie (dla CCA):").grid(row=1, column=0, padx=10, pady=(5,0), sticky="w")
        dilate_slider = ttk.Scale(
            method_frame,
            from_=0, to=3,
            orient=tk.HORIZONTAL,
            variable=self.cca_dilate,
            length=150
        )
        dilate_slider.grid(row=1, column=1, padx=10, pady=(5,0), sticky="w")
        ttk.Label(method_frame, textvariable=self.cca_dilate).grid(row=1, column=2, pady=(5,0), sticky="w")

        # Ramka na wybór metody binaryzacji
        bin_frame = ttk.LabelFrame(main_frame, text="Metoda binaryzacji (dla wczytanych obrazów)", padding=5)
        bin_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky="ew")

        ttk.Radiobutton(
            bin_frame, text="Sauvola (dokumenty)",
            variable=self.binarization_method, value="sauvola"
        ).grid(row=0, column=0, padx=10, sticky="w")

        ttk.Radiobutton(
            bin_frame, text="Adaptacyjna (lokalna)",
            variable=self.binarization_method, value="adaptive"
        ).grid(row=0, column=1, padx=10, sticky="w")

        ttk.Radiobutton(
            bin_frame, text="Globalna (próg 127)",
            variable=self.binarization_method, value="global"
        ).grid(row=0, column=2, padx=10, sticky="w")

        # Ramka na wybór modelu OCR
        ocr_frame = ttk.LabelFrame(main_frame, text="Model OCR do rozpoznawania", padding=5)
        ocr_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky="ew")

        ttk.Radiobutton(
            ocr_frame, text="Tesseract (single letter)",
            variable=self.ocr_model_name, value="tesseract"
        ).grid(row=0, column=0, padx=10, sticky="w")

        ttk.Radiobutton(
            ocr_frame, text="PyTorch CNN",
            variable=self.ocr_model_name, value="pytorch"
        ).grid(row=0, column=1, padx=10, sticky="w")

        ttk.Radiobutton(
            ocr_frame, text="Scikit-learn SVM/RF",
            variable=self.ocr_model_name, value="sklearn"
        ).grid(row=0, column=2, padx=10, sticky="w")

        ttk.Radiobutton(
            ocr_frame, text="Vision Transformer (ViT)",
            variable=self.ocr_model_name, value="vit"
        ).grid(row=1, column=0, padx=10, pady=(5,0), sticky="w")

        ttk.Radiobutton(
            ocr_frame, text="Tesseract (whole text)",
            variable=self.ocr_model_name, value="tesseract_whole"
        ).grid(row=1, column=1, columnspan=2, padx=10, pady=(5,0), sticky="w")

        # Dodaj notatkę o trybie whole text
        ttk.Label(
            ocr_frame,
            text="* Tryb 'whole text' rozpoznaje cały obraz bez segmentacji",
            font=("Arial", 8),
            foreground="gray"
        ).grid(row=2, column=0, columnspan=3, pady=(2,0))

        # Przyciski
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=10)

        ttk.Button(btn_frame, text="Wyczyść", command=self._clear_canvas).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Wczytaj obraz", command=self._load_image).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Segmentuj", command=self._segment_letters).grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="Rozpoznaj (OCR)", command=self._recognize_letters,
                  style="Accent.TButton").grid(row=0, column=3, padx=5)

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)

        # WIZUALIZACJA PROFILU/CCA - UKRYTA (zaoszczędza miejsce)
        # Etykieta wizualizacji
        # self.viz_label = ttk.Label(main_frame, text="Projekcja profilu:", font=("Arial", 10))
        # self.viz_label.grid(row=6, column=0, columnspan=3)

        # Canvas na wizualizację (profil lub komponenty)
        # self.viz_canvas = tk.Canvas(
        #     main_frame,
        #     width=self.canvas_width,
        #     height=80,
        #     bg="white",
        #     relief="solid",
        #     borderwidth=1
        # )
        # self.viz_canvas.grid(row=7, column=0, columnspan=3, pady=5)

        # Separator
        # ttk.Separator(main_frame, orient="horizontal").grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)

        # Tymczasowe zastąpienie - musimy mieć te zmienne żeby nie było błędu
        self.viz_label = ttk.Label(main_frame, text="")
        self.viz_canvas = tk.Canvas(main_frame, width=0, height=0)

        # SEKCJA WYSEGMENTOWANYCH LITER - małe miniatury 15x15
        # Etykieta wyników
        ttk.Label(main_frame, text="Wysegmentowane litery:", font=("Arial", 9)).grid(row=6, column=0, columnspan=3, sticky="w")

        # Info o liczbie liter
        self.result_label = ttk.Label(main_frame, text="(narysuj litery i kliknij Segmentuj)", font=("Arial", 8), foreground="gray")
        self.result_label.grid(row=6, column=0, columnspan=3, pady=2, sticky="e")

        # Ramka na wysegmentowane litery (miniatury 15x15)
        self.letters_frame = ttk.Frame(main_frame)
        self.letters_frame.grid(row=7, column=0, columnspan=3, pady=5, sticky="w")

        # Separator przed wynikiem OCR
        ttk.Separator(main_frame, orient="horizontal").grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)

        # Rozpoznany tekst
        ttk.Label(main_frame, text="Rozpoznany tekst (OCR):", font=("Arial", 10, "bold")).grid(row=9, column=0, columnspan=3)

        self.recognized_text_label = ttk.Label(
            main_frame,
            text="(wykonaj segmentację i kliknij 'Rozpoznaj (OCR)')",
            font=("Arial", 16),
            foreground="blue",
            wraplength=500,
            relief="solid",
            borderwidth=2,
            padding=15
        )
        self.recognized_text_label.grid(row=8, column=0, columnspan=3, pady=10)

    def _setup_drawing_image(self):
        """Tworzy obraz PIL do rysowania."""
        self.drawing_image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.drawing_image)
        self.segmented_letters = []  # Przechowuje wysegmentowane litery

    def _load_ocr_models(self):
        """Ładuje modele OCR (Tesseract, TensorFlow, PyTorch)."""
        print("\n" + "="*70)
        print("ŁADOWANIE MODELI OCR")
        print("="*70)

        # 1. Tesseract OCR
        tesseract_model = recognition.TesseractRecognizer()
        tesseract_model.load()
        self.ocr_models['tesseract'] = tesseract_model

        # 2. PyTorch CNN
        pytorch_path = './models/saved/pytorch_best.pth'
        if os.path.exists(pytorch_path):
            pytorch_model = recognition.PyTorchRecognizer(pytorch_path)
            pytorch_model.load()
            self.ocr_models['pytorch'] = pytorch_model
        else:
            print(f"[!] Model PyTorch nie znaleziony: {pytorch_path}")
            print("  Wytrenuj model używając: python train_pytorch.py")
            self.ocr_models['pytorch'] = None

        # 3. Scikit-learn SVM/Random Forest
        sklearn_svm_path = './models/saved/sklearn_svm.pkl'
        sklearn_rf_path = './models/saved/sklearn_rf.pkl'

        if os.path.exists(sklearn_svm_path):
            sklearn_model = recognition.SklearnRecognizer(sklearn_svm_path, model_type='svm')
            sklearn_model.load()
            self.ocr_models['sklearn'] = sklearn_model
        elif os.path.exists(sklearn_rf_path):
            sklearn_model = recognition.SklearnRecognizer(sklearn_rf_path, model_type='random_forest')
            sklearn_model.load()
            self.ocr_models['sklearn'] = sklearn_model
        else:
            print(f"[!] Model Scikit-learn nie znaleziony")
            print(f"  Sprawdzono: {sklearn_svm_path}")
            print(f"  Sprawdzono: {sklearn_rf_path}")
            print("  Wytrenuj model uzywajac: python train_sklearn.py")
            self.ocr_models['sklearn'] = None

        # 4. Vision Transformer (ViT)
        vit_path = './models/saved/vit_best.pth'
        if os.path.exists(vit_path):
            vit_model = recognition.ViTRecognizer(vit_path)
            vit_model.load()
            self.ocr_models['vit'] = vit_model
        else:
            print(f"[!] Model Vision Transformer nie znaleziony: {vit_path}")
            print("  Wytrenuj model uzywajac: python train_vit.py")
            self.ocr_models['vit'] = None

        # 5. Tesseract Whole Text
        tesseract_whole = recognition.TesseractWholeTextRecognizer()
        tesseract_whole.load()
        self.ocr_models['tesseract_whole'] = tesseract_whole

        print("="*70 + "\n")

    # ==================== RYSOWANIE ====================

    def _start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self._draw_point(event.x, event.y)

    def _draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill="black", width=self.brush_size, capstyle=tk.ROUND, smooth=True
            )
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=0, width=self.brush_size
            )
        self.last_x = event.x
        self.last_y = event.y

    def _draw_point(self, x, y):
        r = self.brush_size // 2
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def _stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    def _clear_canvas(self):
        self.canvas.delete("all")
        # self.viz_canvas.delete("all")  # UKRYTE - wizualizacja wyłączona
        self._setup_drawing_image()
        self.loaded_image = None
        self.photo_image = None
        self.image_source.set("draw")
        self.result_label.config(text="(narysuj litery lub wczytaj obraz)")
        self._clear_letters_display()
        self.segmented_letters = []
        self.current_recognized_text = ""
        self.recognized_text_label.config(
            text="(wykonaj segmentację i kliknij 'Rozpoznaj (OCR)')",
            foreground="blue"
        )

    def _clear_letters_display(self):
        """Usuwa wyświetlone litery."""
        for widget in self.letters_frame.winfo_children():
            widget.destroy()

    # ==================== WCZYTYWANIE OBRAZU ====================

    def _load_image(self):
        """Otwiera dialog wyboru pliku i wczytuje obraz."""
        filetypes = [
            ("Obrazy", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("Wszystkie pliki", "*.*")
        ]

        filepath = filedialog.askopenfilename(
            title="Wybierz obraz z tekstem",
            filetypes=filetypes
        )

        if not filepath:
            return

        try:
            image = Image.open(filepath)
            processed = self._preprocess_image(image)
            self.loaded_image = processed
            self.image_source.set("file")
            self._display_loaded_image(processed)
            self.result_label.config(text=f"Wczytano: {filepath.split('/')[-1]}")
        except Exception as e:
            self.result_label.config(text=f"Błąd wczytywania: {str(e)}")

    def _preprocess_image(self, image):
        """
        Preprocessing obrazu ze skanu/zdjęcia.
        Binaryzacja wykonywana jest w PEŁNEJ rozdzielczości,
        dopiero potem obraz jest skalowany do wyświetlenia.
        """
        # 1. Konwersja do grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image.copy()

        # 2. Przytnij krawędzie (często są ciemne)
        gray = self._crop_dark_edges(gray, margin_percent=0.02)

        # 3. Zwiększ kontrast (w pełnej rozdzielczości)
        gray = ImageOps.autocontrast(gray)

        # 4. Binaryzacja W PEŁNEJ ROZDZIELCZOŚCI
        method = self.binarization_method.get()
        if method == "global":
            binary = binarization.global_threshold(gray, threshold=127)
        elif method == "sauvola":
            binary = binarization.sauvola_threshold(gray, window_size=25, k=0.2)
        else:  # adaptive
            binary = binarization.adaptive_threshold(gray)

        # 5. Usuń szum (w pełnej rozdzielczości - większy min_size)
        orig_w, orig_h = binary.size
        # Skaluj min_size proporcjonalnie do rozdzielczości
        scale_factor = max(orig_w, orig_h) / max(self.canvas_width, self.canvas_height)
        min_noise_size = int(20 * scale_factor * scale_factor)  # Skaluj kwadratowo (piksele to powierzchnia)
        binary = binarization.remove_noise(binary, min_size=max(20, min_noise_size))

        # 6. Operacje morfologiczne (w pełnej rozdzielczości)
        binary = binarization.morphological_clean(binary)

        # 7. Teraz skaluj do rozmiaru okienka
        target_w, target_h = self.canvas_width, self.canvas_height
        margin = 15

        scale = min((target_w - 2*margin) / orig_w, (target_h - 2*margin) / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Użyj NEAREST dla obrazu binarnego (zachowuje ostre krawędzie)
        resized = binary.resize((new_w, new_h), Image.Resampling.NEAREST)

        # 8. Wklej na białe tło
        result = Image.new('L', (target_w, target_h), 255)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        result.paste(resized, (paste_x, paste_y))

        return result

    def _crop_dark_edges(self, image, margin_percent=0.02):
        """Przycina ciemne krawędzie z oryginalnego obrazu."""
        w, h = image.size
        margin_x = int(w * margin_percent)
        margin_y = int(h * margin_percent)

        if margin_x > 0 or margin_y > 0:
            return image.crop((margin_x, margin_y, w - margin_x, h - margin_y))
        return image

    def _display_loaded_image(self, image):
        """Wyświetla wczytany obraz na canvasie."""
        self.canvas.delete("all")
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.drawing_image = image.copy()

    # ==================== SEGMENTACJA ====================

    def _segment_letters(self):
        """Główna funkcja segmentacji - wybiera metodę."""
        method = self.segmentation_method.get()

        if method == "profile":
            self._segment_by_profile()
        elif method == "cca":
            self._segment_by_cca()
        elif method == "grid":
            self._segment_by_grid()
        else:
            self._segment_by_cca()

    def _segment_by_profile(self):
        """Segmentacja metodą projekcji profilu."""
        # self.viz_label.config(text="Projekcja profilu (suma pikseli w kolumnach):")

        # Konwertuj obraz (inwertuj: czarne pismo -> białe wartości)
        img_array = 255 - np.array(self.drawing_image)

        # Użyj modułu segmentation
        profile, boundaries = segmentation.segment_by_profile(img_array)

        if len(boundaries) < 2:
            self.result_label.config(text="Nie wykryto liter!")
            self.segmented_letters = []
            return

        # Wyświetl wizualizację (UKRYTE - zaoszczędza miejsce)
        # self._draw_profile(profile, boundaries)

        # Wytnij litery
        letters = segmentation.extract_letters_profile(img_array, boundaries, profile)

        # Zapisz litery
        self.segmented_letters = letters

        # Wyświetl litery
        self._display_letters(letters, "Projekcja profilu")

    def _segment_by_cca(self):
        """Segmentacja metodą CCA."""
        # self.viz_label.config(text="CCA - Wykryte komponenty (każdy kolor = osobny komponent):")

        # Konwertuj obraz
        img_array = 255 - np.array(self.drawing_image)

        # Użyj modułu segmentation z dylatacją (pogrubienie)
        dilate_iterations = self.cca_dilate.get()
        labeled_array, num_components, merged_components = segmentation.segment_by_cca(
            img_array,
            dilate_iterations=dilate_iterations
        )

        if num_components == 0:
            self.result_label.config(text="Nie wykryto liter!")
            self.segmented_letters = []
            return

        # Wyświetl wizualizację (UKRYTE - zaoszczędza miejsce)
        # self._draw_cca_visualization(labeled_array, num_components, merged_components)

        # Wytnij litery
        letters = segmentation.extract_letters_cca(img_array, merged_components)

        # Zapisz litery
        self.segmented_letters = letters

        # Wyświetl litery
        self._display_letters(letters, "CCA", num_components)

    def _segment_by_grid(self):
        """Segmentacja metodą Grid/Kratki dla formularzy."""
        # Konwertuj obraz
        img_array = 255 - np.array(self.drawing_image)

        # Wykryj kratki
        grid_boxes, num_boxes = segmentation.segment_by_grid(img_array)

        if num_boxes == 0:
            self.result_label.config(text="Nie wykryto kratek! Spróbuj innych parametrów lub metody.")
            self.segmented_letters = []
            return

        # Wytnij zawartość kratek
        letters = segmentation.extract_letters_grid(img_array, grid_boxes)

        if len(letters) == 0:
            self.result_label.config(text=f"Znaleziono {num_boxes} kratek, ale są puste!")
            self.segmented_letters = []
            return

        # Zapisz litery
        self.segmented_letters = letters

        # Wyświetl litery
        self._display_letters(letters, "Grid/Kratki", num_boxes)

    # ==================== WIZUALIZACJA ====================

    def _draw_profile(self, profile, boundaries):
        """Rysuje wykres projekcji profilu z zaznaczonymi granicami."""
        self.viz_canvas.delete("all")

        canvas_height = 80
        max_val = np.max(profile) if np.max(profile) > 0 else 1

        # Rysuj gradient tła
        for x in range(len(profile)):
            intensity = int((profile[x] / max_val) * 255)
            color = f"#{255-intensity:02x}{255-intensity:02x}ff"
            self.viz_canvas.create_line(x, 0, x, canvas_height, fill=color)

        # Rysuj wykres profilu (linia)
        points = []
        for x in range(len(profile)):
            y = canvas_height - (profile[x] / max_val) * (canvas_height - 10)
            points.append((x, y))

        if len(points) > 1:
            for i in range(len(points) - 1):
                self.viz_canvas.create_line(
                    points[i][0], points[i][1],
                    points[i+1][0], points[i+1][1],
                    fill="darkblue", width=2
                )

        # Rysuj granice
        for b in boundaries:
            self.viz_canvas.create_line(b, 0, b, canvas_height, fill="red", width=2, dash=(4, 2))

        # Legenda
        self.viz_canvas.create_text(5, 5, anchor="nw", text="Profil", fill="darkblue", font=("Arial", 8))
        self.viz_canvas.create_text(5, 18, anchor="nw", text="Granice", fill="red", font=("Arial", 8))

    def _draw_cca_visualization(self, labeled_array, num_components, merged_components):
        """Rysuje wizualizację komponentów CCA."""
        self.viz_canvas.delete("all")

        canvas_height = 80
        scale_y = canvas_height / self.canvas_height

        # Generuj kolory
        colors = self._generate_colors(num_components)

        # Rysuj każdy piksel z kolorem jego komponentu
        for y in range(labeled_array.shape[0]):
            for x in range(labeled_array.shape[1]):
                label = labeled_array[y, x]
                if label > 0:
                    color = colors[label - 1]
                    self.viz_canvas.create_rectangle(
                        x, int(y * scale_y),
                        x + 1, int((y + 1) * scale_y),
                        fill=color, outline=""
                    )

        # Rysuj bounding boxy
        for comp in merged_components:
            self.viz_canvas.create_rectangle(
                comp['x_min'], int(comp['y_min'] * scale_y),
                comp['x_max'], int(comp['y_max'] * scale_y),
                outline="red", width=2
            )

        # Legenda
        self.viz_canvas.create_text(5, 5, anchor="nw", text=f"Komponenty: {num_components}", fill="black", font=("Arial", 8))
        self.viz_canvas.create_text(5, 18, anchor="nw", text=f"Po scaleniu: {len(merged_components)}", fill="red", font=("Arial", 8))

    def _generate_colors(self, n):
        """Generuje n różnych kolorów."""
        colors = []
        for i in range(n):
            hue = i / n
            r, g, b = self._hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
        return colors

    def _hsv_to_rgb(self, h, s, v):
        """Konwertuje HSV do RGB."""
        if s == 0:
            return v, v, v

        i = int(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        i = i % 6

        if i == 0: return v, t, p
        if i == 1: return q, v, p
        if i == 2: return p, v, t
        if i == 3: return p, q, v
        if i == 4: return t, p, v
        if i == 5: return v, p, q

    # ==================== ROZPOZNAWANIE OCR ====================

    def _recognize_letters(self):
        """Rozpoznaje wysegmentowane litery lub cały tekst (zależnie od modelu)."""

        # Wybierz model
        model_name = self.ocr_model_name.get()
        model = self.ocr_models.get(model_name)

        if model is None or not model.is_loaded:
            messagebox.showerror(
                "Model niedostępny",
                f"Model {model_name} nie jest dostępny.\n\n"
                f"Dla modeli PyTorch/Sklearn:\n"
                f"Najpierw wytrenuj model używając:\n"
                f"  python train_{model_name}.py\n\n"
                f"Dla Tesseract:\n"
                f"Zainstaluj Tesseract OCR"
            )
            return

        # ========== SPECIAL MODE: TESSERACT WHOLE TEXT ==========
        if model_name == 'tesseract_whole':
            print(f"\n🔍 Rozpoznawanie całego tekstu (bez segmentacji): {model.name}")

            # self.drawing_image is already a PIL Image (created in _setup_drawing_image)
            # Recognize full image
            recognized_text, confidence = model.recognize_full_image(self.drawing_image)

            # Display result
            self.current_recognized_text = recognized_text
            self.recognized_text_label.config(
                text=f'"{recognized_text}"',
                foreground="darkgreen" if recognized_text != "?" else "red"
            )

            # Update info
            self.result_label.config(
                text=f"[{model.name}] Rozpoznano cały tekst (pewność: {confidence:.3f})"
            )

            print(f"\n✓ Rozpoznany tekst: {recognized_text}")
            print(f"  Pewność: {confidence:.3f}\n")
            return

        # ========== STANDARD MODE: LETTER-BY-LETTER RECOGNITION ==========
        # Sprawdź czy są wysegmentowane litery
        if len(self.segmented_letters) == 0:
            messagebox.showwarning(
                "Brak liter",
                "Najpierw wykonaj segmentację (kliknij 'Segmentuj')!"
            )
            return

        print(f"\n🔍 Rozpoznawanie {len(self.segmented_letters)} liter przy użyciu: {model.name}")

        # Normalizuj i rozpoznaj każdą literę
        recognized_letters = []
        confidences = []

        for i, (letter_img, letter_profile) in enumerate(self.segmented_letters):
            # Normalizuj do 28x28
            normalized = binarization.normalize_letter(letter_img, target_size=(28, 28))

            # Rozpoznaj
            letter, confidence = model.predict(normalized)
            recognized_letters.append(letter)
            confidences.append(confidence)

            print(f"  Litera {i+1}: {letter} (pewność: {confidence:.3f})")

        # Stwórz tekst
        recognized_text = "".join(recognized_letters)
        avg_confidence = np.mean(confidences)

        # Wyświetl wynik
        self.current_recognized_text = recognized_text
        self.recognized_text_label.config(
            text=f'"{recognized_text}"',
            foreground="darkgreen"
        )

        # Zaktualizuj info
        self.result_label.config(
            text=f"[{model.name}] Rozpoznano: {len(recognized_letters)} liter(y) "
                 f"(średnia pewność: {avg_confidence:.3f})"
        )

        print(f"\n✓ Rozpoznany tekst: {recognized_text}")
        print(f"  Średnia pewność: {avg_confidence:.3f}\n")

    # ==================== WYŚWIETLANIE LITER ====================

    def _display_letters(self, letters, method_name, num_components=None):
        """Wyświetla wysegmentowane litery."""
        self._clear_letters_display()

        for i, (letter_img, letter_profile) in enumerate(letters, 1):
            self._display_single_letter(letter_img, i, letter_profile)

        if num_components is not None:
            self.result_label.config(text=f"[{method_name}] Znaleziono {len(letters)} liter(y) ({num_components} komponentów)")
        else:
            self.result_label.config(text=f"[{method_name}] Znaleziono {len(letters)} liter(y)")

    def _display_single_letter(self, letter_img, letter_number, letter_profile=None):
        """Wyświetla pojedynczą literę jako mały obrazek 15x15 pikseli."""
        letter_frame = ttk.Frame(self.letters_frame, relief="solid", borderwidth=1)
        letter_frame.pack(side="left", padx=2, pady=2)

        # Canvas 15x15 pikseli
        letter_canvas = tk.Canvas(letter_frame, width=15, height=15, bg="black", highlightthickness=0)
        letter_canvas.pack()

        # Przeskaluj literę do 15x15 z zachowaniem aspect ratio
        from PIL import Image
        letter_h, letter_w = letter_img.shape

        # Konwertuj numpy array na PIL Image
        pil_img = Image.fromarray(letter_img.astype(np.uint8))

        # Oblicz nowy rozmiar z zachowaniem aspect ratio
        aspect = letter_w / letter_h if letter_h > 0 else 1
        if aspect > 1:  # szeroka litera
            new_w = 15
            new_h = max(1, int(15 / aspect))
        else:  # wysoka litera
            new_h = 15
            new_w = max(1, int(15 * aspect))

        # Resize
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Centruj na canvas 15x15
        offset_x = (15 - new_w) // 2
        offset_y = (15 - new_h) // 2

        # Rysuj piksel po pikselu
        resized_array = np.array(resized)
        for y in range(new_h):
            for x in range(new_w):
                val = resized_array[y, x]
                if val > 10:
                    gray = int(val)
                    color = f"#{gray:02x}{gray:02x}{gray:02x}"
                    letter_canvas.create_rectangle(
                        offset_x + x, offset_y + y,
                        offset_x + x + 1, offset_y + y + 1,
                        fill=color, outline=""
                    )


def main():
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
