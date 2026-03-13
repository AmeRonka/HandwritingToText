"""
Aplikacja produkcyjna do OCR dokumentów.

Funkcjonalności:
- Wczytywanie skanów dokumentów
- Zaznaczanie obszaru ROI (Region of Interest) do przetwarzania
- ROI zapisuje się i jest używany dla kolejnych dokumentów
- Stałe ustawienia: Sauvola + CCA (dilation=2)
- Wybór modelu OCR (PyTorch CNN, Vision Transformer, SVM, Tesseract)

Uruchomienie:
    python app_production.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import json

# Importy z projektu
import binarization
import segmentation
import recognition


class ProductionApp:
    """Aplikacja produkcyjna do OCR dokumentów ze stałym ROI."""

    # Plik konfiguracyjny do zapisywania ROI
    CONFIG_FILE = "roi_config.json"

    def __init__(self, root):
        self.root = root
        self.root.title("OCR Dokumentow - Tryb Produkcyjny")
        self.root.geometry("1200x800")

        # Stan aplikacji
        self.current_image = None  # Oryginalny obraz PIL
        self.current_image_path = None
        self.display_image = None  # Obraz do wyświetlenia (przeskalowany)
        self.photo_image = None  # ImageTk dla canvas

        # ROI (Region of Interest) - LISTA wielu obszarów
        self.roi_list = []  # Lista krotek (x1, y1, x2, y2) w pikselach oryginalnego obrazu
        self.roi_rect_ids = []  # Lista ID prostokątów na canvas
        self.drawing_roi = False
        self.roi_start = None

        # Skala wyświetlania (obraz może być większy niż canvas)
        self.display_scale = 1.0

        # Modele OCR
        self.ocr_models = {}
        self.current_model_name = tk.StringVar(value="vit")

        # Buduj GUI
        self._build_gui()

        # Załaduj modele OCR
        self._load_ocr_models()

        # Wczytaj zapisane ROI (jeśli istnieje)
        self._load_roi_config()

    def _build_gui(self):
        """Buduje interfejs użytkownika."""
        # Główna ramka
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === LEWY PANEL: Obraz i ROI ===
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Nagłówek
        ttk.Label(left_frame, text="Podglad dokumentu", font=("Arial", 12, "bold")).pack(pady=5)

        # Canvas na obraz
        canvas_frame = ttk.Frame(left_frame, relief="solid", borderwidth=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bindy do zaznaczania ROI
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # Przyciski pod obrazem
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Wczytaj dokument", command=self._load_document).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Usun ostatni ROI", command=self._remove_last_roi).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Usun wszystkie ROI", command=self._clear_all_roi).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Zapisz ROI", command=self._save_roi_config).pack(side=tk.LEFT, padx=5)

        # Informacja o ROI
        self.roi_label = ttk.Label(left_frame, text="ROI: 0 obszarow (zaznacz myszka)", font=("Arial", 9))
        self.roi_label.pack(pady=2)

        # === PRAWY PANEL: Ustawienia i wyniki ===
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)

        # Nagłówek
        ttk.Label(right_frame, text="Ustawienia OCR", font=("Arial", 12, "bold")).pack(pady=10)

        # Wybór modelu OCR
        model_frame = ttk.LabelFrame(right_frame, text="Model OCR", padding=10)
        model_frame.pack(fill=tk.X, pady=5)

        models = [
            ("Vision Transformer (ViT)", "vit"),
            ("PyTorch CNN", "pytorch"),
            ("Tesseract OCR (caly ROI)", "tesseract"),
        ]

        for text, value in models:
            ttk.Radiobutton(
                model_frame, text=text,
                variable=self.current_model_name, value=value
            ).pack(anchor=tk.W, pady=2)

        # Informacja o algorytmach
        info_frame = ttk.LabelFrame(right_frame, text="Algorytmy (stale)", padding=10)
        info_frame.pack(fill=tk.X, pady=5)

        ttk.Label(info_frame, text="Binaryzacja: Sauvola", font=("Arial", 9)).pack(anchor=tk.W)
        ttk.Label(info_frame, text="Segmentacja: CCA (ViT/CNN)", font=("Arial", 9)).pack(anchor=tk.W)
        ttk.Label(info_frame, text="Tesseract: bez segmentacji", font=("Arial", 9)).pack(anchor=tk.W)

        # Przycisk przetwarzania
        ttk.Button(
            right_frame, text="ROZPOZNAJ TEKST",
            command=self._process_document,
            style="Accent.TButton"
        ).pack(fill=tk.X, pady=20)

        # Wynik OCR
        result_frame = ttk.LabelFrame(right_frame, text="Rozpoznany tekst", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_text = tk.Text(result_frame, height=10, font=("Courier", 14), wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Przycisk kopiowania
        ttk.Button(right_frame, text="Kopiuj do schowka", command=self._copy_to_clipboard).pack(fill=tk.X, pady=5)

        # Podgląd segmentacji (małe miniatury)
        seg_frame = ttk.LabelFrame(right_frame, text="Segmentacja (podglad)", padding=5)
        seg_frame.pack(fill=tk.X, pady=5)

        self.segments_canvas = tk.Canvas(seg_frame, height=50, bg="white")
        self.segments_canvas.pack(fill=tk.X)

        # Status bar
        self.status_label = ttk.Label(right_frame, text="Gotowy", font=("Arial", 9), foreground="gray")
        self.status_label.pack(pady=5)

    def _load_ocr_models(self):
        """Ładuje dostępne modele OCR."""
        self._set_status("Ladowanie modeli OCR...")

        # 1. Vision Transformer
        vit_path = './models/saved/vit_best.pth'
        if os.path.exists(vit_path):
            try:
                vit_model = recognition.ViTRecognizer(vit_path)
                vit_model.load()
                self.ocr_models['vit'] = vit_model
            except Exception as e:
                print(f"Blad ladowania ViT: {e}")

        # 2. PyTorch CNN
        pytorch_path = './models/saved/pytorch_best.pth'
        if os.path.exists(pytorch_path):
            try:
                pytorch_model = recognition.PyTorchRecognizer(pytorch_path)
                pytorch_model.load()
                self.ocr_models['pytorch'] = pytorch_model
            except Exception as e:
                print(f"Blad ladowania PyTorch: {e}")

        # 3. Tesseract (dziala na calym ROI, bez segmentacji)
        try:
            tesseract_model = recognition.TesseractRecognizer()
            tesseract_model.load()
            self.ocr_models['tesseract'] = tesseract_model
        except Exception as e:
            print(f"Blad ladowania Tesseract: {e}")

        # Ustaw domyslny model
        if 'vit' not in self.ocr_models:
            if 'pytorch' in self.ocr_models:
                self.current_model_name.set('pytorch')
            else:
                self.current_model_name.set('tesseract')

        self._set_status(f"Zaladowano {len(self.ocr_models)} modeli OCR")

    def _load_document(self):
        """Wczytuje dokument (obraz)."""
        filetypes = [
            ("Obrazy", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("Wszystkie pliki", "*.*")
        ]

        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if not filepath:
            return

        try:
            self.current_image = Image.open(filepath).convert('L')  # Grayscale
            self.current_image_path = filepath

            self._display_image()
            self._set_status(f"Wczytano: {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Blad", f"Nie mozna wczytac obrazu:\n{e}")

    def _display_image(self):
        """Wyświetla obraz na canvas z dopasowaniem rozmiaru."""
        if self.current_image is None:
            return

        # Pobierz rozmiar canvas
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width, canvas_height = 800, 600

        # Oblicz skalę
        img_width, img_height = self.current_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.display_scale = min(scale_x, scale_y, 1.0)  # Nie powiększaj ponad oryginał

        # Przeskaluj obraz
        new_width = int(img_width * self.display_scale)
        new_height = int(img_height * self.display_scale)

        self.display_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Konwertuj na PhotoImage
        self.photo_image = ImageTk.PhotoImage(self.display_image)

        # Wyświetl na canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        # Narysuj ROI jeśli istnieje
        self._draw_roi()

    def _on_mouse_down(self, event):
        """Rozpoczyna zaznaczanie nowego ROI."""
        if self.current_image is None:
            return

        self.drawing_roi = True
        self.roi_start = (event.x, event.y)
        self._temp_rect_id = None  # Tymczasowy prostokąt podczas rysowania

    def _on_mouse_drag(self, event):
        """Rysuje prostokąt podczas przeciągania."""
        if not self.drawing_roi or self.roi_start is None:
            return

        # Usuń tymczasowy prostokąt
        if hasattr(self, '_temp_rect_id') and self._temp_rect_id:
            self.canvas.delete(self._temp_rect_id)

        # Narysuj nowy tymczasowy
        x1, y1 = self.roi_start
        x2, y2 = event.x, event.y

        self._temp_rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="red", width=2, dash=(5, 5)
        )

    def _on_mouse_up(self, event):
        """Kończy zaznaczanie ROI i dodaje do listy."""
        if not self.drawing_roi or self.roi_start is None:
            return

        self.drawing_roi = False

        # Usuń tymczasowy prostokąt
        if hasattr(self, '_temp_rect_id') and self._temp_rect_id:
            self.canvas.delete(self._temp_rect_id)
            self._temp_rect_id = None

        x1, y1 = self.roi_start
        x2, y2 = event.x, event.y

        # Upewnij się że x1 < x2 i y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Przelicz na współrzędne oryginalnego obrazu
        orig_x1 = int(x1 / self.display_scale)
        orig_y1 = int(y1 / self.display_scale)
        orig_x2 = int(x2 / self.display_scale)
        orig_y2 = int(y2 / self.display_scale)

        # Ogranicz do rozmiaru obrazu
        img_w, img_h = self.current_image.size
        orig_x1 = max(0, min(orig_x1, img_w))
        orig_y1 = max(0, min(orig_y1, img_h))
        orig_x2 = max(0, min(orig_x2, img_w))
        orig_y2 = max(0, min(orig_y2, img_h))

        # Sprawdź minimalny rozmiar
        if orig_x2 - orig_x1 < 10 or orig_y2 - orig_y1 < 10:
            self._set_status("ROI zbyt maly - zignorowany")
            return

        # DODAJ do listy ROI
        new_roi = (orig_x1, orig_y1, orig_x2, orig_y2)
        self.roi_list.append(new_roi)

        # Przerysuj wszystkie ROI
        self._draw_all_roi()

        # Aktualizuj etykietę
        self._update_roi_label()

        self._set_status(f"Dodano ROI #{len(self.roi_list)}: {orig_x2-orig_x1}x{orig_y2-orig_y1} px")

    def _draw_all_roi(self):
        """Rysuje wszystkie ROI na canvas z numerami."""
        if self.current_image is None:
            return

        # Usuń stare prostokąty
        for rect_id in self.roi_rect_ids:
            self.canvas.delete(rect_id)
        self.roi_rect_ids = []

        # Kolory dla różnych ROI
        colors = ["green", "blue", "purple", "orange", "cyan", "magenta"]

        # Narysuj każdy ROI
        for i, roi in enumerate(self.roi_list):
            x1 = int(roi[0] * self.display_scale)
            y1 = int(roi[1] * self.display_scale)
            x2 = int(roi[2] * self.display_scale)
            y2 = int(roi[3] * self.display_scale)

            color = colors[i % len(colors)]

            # Prostokąt
            rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=3
            )
            self.roi_rect_ids.append(rect_id)

            # Numer ROI
            text_id = self.canvas.create_text(
                x1 + 5, y1 + 5,
                text=str(i + 1),
                anchor=tk.NW,
                fill=color,
                font=("Arial", 14, "bold")
            )
            self.roi_rect_ids.append(text_id)

    def _draw_roi(self):
        """Rysuje wszystkie ROI (alias dla kompatybilności)."""
        self._draw_all_roi()

    def _remove_last_roi(self):
        """Usuwa ostatni ROI z listy."""
        if len(self.roi_list) == 0:
            self._set_status("Brak ROI do usuniecia")
            return

        self.roi_list.pop()
        self._draw_all_roi()
        self._update_roi_label()
        self._set_status(f"Usunieto ostatni ROI. Pozostalo: {len(self.roi_list)}")

    def _clear_all_roi(self):
        """Czyści wszystkie ROI."""
        self.roi_list = []
        for rect_id in self.roi_rect_ids:
            self.canvas.delete(rect_id)
        self.roi_rect_ids = []
        self._update_roi_label()
        self._set_status("Wszystkie ROI usuniete")

    def _update_roi_label(self):
        """Aktualizuje etykietę ROI."""
        count = len(self.roi_list)
        if count == 0:
            self.roi_label.config(text="ROI: 0 obszarow (zaznacz myszka)")
        elif count == 1:
            self.roi_label.config(text=f"ROI: 1 obszar")
        else:
            self.roi_label.config(text=f"ROI: {count} obszarow")

    def _save_roi_config(self):
        """Zapisuje listę ROI do pliku konfiguracyjnego."""
        if len(self.roi_list) == 0:
            messagebox.showinfo("Info", "Najpierw zaznacz ROI na dokumencie.")
            return

        config = {
            'roi_list': self.roi_list  # Lista krotek (x1, y1, x2, y2)
        }

        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            self._set_status(f"ROI zapisany do {self.CONFIG_FILE}")
            messagebox.showinfo("Sukces", f"Zapisano {len(self.roi_list)} obszarow ROI.\nBeda uzyte dla kolejnych dokumentow.")
        except Exception as e:
            messagebox.showerror("Blad", f"Nie mozna zapisac ROI:\n{e}")

    def _load_roi_config(self):
        """Wczytuje listę ROI z pliku konfiguracyjnego."""
        if not os.path.exists(self.CONFIG_FILE):
            return

        try:
            with open(self.CONFIG_FILE, 'r') as f:
                config = json.load(f)

            # Nowy format - lista ROI
            if 'roi_list' in config:
                self.roi_list = [tuple(roi) for roi in config['roi_list']]
                self._update_roi_label()
                self._set_status(f"Wczytano {len(self.roi_list)} obszarow ROI z konfiguracji")
            # Kompatybilność wsteczna - stary format z pojedynczym ROI
            elif 'roi' in config:
                self.roi_list = [tuple(config['roi'])]
                self._update_roi_label()
                self._set_status("ROI wczytany z konfiguracji (stary format)")
        except Exception as e:
            print(f"Blad wczytywania ROI: {e}")

    def _process_document(self):
        """Przetwarza dokument: binaryzacja -> segmentacja -> OCR dla wszystkich ROI."""
        if self.current_image is None:
            messagebox.showinfo("Info", "Najpierw wczytaj dokument.")
            return

        if len(self.roi_list) == 0:
            messagebox.showinfo("Info", "Najpierw zaznacz obszar ROI na dokumencie.")
            return

        model_name = self.current_model_name.get()
        if model_name not in self.ocr_models:
            messagebox.showerror("Blad", f"Model '{model_name}' nie jest dostepny.")
            return

        model = self.ocr_models[model_name]

        self._set_status("Przetwarzanie...")
        self.root.update()

        # Lista wyników dla każdego ROI
        all_results = []
        all_letters_for_display = []
        total_letters = 0

        try:
            # Przetwarzaj każdy ROI osobno
            for roi_idx, roi in enumerate(self.roi_list):
                self._set_status(f"Przetwarzanie ROI {roi_idx + 1}/{len(self.roi_list)}...")
                self.root.update()

                # 1. Wytnij ROI
                x1, y1, x2, y2 = roi
                roi_image = self.current_image.crop((x1, y1, x2, y2))

                # 2. Binaryzacja Sauvola
                binary_image = binarization.sauvola_threshold(roi_image)

                # TESSERACT: rozpoznaje caly ROI bez segmentacji
                if model_name == 'tesseract':
                    recognized_text = model.predict_text(binary_image)
                    all_results.append(recognized_text)
                    continue

                # ViT/CNN: segmentacja na pojedyncze litery
                # 3. Konwersja do numpy (inwersja: tekst = bialy)
                img_array = 255 - np.array(binary_image)

                # 4. Segmentacja CCA z dilation=1
                labeled_array, num_components, merged_components = segmentation.segment_by_cca(
                    img_array, dilate_iterations=1
                )

                if num_components == 0:
                    all_results.append(f"(ROI {roi_idx + 1}: brak liter)")
                    continue

                # 5. Wytnij litery
                letters = segmentation.extract_letters_cca(img_array, merged_components)
                total_letters += len(letters)

                # Zachowaj litery do podgladu (tylko pierwsze kilka z kazdego ROI)
                all_letters_for_display.extend(letters[:5])

                # 6. Rozpoznaj kazda litere
                recognized_text = ""
                for letter_img, _ in letters:
                    # Przeskaluj do 28x28 (format EMNIST)
                    letter_resized = binarization.normalize_letter(letter_img, target_size=(28, 28))

                    # OCR
                    letter, confidence = model.predict(letter_resized)
                    recognized_text += letter

                all_results.append(recognized_text)

            # 7. Wyświetl podgląd segmentacji (reprezentatywne litery)
            self._display_segments(all_letters_for_display)

            # 8. Wyświetl wyniki - każdy ROI w osobnej linii
            self.result_text.delete(1.0, tk.END)
            for i, result in enumerate(all_results):
                if i > 0:
                    self.result_text.insert(tk.END, "\n")
                self.result_text.insert(tk.END, result)

            self._set_status(f"Rozpoznano {total_letters} liter z {len(self.roi_list)} obszarow")

        except Exception as e:
            messagebox.showerror("Blad", f"Blad przetwarzania:\n{e}")
            self._set_status("Blad przetwarzania")
            import traceback
            traceback.print_exc()

    def _display_segments(self, letters):
        """Wyświetla małe podglądy segmentacji."""
        self.segments_canvas.delete("all")

        x_offset = 5
        for i, (letter_img, _) in enumerate(letters):
            if x_offset > 300:  # Limit szerokości
                break

            # Przeskaluj do 20x20
            from PIL import Image as PILImage
            h, w = letter_img.shape
            aspect = w / h if h > 0 else 1

            if aspect > 1:
                new_w = 20
                new_h = max(1, int(20 / aspect))
            else:
                new_h = 20
                new_w = max(1, int(20 * aspect))

            pil_img = PILImage.fromarray(letter_img.astype(np.uint8))
            pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

            # Rysuj na canvas
            for y in range(new_h):
                for x in range(new_w):
                    pixel = pil_img.getpixel((x, y))
                    if pixel > 30:
                        gray = int(pixel)
                        color = f"#{gray:02x}{gray:02x}{gray:02x}"
                        self.segments_canvas.create_rectangle(
                            x_offset + x, 5 + y,
                            x_offset + x + 1, 5 + y + 1,
                            fill=color, outline=""
                        )

            x_offset += new_w + 3

    def _copy_to_clipboard(self):
        """Kopiuje rozpoznany tekst do schowka."""
        text = self.result_text.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._set_status("Skopiowano do schowka")

    def _set_status(self, text):
        """Ustawia tekst na pasku statusu."""
        self.status_label.config(text=text)


def main():
    root = tk.Tk()

    # Styl
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Arial", 11, "bold"))

    app = ProductionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
