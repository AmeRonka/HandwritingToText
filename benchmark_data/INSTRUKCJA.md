# 📸 Instrukcja przygotowania danych testowych

## Słowa do napisania (każde 5 razy):

```
1. WALTZ
2. NYMPH
3. VEXED
4. QUICK
5. FJORD
6. BIG
```

**Razem: 6 słów × 5 powtórzeń = 30 zdjęć**

---

## 📝 Jak przygotować:

### 1. Napisz słowa:
- **Długopisem** na **białej kartce**
- **WIELKIE LITERY** (drukowane, nie pisane)
- **Wyraźnie**, odstępy między literami
- **Każde słowo na osobnej kartce** (lub z dużym odstępem)

### 2. Zrób zdjęcia:
- **Dobre oświetlenie** (równomierne, bez cieni)
- **Telefon prosto nad kartką** (nie pod kątem)
- **Wysoka rozdzielczość** (min. 1000px szerokości)
- **Tylko słowo w kadrze** (przytnij zbędne tło)

### 3. Nazwij pliki według schematu:

```
waltz_1.jpg
waltz_2.jpg
waltz_3.jpg
waltz_4.jpg
waltz_5.jpg

nymph_1.jpg
nymph_2.jpg
nymph_3.jpg
nymph_4.jpg
nymph_5.jpg

vexed_1.jpg
vexed_2.jpg
vexed_3.jpg
vexed_4.jpg
vexed_5.jpg

quick_1.jpg
quick_2.jpg
quick_3.jpg
quick_4.jpg
quick_5.jpg

fjord_1.jpg
fjord_2.jpg
fjord_3.jpg
fjord_4.jpg
fjord_5.jpg

big_1.jpg
big_2.jpg
big_3.jpg
big_4.jpg
big_5.jpg
```

**WAŻNE:**
- Nazwa pliku MUSI zawierać słowo (małymi literami)
- Program automatycznie rozpozna groundtruth z nazwy pliku
- Format: `.jpg`, `.png`, `.jpeg` - wszystko OK

---

## 📁 Struktura folderów:

Umieść wszystkie zdjęcia tutaj:
```
benchmark_data/
  ├── waltz_1.jpg
  ├── waltz_2.jpg
  ├── ...
  ├── nymph_1.jpg
  ├── ...
  └── big_5.jpg
```

---

## 🚀 Uruchomienie testu:

Gdy masz wszystkie 30 zdjęć, uruchom:

```bash
python benchmark_words.py
```

Program automatycznie:
1. Wczyta wszystkie obrazy z `benchmark_data/`
2. Przetestuje 7 konfiguracji modeli
3. Porówna wyniki z groundtruth
4. Wygeneruje raport: `benchmark_report.html`

---

## 💡 Wskazówki dla lepszych wyników:

### ✅ DOBRE praktyki:
- Pisz wyraźnie i duże litery
- Równy nacisk długopisu
- Białe tło, czarny/niebieski długopis
- Dobre oświetlenie (dzienne światło)
- Litery nie łączą się ze sobą

### ❌ UNIKAJ:
- Pisanie pisanym (używaj drukowanych liter!)
- Ciemne cienie na zdjęciu
- Rozmazane zdjęcia
- Zbyt małe litery (<1cm wysokości)
- Kolorowe tło

---

## 🎯 Testowane konfiguracje:

Program przetestuje:

1. **PyTorch CNN + Profile Projection**
2. **PyTorch CNN + CCA**
3. **Scikit-learn SVM + Profile Projection**
4. **Scikit-learn SVM + CCA**
5. **Tesseract Single + Profile Projection**
6. **Tesseract Single + CCA**
7. **Tesseract Whole Text** (bez segmentacji)

Wszystkie z binaryzacją **Sauvola** (najlepsza dla dokumentów).

---

## 📊 Czego się spodziewać:

Po uruchomieniu dostaniesz:
- **Wykres słupkowy** z accuracy dla każdego modelu
- **Tabelę wyników** per słowo
- **Przykłady błędów** (które litery były mylone)
- **Confusion matrix** (najczęstsze pomyłki)
- **Czas wykonania** dla każdego modelu

---

Gotowa? Napisz słowa, zrób zdjęcia i uruchom benchmark! 🚀
