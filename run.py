"""
Launcher aplikacji OCR.
Pozwala wybrac tryb: testowy (main.py) lub produkcyjny (app_production.py)

Uruchomienie:
    python run.py
"""

import subprocess
import sys
import os


def main():
    # Zmien katalog roboczy na katalog skryptu
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print()
    print("=" * 55)
    print("   ROZPOZNAWANIE PISMA RECZNEGO - OCR")
    print("=" * 55)
    print()
    print("Dostepne tryby:")
    print()
    print("  1. TESTOWY")
    print("     - Rysowanie liter na canvas")
    print("     - Testowanie roznych algorytmow binaryzacji")
    print("     - Testowanie roznych algorytmow segmentacji")
    print("     - Porownywanie modeli OCR")
    print()
    print("  2. PRODUKCYJNY")
    print("     - Przetwarzanie skanow dokumentow")
    print("     - Zaznaczanie obszaru ROI (Region of Interest)")
    print("     - ROI zapisywane dla kolejnych dokumentow")
    print("     - Stale algorytmy: Sauvola + CCA")
    print()
    print("-" * 55)

    choice = input("Wybierz tryb (1 lub 2): ").strip()

    print()

    if choice == "2":
        print("Uruchamiam tryb PRODUKCYJNY...")
        subprocess.run([sys.executable, "app_production.py"])
    else:
        print("Uruchamiam tryb TESTOWY...")
        subprocess.run([sys.executable, "main.py"])


if __name__ == "__main__":
    main()
