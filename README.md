# Handwriting to Text

OCR system for recognizing handwritten uppercase letters (A–Z) from scanned documents. The project compares two segmentation approaches (Profile Projection and CCA) and four recognition models (PyTorch CNN, Vision Transformer, SVM, Tesseract) to evaluate their accuracy and speed on real handwriting.

## Requirements

- Python 3.10+ 64-bit
- Tesseract OCR — optional, only needed for the Tesseract model ([Windows installer](https://github.com/UB-Mannheim/tesseract/wiki), `apt-get install tesseract-ocr` on Linux)
- CUDA GPU — optional, speeds up training significantly

## Installation

```bash
git clone https://github.com/your-username/HandwritingToText.git
cd HandwritingToText
pip install -r requirements.txt
```

## Usage

```bash
python run.py
```

This opens a launcher to choose between two modes:

**Test mode** — draw letters on a canvas or load an image, choose a segmentation method and OCR model, see the recognized text with confidence scores.

**Production mode** — load a scanned document, draw ROI regions over text lines, process. ROI regions are saved and reused automatically for future documents.

To train the models yourself (pretrained weights are already included in `models/saved/`):
```bash
python prepare_dataset.py   # downloads EMNIST, run once
python train_pytorch.py
python train_vit.py
python train_sklearn.py
```
