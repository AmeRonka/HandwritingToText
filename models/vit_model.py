"""
Vision Transformer (ViT) - własna implementacja od zera.

Vision Transformer to architektura wprowadzona w 2020 roku przez Google
w artykule "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".

Kluczowe innowacje:
1. Dzieli obraz na "patche" (kawałki) jak słowa w zdaniu
2. Używa mechanizmu Self-Attention zamiast konwolucji
3. Każdy patch może "patrzeć" na wszystkie inne patche jednocześnie

Architektura:
┌─────────────────────────────────────────────────────────────┐
│  Obraz 28x28  →  Patche 7x7  →  Embeddingi  →  Transformer  │
│                     (16 patchy)   (wektory)     (Attention)  │
│                                                      ↓       │
│                                              Klasyfikacja    │
│                                              (26 liter)      │
└─────────────────────────────────────────────────────────────┘

Autor: Własna implementacja dla projektu OCR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    Dzieli obraz na patche i konwertuje je na wektory (embeddingi).

    Dla obrazu 28x28 z patchami 7x7:
    - 28 / 7 = 4 patche w poziomie
    - 28 / 7 = 4 patche w pionie
    - Razem: 4 × 4 = 16 patchy

    Każdy patch 7×7 = 49 pikseli → przekształcamy na wektor o wymiarze embed_dim

    Wizualizacja:
    ┌──┬──┬──┬──┐
    │P1│P2│P3│P4│     Każdy patch Pn ma wymiar 7×7 pikseli
    ├──┼──┼──┼──┤     Po embedding: wektor o długości embed_dim
    │P5│P6│P7│P8│
    ├──┼──┼──┼──┤
    │P9│P10│P11│P12│
    ├──┼──┼──┼──┤
    │P13│P14│P15│P16│
    └──┴──┴──┴──┘
    """

    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 16 dla 28/7

        # Konwolucja która wyciąga patche i od razu je embedduje
        # kernel_size=patch_size, stride=patch_size → nie nakładające się patche
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: Tensor obrazu [batch_size, channels, height, width]
               np. [32, 1, 28, 28]

        Returns:
            Tensor patchy [batch_size, num_patches, embed_dim]
            np. [32, 16, 64]
        """
        # x: [B, 1, 28, 28] → [B, embed_dim, 4, 4]
        x = self.projection(x)

        # Spłaszcz przestrzenne wymiary: [B, embed_dim, 4, 4] → [B, embed_dim, 16]
        x = x.flatten(2)

        # Zamień wymiary: [B, embed_dim, 16] → [B, 16, embed_dim]
        x = x.transpose(1, 2)

        return x


class PositionalEncoding(nn.Module):
    """
    Dodaje informację o pozycji każdego patcha.

    Problem: Self-Attention nie wie gdzie jest który patch!
    Rozwiązanie: Dodajemy do każdego patcha unikalny wektor pozycji.

    Używamy uczyalnych (learnable) pozycji - model sam nauczy się
    najlepszych wektorów pozycyjnych.

    Wizualizacja:
    Patch 1 + Pozycja 1 = Patch z informacją "jestem w lewym górnym rogu"
    Patch 16 + Pozycja 16 = Patch z informacją "jestem w prawym dolnym rogu"
    """

    def __init__(self, num_patches, embed_dim):
        super().__init__()

        # Uczylane wektory pozycji - jeden dla każdego patcha
        # +1 dla tokenu [CLS] (token klasyfikacji)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )

        # Token [CLS] - specjalny token do klasyfikacji
        # Jego embedding na końcu używamy do przewidywania klasy
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )

    def forward(self, x):
        """
        Args:
            x: Tensor patchy [batch_size, num_patches, embed_dim]

        Returns:
            Tensor z dodanymi pozycjami [batch_size, num_patches + 1, embed_dim]
        """
        batch_size = x.shape[0]

        # Rozszerz token [CLS] dla całego batcha
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Dodaj token [CLS] na początek sekwencji
        # [B, 16, 64] → [B, 17, 64]
        x = torch.cat([cls_tokens, x], dim=1)

        # Dodaj pozycje do wszystkich tokenów
        x = x + self.position_embeddings

        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention - serce Transformera!

    Self-Attention pozwala każdemu tokenowi "patrzeć" na wszystkie inne tokeny
    i zdecydować które są ważne.

    Mechanizm Query-Key-Value (QKV):
    - Query (Q): "Czego szukam?"
    - Key (K): "Co oferuję?"
    - Value (V): "Jaką mam wartość?"

    Attention(Q, K, V) = softmax(Q × K^T / √d) × V

    Multi-Head: Uruchamiamy attention równolegle N razy (N głów)
    Każda głowa może skupić się na czymś innym!

    Wizualizacja dla 4 głów:
    ┌─────────────────────────────────────────┐
    │  Głowa 1: Patrzy na kształt górnej części litery    │
    │  Głowa 2: Patrzy na kształt dolnej części litery    │
    │  Głowa 3: Patrzy na proporcje litery                │
    │  Głowa 4: Patrzy na kontekst sąsiednich patchy      │
    └─────────────────────────────────────────┘
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Wymiar każdej głowy

        assert embed_dim % num_heads == 0, "embed_dim musi być podzielny przez num_heads"

        # Projekcje Q, K, V - osobna warstwa dla każdej
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Projekcja wyjściowa - łączy wszystkie głowy
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Dropout dla regularyzacji
        self.dropout = nn.Dropout(dropout)

        # Skalowanie (√d) dla stabilności numerycznej
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, seq_len, embed_dim]
               np. [32, 17, 64] (17 = 16 patchy + 1 token CLS)

        Returns:
            Tensor po attention [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 1. Oblicz Q, K, V
        Q = self.query(x)  # [B, seq_len, embed_dim]
        K = self.key(x)
        V = self.value(x)

        # 2. Podziel na głowy
        # [B, seq_len, embed_dim] → [B, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Oblicz attention scores: Q × K^T / √d
        # [B, heads, seq, head_dim] × [B, heads, head_dim, seq] → [B, heads, seq, seq]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 4. Softmax - normalizacja do prawdopodobieństw
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 5. Pomnóż przez wartości V
        # [B, heads, seq, seq] × [B, heads, seq, head_dim] → [B, heads, seq, head_dim]
        context = torch.matmul(attention_probs, V)

        # 6. Połącz głowy z powrotem
        # [B, heads, seq, head_dim] → [B, seq, heads, head_dim] → [B, seq, embed_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 7. Projekcja wyjściowa
        output = self.output_projection(context)

        return output


class TransformerBlock(nn.Module):
    """
    Pojedynczy blok Transformera.

    Składa się z:
    1. Layer Normalization (przed attention)
    2. Multi-Head Self-Attention
    3. Residual Connection (skip connection)
    4. Layer Normalization (przed MLP)
    5. MLP (Feed-Forward Network)
    6. Residual Connection

    Schemat:
    ┌─────────────────────────────────────┐
    │  Input                              │
    │    ↓                                │
    │  LayerNorm → Attention → + (skip)   │
    │    ↓                                │
    │  LayerNorm → MLP → + (skip)         │
    │    ↓                                │
    │  Output                             │
    └─────────────────────────────────────┘
    """

    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()

        # Layer Normalization - stabilizuje uczenie
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # MLP (Feed-Forward Network)
        # Standardowo: embed_dim → 4*embed_dim → embed_dim
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),  # Activation function (lepsza niż ReLU dla Transformerów)
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, seq_len, embed_dim]

        Returns:
            Tensor [batch_size, seq_len, embed_dim]
        """
        # Attention z residual connection
        x = x + self.attention(self.norm1(x))

        # MLP z residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Kompletny Vision Transformer (ViT) dla rozpoznawania liter.

    Architektura:
    1. Patch Embedding - dzieli obraz na patche
    2. Positional Encoding - dodaje informację o pozycji
    3. Transformer Encoder - N bloków z attention
    4. Classification Head - przewiduje klasę

    Dla EMNIST (28×28, 26 klas):
    - Obraz 28×28 → 16 patchy 7×7
    - Każdy patch → wektor 64-wymiarowy
    - 4 bloki Transformer
    - Wyjście: 26 klas (litery A-Z)

    Porównanie z CNN:
    ┌─────────────────────────────────────────────────────┐
    │  CNN:         Lokalny kontekst (filtr 3×3)         │
    │  Transformer: Globalny kontekst (attention)         │
    │                                                     │
    │  CNN:         Hierarchia cech (warstwy)            │
    │  Transformer: Bezpośredni dostęp do wszystkiego    │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=26,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()

        self.num_classes = num_classes

        # 1. Patch Embedding
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embedding.num_patches

        # 2. Positional Encoding + CLS token
        self.positional_encoding = PositionalEncoding(num_patches, embed_dim)

        # 3. Transformer Encoder - stos bloków
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # 4. Layer Norm przed klasyfikacją
        self.norm = nn.LayerNorm(embed_dim)

        # 5. Classification Head
        # Używamy tylko tokenu [CLS] do klasyfikacji
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Inicjalizacja wag
        self._init_weights()

    def _init_weights(self):
        """Inicjalizacja wag - ważne dla stabilnego uczenia."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass przez cały model.

        Args:
            x: Tensor obrazu [batch_size, channels, height, width]
               np. [32, 1, 28, 28]

        Returns:
            logits: Tensor [batch_size, num_classes]
                    np. [32, 26]
        """
        # 1. Patch Embedding: [B, 1, 28, 28] → [B, 16, 64]
        x = self.patch_embedding(x)

        # 2. Dodaj pozycje i token CLS: [B, 16, 64] → [B, 17, 64]
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # 3. Przepuść przez bloki Transformer
        for block in self.transformer_blocks:
            x = block(x)

        # 4. Normalizacja
        x = self.norm(x)

        # 5. Weź tylko token [CLS] (pierwszy token)
        cls_token = x[:, 0]  # [B, 64]

        # 6. Klasyfikacja
        logits = self.classifier(cls_token)  # [B, 26]

        return logits

    def predict(self, x):
        """
        Predykcja z prawdopodobieństwami.

        Args:
            x: Tensor obrazu [1, 1, 28, 28] lub [28, 28]

        Returns:
            predicted_class: int (0-25)
            probabilities: Tensor [26]
        """
        self.eval()

        with torch.no_grad():
            # Upewnij się że wymiary są poprawne
            if x.dim() == 2:
                x = x.unsqueeze(0).unsqueeze(0)  # [28, 28] → [1, 1, 28, 28]
            elif x.dim() == 3:
                x = x.unsqueeze(0)  # [1, 28, 28] → [1, 1, 28, 28]

            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()

        return predicted_class, probabilities.squeeze()


def count_parameters(model):
    """Liczy parametry modelu."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test modelu
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Vision Transformer (ViT)")
    print("=" * 60)

    # Stwórz model
    model = VisionTransformer(
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=26,
        embed_dim=64,
        num_heads=4,
        num_layers=4
    )

    print(f"\nArchitektura:")
    print(f"  - Obraz: 28x28")
    print(f"  - Patch size: 7x7")
    print(f"  - Liczba patchy: {model.patch_embedding.num_patches}")
    print(f"  - Embedding dim: 64")
    print(f"  - Liczba głów attention: 4")
    print(f"  - Liczba bloków Transformer: 4")
    print(f"  - Parametry: {count_parameters(model):,}")

    # Test forward pass
    print(f"\nTest forward pass:")
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 images
    print(f"  Input shape: {x.shape}")

    output = model(x)
    print(f"  Output shape: {output.shape}")

    # Test predykcji
    print(f"\nTest predykcji:")
    pred_class, probs = model.predict(torch.randn(28, 28))
    print(f"  Predicted class: {pred_class}")
    print(f"  Top 5 probabilities: {probs.topk(5)}")

    print("\n" + "=" * 60)
    print("Model dziala poprawnie!")
    print("=" * 60)
