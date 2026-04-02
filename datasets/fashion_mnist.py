# =============================================================================
# datasets/fashion_mnist.py — Carga y preparación de Fashion-MNIST
# =============================================================================
# Fashion-MNIST es el dataset principal del TFG (Capítulo 3).
# - 60.000 imágenes de entrenamiento / 10.000 de test
# - 10 clases de prendas de ropa
# - 28×28 píxeles, 1 canal (escala de grises)
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# Estadísticas de normalización de Fashion-MNIST (media y std del canal único)
# Estos valores están calculados sobre el conjunto de entrenamiento completo
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD  = (0.3530,)


def get_transform():
    """
    Devuelve la transformación estándar para Fashion-MNIST.
    Solo normalizamos: convertimos a tensor y ajustamos media/std.
    NO se aplica data augmentation (tal como se define en el Capítulo 3).
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    ])


def get_dataset(seed: int, initial_pool_size: int = 100):
    """
    Carga Fashion-MNIST y divide el conjunto de entrenamiento en:
      - labeled_indices:   índices de las 'initial_pool_size' muestras iniciales (L₀)
      - unlabeled_indices: índices del resto del train (pool no etiquetado U₀)
      - train_dataset:     el dataset de train completo (necesario para acceder a las muestras)
      - test_dataset:      el dataset de test completo (para evaluar accuracy)

    La semilla inicial L₀ se selecciona de forma ESTRATIFICADA: se toman
    (initial_pool_size // n_clases) muestras de cada clase, garantizando que
    todas las clases están representadas desde el principio.

    Parámetros
    ----------
    seed : int
        Semilla para la selección aleatoria de L₀. Garantiza reproducibilidad.
    initial_pool_size : int
        Tamaño de la semilla inicial (n₀ en el Capítulo 2). Por defecto 100.

    Retorna
    -------
    labeled_indices : list[int]
        Índices (sobre train_dataset) de las muestras inicialmente etiquetadas.
    unlabeled_indices : list[int]
        Índices (sobre train_dataset) de las muestras no etiquetadas.
    train_dataset : torchvision.datasets.FashionMNIST
        Dataset de entrenamiento completo (con transformaciones aplicadas).
    test_dataset : torchvision.datasets.FashionMNIST
        Dataset de test completo.
    """
    transform = get_transform()

    # Descargar (si no están ya) y cargar los datos
    # data_root='./data' guarda los archivos en una carpeta 'data' local
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform,
    )

    # Obtener todas las etiquetas del train para hacer la selección estratificada
    all_labels = np.array(train_dataset.targets)
    n_classes = len(np.unique(all_labels))
    n_per_class = initial_pool_size // n_classes  # ej: 100 // 10 = 10 por clase

    # Fijar la semilla aleatoria para reproducibilidad
    rng = np.random.default_rng(seed)

    labeled_indices = []
    for clase in range(n_classes):
        # Índices de todas las muestras de esta clase
        indices_clase = np.where(all_labels == clase)[0]
        # Seleccionar n_per_class al azar de esta clase
        seleccionados = rng.choice(indices_clase, size=n_per_class, replace=False)
        labeled_indices.extend(seleccionados.tolist())

    # El conjunto no etiquetado es todo el train MENOS los índices ya seleccionados
    labeled_set = set(labeled_indices)
    unlabeled_indices = [i for i in range(len(train_dataset)) if i not in labeled_set]

    return labeled_indices, unlabeled_indices, train_dataset, test_dataset


# Nombres de las 10 clases de Fashion-MNIST (útil para las figuras)
FASHION_MNIST_CLASSES = [
    "T-shirt/top",  # 0
    "Trouser",      # 1
    "Pullover",     # 2
    "Dress",        # 3
    "Coat",         # 4
    "Sandal",       # 5
    "Shirt",        # 6
    "Sneaker",      # 7
    "Bag",          # 8
    "Ankle boot",   # 9
]
