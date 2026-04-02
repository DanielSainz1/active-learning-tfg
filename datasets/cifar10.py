# =============================================================================
# datasets/cifar10.py — Carga y preparación de CIFAR-10
# =============================================================================
# CIFAR-10 es el dataset de extensión (para GitHub/CV, Sesión 9).
# - 50.000 imágenes de entrenamiento / 10.000 de test
# - 10 clases de objetos (avión, coche, pájaro, gato, ...)
# - 32×32 píxeles, 3 canales (RGB)
#
# La interfaz es IDÉNTICA a fashion_mnist.py: misma función get_dataset(),
# mismos parámetros, mismo valor de retorno. Esto permite cambiar de dataset
# con un solo string en config.py.
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# Estadísticas de normalización de CIFAR-10 (media y std por canal RGB)
# Calculadas sobre el conjunto de entrenamiento completo
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_transform():
    """
    Devuelve la transformación estándar para CIFAR-10.
    Solo normalizamos por canal. NO se aplica data augmentation.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_dataset(seed: int, initial_pool_size: int = 100):
    """
    Carga CIFAR-10 y divide el conjunto de entrenamiento en:
      - labeled_indices:   índices de las 'initial_pool_size' muestras iniciales (L₀)
      - unlabeled_indices: índices del resto del train (pool no etiquetado U₀)
      - train_dataset:     el dataset de train completo
      - test_dataset:      el dataset de test completo

    La selección de L₀ es estratificada: ~10 muestras por clase.
    Interfaz idéntica a datasets/fashion_mnist.py.

    Parámetros
    ----------
    seed : int
        Semilla para la selección aleatoria de L₀.
    initial_pool_size : int
        Tamaño de la semilla inicial. Por defecto 100.

    Retorna
    -------
    labeled_indices : list[int]
    unlabeled_indices : list[int]
    train_dataset : torchvision.datasets.CIFAR10
    test_dataset : torchvision.datasets.CIFAR10
    """
    transform = get_transform()

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform,
    )

    # CIFAR-10 guarda las etiquetas en un atributo diferente al de FashionMNIST
    all_labels = np.array(train_dataset.targets)
    n_classes = len(np.unique(all_labels))
    n_per_class = initial_pool_size // n_classes

    rng = np.random.default_rng(seed)

    labeled_indices = []
    for clase in range(n_classes):
        indices_clase = np.where(all_labels == clase)[0]
        seleccionados = rng.choice(indices_clase, size=n_per_class, replace=False)
        labeled_indices.extend(seleccionados.tolist())

    labeled_set = set(labeled_indices)
    unlabeled_indices = [i for i in range(len(train_dataset)) if i not in labeled_set]

    return labeled_indices, unlabeled_indices, train_dataset, test_dataset


# Nombres de las 10 clases de CIFAR-10
CIFAR10_CLASSES = [
    "Airplane",     # 0
    "Automobile",   # 1
    "Bird",         # 2
    "Cat",          # 3
    "Deer",         # 4
    "Dog",          # 5
    "Frog",         # 6
    "Horse",        # 7
    "Ship",         # 8
    "Truck",        # 9
]
