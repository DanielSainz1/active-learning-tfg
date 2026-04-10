# =============================================================================
# config.py — Configuración central del experimento de Active Learning
# =============================================================================
# Este archivo contiene TODOS los hiperparámetros del experimento.
# =============================================================================

CONFIG = {
    # -------------------------------------------------------------------------
    # Ciclo de Active Learning
    # -------------------------------------------------------------------------
    # n₀: tamaño del conjunto etiquetado inicial (semilla L₀)
    "initial_pool_size": 100,

    # b: número de instancias adquiridas en cada iteración del ciclo AL
    "query_batch_size": 100,

    # B: presupuesto total de etiquetas (número máximo de muestras a etiquetar)
    # El ciclo se detiene cuando |L| alcanza este valor
    "budget": 500,

    # Número de repeticiones con semillas distintas para estimar variabilidad
    "n_seeds": 5,

    # -------------------------------------------------------------------------
    # Entrenamiento de la CNN
    # -------------------------------------------------------------------------
    # Épocas de entrenamiento en cada ciclo AL (el modelo se entrena desde cero)
    "train_epochs": 2,

    # Tamaño del mini-batch para el DataLoader de entrenamiento
    "train_batch_size": 64,

    # Tasa de aprendizaje del optimizador Adam (igual que en el Capítulo 2)
    "learning_rate": 0.001,

    # Batch size para la fase de evaluación/predicción (más grande = más rápido)
    # No afecta a la calidad del modelo, solo a la velocidad de inferencia
    "eval_batch_size": 256,

    # -------------------------------------------------------------------------
    # MC Dropout para BALD (estrategia 5)
    # -------------------------------------------------------------------------
    # T: número de pasadas forward estocásticas para estimar la incertidumbre
    "mc_forward_passes": 5,

    # -------------------------------------------------------------------------
    # Reproducibilidad
    # -------------------------------------------------------------------------
    # Semillas fijas para cada una de las n_seeds repeticiones.
    # Fijar la semilla garantiza que los experimentos son reproducibles:
    # si alguien ejecuta el mismo código con las mismas semillas, obtendrá
    # exactamente los mismos resultados.
    "base_seeds": [42, 123, 456, 789, 1024],

    # -------------------------------------------------------------------------
    # Métricas
    # -------------------------------------------------------------------------
    # Umbrales de accuracy para calcular la Label Efficiency (LE).
    # LE(a*) = mínimo número de etiquetas para alcanzar el accuracy a*.
    "label_efficiency_thresholds": [0.80, 0.85, 0.88, 0.90],
}

# Estrategias de adquisición disponibles.
# El orden aquí define el orden en las tablas y gráficas del TFG.
STRATEGIES = ["random", "least_confidence", "margin", "entropy", "bald"]

# Datasets disponibles
DATASETS = ["fashion_mnist", "cifar10"]

# Colores para las gráficas (uno por estrategia, distinguibles en escala de grises)
# Orden: random, least_confidence, margin, entropy, bald
PLOT_COLORS = {
    "random":           "#888888",  # gris
    "least_confidence": "#E69F00",  # naranja
    "margin":           "#56B4E9",  # azul claro
    "entropy":          "#009E73",  # verde
    "bald":             "#CC79A7",  # rosa/magenta
}

# Nombres legibles para las leyendas de las gráficas
STRATEGY_LABELS = {
    "random":           "Random",
    "least_confidence": "Least Confidence",
    "margin":           "Margin Sampling",
    "entropy":          "Entropy Sampling",
    "bald":             "BALD (MC Dropout)",
}
