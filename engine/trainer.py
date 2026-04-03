import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluar(modelo, loader, dispositivo):
    modelo.eval()
    correctas = 0
    total = 0
    
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            # 1. mover al dispositivo
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)
            # 2. pasar por el modelo
            logits = modelo(imagenes)
            # 3. la predicción es el índice más alto
            predicciones = torch.argmax(logits, dim=1)
            # 4. contar aciertos
            correctas += (predicciones == etiquetas).sum().item()
            total += etiquetas.size(0)

    return correctas / total

def entrenar_epoca(modelo, loader, optimizador, criterio, dispositivo):
    modelo.train()
    perdida_total = 0

    for imagenes, etiquetas in loader:
        # 1. mover al dispositivo
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)
        # 2. gradientes a cero
        optimizador.zero_grad()
        # 3. pasar por el modelo
        logits = modelo(imagenes)
        # 4. calcular pérdida
        perdida = criterio(logits, etiquetas)
        # 5. backpropagation
        perdida.backward()
        # 6. actualizar pesos
        optimizador.step()
        perdida_total += perdida.item()

    return perdida_total / len(loader)

def entrenar(modelo, dataset_etiquetado, config, dispositivo):
    loader = torch.utils.data.DataLoader(dataset_etiquetado, batch_size=config["train_batch_size"], shuffle=True)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=config["learning_rate"])
    criterio = nn.CrossEntropyLoss()
    
    for epoca in range(config["train_epochs"]):
        perdida = entrenar_epoca(modelo, loader, optimizador, criterio, dispositivo)


