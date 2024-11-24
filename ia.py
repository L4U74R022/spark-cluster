import torch
from torch import nn, optim
from torchvision import transforms, models
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split

# Verifica si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Cargar el dataset desde Hugging Face
dataset = load_dataset("jbarat/plant_species")

# Convertir las imágenes y etiquetas a un DataLoader de PyTorch
def prepare_data(dataset, split="train"):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Redimensionar a 128x128 píxeles
        transforms.ToTensor(),         # Convertir a tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalizar los valores de píxeles
    ])

    images = []
    labels = []

    for example in dataset[split]:
        image = example['image']  # La imagen ya está en formato PIL
        images.append(transform(image))  # Aplicar transformaciones
        labels.append(example['label'])

    # Crear tensores
    X = torch.stack(images)
    y = torch.tensor(labels)

    return X, y

# Preparar los datos de entrenamiento y validación
X, y = prepare_data(dataset, split="train")

# Dividir los datos en entrenamiento y validación
train_size = int(0.8 * len(X))
val_size = len(X) - train_size
train_data, val_data = random_split(list(zip(X, y)), [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 2. Crear un modelo usando transferencia de aprendizaje (ResNet18)
model = models.resnet18(weights="IMAGENET1K_V1")

# Ajustar la última capa para clasificar las especies de plantas
num_classes = len(set(y.tolist()))
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# 3. Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Entrenar el modelo
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Evaluar en el conjunto de validación
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

train(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# 5. Guardar el modelo entrenado
torch.save(model.state_dict(), "plant_recognition_model.pth")
print("Modelo guardado en 'plant_recognition_model.pth'")
