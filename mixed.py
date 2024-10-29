import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define your class labels
class_labels = {
    "Combat": 0,
    "Humanitarian Aid and rehabilitation": 1,
    "Military vehicles and weapons": 2,
    "Fire": 3,
    "DestroyedBuildings": 4
}

# Define a function to preprocess and apply data augmentation to an image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image_pil = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    augmented_image = transform(image_pil)
    return augmented_image

# Define the directory for your training data
train_data_dir = "/home/dipshikha/Pictures/EYANTRA-ttt/training"  # Replace with the folder containing subfolders for each class

# Create output directories for preprocessed images
output_dir = "/home/dipshikha/Pictures/EYANTRA-ttt/output"  # Replace with the output folder path
os.makedirs(output_dir, exist_ok=True)

for class_name, label in class_labels.items():
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

# Iterate through the images in the training data directory, preprocess them, and save in corresponding output directories
for class_name, label in class_labels.items():
    class_dir = os.path.join(train_data_dir, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith(".jpeg"):
            image_path = os.path.join(class_dir, filename)
            preprocessed_image = preprocess_image(image_path)
            output_path = os.path.join(output_dir, class_name, filename)
            save_image(preprocessed_image, output_path)
            print(f"Preprocessed and augmented: {filename} in class {class_name}")

# Define the directory for your test images
test_data_dir = "/home/dipshikha/Pictures/EYANTRA-ttt/testing"  # Replace with the folder containing test images

# Create an output directory for preprocessed test images
test_output_dir = "/home/dipshikha/Pictures/EYANTRA-ttt/test_output"  # Replace with the output folder path
os.makedirs(test_output_dir, exist_ok=True)

# Define data transformations for validation data (similar to training)
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define default label for test images
default_label = 0  # Change this to the appropriate label

# Create a custom test dataset with default labels
class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, default_label=default_label):
        self.root_dir = root_dir
        self.transform = transform
        self.default_label = default_label
        self.images = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith(".jpeg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, self.default_label

# Create a custom test dataset with default labels
test_dataset = CustomTestDataset(test_data_dir, transform=test_transform, default_label=default_label)

# Set batch size and create a data loader for the test dataset
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the VGG16 model pre-trained on ImageNet
vgg_model = vgg16(pretrained=True)

# Freeze all layers except the final classification layer
for param in vgg_model.parameters():
    param.requires_grad = False

# Modify the classification layer for your specific problem
in_features = vgg_model.classifier[6].in_features
num_classes = len(class_labels)

classifier = nn.Sequential(
    nn.Linear(in_features, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, num_classes)
)

vgg_model.classifier[6] = classifier

# Set the device (CPU or GPU) for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model.to(device)

# Define the dataset and data loaders for training
train_dataset = ImageFolder(train_data_dir, transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# Set batch size and create data loaders for training
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the loss function as categorical cross-entropy
criterion = nn.CrossEntropyLoss()

# Choose an optimizer (Adam in this example)
optimizer = torch.optim.Adam(vgg_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    vgg_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Save the trained model and class labels
model_save_path = "/home/dipshikha/Pictures/EYANTRA-ttt/vgg_model.pth"
torch.save(vgg_model.state_dict(), model_save_path)

labels_save_path = "/home/dipshikha/Pictures/EYANTRA-ttt/class_labels.pth"
torch.save(class_labels, labels_save_path)

print("Model and class labels saved.")

# Model Evaluation
vgg_model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = vgg_model(inputs)
        _, predictions = torch.max(outputs, 1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate evaluation metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
