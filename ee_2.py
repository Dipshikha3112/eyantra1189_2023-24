import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Utility Functions

def data_preprocessing(task_1a_dataframe):
    label_encoder = LabelEncoder()
    non_numeric_cols = ['Education', 'City', 'Gender', 'EverBenched']
    
    for col in non_numeric_cols:
        task_1a_dataframe[col] = label_encoder.fit_transform(task_1a_dataframe[col])
    
    scaler = StandardScaler()
    numerical_cols = ['Age', 'PaymentTier', 'ExperienceInCurrentDomain']
    task_1a_dataframe[numerical_cols] = scaler.fit_transform(task_1a_dataframe[numerical_cols])
    
    encoded_dataframe = task_1a_dataframe
    
    return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
    features = encoded_dataframe.drop(columns=['LeaveOrNot'])
    target = encoded_dataframe['LeaveOrNot']
    return [features.values, target.values]

def load_as_tensors(features_and_targets):
    X_train, X_test, y_train, y_test = train_test_split(features_and_targets[0], features_and_targets[1], test_size=0.2, random_state=42)
    
    X_train_tensor = torch.Tensor(X_train)
    X_test_tensor = torch.Tensor(X_test)
    y_train_tensor = torch.Tensor(y_train)
    y_test_tensor = torch.Tensor(y_test)
    
    return [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor]

class DynamicSizePredictor(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(DynamicSizePredictor, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def model_loss_function(loss_type):
    if loss_type == 'BCE':
        return nn.BCELoss()
    elif loss_type == 'MSE':
        return nn.MSELoss()
    else:
        raise ValueError("Invalid loss_type")

def model_optimizer(optimizer_type, model):
    if optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=0.0001)
    elif optimizer_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=0.0001)
    elif optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer_type")

def model_number_of_epochs():
    return 400
def train_and_evaluate(X_train, X_test, y_train, y_test, model, loss_function, optimizer, number_of_epochs):
    model.train()
    train_dataset = TensorDataset(X_train, y_train.view(-1, 1))  # Reshape y_train to match the output size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(number_of_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, target)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = (y_pred > 0.5).float()  # Convert to binary predictions (0 or 1)
    
    accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
    return accuracy

if __name__ == "__main__":
    task_1a_dataframe = pd.read_csv('task_1a_dataset.csv')
    
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_as_tensors(features_and_targets)
    
    input_size = X_train_tensor.shape[1]
    hidden_layers = [64, 32, 16]  # You can adjust the number of hidden units and layers
    num_classes = 1  # Binary classification
    
    # Experiment with different loss functions and optimizers
    loss_functions = ['BCE', 'MSE']
    optimizers = ['Adam', 'AdamW', 'SGD']
    
    best_accuracy = 0.0
    best_loss_type = ''
    best_optimizer_type = ''
    
    for loss_type in loss_functions:
        for optimizer_type in optimizers:
            model = DynamicSizePredictor(input_size, hidden_layers, num_classes)
            optimizer = model_optimizer(optimizer_type, model)
            number_of_epochs = model_number_of_epochs()
            
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            cv_accuracy = train_and_evaluate(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, model, model_loss_function(loss_type), optimizer, number_of_epochs)
            
            if cv_accuracy > best_accuracy:
                best_accuracy = cv_accuracy
                best_loss_type = loss_type
                best_optimizer_type = optimizer_type
    
    print(f"Test Set Accuracy: {best_accuracy:.4f}")
