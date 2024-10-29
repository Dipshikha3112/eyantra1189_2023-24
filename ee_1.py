'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ 1189 ]
# Author List:		[ Dipshikha, Aditya, Sarath, MAnu John]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas 
import torch
import numpy as np
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

def cross_validate(X, y, model, loss_function, optimizer, number_of_epochs, cv):
    accuracies = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        training_function(X_train, y_train, model, loss_function, optimizer, number_of_epochs)
        accuracy = validation_function(X_test, y_test, model)
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

def learning_rate_scheduler(optimizer, step_size=30, gamma=0.5):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler

def main():
    task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')
    
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_as_tensors(features_and_targets)
    
    input_size = X_train_tensor.shape[1]
    hidden_layers = [64, 32, 16]  # You can adjust the number of hidden units and layers
    num_classes = 1  # Binary classification
    
    # Experiment with different loss functions and optimizers
    loss_functions = ['BCE', 'MSE']
    optimizers = ['Adam', 'AdamW', 'SGD']
    model = Salary_Predictor(input_size, hidden_layers, num_classes)
    
    best_accuracy = 0.0
    best_loss_type = ''
    best_optimizer_type = ''
    
    for loss_type in loss_functions:
        for optimizer_type in optimizers:
            model = Salary_Predictor(input_size, hidden_layers, num_classes)
            optimizer = model_optimizer(model, optimizer_type)
            number_of_epochs = model_number_of_epochs()
            
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            accuracies = []
            
            for train_idx, test_idx in cv.split(X_train_tensor):
                X_train_cv, X_val_cv = X_train_tensor[train_idx], X_train_tensor[test_idx]
                y_train_cv, y_val_cv = y_train_tensor[train_idx], y_train_tensor[test_idx]
                
                trained_model = training_function(model, number_of_epochs, (X_train_cv, y_train_cv), model_loss_function(loss_type), optimizer)
                accuracy = validation_function(trained_model, (X_val_cv, y_val_cv))
                accuracies.append(accuracy)
            
            cv_accuracy = np.mean(accuracies)
            
            if cv_accuracy > best_accuracy:
                best_accuracy = cv_accuracy
                best_loss_type = loss_type
                best_optimizer_type = optimizer_type


##############################################################

def data_preprocessing(task_1a_dataframe):
    ''' 
	Purpose:
	---
	This function will be used to load your csv dataset and preprocess it.
	Preprocessing involves cleaning the dataset by removing unwanted features,
	decision about what needs to be done with missing values etc. Note that 
	there are features in the csv file whose values are textual (eg: Industry, 
	Education Level etc)These features might be required for training the model
	but can not be given directly as strings for training. Hence this function 
	should return encoded dataframe in which all the textual features are 
	numerically labeled.
	
	Input Arguments:
	---
	`task_1a_dataframe`: [Dataframe]
						  Pandas dataframe read from the provided dataset 	
	
	Returns:
	---
	`encoded_dataframe` : [ Dataframe ]
						  Pandas dataframe that has all the features mapped to 
						  numbers starting from zero

	Example call:
	---
	encoded_dataframe = data_preprocessing(task_1a_dataframe)
	'''

	#################	ADD YOUR CODE HERE	##################    
    label_encoder = LabelEncoder()
    non_numeric_cols = ['Education', 'City', 'Gender', 'EverBenched']
    
    for col in non_numeric_cols:
        task_1a_dataframe[col] = label_encoder.fit_transform(task_1a_dataframe[col])
    
    scaler = StandardScaler()
    numerical_cols = ['Age', 'PaymentTier', 'ExperienceInCurrentDomain']
    task_1a_dataframe[numerical_cols] = scaler.fit_transform(task_1a_dataframe[numerical_cols])
    
    encoded_dataframe = task_1a_dataframe

	##########################################################

    return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
    '''
	Purpose:
	---
	The purpose of this function is to define the features and
	the required target labels. The function returns a python list
	in which the first item is the selected features and second 
	item is the target label

	Input Arguments:
	---
	`encoded_dataframe` : [ Dataframe ]
						Pandas dataframe that has all the features mapped to 
						numbers starting from zero
	
	Returns:
	---
	`features_and_targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label

	Example call:
	---
	features_and_targets = identify_features_and_targets(encoded_dataframe)
	'''

	#################	ADD YOUR CODE HERE	##################
    features = encoded_dataframe.drop(columns=['LeaveOrNot'])
    target = encoded_dataframe['LeaveOrNot']
    features_and_targets =[features.values, target.values]
	##########################################################

    return features_and_targets


def load_as_tensors(features_and_targets):
    ''' 
	Purpose:
	---
	This function aims at loading your data (both training and validation)
	as PyTorch tensors. Here you will have to split the dataset for training 
	and validation, and then load them as as tensors. 
	Training of the model requires iterating over the training tensors. 
	Hence the training sensors need to be converted to iterable dataset
	object.
	
	Input Arguments:
	---
	`features_and targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label
	
	Returns:
	---
	`tensors_and_iterable_training_data` : [ list ]
											Items:
											[0]: X_train_tensor: Training features loaded into Pytorch array
											[1]: X_test_tensor: Feature tensors in validation data
											[2]: y_train_tensor: Training labels as Pytorch tensor
											[3]: y_test_tensor: Target labels as tensor in validation data
											[4]: Iterable dataset object and iterating over it in 
												 batches, which are then fed into the model for processing

	Example call:
	---
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	'''

	#################	ADD YOUR CODE HERE	##################
    X_train, X_test, y_train, y_test = train_test_split(features_and_targets[0], features_and_targets[1], test_size=0.2, random_state=42)
    
    X_train_tensor = torch.Tensor(X_train)
    X_test_tensor = torch.Tensor(X_test)
    y_train_tensor = torch.Tensor(y_train)
    y_test_tensor = torch.Tensor(y_test)
    
    tensors_and_iterable_training_data=[X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor]
	##########################################################
    return tensors_and_iterable_training_data

class Salary_Predictor(torch.nn.Module):
    '''
	Purpose:
	---
	The architecture and behavior of your neural network model will be
	defined within this class that inherits from nn.Module. Here you
	also need to specify how the input data is processed through the layers. 
	It defines the sequence of operations that transform the input data into 
	the predicted output. When an instance of this class is created and data
	is passed through it, the `forward` method is automatically called, and 
	the output is the prediction of the model based on the input data.
	
	Returns:
	---
	`predicted_output` : Predicted output for the given input data
	'''
    def __init__(self, input_size, hidden_layers, num_classes):
        super(Salary_Predictor, self).__init__()
        '''
		Define the type and number of layers
		'''
		#######	ADD YOUR CODE HERE	#######
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())

        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
		###################################	
    def forward(self, x):
        '''
		Define the activation functions
		'''
		#######	ADD YOUR CODE HERE	#######
        predicted_output=self.net(x)
		###################################

        return predicted_output

def model_loss_function(loss_type):
    '''
	Purpose:
	---
	To define the loss function for the model. Loss function measures 
	how well the predictions of a model match the actual target values 
	in training data.
	
	Input Arguments:
	---
	None

	Returns:
	---
	`loss_function`: This can be a pre-defined loss function in PyTorch
					or can be user-defined

	Example call:
	---
	loss_function = model_loss_function()
	'''
	#################	ADD YOUR CODE HERE	##################
    if loss_type == 'BCE':
     return nn.BCELoss()
    elif loss_type == 'MSE':
        return nn.MSELoss()
    else:
        raise ValueError("Invalid loss_type")
	##########################################################
    return loss_function

def model_optimizer(model,optimizer_type):
    '''
	Purpose:
	---
	To define the optimizer for the model. Optimizer is responsible 
	for updating the parameters (weights and biases) in a way that 
	minimizes the loss function.
	
	Input Arguments:
	---
	`model`: An object of the 'Salary_Predictor' class

	Returns:
	---
	`optimizer`: Pre-defined optimizer from Pytorch

	Example call:
	---
	optimizer = model_optimizer(model)
	'''
	#################	ADD YOUR CODE HERE	##################
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer_type")
	##########################################################

    return optimizer

def model_number_of_epochs():
    '''
	Purpose:
	---
	To define the number of epochs for training the model

	Input Arguments:
	---
	None

	Returns:
	---
	`number_of_epochs`: [integer value]

	Example call:
	---
	number_of_epochs = model_number_of_epochs()
	'''
	#################	ADD YOUR CODE HERE	##################
    number_of_epochs=10
	##########################################################

    return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
	Purpose:
	---
	All the required parameters for training are passed to this function.

	Input Arguments:
	---
	1. `model`: An object of the 'Salary_Predictor' class
	2. `number_of_epochs`: For training the model
	3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
											 and iterable dataset object of training tensors
	4. `loss_function`: Loss function defined for the model
	5. `optimizer`: Optimizer defined for the model

	Returns:
	---
	trained_model

	Example call:
	---
	trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

	'''	
	#################	ADD YOUR CODE HERE	##################
    X_train_tensor, y_train_tensor = tensors_and_iterable_training_data
    model.train()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.view(-1, 1))  # Reshape y_train to match the output size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_losses=[]
    
    for epoch in range(number_of_epochs):
        total_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))
    
    trained_model = model
	##########################################################

    return trained_model

def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
	Purpose:
	---
	This function will utilise the trained model to do predictions on the
	validation dataset. This will enable us to understand the accuracy of
	the model.

	Input Arguments:
	---
	1. `trained_model`: Returned from the training function
	2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
											 and iterable dataset object of training tensors

	Returns:
	---
	model_accuracy: Accuracy on the validation dataset

	Example call:
	---
	model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

	'''	
	#################	ADD YOUR CODE HERE	##################
    trained_model.eval()
    X_eval_tensor, y_eval_tensor = tensors_and_iterable_training_data
    
    with torch.no_grad():
        y_pred = trained_model(X_eval_tensor)
        y_pred = (y_pred > 0.5).float()  # Convert to binary predictions (0 or 1)
    
    model_accuracy = accuracy_score(y_eval_tensor.numpy(), y_pred.numpy())

	##########################################################

    return model_accuracy
main()
########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''
if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	
	# model is an instance of the class that defines the architecture of the model
	model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
					loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")