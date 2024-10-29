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
# Functions:	    [`data_preprocessing`, `identify_features_and_targets`, `load_as_tensors`,
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################






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
    task_1a_dataframe = task_1a_dataframe.dropna()
    # categorical_columns = ['City', 'Education', 'Gender', 'EverBenched']
    # task_1a_dataframe = pandas.get_dummies(task_1a_dataframe, columns=categorical_columns)

    # scaler = StandardScaler()
    # task_1a_dataframe[['Age', 'ExperienceInCurrentDomain']] = scaler.fit_transform(task_1a_dataframe[['Age', 'ExperienceInCurrentDomain']])
    label_encoder = LabelEncoder()
    non_numeric_cols = ['Education', 'City', 'Gender', 'EverBenched']
    for col in non_numeric_cols:
        task_1a_dataframe[col] = label_encoder.fit_transform(task_1a_dataframe[col])
    for column in task_1a_dataframe:
        if task_1a_dataframe[column].dtype == 'bool':
            task_1a_dataframe[column] = task_1a_dataframe[column].astype('int32')

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
    class_0 = encoded_dataframe[encoded_dataframe['LeaveOrNot'] == 0]
    class_1 = encoded_dataframe[encoded_dataframe['LeaveOrNot'] == 1]

    if len(class_0) > len(class_1):
        majority = class_0
        minority = class_1
    else:
        minority = class_0
        majority = class_1
    
    oversampled = resample(minority, replace=True, n_samples=len(majority), random_state=12)

    balanced_df = pandas.concat([encoded_dataframe, oversampled])
    features = balanced_df.drop("LeaveOrNot", axis=1)
    target = balanced_df[["LeaveOrNot"]]
    features_and_targets = [features, target]
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
    X_train, X_val, y_train, y_val = train_test_split(*features_and_targets, test_size=0.2, random_state=12, shuffle=True)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    training_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)

    tensors_and_iterable_training_data = [X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, train_loader]
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
    def __init__(self):
        super(Salary_Predictor, self).__init__()
        '''
		Define the type and number of layers
		'''
		#######	ADD YOUR CODE HERE	#######
        self.fc1 = torch.nn.Linear(1, 64)  # Input size will be determined dynamically
        self.fc2 = torch.nn.Linear(16, 4)
        self.fc3 = torch.nn.Linear(4, 2)
        self.fc4 = torch.nn.Linear(2, 1)
		###################################	
    def forward(self, x):
        '''
		Define the activation functions
		'''
		#######	ADD YOUR CODE HERE	#######
        if self.fc1.in_features == 1:
            self.fc1 = torch.nn.Linear(x.size(1), 16)  # Adjust the input size dynamically
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        predicted_output = torch.sigmoid(self.fc4(x))
		###################################

        return predicted_output

def model_loss_function():
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
	loss_function = torch.nn.BCELoss()
	##########################################################
	
	return loss_function

def model_optimizer(model):
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
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
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
    number_of_epochs = 100
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
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader = tensors_and_iterable_training_data

    model.train()
    for epoch in range(number_of_epochs):
        running_loss = 0.0 
        val_loss = 0.0
        i = 0
        for inputs, labels in train_loader:
            val_inputs = X_test_tensor[i]
            val_labels = y_test_tensor[i]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            with torch.no_grad():
                val_outputs = model(val_inputs)
                val_loss += loss_function(val_outputs, val_labels).item()
            i += 1

        loss = running_loss / len(train_loader)
        val_loss = val_loss / len(train_loader)
        # print(f"Epoch {epoch + 1}/{number_of_epochs},Average Loss: {loss:.4f}, Val = {val_loss:.4f}")
   
    trained_model=model
	##########################################################

    return trained_model

def validation_function(trained_model, tenSors_and_iterable_training_data):
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
    X_val_tensor, y_val_tensor, _ = tensors_and_iterable_training_data[1], tensors_and_iterable_training_data[3], tensors_and_iterable_training_data[4]
    
    with torch.no_grad():
        predictions = trained_model(X_val_tensor)
        predicted_labels = (predictions >= 0.5).float()
        correct = (predicted_labels == y_val_tensor.view_as(predicted_labels)).sum()
        total = y_val_tensor.size(0)
        accuracy = correct / total

    # print(predictions)
    # average_prediction = torch.mean(predictions).item()
    # print(f"Average Predicted Value: {average_prediction}")

    # count_ones = (predicted_labels == 1).sum().item()
    # count_zeros = (predicted_labels == 0).sum().item()

    # print(f"Count of predicted 1s: {count_ones}")
    # print(f"Count of predicted 0s: {count_zeros}")


    trained_model.train()
    model_accuracy = accuracy.item()
	##########################################################

    return model_accuracy

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