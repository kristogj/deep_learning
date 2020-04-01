# Facial Expression Classification Using Logistic- and Softmax Regression

## Setup
Install the dependencies in the requirments.txt file.

## Dataloader
**dataloader.py**
A dataloader was provided to us from the start of the project, but we defined new method for doing cross validation.

## PCA
**pca.py**
We defined our own PCA class. This was the recommended way of doing it, and contains all the methods for doing the 
transformation. 

## Models
**model.py**
Both Logistic and Softmax regression model are defined in this file.

## Objective functions
**objective.py**
Here you will find the cross entropy loss function used for our models as well as the calculation of accuracy.

## Optimization of model
**optimization.py**
The training loop for optimizing our models can be found in this file. This is sort of where everything is put together

## Graphing
**graphing.py**
We decided to seperate all the graph functions in its own file. You will find all graphing methods used in the project here.

## Runners and other
**runner.py**  
Here is the main method for running all the different tasks.

**constants.py**
Contains some constants we used for our settings. 

**utils.py**
Some help functions that we have been using through the project. 
