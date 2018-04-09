__author__ = 'Trenton'

import network2
import mnist_loader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier


#training_data, validation_data, test_data = mnist_loader.load_data_wrapper(False)
training_data = pd.io.parsers.read_csv('mnist_rotation_train.txt', delim_whitespace=True,header=None)
test_data = pd.io.parsers.read_csv('mnist_rotation_test.txt', delim_whitespace=True,header=None)

training = training_data.as_matrix()
test = test_data.as_matrix()
training_data = []
training_targets = []
test_data =  []
test_targets = []

for i in range(len(test)):
    test_data.append(test[i][0:len(test[i]) - 1])
    test_targets.append(test[i][-1])


for i in range(len(training)):
    training_data.append(training[i][0:len(training[i]) - 1])
    training_targets.append(training[i][-1])

#hidden_layer_sizes=(30,30), activation='logistic', solver='adam',
 #                   batch_size='auto', learning_rate='adaptive', learning_rate_init=0.5,
  #                  shuffle=False, random_state=1, tol=-5)
for i in range(200, 2100, 300):
    rotated = MLPClassifier(hidden_layer_sizes=(i,int(i/2),int(i/4)), max_iter=500)
    rotated.fit(training_data, training_targets)
    guesses = rotated.predict(test_data)

    correct = 0
    for guess, answer in zip(guesses, test_targets):
        if guess == answer:
            correct += 1
    accuracy = correct/len(guesses) * 100.00
    print(i)
    print(accuracy)
#----------------------------------------------------------------

#test_data = test_data.as_matrix()


#pass in a list for how many nodes in each layer
#net = network2.Network([784, 30, 10])

#evaluationAccList = []
#for x in range (5, 25, 5):
#first number is the number of epochs the second number is mini bach- or the bach size, the last number is the learning rate
#evaluation_cost, evaluation_accuracy, training_cost, training_accuracy= net.SGD(training_data, 1, 10, .5, evaluation_data=test_data, monitor_training_accuracy=False,
                                                                                #monitor_evaluation_accuracy= True, monitor_evaluation_cost=False,monitor_training_cost=False)

#evaluationAccList.append(evaluation_accuracy[-1])
#evaluationAccList = evaluation_accuracy


#plt.plot(list(range(1000)), evaluationAccList)
#plt.show()

