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

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(training_data, training_targets)
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
answers = clf.predict(test_data)

z = 0
for h, i in zip(answers, test_targets):
    if h == i:
        z = z + 1
z = z/ len(answers)
print(z)
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