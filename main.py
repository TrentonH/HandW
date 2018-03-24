__author__ = 'Trenton'

import network2
import mnist_loader
import matplotlib.pyplot as plt



training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print ("traingin_data{}".format(training_data[-9:]))

#pass in a list for how many nodes in each layer
net = network2.Network([784, 30, 10])

evaluationAccList = []
for x in range (5, 25, 5):
#first number is the number of epochs the second number is mini bach- or the bach size, the last number is the learning rate
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy= net.SGD(training_data, 1000, 10, .5, evaluation_data=test_data, monitor_training_accuracy=False,
                                                                                monitor_evaluation_accuracy= True, monitor_evaluation_cost=False,monitor_training_cost=False)

evaluationAccList.append(evaluation_accuracy[-1])
evaluationAccList = evaluation_accuracy



#plt.plot(list(range(1000)), evaluationAccList)
#plt.show()