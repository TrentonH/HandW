__author__ = 'Trenton'

import network2
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10])

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy= net.SGD(training_data, 5, 10, 3.0, evaluation_data=test_data, monitor_training_accuracy=True,
               monitor_evaluation_accuracy= True, monitor_evaluation_cost=True,monitor_training_cost=True)
