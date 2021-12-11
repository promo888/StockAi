# #https://www.oreilly.com/radar/uncovering-hidden-patterns-through-machine-learning/
# Structure the problem as a multi-class classification problem
# Generate the fizzbuzz data
# Divide the data into train and test
# Build a logistic regression model in MXNet from scratch
# Introduce gluon
# Build a multi-layer-perceptron model using gluon
# we only have the data. Machine learning helps us create a model of the data. In this aspect, fizzbuzz provides us with an easy-to-understand data set and allows us to understand and explore the algorithms.

import numpy as np
import mxnet as mx
import os
mx.random.seed(1)

#Define the context to be CPU
ctx = mx.cpu()

#function to encode the integer to its binary representation
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

#function to encode the target into multi-class
def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 0
    elif i % 5  == 0:
        return 1
    elif i % 3  == 0:
        return 2
    else:
        return 3

#Given prediction, map it to the correct output label
def fizz_buzz(i, prediction):
    if prediction == 0:
        return "fizzbuzz"
    elif prediction == 1:
        return "buzz"
    elif prediction == 2:
        return "fizz"
    else:
        return str(i)

#Number of integers to generate
MAX_NUMBER = 100000

#The input feature vector is determined by NUM_DIGITS
NUM_DIGITS = np.log2(MAX_NUMBER).astype(np.int)+1

#Generate training dataset - both features and labels
trainX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, np.int(MAX_NUMBER/2))])
trainY = np.array([fizz_buzz_encode(i)          for i in range(101, np.int(MAX_NUMBER/2))])

#Generate validation dataset - both features and labels
valX = np.array([binary_encode(i, NUM_DIGITS) for i in range(np.int(MAX_NUMBER/2), MAX_NUMBER)])
valY = np.array([fizz_buzz_encode(i)          for i in range(np.int(MAX_NUMBER/2), MAX_NUMBER)])

#Generate test dataset - both features and labels
testX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 201)])
testY = np.array([fizz_buzz_encode(i)          for i in range(101, 201)])

#Define the parameters
batch_size = 100
num_inputs = NUM_DIGITS
num_outputs = 4

#Create iterator for train, test and validation datasets
train_data = mx.io.NDArrayIter(trainX, trainY,
                               batch_size, shuffle=True)
val_data = mx.io.NDArrayIter(valX, valY,
                               batch_size, shuffle=True)
test_data = mx.io.NDArrayIter(testX, testY,
                              batch_size, shuffle=False)

#Function to evaluate accuracy of the model

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return predictions, acc.get()[1]

#import autograd package
from mxnet import autograd, nd

#Initialize the weight and bias matrix

#weights matrix
W = nd.random_normal(shape=(num_inputs, num_outputs))
#bias matrix
b = nd.random_normal(shape=num_outputs)

#Model parameters
params = [W, b]
for param in params:
    param.attach_grad()

def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    norms = nd.sum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / norms

#loss function
def softmax_cross_entropy(yhat, y):
    return - nd.nansum(y * nd.log(yhat), axis=0, exclude=True)

def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = softmax(y_linear)
    return yhat

#Define the optimizer
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

#hyper parameters for the training
epochs = 100
learning_rate = .01
smoothing_constant = .01

#training
# for e in range(epochs):
#     #at the start of each epoch, the train data iterator is reset
#     train_data.reset()
#     for i, batch in enumerate(train_data):
#         data = batch.data[0].as_in_context(ctx)
#         label = batch.label[0].as_in_context(ctx)
#         label_one_hot = nd.one_hot(label, 4)
#         with autograd.record():
#             output = net(data)
#             loss = softmax_cross_entropy(output, label_one_hot)
#         loss.backward()
#         SGD(params, learning_rate)
#         curr_loss = nd.mean(loss).asscalar()
#         moving_loss = (curr_loss if ((i == 0) and (e == 0))
#                        else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
#
#     #the training and validation accuracies are computed
#     _,val_accuracy = evaluate_accuracy(val_data, net)
#     _,train_accuracy = evaluate_accuracy(train_data, net)
#     print("Epoch %s. Loss: %s, Train_acc %s, Val_acc %s" %
#           (e, moving_loss, train_accuracy, val_accuracy))


#model accuracy on the test dataset
# predictions,test_accuracy = evaluate_accuracy(test_data, net)
# output = np.vectorize(fizz_buzz)(np.arange(1, 101), predictions.asnumpy().astype(np.int))
# print(output)
# print("Test Accuracy : ", test_accuracy)

#import gluon
from mxnet import gluon

#reset the training, test and validation iterators
train_data.reset()
val_data.reset()
test_data.reset()

#Define number of neurons in each hidden layer
num_hidden = 64
#Define the sequential network
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_inputs, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))


#initialize parameters
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

#define the loss function
loss = gluon.loss.SoftmaxCrossEntropyLoss()

#Define the optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .02, 'momentum': 0.9})

#define variables/hyper-paramters
epochs = 100
moving_loss = 0.
best_accuracy = 0.
best_epoch = -1

#train the model
def train_gluon():
    global best_accuracy, best_epoch, moving_loss, epochs
    for e in range(epochs):
        train_data.reset()
        for i, batch in enumerate(train_data):
            data = batch.data[0].as_in_context(ctx)
            label = batch.label[0].as_in_context(ctx)
            with autograd.record():
                output = net(data)
                cross_entropy = loss(output, label)
                cross_entropy.backward()
            trainer.step(data.shape[0])
            if i == 0:
                moving_loss = nd.mean(cross_entropy).asscalar()
            else:
                moving_loss = .99 * moving_loss + .01 * nd.mean(cross_entropy).asscalar()

        _, val_accuracy = evaluate_accuracy(val_data, net)
        _, train_accuracy = evaluate_accuracy(train_data, net)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if best_epoch!=-1:
                print('deleting previous checkpoint...')
                os.remove('mlp-%d.params'%(best_epoch))
            best_epoch = e
            print('Best validation accuracy found. Checkpointing...')
            net.save_params('mlp-%d.params'%(e))
        print("Epoch %s. Loss: %s, Train_acc %s, Val_acc %s" %
              (e, moving_loss, train_accuracy, val_accuracy))

##train_gluon()

#Load the parameters
#net.load_params('mlp-%d.params'%(best_epoch), ctx)
# net.load_params('mlp-0.params', ctx)
#
# #predict on the test dataset
# predictions, test_accuracy = evaluate_accuracy(test_data, net)
# output = np.vectorize(fizz_buzz)(np.arange(101, 201), predictions.asnumpy().astype(np.int))
# print(output)
# print("Gluon Test Accuracy : ", test_accuracy)


#fizzbuzz keras1
#https://github.com/ad34/FizzBuzz-keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers.advanced_activations import ReLU,PReLU,ELU,Softmax,ThresholdedReLU #SReLU
from keras.optimizers import SGD,RMSprop,Adam,Adagrad,Adamax
from keras.utils import np_utils
from keras.callbacks import Callback,EarlyStopping
#
import numpy
#
num_digits = 10 # binary encode numbers
nb_classes = 4 # 4 classes : number/fizz/buzz/fizzbuzz
batch_size = 128
#
def fb_encode(i):
    if   i % 15 == 0: return [3]
    elif i % 5  == 0: return [2]
    elif i % 3  == 0: return [1]
    else:             return [0]
#
def bin_encode(i):
    return [i >> d & 1 for d in range(num_digits)]

def fizz_buzz_pred(i, pred):
    return [str(i), "fizz", "buzz", "fizzbuzz"][pred.argmax()]

def fizz_buzz(i):
    if   i % 15 == 0: return "fizzbuzz"
    elif i % 5  == 0: return "buzz"
    elif i % 3  == 0: return "fizz"
    else:             return str(i)

def create_dataset():
    dataX,dataY = [],[]
    for i in range(101,1024):
         dataX.append(bin_encode(i))
         dataY.append(fb_encode(i))

    return numpy.array(dataX), np_utils.to_categorical(numpy.array(dataY), nb_classes)


dataX,dataY = create_dataset()

class EarlyStopping(Callback):
    def __init__(self, monitor='accuracy', value=1.0, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


model = Sequential()
model.add(Dense(1000, input_shape=(num_digits,))) #64 #100% dropout?
#model.add(Dense(256, input_shape=(num_digits,)))
model.add(ReLU()) #SReLU()
model.add(Dropout(0.2)) #0.2
# model.add(Dense(128))
# model.add(ReLU()) #SReLU()
# model.add(Dropout(0.2)) #0.2
model.add(Dense(4))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(0.02, 0.9)) #RMSprop() SGD(0.02, 0.9)

#callbacks = [EarlyStopping(monitor='loss',value=1.193e-07,verbose=1)]
model.fit(dataX, dataY, nb_epoch=1000, batch_size=batch_size) #nb_epoch=10000
#print('Best SGD model params: ', model.losses.sort())

errors = 0
correct = 0
#
for i in range(1,101):
    x = bin_encode(i)
    y = model.predict(numpy.array(x).reshape(-1,10))
    print (fizz_buzz_pred(i,y))
    if fizz_buzz_pred(i,y) == fizz_buzz(i):
        correct = correct + 1
    else:
        errors = errors + 1

print("SGD Errors:", errors, " Correct:", correct)

#fizbuzz keras 2
#https://programmersought.com/article/46733523225/

from keras.models import Sequential
import numpy as np
from keras.layers import Dense


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return np.array([0, 0, 0, 1])
    elif i % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif i % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])


NUM_TRAIN = 10
x_train = np.array([binary_encode(i, NUM_TRAIN) for i in range(101, 2 ** NUM_TRAIN)])
y_train = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_TRAIN)])

x_test = np.array([binary_encode(i, NUM_TRAIN) for i in range(0, 100)])
y_test = np.array([fizz_buzz_encode(i) for i in range(0, 100)])

model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=10)) #units=1000, , input_dim=10
model.add(Dense(units=4, activation='softmax')) #units=4
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32)
loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
print('train:', loss_and_metrics[1])
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print('test:', loss_and_metrics[1])

from random import randrange
for i in range(1, 101):
    num = randrange(1000) ##num->i
    x = bin_encode(num) ##num->i
    y = model.predict(numpy.array(x).reshape(-1,10))
    print (fizz_buzz_pred(num, y)) ##num->i
    if fizz_buzz_pred(num, y) == fizz_buzz(num): #num->i
        correct = correct + 1
    else:
        errors = errors + 1

print("Errors :", errors, " Correct :", correct)

from keras.datasets import *


#
#
#
# #https://ardsite.medium.com/introduction-in-using-machine-learning-for-pattern-recognition-in-python-892104422df2
# from sklearn import tree
# from sklearn.datasets import load_iris
# from sklearn.metrics import accuracy_score
# import numpy
# #Preparing the data set - Loading the data via iris.data - Loading the descriptions of the data via iris.target
# #The names of the plant species can be retrieved via "iris_target_names". The names are stored as IDs (numbers) in "data".
# iris = load_iris()
# x_coordinate = iris.data
# y_coordinate =  iris.target
# plant_names = iris.target_names
# #Create random indexes used to retrieve the data in the iris dataset
# array_ids = numpy.random.permutation(len(x_coordinate))
# #In "train" the data is used for learning for the Machine Learning program.
# #In "real" the actual data is stored, which is used to check the predicted data.
# #The last 15 values are used for "real" for checking, the rest for "train".
# x_coordinate_train = x_coordinate[array_ids[:-15]]
# x_coordinate_real = x_coordinate[array_ids[-15:]]
# y_coordinate_train = y_coordinate[array_ids[:-15]]
# y_coordinate_real = y_coordinate[array_ids[-15:]]
# #Classify the data using a decision tree and train it with the previously created data.
# data_classification = tree.DecisionTreeClassifier()
# data_classification.fit(x_coordinate_train,y_coordinate_train)
# #Create predictions from existing data (in data set "real")
# prediction = data_classification.predict(x_coordinate_real)
# #Display the predicted names
# print(prediction)
# #The actual values
# print(y_coordinate_real)
# #Calculate the accuracy of the predicted data -
# # Method accuracy_score() gets the predicted value and the actual value returned
# print("Accuracy in percent: %.2f" %((accuracy_score(prediction,y_coordinate_real)) * 100))