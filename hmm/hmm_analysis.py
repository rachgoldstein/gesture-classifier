from __future__ import division
import os
import scipy.io
import numpy as np
from sklearn.svm import SVC # "Support vector classifier"
from hmmlearn import hmm
from sklearn.utils import check_array
import timeit
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_training(path, marker):
    os.chdir(path)
    filenames = []
    count=0
    for filename in os.listdir(path):
        if (filename.startswith("S1") or filename.startswith("S2")):
            filenames.append(filename)
    data = []
    with open(filenames[0], 'r') as f:
        try:
            data = [[float(val) for val in line.split(',')[:6]] for line in f] #columns in dataset
        except ValueError:
            pass
    data = np.asarray(data)

    i = 0
    for f in filenames:
        print(f)
        if i==0:
            i=1
            continue
        else:
            with open(f, 'r') as file:
                filex = [[float(val) for val in line.split(',')[:6]] for line in file]
            data = np.concatenate((data, filex), axis=0)

    for row in data:
        row[0] = marker

    train_x = train_y = []
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_x = data[:,1:6]
    train_y = data[:,0]

    #np.savetxt("trainx.csv", train_x)
    #np.savetxt("trainy.csv", train_y)
    #np.savetxt("train_data.csv", data)
    #print("training data saved!")

    return(train_x, train_y)

def create_testing(path, marker):
    os.chdir(path)
    testing_filenames = []

    for filename in os.listdir(path):
        if (filename.startswith("S1") or filename.startswith("S2")):
            testing_filenames.append(filename)

    testing = []
    with open(testing_filenames[0], 'r') as f:
        try:
            testing = [[float(val) for val in line.split(',')[:6]] for line in f] #columns in dataset
        except ValueError:
            pass
    testing = np.asarray(testing)

    i=0
    filey = []
    for f in testing_filenames:
        if i==0:
            i=1
            continue
        else:
            with open(f, 'r') as file:
                try:
                    filey = [[float(val) for val in line.split(',')[:6]] for line in file]
                except ValueError:
                    pass  # do nothing!
            testing = np.concatenate((testing, filey), axis=0)

    for row in testing:
        row[0] = marker

    test_x = test_y = []
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)
    test_x = testing[:,1:6]
    test_y = testing[:,0]

    return(test_x, test_y)


def get_accuracy(matrix, true_y, y_true, y_predict):
    count=0
    correct=0
    for row in matrix:
        predictions = []
        for model in models:
            prediction = model.decode(row.reshape(1,-1), algorithm="viterbi")
            predictions.append(prediction)
        maxpos = predictions.index(max(predictions))
        y_true.append(true_y[count])
        y_predict.append(maxpos)
        if maxpos == true_y[count]:
            correct+=1
        count+=1

    accuracy = correct/count
    print(accuracy)
    return(accuracy)

start_time = timeit.default_timer()
gestures = ['beats','pointf', 'pointl', 'pointr', 'raise', 'shake', 'shrug', 'thumbsdown', 'thumbsup','wave']
i=0
all_training = []

for gesture in gestures:
    data = create_training(('/home/robotlab/classifiers/hmm/csv/training/%s'%(gesture)), i)
    all_training.append(data)
    i+=1

i=0
all_testing = []
for gesture in gestures:
    data = create_testing(('/home/robotlab/classifiers/hmm/csv/testing/%s'%(gesture)), i)
    all_testing.append(data)
    i+=1

models = []
for i in range(0,9):
    model = hmm.GaussianHMM(n_components=1, covariance_type="full", n_iter=100).fit(all_training[i][0])
    models.append(model)


training_accuracies = []
print("training accuracies: ")
training_y_pred = []
training_y_true = []
for i in range(0,9):
    accuracy = get_accuracy(all_training[i][0], all_training[i][1], training_y_true, training_y_pred)
    training_accuracies.append(accuracy)
elapsed = timeit.default_timer() - start_time
print("total training time: ", elapsed)
train_conf_matrix = confusion_matrix(training_y_true, training_y_pred)
print("training confusion matrix: ", train_conf_matrix)
plt.show()

start_time = timeit.default_timer()
testing_accuracies = []
testing_y_pred = []
testing_y_true = []
print("testing accuracies: ")
for i in range(0,9):
    accuracy = get_accuracy(all_testing[i][0], all_testing[i][1], testing_y_true, testing_y_pred)
    testing_accuracies.append(accuracy)
elapsed = timeit.default_timer() - start_time
print("total testing time: ", elapsed)
test_conf_matrix =  confusion_matrix(testing_y_true, testing_y_pred)
print("testing confusion matrix: ", test_conf_matrix)
#plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
#cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_conf_matrix, classes=gestures,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_conf_matrix, classes=gestures, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
