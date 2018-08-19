from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import KFold
import itertools
import matplotlib.pyplot as plt

#### 5 fold cross validation ####
with open('5subjects.csv', 'r') as f:
    data = [[float(val) for val in line.split(',')[:109]] for line in f]
data = np.asarray(data)
data_x, data_y = data[:,:108], data[:,108]

clf = SVC(kernel='linear')

kf = KFold(n_splits=5)
kf.get_n_splits(data_x)
print(kf)

test_y_pred = []
test_y_pred = np.asarray(test_y_pred)
test_y_true = []
test_y_true = np.asarray(test_y_true)

#will loop 5 times. train, test hold indices to be used.
for train, test in kf.split(data_x):
    f = 0
    j = 0
    for index in train:
        if (f==0):
            train_x = data[index,:108]
            train_y = data[index,108]
            f=1
        else:
            train_x = np.vstack((train_x, data[index,:108]))
            train_y = np.vstack((train_y, data[index, 108]))
    for index in test:
        if (j==0):
            test_x = data[index,:108]
            test_y = data[index, 108]
            j=1
        else:
            test_x = np.vstack((test_x, data[index,:108]))
            test_y = np.vstack((test_y, data[index, 108]))
    clf.fit(train_x, train_y)
    test_y_true = np.append(test_y_true, test_y)
    test_y_pred = np.append(test_y_pred, clf.predict(test_x))


##### PLOT CONFUSION MATRIX ####

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

######## compute confusion matrix ########
# Compute confusion matrix
class_names = ['beats', 'point forward', 'point left', 'point right', 'raise hand', 'shrug', 'thumbs down', 'thumbs up', 'wave']

cnf_matrix = confusion_matrix(test_y_true, test_y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='SVM confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='SVM Normalized Confusion Matrix')

plt.show()
