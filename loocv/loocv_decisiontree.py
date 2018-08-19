from sklearn import tree
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import KFold
import itertools
import matplotlib.pyplot as plt

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

    return

def loocv_csv(training, testing, loocv_subject):
    with open(training, 'r') as f:
        train_data = [[float(val) for val in line.split(',')[:109]] for line in f]
    with open(testing, 'r') as f:
        test_data = [[float(val) for val in line.split(',')[:109]] for line in f]

    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)
    training_x, training_y, test_x, test_y = train_data[:,:108], train_data[:,108], test_data[:,:108], test_data[:,108]

    clf = tree.DecisionTreeClassifier()
    clf.fit(training_x, training_y)

    test_y_pred = clf.predict(test_x)


    ##### PLOT CONFUSION MATRIX ####



    ######## compute confusion matrix ########
    # Compute confusion matrix
    class_names = ['beats', 'point forward', 'point left', 'point right', 'raise hand', 'shrug', 'thumbs down', 'thumbs up', 'wave']

    cnf_matrix = confusion_matrix(test_y, test_y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='LOOCV %s Decision Tree Confusion Matrix, Without Normalization'%(loocv_subject))

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='LOOCV %s Decision Tree Normalized Confusion Matrix'%(loocv_subject))

    plt.show()

    return

loocv_csv('loocvs1training.csv', 'loocvs1testing.csv', 's1')
loocv_csv('loocvs2training.csv', 'loocvs2testing.csv', 's2')
loocv_csv('loocvs3training.csv', 'loocvs3testing.csv', 's3')
loocv_csv('loocvs4training.csv', 'loocvs4testing.csv', 's4')
loocv_csv('loocvs5training.csv', 'loocvs5testing.csv', 's5')
