import numpy as np 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import itertools

def plot_confusion_matrix(cm, classes=['NonTroll','Troll'],
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """ Prints out and plots a confusion matrix 

    Parameters: 
        cm (confusion matrix): output of sklearn.metrics.confusion_matrix function
        classes (list) (optional): list of classes, default is NonTroll and Troll 
        normalize (boolean) (optional): normalize labels if true
        title (string): title of the plot, default "Confusion Matrix"
        cmap (matplotlib colormap) (optional): default is plt.cm.Blues

    Returns:
        None
    """
    

    #normalize the labels if normalize parameter is True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print the confusion matrix
    print(cm)

    #format and display the confusion matrix
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
    plt.show();


def get_auc_score(y_test, y_score):
    """ Creates a receiver operating characteristic curve for y_test and y_score, and returns the area under the curve

    Parameters:
        y_test (series): class labels for test subset from train_test_split
        y_score (np.array): array of probabiliites of prediction for one of the classes 

    Return:
        auc_score (int): area under the ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    return auc(fpr,tpr)


def plot_roc(y_test, y_score):
    """ Plots a receiver operating characteristic curve (true positive rate against false positive rate at various thresholds)
    for y_test and y_scores

    Parameters:
        y_test (series): test class labels from train_test_split
        y_score (np.array): array of probabiliites of prediction for one of the classes
    """

    # use y_test and y_score to get false positive rates and true positive rates for different thresholds 
    fpr, tpr, thresholds = roc_curve(y_test, y_score)


    #print the area under the ROC curve based on false positive rates and true positive rates
    print('AUC: {}'.format(auc(fpr, tpr)))

    # format and display the ROC curve
    plt.figure(figsize=(10,8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)], fontsize = 12)
    plt.xticks([i/20.0 for i in range(21)], fontsize = 12)
    plt.xlabel('False Positive Rate', fontsize = 18)
    plt.ylabel('True Positive Rate', fontsize = 18)
    plt.title('Receiver operating characteristic (ROC) Curve', fontsize = 20)
    plt.legend(loc="lower right")
    plt.show();
