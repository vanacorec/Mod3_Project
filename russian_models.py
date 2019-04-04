import numpy as np
import plot
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score


def strut_models(classifiers,data_split,output=False):
    """ Takes in a list of models and training and testing data, 
    fits each model on the training data, and prints out the classification report and confusion matrix

    Parameters:
        classifiers (list): list of model objects
        data_split (list): list of outputs from train_test_split [X_train, X_test, y_train, y_test]

    Returns:
        Prints the classification report and confusion matrix for each model.
        if the output parameter is True:
            model (model): fitted input model if output is true
            prob (np.array): array of two columns, each col is probabilities of predictions for a class
        else:
            None
    """
    for classifier in classifiers:
        model = classifier.fit(data_split[0],data_split[2])
        prediction = classifier.predict(data_split[1])
        print(classifier)
        print(classification_report(data_split[3],prediction))
        print('\n')
        plot.plot_confusion_matrix(confusion_matrix(data_split[3],prediction))
    if output:
        prob = model.predict_proba(data_split[1])
        return model, prob

def predict(function,nt_pred,tfidf_pred,y_test,weight=1):
    """ Takes in arrays of probabilites of class predictions for 2 different models, combines them based on 
    the input function, and prints out the classification report and plots the confusion matrix

    Parameters:
        function (function): function that determines how probability predictions will be combined
        nt_pred (np.array): probabilities of prediction for each class from one model (non-text)
        tfidf_pred (np.array):  probabilities of prediction for each class from another model (text)
        y_test (series): class label values for test data
        weight (int) (optional): number by which to multiply nt_pred values to weight them more or less,
            Default 1
    Returns:
        None
    """

    votes = function(nt_pred*weight,tfidf_pred)
    print('Combined Votes:')
    print(classification_report(y_test, votes))
    plot.plot_confusion_matrix(confusion_matrix(y_test,votes))


"""
For the following functions (pred_* ):
    Parameters: 
        non_text (np.array): 2 column array, first and second columns are predicted probabilities of the corresponding row being a nontroll 
            and troll, respectively. Array is output of model.predict_proba(y_test), where model is fit with non text data
        text (np.array): 2 column array, first and second columns are predicted probabilities of the corresponding row being a nontroll 
            and troll, respectively. Array is output of model.predict_proba(y_test), where model is fit with text data


    Returns:
        votes (np.array): 1 column array with final predictions of nontroll (0) or troll (1) for each row
"""

    
def pred_geometric(non_text,text):
    """ 
    Combines the probabilities from two models by taking the mean of non-troll probabilities for each row 
    and mean of troll probabilities for each row, and returning an array with 0 if non-troll mean is higher
    or 1 if lower
    """
    votes = []
    for nt_pred,t_pred in zip(non_text,text):
        vote_null = (nt_pred[0]*t_pred[0])**.5
        vote_troll = (nt_pred[1]*t_pred[1])**.5
        votes.append(0 if vote_null>vote_troll else 1)
    return votes 

def pred_harmonic(non_text,text):
    """ 
    Combines the probabilities from two models by taking the harmonic mean of non-troll probabilities for each row 
    and harmonic mean of troll probabilities for each row, and returning an array with 0 if non-troll mean is higher
    or 1 if lower
    """
    votes = []
    for nt_pred,t_pred in zip(non_text,text):
        vote_null = 2*nt_pred[0]*t_pred[0]/(nt_pred[0]+t_pred[0])
        vote_troll = 2*nt_pred[1]*t_pred[1]/(nt_pred[1]+t_pred[1])
        votes.append(0 if vote_null>vote_troll else 1)
    return votes    

def pred_confidence(non_text,text):
    """ 
    Combines the probabilities from two models by determinining the differences in probabilities  
    between the two models for each class, and setting the final vote of whichever model is more confident
    """
    votes = []
    for nt_pred,t_pred in zip(non_text,text):
        vote_text = t_pred[1]-t_pred[0]
        vote_nt = nt_pred[1]-nt_pred[0]
        votes.append(1 if vote_text+vote_nt>0 else 0)
    return votes  

def pred_max(non_text,text):
    """ 
    Combines predictionprobabilities from two models by adding non-troll probabilities for each row together 
    and the troll probabalities for each row and using whichever sum is greater as the prediction
    """

    votes = []
    for nt_pred,t_pred in zip(non_text,text):
        vote = nt_pred+t_pred
        votes.append(0 if vote[0]>vote[1] else 1)
    return votes    

def decision_func(non_text,text):
    """
    Combines predictions of two models by summing predictions for each class and outputs total predictions
    for one of the classes, to be used to as y_cores in the ROC curve 
    """
    confidences = []
    for nt_pred,t_pred in zip(non_text,text):
        vote = nt_pred+t_pred
        confidences.append(vote[1])
    return confidences
    