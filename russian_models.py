import numpy as np
from sklearn.metrics import confusion_matrix
import plot
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score


def strut_models(classifiers,data_split,output=False):
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
    tfidf_votes = function(nt_pred*weight,tfidf_pred)
    
    print('Combined Votes:')
    plot.plot_confusion_matrix(confusion_matrix(y_test,tfidf_votes))
    
def pred_geometric(non_text,text):
    votes = []
    for nt_pred,t_pred in zip(non_text,text):
        vote_null = (nt_pred[0]*t_pred[0])**.5
        vote_troll = (nt_pred[1]*t_pred[1])**.5
        votes.append(0 if vote_null>vote_troll else 1)
    return votes 

def pred_harmonic(non_text,text):
    votes = []
    for nt_pred,t_pred in zip(non_text,text):
        vote_null = 2*nt_pred[0]*t_pred[0]/(nt_pred[0]+t_pred[0])
        vote_troll = 2*nt_pred[1]*t_pred[1]/(nt_pred[1]+t_pred[1])
        votes.append(0 if vote_null>vote_troll else 1)
    return votes    

def pred_confidence(non_text,text):
    votes = []
    for nt_pred,t_pred in zip(non_text,text):
        vote_text = t_pred[1]-t_pred[0]
        vote_nt = nt_pred[1]-nt_pred[0]
        votes.append(1 if vote_text+vote_nt>0 else 0)
    return votes  

def pred_max(non_text,text):
    votes = []
    for nt_pred,t_pred in zip(non_text,text):
        vote = nt_pred+t_pred
        votes.append(0 if vote[0]>vote[1] else 1)
    return votes    

def decision_func(non_text,text,weight=1):
    confidences = []
    for nt_pred,t_pred in zip(non_text,text):
        vote = nt_pred+t_pred
        confidences.append(vote[1])
    return confidences
    