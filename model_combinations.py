




def predict(function,weight=1,nt_pred=nt_test_pred,tf_pred=tf_test_pred,tfidf_pred=tfidf_test_pred,y_test=y_test):
    tf_votes = function(nt_pred*weight,tf_pred)
    tfidf_votes = function(nt_pred*weight,tfidf_pred)
    print('TF Values:')
    plot_confusion_matrix(confusion_matrix(y_test,tf_votes))
    
    print('TFIDF Values:')
    plot_confusion_matrix(confusion_matrix(y_test,tfidf_votes))
    
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

def decision_func(weight=1, non_text=nt_test_pred,text=tfidf_test_pred):
    confidences = []
    for nt_pred,t_pred in zip(non_text,text):
        vote = nt_pred+t_pred
        confidences.append(vote[1])
    return confidences
    